"""Quality gate: prevent propagating clearly bad products downstream.

This module is intentionally lightweight and does **not** depend on any UI.

The contract:
  - selected stages write a stage-local done JSON that includes ``qc.flags``
    (each flag has at least ``code``, ``severity`` and ``message``).
  - before running a downstream stage, the runner can optionally stop if any
    upstream stage has flags with severity >= ERROR.

The goal is not to be perfect; it is to be *fail-fast* and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import json
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.stage_registry import STAGES
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir


log = logging.getLogger(__name__)


_SEV_ORDER = {
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3,
}


def normalize_severity(sev: str | None) -> str:
    s = (sev or "").strip().upper()
    if not s:
        return "INFO"
    if s in _SEV_ORDER:
        return s
    # common aliases
    if s in {"BAD", "FAIL", "FATAL", "CRIT", "CRITICAL"}:
        return "ERROR"
    if s in {"WARNING"}:
        return "WARN"
    if s in {"OK"}:
        return "INFO"
    return "INFO"


def max_severity(flags: Iterable[dict[str, Any]] | None) -> str:
    best = "INFO"
    for f in flags or []:
        s = normalize_severity(str(f.get("severity") or ""))
        if _SEV_ORDER.get(s, 0) > _SEV_ORDER.get(best, 0):
            best = s
    return best


@dataclass(frozen=True)
class QCGateError(RuntimeError):
    """Raised when the QC gate blocks running a stage."""

    task: str
    blockers: list[dict[str, Any]]
    upstream_max_severity: str

    def summary(self, max_items: int = 8) -> str:
        items = self.blockers[:max_items]
        lines = [
            f"QC gate blocked task '{self.task}' (upstream max severity: {self.upstream_max_severity}).",
            "Blocking flags:",
        ]
        for b in items:
            st = str(b.get("stage") or "?")
            code = str(b.get("code") or "?")
            sev = normalize_severity(str(b.get("severity") or ""))
            msg = str(b.get("message") or "")
            lines.append(f" - [{sev}] {st}: {code} — {msg}")
        if len(self.blockers) > max_items:
            lines.append(f" ... and {len(self.blockers) - max_items} more")
        return "\n".join(lines)


def _stage_key_order() -> list[str]:
    """Ordered stage keys (excluding UI-only stages)."""
    keys: list[str] = []
    for s in STAGES:
        if getattr(s, "ui_only", False):
            continue
        keys.append(str(s.key))
    return keys


def _task_to_stage_key(task: str) -> str | None:
    """Map runner tasks to stage keys.

    Most tasks share the same identifier as the stage key.
    """
    t = (task or "").strip().lower()
    if not t:
        return None

    # runner task aliases
    if t == "lineid_prepare":
        return "arc_line_id"
    if t in {"extract", "extract1d"}:
        return "extract1d"
    # default: same as task
    return t


def _find_done_json(cfg: dict[str, Any], stage_key: str) -> Path | None:
    wd = resolve_work_dir(cfg)

    if stage_key == "wavesolution":
        base = wavesol_dir(cfg)
    else:
        try:
            base = stage_dir(wd, stage_key)
        except Exception:
            return None

    cand = [
        base / "done.json",
        base / f"{stage_key}_done.json",
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def _read_done_flags(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}

    qc = data.get("qc") if isinstance(data.get("qc"), dict) else {}
    flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []
    out: list[dict[str, Any]] = []
    for f in flags:
        if isinstance(f, dict):
            out.append(dict(f))
    return out, data if isinstance(data, dict) else {}


def check_qc_gate(
    cfg: dict[str, Any],
    *,
    task: str,
    allow_override: bool = False,
) -> None:
    """Raise :class:`QCGateError` if upstream QC blocks running *task*.

    Parameters
    ----------
    cfg:
        Parsed config dict.
    task:
        Runner task name.
    allow_override:
        If True, the runner *requests* bypassing ERROR blockers.
        For safety, bypassing errors is only enabled when the config contains
        ``qc.allow_override_errors: true``.
    """
    # Override policy: allow bypassing blockers entirely.
    # (Older versions distinguished FATAL; P2 standardizes to INFO/WARN/ERROR.)
    cur_stage = _task_to_stage_key(task)
    if not cur_stage:
        return

    qc_cfg = cfg.get("qc") if isinstance(cfg.get("qc"), dict) else {}
    allow_override_errors = bool(qc_cfg.get("allow_override_errors", False))
    if allow_override and not allow_override_errors:
        log.warning(
            "QC gate override requested for task '%s' but qc.allow_override_errors is false; keeping strict gate.",
            task,
        )

    ordered = _stage_key_order()
    if cur_stage not in ordered:
        return

    idx = ordered.index(cur_stage)
    upstream = ordered[:idx]

    blockers: list[dict[str, Any]] = []
    for sk in upstream:
        done = _find_done_json(cfg, sk)
        if not done:
            continue
        flags, _ = _read_done_flags(done)
        for f in flags:
            sev = normalize_severity(str(f.get("severity") or ""))
            if allow_override and allow_override_errors:
                continue
            if _SEV_ORDER.get(sev, 0) >= _SEV_ORDER["ERROR"]:
                x = dict(f)
                x.setdefault("stage", sk)
                x.setdefault("severity", sev)
                blockers.append(x)

    if blockers:
        raise QCGateError(
            task=task,
            blockers=blockers,
            upstream_max_severity=max_severity(blockers),
        )
