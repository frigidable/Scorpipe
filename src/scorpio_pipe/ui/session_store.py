"""Backward-compatible re-export for GUI session persistence.

P0-ARCH-12
----------
UI code must not import pipeline internals directly.
The frozen boundary is :mod:`scorpio_pipe.pipeline.public_api`.

This module remains as a stable import path for already-shipped UI code.
"""

from __future__ import annotations

from scorpio_pipe.pipeline.public_api import (  # noqa: F401
    load_session,
    save_session,
    session_path,
    snapshot,
    update_stage,
)

__all__ = [
    "load_session",
    "save_session",
    "session_path",
    "snapshot",
    "update_stage",
]


# -----------------------------------------------------------------------------
# PERF-10.4: UI caches for "late stages" (plan/status) to avoid redoing expensive
# work on every click. Best-effort, invalidated by manifest mtimes.
# -----------------------------------------------------------------------------

import json
import os
import hashlib
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _file_sig_ns(p: Path) -> tuple[int, int]:
    """(mtime_ns, size) or (0,0) if missing."""
    try:
        st = p.stat()
        return (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))
    except Exception:
        return (0, 0)


def _norm_status(s: Any) -> str:
    x = str(s or "").strip().lower()
    return {
        "failed": "fail",
        "failure": "fail",
        "skip": "skipped",
        "skipped": "skipped",
        "canceled": "cancelled",
        "cancelled": "cancelled",
    }.get(x, x or "unknown")


def _stable_cfg_hash(cfg: Mapping[str, Any]) -> str:
    """Hash a config mapping for cache keys (best-effort, must never raise)."""
    try:
        payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
    except Exception:
        return "cfg-unknown"


@dataclass
class UiRunCache:
    """Per-workdir UI cache (plan + manifest-derived statuses)."""

    work_dir: Path
    _manifest_sig: tuple[tuple[int, int], tuple[int, int]] = field(default_factory=lambda: ((0, 0), (0, 0)))
    _plan_cache: dict[tuple, Any] = field(default_factory=dict)
    _status_cache: dict[str, Any] | None = None

    def _current_manifest_sig(self) -> tuple[tuple[int, int], tuple[int, int]]:
        st = self.work_dir / "manifest" / "stage_state.json"
        mt = self.work_dir / "manifest" / "metrics.json"
        return (_file_sig_ns(st), _file_sig_ns(mt))

    def invalidate_if_needed(self) -> None:
        sig = self._current_manifest_sig()
        if sig != self._manifest_sig:
            self._manifest_sig = sig
            self._plan_cache.clear()
            self._status_cache = None

    def get_stage_statuses(self) -> dict[str, Any]:
        """Return a snapshot mapping stage->info (best-effort).

        Uses only small manifest files (metrics.json + stage_state.json), and falls
        back to per-task done.json if a task is missing from those summaries.
        """
        self.invalidate_if_needed()
        if self._status_cache is not None:
            return dict(self._status_cache)

        out: dict[str, Any] = {}

        # 1) metrics.json is the preferred, UI-friendly summary (contains status+artifacts+metrics).
        mp = self.work_dir / "manifest" / "metrics.json"
        if mp.exists():
            try:
                js = json.loads(mp.read_text(encoding="utf-8"))
                stages = (js or {}).get("stages") or {}
                if isinstance(stages, dict):
                    for k, v in stages.items():
                        if not isinstance(v, dict):
                            continue
                        out[str(k)] = {
                            "status": _norm_status(v.get("status")),
                            "hash": v.get("hash"),
                            "updated_at": v.get("updated_at"),
                            "metrics": v.get("metrics") if isinstance(v.get("metrics"), dict) else None,
                        }
            except Exception:
                pass

        # 2) stage_state.json is a lightweight runner state (status/hash/message/trace).
        sp = self.work_dir / "manifest" / "stage_state.json"
        if sp.exists():
            try:
                js = json.loads(sp.read_text(encoding="utf-8"))
                stages = (js or {}).get("stages") or {}
                if isinstance(stages, dict):
                    for k, v in stages.items():
                        if not isinstance(v, dict):
                            continue
                        kk = str(k)
                        if kk not in out:
                            out[kk] = {}
                        out[kk].setdefault("status", _norm_status(v.get("status")))
                        out[kk].setdefault("hash", v.get("hash"))
                        out[kk].setdefault("updated_at", v.get("updated_at"))
                        out[kk].setdefault("message", v.get("message"))
            except Exception:
                pass

        # 3) done.json fallback for task-like directories not covered by manifest summaries.
        # Intentionally avoid importing StageCatalog here: UI code must stay behind
        # the frozen public boundary.
        try:
            candidate_dirs: list[Path] = []
            for special in (self.work_dir / "manifest", self.work_dir / "qc", self.work_dir / "ui"):
                if special.is_dir():
                    candidate_dirs.append(special)
            for d in sorted(self.work_dir.iterdir()):
                if not d.is_dir():
                    continue
                if d.name in {"manifest", "qc", "ui", "raw", "resources", "compare"}:
                    continue
                candidate_dirs.append(d)

            for d in candidate_dirs:
                p = d / "done.json"
                if not p.exists():
                    continue
                key = str(d.name)
                if "_" in key[:4]:
                    key = key.split("_", 1)[1]
                if key in out:
                    continue
                try:
                    js = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(js, dict):
                        out[key] = {
                            "status": _norm_status(js.get("status")),
                            "updated_at": js.get("updated_at") or js.get("created_utc"),
                            "message": js.get("error_message") or js.get("message"),
                        }
                except Exception:
                    continue
        except Exception:
            pass

        self._status_cache = dict(out)
        return dict(out)

def plan_sequence_cached(
        self,
        cfg: Mapping[str, Any],
        tasks: Iterable[str],
        *,
        resume: bool = True,
        force: bool = False,
        config_path: Path | None = None,
    ):
        """Cached wrapper for pipeline.plan_sequence() (can be expensive).

        Cache key includes:
        - manifest signature (stage_state/metrics mtimes) via invalidate_if_needed()
        - task list + flags
        - config signature: config_path (mtime,size) if available, else cfg hash
        """
        self.invalidate_if_needed()

        tasks_t = tuple([str(t) for t in tasks])
        cfg_sig: tuple[str, Any]
        if config_path is not None:
            cfg_sig = ("path", str(Path(config_path).expanduser().resolve()), _file_sig_ns(Path(config_path)))
        else:
            cfg_sig = ("hash", _stable_cfg_hash(cfg))

        key = (tasks_t, bool(resume), bool(force), cfg_sig)
        if key in self._plan_cache:
            return self._plan_cache[key]

        # Import lazily: plan_sequence pulls in the engine + reference context.
        from scorpio_pipe.pipeline.public_api import plan_sequence as _plan_sequence  # noqa: PLC0415

        plan = _plan_sequence(dict(cfg), tasks_t, resume=resume, force=force, config_path=config_path)
        self._plan_cache[key] = plan
        return plan


_UI_RUN_CACHES: "OrderedDict[str, UiRunCache]" = OrderedDict()
_UI_RUN_CACHES_MAX = 8


def get_ui_run_cache(work_dir: str | Path) -> UiRunCache:
    """Get (or create) UI cache for a work_dir. Keeps a small LRU to avoid leaks."""
    wd = Path(work_dir).expanduser().resolve()
    key = str(wd)
    c = _UI_RUN_CACHES.get(key)
    if c is not None:
        # LRU bump
        try:
            _UI_RUN_CACHES.move_to_end(key)
        except Exception:
            pass
        return c
    c = UiRunCache(work_dir=wd)
    _UI_RUN_CACHES[key] = c
    try:
        while len(_UI_RUN_CACHES) > int(_UI_RUN_CACHES_MAX):
            _UI_RUN_CACHES.popitem(last=False)
    except Exception:
        pass
    return c


__all__ = list(__all__) + ["UiRunCache", "get_ui_run_cache"]
