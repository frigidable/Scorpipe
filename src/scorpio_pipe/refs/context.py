"""Reference context manifest (CRDS-lite).

The goal is to make science runs reproducible when reference resources change.
We capture a frozen set of references (linelists, atlases, etc.) with their
content hashes and compute a stable ``context_id``.

Contract (P0-B3)
----------------
* ``manifest/reference_context.json`` is obligatory.
* ``context_id`` is a stable hash derived from the *content hashes* of
  references (not their absolute paths).
* The engine includes ``context_id`` in stage hashes so changes trigger reruns.
* ``context_id`` is attached to stage ``done.json`` and surfaced in QC report.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.version import PIPELINE_VERSION

from .store import ReferenceResolution, file_hash, reference_stat, resolve_reference


@dataclass(frozen=True)
class ReferenceEntry:
    """One entry in reference_context.json."""

    name: str  # logical name, e.g. "wavesol.linelist_csv"
    requested: str
    resolved_path: Path
    source: str
    sha256: str
    size: int | None = None
    mtime_utc: str | None = None
    version: str | None = None

    def to_dict(self, *, work_dir: Path | None = None) -> dict[str, Any]:
        p = self.resolved_path
        resolved = str(p)
        if work_dir is not None:
            try:
                resolved = str(p.resolve().relative_to(work_dir.resolve())).replace("\\", "/")
            except Exception:
                resolved = str(p)
        return {
            "name": self.name,
            "requested": self.requested,
            "resolved": resolved,
            "source": self.source,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_utc": self.mtime_utc,
            "version": self.version,
        }


def reference_context_id(entries: Iterable[ReferenceEntry]) -> str:
    """Compute a stable context id.

    The id must be machine-independent. We therefore hash only logical names and
    content hashes.
    """

    rows = [{"name": e.name, "sha256": e.sha256} for e in entries]
    rows = sorted(rows, key=lambda r: str(r.get("name")))
    s = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _parse_text_version(path: Path) -> str | None:
    """Best-effort version extraction from small text refs.

    We intentionally avoid heavy parsing. For CSV-like resources, we accept a
    "# version:" / "# VERSION:" comment line in the first ~40 lines.
    """

    try:
        if path.suffix.lower() not in (".csv", ".txt", ".yaml", ".yml", ".json", ".md"):
            return None
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
        for ln in lines:
            s = ln.strip().lstrip("#").strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("version"):
                # version: 2024-01
                parts = s.split(":", 1)
                if len(parts) == 2:
                    v = parts[1].strip()
                    return v[:120] if v else None
        return None
    except Exception:
        return None


def _resolve_ref(
    requested: str,
    *,
    resources_dir: str | Path | None,
    work_dir: Path,
    config_dir: str | Path | None,
    project_root: str | Path | None,
    allow_package: bool = True,
) -> ReferenceResolution:
    return resolve_reference(
        requested,
        resources_dir=resources_dir,
        work_dir=work_dir,
        config_dir=config_dir,
        project_root=project_root,
        allow_package=allow_package,
    )


def _gather_requests(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (logical_name, requested_path) pairs.

    Keep this minimal and explicit: only references that materially affect
    science results or interactive calibration.
    """

    wcfg = cfg.get("wavesol") if isinstance(cfg.get("wavesol"), dict) else {}
    wcfg = wcfg or {}

    # Linelist CSV: new key wavesol.linelist_csv; legacy neon_lines_csv.
    lamp_type = str(wcfg.get("lamp_type") or wcfg.get("lamp") or "").strip()
    if not lamp_type:
        # Default aligns with historical pipeline behavior.
        lamp_type = "Ne"

    try:
        from scorpio_pipe.lamp_contract import choose_linelist_name

        default_ll = choose_linelist_name(lamp_type)
    except Exception:
        default_ll = "neon_lines.csv"

    linelist_req = (
        str(wcfg.get("linelist_csv") or wcfg.get("line_list_csv") or wcfg.get("neon_lines_csv") or "").strip()
        or default_ll
    )
    reqs: list[tuple[str, str]] = [("wavesol.linelist_csv", linelist_req)]

    # Atlas PDF for interactive checks (UI): optional override.
    atlas_req = str(wcfg.get("atlas_pdf") or "").strip() or "HeNeAr_atlas.pdf"
    reqs.append(("wavesol.atlas_pdf", atlas_req))

    # Instrument DB is shipped with the package (affects auto-config / metadata).
    reqs.append(("instrument_db", "instruments/scorpio_instruments.yaml"))

    return reqs


def build_reference_context(
    cfg: dict[str, Any],
    resources_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a reference context payload.

    Raises
    ------
    FileNotFoundError
        If a requested reference cannot be found.
    OSError
        If a reference cannot be read/hashed.
    """

    work_dir = resolve_work_dir(cfg)
    config_dir = cfg.get("config_dir")
    project_root = cfg.get("data_dir") or cfg.get("project_root")

    entries: list[ReferenceEntry] = []
    for logical, requested in _gather_requests(cfg):
        rr = _resolve_ref(
            requested,
            resources_dir=resources_dir,
            work_dir=work_dir,
            config_dir=config_dir,
            project_root=project_root,
            allow_package=True,
        )
        h = file_hash(rr.resolved_path)
        st = reference_stat(rr.resolved_path)
        ver = _parse_text_version(rr.resolved_path)
        entries.append(
            ReferenceEntry(
                name=str(logical),
                requested=str(requested),
                resolved_path=Path(rr.resolved_path),
                source=str(rr.source),
                sha256=h,
                size=st.get("size"),
                mtime_utc=st.get("mtime_utc"),
                version=ver,
            )
        )

    cid = reference_context_id(entries)

    payload: dict[str, Any] = {
        "schema": "scorpio_pipe.reference_context",
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context_id": cid,
        "references": [e.to_dict(work_dir=work_dir) for e in entries],
    }
    return payload


def reference_context_path(work_dir: str | Path) -> Path:
    wd = Path(work_dir)
    d = wd / "manifest"
    d.mkdir(parents=True, exist_ok=True)
    return d / "reference_context.json"


def ensure_reference_context(
    cfg: dict[str, Any],
    *,
    resources_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ensure ``manifest/reference_context.json`` exists and is up to date.

    Returns the context payload.
    """

    work_dir = resolve_work_dir(cfg)
    out_path = reference_context_path(work_dir)

    ctx = build_reference_context(cfg, resources_dir=resources_dir)

    # Write if missing or changed.
    try:
        old = None
        if out_path.exists():
            try:
                old = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                old = None
        if not isinstance(old, dict) or old.get("context_id") != ctx.get("context_id"):
            atomic_write_json(out_path, ctx, indent=2, ensure_ascii=False)
    except Exception:
        # If we cannot write, fail loudly: provenance is a hard contract for B3.
        raise

    return ctx


def load_reference_context(work_dir: str | Path) -> dict[str, Any] | None:
    p = reference_context_path(work_dir)
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
