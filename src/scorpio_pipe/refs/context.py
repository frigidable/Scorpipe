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
from dataclasses import dataclass, field
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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "metadata": dict(self.metadata),
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


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s or "").strip()).strip("_") or "default"


def _parse_text_header_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        if path.suffix.lower() not in (".csv", ".txt", ".yaml", ".yml", ".json", ".md"):
            return out
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
        for ln in lines:
            s = ln.strip()
            if not s.startswith("#"):
                continue
            s = s.lstrip("#").strip()
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            kk = k.strip().lower()
            vv = v.strip()
            if kk and vv:
                out[kk] = vv
    except Exception:
        return out
    return out


def _extract_reference_metadata(logical_name: str, path: Path) -> dict[str, Any]:
    logical = str(logical_name or "")
    hdr = _parse_text_header_metadata(path)
    meta: dict[str, Any] = {}

    if logical.startswith("standards.") and logical.endswith(".table"):
        meta["reference_kind"] = "standard_table"
        meta["source_name"] = hdr.get("standard_name") or logical.split(".")[1]
        meta.setdefault("interpolation_assumption", "linear_in_wavelength")
    elif logical.startswith("standards.library"):
        meta["reference_kind"] = "standard_library"
        meta["source_name"] = path.name
        meta.setdefault("interpolation_assumption", "lookup_only")
    elif logical.startswith("extinction_curve."):
        meta["reference_kind"] = "extinction_curve"
        meta["source_name"] = logical.split(".", 1)[1]
        meta.setdefault("interpolation_assumption", "linear_in_wavelength")
    elif logical.startswith("telluric_mask."):
        meta["reference_kind"] = "telluric_mask"
        meta["source_name"] = logical.split(".", 1)[1]
        meta.setdefault("interpolation_assumption", "interval_mask_no_interp")
    elif logical.startswith("telluric_template."):
        meta["reference_kind"] = "telluric_template"
        meta["source_name"] = logical.split(".", 1)[1]
        meta.setdefault("interpolation_assumption", "linear_in_wavelength")

    if not meta:
        return {}

    for src_key, dst_key in (("units", "units"), ("wavelength_convention", "wavelength_convention"), ("wavelength_ref", "wavelength_convention"), ("sampling", "sampling"), ("standard_name", "standard_name"), ("version", "version")):
        if hdr.get(src_key):
            meta[dst_key] = hdr[src_key]

    if meta.get("reference_kind") == "standard_table":
        meta.setdefault("units", "f_lambda")
        meta.setdefault("unit_family", "flux_density")
        meta.setdefault("wavelength_convention", "air")
        meta.setdefault("sampling", "tabulated")
    elif meta.get("reference_kind") == "extinction_curve":
        meta.setdefault("units", "mag_per_airmass")
        meta.setdefault("unit_family", "extinction")
        meta.setdefault("wavelength_convention", "air")
        meta.setdefault("sampling", "tabulated")
    elif meta.get("reference_kind") == "telluric_mask":
        meta.setdefault("units", "transmission_mask")
        meta.setdefault("unit_family", "mask")
        meta.setdefault("wavelength_convention", "air")
        meta.setdefault("sampling", "interval_list")
    elif meta.get("reference_kind") == "telluric_template":
        meta.setdefault("units", "relative_transmission")
        meta.setdefault("unit_family", "transmission")
        meta.setdefault("wavelength_convention", "air")
        meta.setdefault("sampling", "tabulated")
    elif meta.get("reference_kind") == "standard_library":
        meta.setdefault("unit_family", "catalog")
        meta.setdefault("sampling", "lookup")

    meta.setdefault("checksum_algorithm", "sha256")
    meta.setdefault("resampling_assumption", meta.get("interpolation_assumption"))
    meta.setdefault("logical_family", meta.get("reference_kind"))
    return meta


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


def _iter_package_resources_for_tracking() -> list[str]:
    """Return package resource paths (relative to scorpio_pipe/resources).

    P0-B10: track *all* built-in scientific reference resources by default so
    any change triggers a new ``context_id`` and appropriate stage reruns.

    We intentionally exclude documentation-only files to keep the context
    stable with respect to prose changes.
    """

    try:
        base = (Path(__file__).resolve().parents[1] / "resources").resolve()
        if not base.exists():
            return []
    except Exception:
        return []

    exts = {".csv", ".yaml", ".yml", ".json", ".pdf"}
    out: list[str] = []
    for p in base.rglob("*"):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            rel = p.relative_to(base).as_posix()
            # Exclude docs and README-like files.
            low = rel.lower()
            if low.startswith("docs/"):
                continue
            if p.name.lower().startswith("readme"):
                continue
            out.append(rel)
        except Exception:
            continue
    return sorted(set(out))


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
    fcfg = cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}
    fcfg = fcfg or {}

    # Linelist CSV: new key wavesol.linelist_csv; legacy neon_lines_csv.
    # P0-B10: do not hardcode historical lamp defaults here. Use the lamp
    # contract resolver so the default (HeNeAr for long-slit SCORPIO/SCORPIO-2)
    # and its provenance are consistent across the pipeline.
    try:
        from scorpio_pipe.lamp_contract import choose_linelist_name, resolve_lamp_type

        lr = resolve_lamp_type(cfg, strict=False)
        default_ll = choose_linelist_name(lr.lamp_type)
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

    # Calibration-domain references (P0.4): standards, extinction, telluric assets.
    std_lib_req = str(fcfg.get("standard_library") or "").strip() or "standards/library.yaml"
    reqs.append(("standards.library", std_lib_req))

    std_name = _slug(str(fcfg.get("standard_name") or "default"))
    std_table_req = str(fcfg.get("standard_table") or fcfg.get("standards_table") or "").strip() or "standards/bd284211.csv"
    reqs.append((f"standards.{std_name}.table", std_table_req))

    site = _slug(str(fcfg.get("site_name") or fcfg.get("observatory") or "bta"))
    extinction_req = str(fcfg.get("extinction_curve") or "").strip() or f"extinction/{site}.csv"
    reqs.append((f"extinction_curve.{site}", extinction_req))

    tell_mask_req = str(fcfg.get("telluric_mask") or "").strip() or "telluric/default_mask.csv"
    reqs.append(("telluric_mask.default", tell_mask_req))

    tell_tmpl_req = str(fcfg.get("telluric_template") or "").strip() or "telluric/default_template.csv"
    reqs.append(("telluric_template.default", tell_tmpl_req))

    # Track all built-in science reference resources (CSV/YAML/PDF/JSON) so
    # any change alters the context id and forces correct reruns.
    # These are additional to the explicit logical entries above.
    try:
        explicit_requested = {str(r[1]) for r in reqs}
        for rel in _iter_package_resources_for_tracking():
            if rel in explicit_requested:
                continue
            reqs.append((f"pkg_resource.{rel}", rel))
    except Exception:
        pass

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
        meta = _extract_reference_metadata(str(logical), Path(rr.resolved_path))
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
                metadata=meta,
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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Ensure ``manifest/reference_context.json`` exists and is up to date.

    Returns the context payload.
    """

    work_dir = resolve_work_dir(cfg)
    out_path = reference_context_path(work_dir)

    ctx = build_reference_context(cfg, resources_dir=resources_dir)

    # Write if missing or changed (or explicitly forced).
    try:
        old = None
        if out_path.exists():
            try:
                old = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                old = None
        if overwrite or (not isinstance(old, dict)) or old.get("context_id") != ctx.get("context_id"):
            atomic_write_json(out_path, ctx, indent=2, ensure_ascii=False)
    except Exception:
        # If we cannot write, fail loudly: provenance is a hard contract for B3.
        raise

    # Return payload, but also include an absolute path pointer for callers (tests/UI)
    # without polluting the serialized JSON (paths are environment-dependent).
    ret = dict(ctx)
    ret["reference_context_json"] = str(out_path)
    return ret


def load_reference_context(work_dir: str | Path) -> dict[str, Any] | None:
    p = reference_context_path(work_dir)
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
