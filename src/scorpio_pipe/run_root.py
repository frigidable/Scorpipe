from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping


def _as_header_mapping(h: Any) -> Mapping[str, Any]:
    """Best-effort adapter for FITS headers / dict-like objects."""

    if h is None:
        return {}
    if isinstance(h, Mapping):
        return h
    # astropy.io.fits.Header behaves like Mapping but is not a real Mapping
    if hasattr(h, "__getitem__") and hasattr(h, "keys"):
        try:
            return {str(k): h[k] for k in list(h.keys())}
        except Exception:
            pass
    return {}


def _pick(h: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if k in h and h[k] not in (None, ""):
            try:
                return str(h[k]).strip()
            except Exception:
                continue
    return default


def _parse_date_obs(s: str) -> datetime | None:
    """Parse DATE-OBS like values robustly."""

    if not s:
        return None
    ss = str(s).strip()
    # common FITS variants
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(ss, fmt)
        except Exception:
            continue
    # last resort: strip timezone suffix like "+03:00" / "Z"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?)?", ss)
    if m:
        d = m.group(1)
        t = m.group(2)
        try:
            return datetime.strptime(d + ("T" + t if t else ""), "%Y-%m-%d" + ("T%H:%M:%S" if t else ""))
        except Exception:
            return None
    return None


def night_date_from_header(header: Any) -> date:
    """Infer the *night-of-observation* date from a FITS header.

    Heuristic (robust + practical):
    - Prefer an explicit night key if present.
    - Else use DATE-OBS (or DATE) and if a time is present and the hour < 12,
      shift to the previous day (astronomical night spans midnight).

    This avoids naming runs by the pipeline execution date.
    """

    h = _as_header_mapping(header)

    # Some instruments write a direct night tag.
    # Keep the list permissive: if one exists, it's almost certainly what the
    # observer expects.
    night = _pick(h, "NIGHT", "NIGHTID", "UTDATE", "OBS-NGT", default="")
    if night:
        # Accept both YYYYMMDD and YYYY-MM-DD
        m = re.match(r"^(\d{4})[-/]?(\d{2})[-/]?(\d{2})$", night)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    dt = _parse_date_obs(_pick(h, "DATE-OBS", "DATE", default=""))
    if dt is None:
        # Fallback: today (better than crashing); caller may still override.
        return datetime.utcnow().date()

    d = dt.date()
    # If the observation time is after midnight (early morning), treat it as
    # the same observing night as the previous evening.
    if dt.hour < 12 and ("T" in str(_pick(h, "DATE-OBS", "DATE", default="")) or ":" in str(_pick(h, "DATE-OBS", "DATE", default=""))):
        d = (dt - timedelta(days=1)).date()
    return d


def format_night_folder(d: date) -> str:
    return f"{d.day:02d}_{d.month:02d}_{d.year:04d}"


def _sanitize_key(s: str, *, fallback: str) -> str:
    s0 = (s or "").strip()
    if not s0:
        return fallback
    try:
        from scorpio_pipe.workdir import safe_slug

        out = safe_slug(s0)
        return out or fallback
    except Exception:
        # Minimal fallback
        out = re.sub(r"[^A-Za-z0-9_-]+", "_", s0).strip("_")
        return out or fallback


def object_key_from_header(header: Any) -> str:
    h = _as_header_mapping(header)
    obj = _pick(h, "OBJECT", "OBJNAME", "TARGNAME", "TARGET", default="")
    return _sanitize_key(obj, fallback="unknown")


def disperser_key_from_header(header: Any) -> str:
    h = _as_header_mapping(header)
    disp = _pick(h, "GRISM", "DISPERSER", "GRATING", "GRAT", default="")
    return _sanitize_key(disp, fallback="unknown")


@dataclass(frozen=True)
class RunSignature:
    night_folder: str
    object_key: str
    disperser_key: str

    def run_name(self, run: int) -> str:
        return f"{self.object_key}_{self.disperser_key}_{run:02d}"


_RUN_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<n>\d{2})$")


def _max_existing_run_id(base_dir: Path, sig: RunSignature) -> int:
    """Return the max existing NN for this signature, or 0 if none."""

    if not base_dir.is_dir():
        return 0
    prefix = f"{sig.object_key}_{sig.disperser_key}_"
    mx = 0
    try:
        for p in base_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if not name.startswith(prefix):
                continue
            m = _RUN_SUFFIX_RE.match(name)
            if not m:
                continue
            try:
                n = int(m.group("n"))
            except Exception:
                continue
            mx = max(mx, n)
    except Exception:
        return 0
    return mx


def detect_legacy_layout(root: Path) -> bool:
    """Detect an old workspace layout rooted directly at ``root``.

    Legacy signals:
    - products/ exists
    - raw/ exists
    - stage-like directories exist under products/ (e.g. products/03_*)
    """

    r = Path(root)
    if (r / "products").is_dir():
        # extra confidence: products/NN_* exists
        try:
            for p in (r / "products").iterdir():
                if p.is_dir() and re.match(r"^\d{2}_.+", p.name):
                    return True
        except Exception:
            return True
        return True
    if (r / "raw").is_dir():
        return True
    return False


def get_or_create_run_root(
    workspace_root: str | Path,
    dataset_path: str | Path | None,
    headers: Any,
    *,
    run_id: int | None = None,
    create: bool = True,
) -> Path:
    """Compute (and optionally create) ``run_root``.

    Parameters
    ----------
    workspace_root
        Top-level workspace folder.
    dataset_path
        Path to the raw dataset folder (optional, used only for provenance).
    headers
        Representative FITS header (preferred: SCI object frame; fallback: ARC).
    run_id
        Optional fixed run number (1..99). If None, auto-increment.
    create
        If True, create the run folder and its parents.
    """

    ws = Path(workspace_root).expanduser().resolve()

    ndate = night_date_from_header(headers)
    sig = RunSignature(
        night_folder=format_night_folder(ndate),
        object_key=object_key_from_header(headers),
        disperser_key=disperser_key_from_header(headers),
    )

    base = ws / sig.night_folder
    if run_id is None:
        mx = _max_existing_run_id(base, sig)
        run_id = mx + 1
    if run_id < 1:
        run_id = 1
    if run_id > 99:
        # keep names sortable / fixed width
        run_id = 99

    run_root = (base / sig.run_name(run_id)).resolve()
    if create:
        run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def scan_recent_runs(workspace_root: str | Path, *, limit: int = 10) -> list[Path]:
    """Return recent run folders under ``workspace_root``.

    We define a "run folder" as a directory that contains either:
      - config.yaml, or
      - manifest/ directory

    Returned list is sorted by directory modification time (desc).
    """

    ws = Path(workspace_root).expanduser().resolve()
    runs: list[tuple[float, Path]] = []
    if not ws.is_dir():
        return []

    # Expected structure: ws/<DD_MM_YYYY>/<something>_<something>_<NN>/
    try:
        for night in ws.iterdir():
            if not night.is_dir():
                continue
            # allow non-night directories (e.g. legacy) but keep them out of recent
            if not re.match(r"^\d{2}_\d{2}_\d{4}$", night.name):
                continue
            for run in night.iterdir():
                if not run.is_dir():
                    continue
                if (run / "config.yaml").is_file() or (run / "manifest").is_dir():
                    try:
                        mtime = run.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    runs.append((mtime, run))
    except Exception:
        return []

    runs.sort(key=lambda t: t[0], reverse=True)
    out = [p for _t, p in runs[: max(0, int(limit))]]
    return out


def format_run_label(workspace_root: str | Path, run_root: str | Path) -> str:
    """Human-readable label for UI lists."""

    ws = Path(workspace_root).expanduser().resolve()
    rr = Path(run_root).expanduser().resolve()
    try:
        rel = rr.relative_to(ws)
        parts = rel.parts
        if len(parts) >= 2:
            # night/run
            night, run = parts[0], parts[1]
            return f"🌙 {night} → {run}"
        return str(rel)
    except Exception:
        return str(rr)
