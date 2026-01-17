from __future__ import annotations

"""Global exclude policy helpers (P0-K).

We treat exclude as an *absolute* rule across the whole pipeline:

- Source of truth: ``<data_dir>/project_manifest.yaml`` (section ``exclude``).
- Optional extras: config/CLI ``exclude_frames``.

Some stages may operate on a previously generated ``dataset_manifest.json``.
To protect against stale manifests and regressions, those stages should call
:func:`resolve_exclude_set` and filter any manifest-selected frame IDs/paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ExcludeResolution:
    excluded_abs: set[str]
    summary: dict[str, Any]


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    return [str(v).strip()] if str(v).strip() else []


def resolve_data_dir(cfg: dict[str, Any]) -> Path | None:
    """Best-effort resolve of ``data_dir``.

    We prefer ``cfg['data_dir']`` if present. Some call-sites store it under
    ``cfg['paths']['data_dir']``; handle that too.
    """
    if not isinstance(cfg, dict):
        return None
    dd = cfg.get("data_dir")
    if not dd and isinstance(cfg.get("paths"), dict):
        dd = cfg["paths"].get("data_dir")
    if not dd:
        return None
    try:
        p = Path(str(dd)).expanduser()
        return p.resolve()
    except Exception:
        return None


def resolve_exclude_set(
    cfg: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    extra_exclude_paths: Sequence[str] | None = None,
) -> ExcludeResolution:
    """Resolve the absolute exclude set + provenance summary."""

    # Import locally: dataset.builder is intentionally Astropy-free.
    from scorpio_pipe.dataset.builder import resolve_global_exclude

    dd = Path(data_dir) if data_dir is not None else resolve_data_dir(cfg)
    if dd is None:
        return ExcludeResolution(excluded_abs=set(), summary={})

    # Config convention: cfg.exclude_frames
    cfg_ex = _as_str_list(cfg.get("exclude_frames"))
    # Some older configs used cfg.exclude
    cfg_ex += _as_str_list(cfg.get("exclude"))

    ex = list(cfg_ex)
    if extra_exclude_paths:
        ex += [str(x) for x in extra_exclude_paths if str(x).strip()]

    excluded_abs_list, summary = resolve_global_exclude(dd, exclude_paths=ex)
    return ExcludeResolution(excluded_abs=set(excluded_abs_list), summary=summary)


def filter_paths_by_exclude(
    paths: Iterable[str | Path],
    excluded_abs: set[str],
) -> tuple[list[Path], list[Path]]:
    """Return (kept, dropped) after applying an absolute exclude set."""
    kept: list[Path] = []
    dropped: list[Path] = []
    for p in paths:
        pp = Path(str(p)).expanduser()
        try:
            pp_abs = pp.resolve()
        except Exception:
            pp_abs = pp
        if str(pp_abs) in excluded_abs:
            dropped.append(pp_abs)
        else:
            kept.append(pp_abs)
    return kept, dropped
