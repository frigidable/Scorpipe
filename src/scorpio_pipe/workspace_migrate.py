"""Workspace/run migration helpers.

This module is intentionally conservative:
- We **never** delete anything from the source run.
- Migration is opt-in and implemented as a *copy* into a new run folder.

Currently supported:
- Detecting pre-v5.39.1 stage directory layouts (e.g. different numbering and
  two-step sky subtraction directories).
- Optional copy-migration into the v5.39.1 canonical layout (12 stages).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from scorpio_pipe.stage_registry import REGISTRY
from scorpio_pipe.work_layout import ensure_work_layout

_RE_STAGE_DIR = re.compile(r"^(\d{2})_([A-Za-z0-9_\-]+)$")


@dataclass(frozen=True)
class LegacyLayoutInfo:
    is_legacy: bool
    reason: str
    found: tuple[str, ...] = ()


def _stage_dirs(run_root: Path) -> list[Path]:
    try:
        return [p for p in run_root.iterdir() if p.is_dir() and _RE_STAGE_DIR.match(p.name)]
    except Exception:
        return []


def detect_legacy_stage_layout(run_root: str | Path) -> LegacyLayoutInfo:
    """Detect older stage directory layouts.

    We flag legacy if we see any of:
    - pre-stage-layout "products/*" folders (e.g. products/lin, products/stack)
    - old numbering after the Sky↔Linearize swap (e.g. 09_linearize, 10_sky)
    - deprecated slugs (stack2d, extract1d) or two-step sky dirs (skyraw/skyrect)

    This is *only* a hint for the GUI: the pipeline can still operate read-only
    using directory fallbacks in :mod:`scorpio_pipe.workspace_paths`.
    """

    rr = Path(run_root)
    dirs = _stage_dirs(rr)
    found: list[str] = []

    # Pre-stage-layout products tree.
    prod = rr / "products"
    if prod.is_dir():
        for name in ["lin", "stack", "stack2d", "extract", "extract1d", "sky", "skyraw", "skyrect"]:
            p = prod / name
            if p.exists():
                found.append(str(p.relative_to(rr)))

    # Any legacy sky directories (two-step sky subtraction).
    sky_legacy = [d for d in dirs if d.name.endswith("_skyraw") or d.name.endswith("_skyrect")]
    if sky_legacy:
        found.extend([d.name for d in sky_legacy])

    # Old order before v5.39.1 swap.
    if (rr / "09_linearize").exists() or (rr / "10_sky").exists():
        if (rr / "09_linearize").exists():
            found.append("09_linearize")
        if (rr / "10_sky").exists():
            found.append("10_sky")

    # Canonical since v5.39.1: 09_sky, 10_linearize, 11_stack, 12_extract.
    linearize = [d for d in dirs if d.name.endswith("_linearize")]
    if linearize and not (rr / "10_linearize").exists():
        found.extend([d.name for d in linearize])

    sky = [d for d in dirs if d.name.endswith("_sky")]
    if sky and not (rr / "09_sky").exists():
        found.extend([d.name for d in sky])

    stack_legacy = [d for d in dirs if d.name.endswith("_stack2d")]
    if stack_legacy and not (rr / "11_stack").exists():
        found.extend([d.name for d in stack_legacy])

    extract_legacy = [d for d in dirs if d.name.endswith("_extract1d")]
    if extract_legacy and not (rr / "12_extract").exists():
        found.extend([d.name for d in extract_legacy])

    if not found:
        return LegacyLayoutInfo(is_legacy=False, reason="")

    found_u = tuple(sorted(set(found)))
    reason = (
        "This run folder uses an older stage directory layout. "
        "It can be opened as-is (read-only browsing), or copied into the new v5.39.1 layout."
    )
    return LegacyLayoutInfo(is_legacy=True, reason=reason, found=found_u)


def _unique_sibling(root: Path, stem: str) -> Path:
    base = root.parent / stem
    if not base.exists():
        return base
    for i in range(2, 1000):
        p = root.parent / f"{stem}_{i:02d}"
        if not p.exists():
            return p
    return root.parent / f"{stem}_999"


def migrate_run_to_v5391(src_run_root: str | Path, dst_run_root: str | Path | None = None) -> Path:
    """Copy a legacy run into the v5.39.1 canonical layout.

    The migration is a best-effort copy of stage directories into their new
    canonical names. The source run is left untouched.

    Returns the new run_root path.
    """

    src = Path(src_run_root)
    if dst_run_root is None:
        dst = _unique_sibling(src, f"{src.name}_migrated")
    else:
        dst = Path(dst_run_root)

    dst.mkdir(parents=True, exist_ok=True)
    ensure_work_layout(dst)

    # Copy config.yaml if present.
    cfg_src = src / "config.yaml"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, dst / "config.yaml")

    # Copy manifest/ + top-level QC artifacts if present.
    for name in ["manifest", "index.html"]:
        p = src / name
        if p.is_dir():
            shutil.copytree(p, dst / name, dirs_exist_ok=True)
        elif p.is_file():
            shutil.copy2(p, dst / name)

    # Stage directory copy.
    # We resolve by directory suffix (slug), not by stage key.
    legacy_slug_candidates: dict[str, list[str]] = {
        "biascorr": ["bias", "biascorr"],
        "flatfield": ["flat", "flatfield"],
        "cosmics": ["cosmics"],
        "superneon": ["superneon"],
        "arclineid": ["lineid", "arclineid"],
        "wavesol": ["wavesol"],
        "linearize": ["linearize"],
        # unified sky: prefer rectified, then raw, then any sky dir
        "sky": ["sky", "skyrect", "skyraw"],
        "stack2d": ["stack", "stack2d"],
        "extract1d": ["extract", "extract1d"],
    }

    def _find_stage_dir(slugs: list[str]) -> Path | None:
        dirs = _stage_dirs(src)
        # Try exact "??_<slug>" first in provided order.
        for slug in slugs:
            hits = [d for d in dirs if d.name.endswith(f"_{slug}")]
            if hits:
                # Choose the highest prefix if multiple.
                hits.sort(key=lambda p: int(p.name.split("_", 1)[0]))
                return hits[-1]
        # Sky catch-all: any stage dir starting with "??_sky".
        if any(s.startswith("sky") for s in slugs):
            hits = [d for d in dirs if d.name.split("_", 1)[1].startswith("sky")]
            if hits:
                hits.sort(key=lambda p: int(p.name.split("_", 1)[0]))
                return hits[-1]
        return None

    for stage in REGISTRY.iter_pipeline_stages():
        key = stage.key
        cands = legacy_slug_candidates.get(key, [stage.slug])
        src_dir = _find_stage_dir(cands)
        if src_dir is None:
            continue
        dst_dir = dst / stage.dir_name
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    return dst


# Backward compatible alias

def migrate_run_to_v5386(src_run_root: str | Path, dst_run_root: str | Path | None = None) -> Path:
    """Alias for :func:`migrate_run_to_v5391` (kept for older UI builds)."""
    return migrate_run_to_v5391(src_run_root, dst_run_root)
