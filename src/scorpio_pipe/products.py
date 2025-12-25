from __future__ import annotations

"""Registry of pipeline products.

The pipeline has many optional stages. A small, explicit registry makes it
 easier to:
  - build QC reports
  - show "what should exist" vs "what exists" in UI/CLI
  - keep backward compatibility when layouts change
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.wavesol_paths import resolve_work_dir, wavesol_dir


@dataclass(frozen=True)
class Product:
    key: str
    stage: str
    path: Path
    kind: str  # fits | png | csv | json | txt | html
    optional: bool = True
    description: str = ""

    def exists(self) -> bool:
        return self.path.exists()

    def size(self) -> int | None:
        try:
            return self.path.stat().st_size if self.path.exists() else None
        except Exception:
            return None


def list_products(cfg: dict[str, Any]) -> list[Product]:
    """Return a flat list of expected products for the given config."""

    wd = resolve_work_dir(cfg)
    wsol = wavesol_dir(cfg)

    rep = wd / "report"
    calib = wd / "calib"
    cosm = wd / "cosmics"

    # Canonical v5.13+ product tree
    prod = wd / "products"
    lin = prod / "lin"
    sky = prod / "sky"
    stack = prod / "stack"
    spec = prod / "spec"

    out: list[Product] = [
        # report
        Product("manifest", "report", rep / "manifest.json", "json", optional=False, description="Reproducibility manifest"),
        Product("qc_json", "report", rep / "qc_report.json", "json", optional=True, description="QC summary (machine-readable)"),
        Product("qc_html", "report", rep / "index.html", "html", optional=True, description="QC index (human-readable)"),
        Product("timings", "report", rep / "timings.json", "json", optional=True, description="Stage timings"),

        # calib
        Product("superbias", "calib", calib / "superbias.fits", "fits", optional=False),
        Product("superflat", "calib", calib / "superflat.fits", "fits", optional=False),

        # cosmics
        Product("cosmics_summary", "cosmics", cosm / "summary.json", "json", optional=False),
        Product("cosmics_coverage_png", "cosmics", cosm / "coverage.png", "png", optional=True),
        Product("cosmics_sum_png", "cosmics", cosm / "sum_excl_cosmics.png", "png", optional=True),

        # flatfield
        Product("flatfield_done", "flatfield", wd / "flatfield" / "flatfield_done.json", "json", optional=False),

        # wavesol (disperser-specific)
        Product("superneon_fits", "wavesol", wsol / "superneon.fits", "fits", optional=False),
        Product("superneon_png", "wavesol", wsol / "superneon.png", "png", optional=True),
        Product("peaks_candidates", "wavesol", wsol / "peaks_candidates.csv", "csv", optional=False),
        Product("hand_pairs", "wavesol", wsol / "hand_pairs.txt", "txt", optional=False),
        Product("wavesol_1d_png", "wavesol", wsol / "wavesolution_1d.png", "png", optional=True),
        Product("wavesol_1d_json", "wavesol", wsol / "wavesolution_1d.json", "json", optional=True),
        Product("residuals_1d", "wavesol", wsol / "residuals_1d.csv", "csv", optional=True),
        Product("control_points_2d", "wavesol", wsol / "control_points_2d.csv", "csv", optional=True),
        Product("wavesol_2d_json", "wavesol", wsol / "wavesolution_2d.json", "json", optional=False),
        Product("residuals_2d", "wavesol", wsol / "residuals_2d.csv", "csv", optional=True),
        Product("lambda_map", "wavesol", wsol / "lambda_map.fits", "fits", optional=False),
        Product("wavelength_matrix", "wavesol", wsol / "wavelength_matrix.png", "png", optional=True),
        Product("residuals_2d_png", "wavesol", wsol / "residuals_2d.png", "png", optional=True),

        # v5.13+ linearize / rectification (λ,y)
        Product("linearize_done", "lin", lin / "linearize_done.json", "json", optional=True),
        Product("lin_preview_fits", "lin", lin / "lin_preview.fits", "fits", optional=False, description="Rectified preview stack for ROI/QC"),
        Product("lin_preview_png", "lin", lin / "lin_preview.png", "png", optional=True),
        Product("lin_per_exp_dir", "lin", lin / "per_exp", "dir", optional=False, description="Per-exposure rectified frames (SCI/VAR/MASK)"),

        # v5.13+ sky subtraction (Kelson-style) per exposure
        Product("sky_done", "sky", sky / "sky_sub_done.json", "json", optional=True),
        Product("sky_per_exp_dir", "sky", sky / "per_exp", "dir", optional=False, description="Per-exposure sky-subtracted frames"),
        # Optional combined products (if running sky in single-frame mode)
        Product("sky_sub_fits", "sky", sky / "obj_sky_sub.fits", "fits", optional=True, description="Sky-subtracted combined 2D (legacy/quick)"),
        Product("sky_model_fits", "sky", sky / "sky_model.fits", "fits", optional=True, description="Sky model (legacy/quick)"),

        # v5.13+ stacking in (λ,y)
        Product("stack2d_done", "stack", stack / "stack2d_done.json", "json", optional=True),
        Product("stacked2d_fits", "stack", stack / "stacked2d.fits", "fits", optional=False, description="Final stacked 2D (SCI/VAR/MASK/COV)"),
        Product("coverage_png", "stack", stack / "coverage.png", "png", optional=True),

        # v5.13+ extraction
        Product("extract1d_done", "spec", spec / "extract1d_done.json", "json", optional=True),
        Product("spec1d_fits", "spec", spec / "spec1d.fits", "fits", optional=False, description="1D spectrum (FLUX/VAR/MASK)"),
        Product("spec1d_png", "spec", spec / "spec1d.png", "png", optional=True),
        Product("trace_json", "spec", spec / "trace.json", "json", optional=True),
    ]

    return out


def group_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    out: dict[str, list[Product]] = {}
    for p in products:
        out.setdefault(p.stage, []).append(p)
    return out


# ---------------------------- resume helpers ----------------------------

# Minimal product sets used to decide whether a stage is "complete".
# This drives CLI/UI resume mode (skip stages that have already produced
# their outputs).

TASK_PRODUCT_KEYS: dict[str, list[str]] = {
    "manifest": ["manifest"],
    "qc_report": ["qc_html", "qc_json"],
    "superbias": ["superbias"],
    "superflat": ["superflat"],
    "cosmics": ["cosmics_summary"],
    "superneon": ["superneon_fits", "peaks_candidates"],
    "lineid_prepare": ["hand_pairs"],
    "wavesolution": ["wavesol_2d_json", "lambda_map", "wavesol_1d_json", "wavesol_1d_png", "residuals_1d"],
    "linearize": ["linearize_done", "lin_preview_fits", "lin_per_exp_dir"],
    "sky": ["sky_done", "sky_per_exp_dir"],
    "stack2d": ["stack2d_done", "stacked2d_fits"],
    "extract1d": ["extract1d_done", "spec1d_fits"],
}


def products_for_task(cfg: dict[str, Any], task_name: str) -> list[Product]:
    """Return products associated with a task name."""

    keys = TASK_PRODUCT_KEYS.get(task_name, [])
    if not keys:
        return []
    prods = list_products(cfg)
    keys_set = set(keys)
    return [p for p in prods if p.key in keys_set]


def task_is_complete(cfg: dict[str, Any], task_name: str) -> bool:
    """True if all required products for the task exist."""

    ps = products_for_task(cfg, task_name)
    if not ps:
        return False
    required = [p for p in ps if not p.optional]
    return all(p.exists() for p in required)
