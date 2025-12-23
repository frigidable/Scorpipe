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

    out: list[Product] = [
        # report
        Product("manifest", "report", rep / "manifest.json", "json", optional=False, description="Reproducibility manifest"),
        Product("qc_json", "report", rep / "qc_report.json", "json", optional=True, description="QC summary (machine-readable)"),
        Product("qc_html", "report", rep / "index.html", "html", optional=True, description="QC index (human-readable)"),
        Product("timings", "report", rep / "timings.json", "json", optional=True, description="Stage timings"),

        # calib
        Product("superbias", "calib", calib / "superbias.fits", "fits", optional=True),
        Product("superflat", "calib", calib / "superflat.fits", "fits", optional=True),

        # cosmics
        Product("cosmics_summary", "cosmics", cosm / "summary.json", "json", optional=True),
        Product("cosmics_coverage_png", "cosmics", cosm / "coverage.png", "png", optional=True),
        Product("cosmics_sum_png", "cosmics", cosm / "sum_excl_cosmics.png", "png", optional=True),

        # wavesol (disperser-specific)
        Product("superneon_fits", "wavesol", wsol / "superneon.fits", "fits", optional=True),
        Product("superneon_png", "wavesol", wsol / "superneon.png", "png", optional=True),
        Product("peaks_candidates", "wavesol", wsol / "peaks_candidates.csv", "csv", optional=True),
        Product("hand_pairs", "wavesol", wsol / "hand_pairs.txt", "txt", optional=True),
        Product("wavesol_1d_png", "wavesol", wsol / "wavesolution_1d.png", "png", optional=True),
        Product("wavesol_1d_json", "wavesol", wsol / "wavesolution_1d.json", "json", optional=True),
        Product("residuals_1d", "wavesol", wsol / "residuals_1d.csv", "csv", optional=True),
        Product("control_points_2d", "wavesol", wsol / "control_points_2d.csv", "csv", optional=True),
        Product("wavesol_2d_json", "wavesol", wsol / "wavesolution_2d.json", "json", optional=True),
        Product("residuals_2d", "wavesol", wsol / "residuals_2d.csv", "csv", optional=True),
        Product("lambda_map", "wavesol", wsol / "lambda_map.fits", "fits", optional=True),
        Product("wavelength_matrix", "wavesol", wsol / "wavelength_matrix.png", "png", optional=True),
        Product("residuals_2d_png", "wavesol", wsol / "residuals_2d.png", "png", optional=True),
    ]

    # Placeholders for future pipeline stages (kept to stabilize QC layout)
    out += [
        Product("sky_done", "sky", wd / "sky" / "sky_sub_done.json", "json", optional=True),
        Product("linearize_done", "lin", wd / "lin" / "linearize_done.json", "json", optional=True),
        Product("stack_done", "stack", wd / "stack" / "stack_done.json", "json", optional=True),
        Product("spectrum_1d", "spec", wd / "spec" / "spectrum_1d.fits", "fits", optional=True),
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
    "wavesolution": ["wavesol_1d_json", "wavesol_2d_json", "lambda_map"],
}


def products_for_task(cfg: dict[str, Any], task_name: str) -> list[Product]:
    """Return products associated with a task name."""

    keys = TASK_PRODUCT_KEYS.get(task_name, [])
    if not keys:
        return []
    prods = list_products(cfg)
    return [p for p in prods if p.key in set(keys)]


def task_is_complete(cfg: dict[str, Any], task_name: str) -> bool:
    """True if all required products for the task exist."""

    ps = products_for_task(cfg, task_name)
    if not ps:
        return False
    return all(p.exists() for p in ps)
