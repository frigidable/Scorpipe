from __future__ import annotations

"""Registry of pipeline products.

This module provides a small, explicit product registry used by QC/UI.
It intentionally does NOT try to be exhaustive for every intermediate file;
instead it lists stable, canonical artifacts users can expect.

v5.17 notes
----------
- QC outputs live in work/qc/ (legacy mirror in work/report/)
- Calibrations live in work/calibs/ (legacy mirror/compat in work/calib/)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.wavesol_paths import resolve_work_dir, wavesol_dir
from scorpio_pipe.work_layout import ensure_work_layout


@dataclass(frozen=True)
class Product:
    key: str
    stage: str
    path: Path
    kind: str  # "fits", "png", "json", "html", "txt", ...
    optional: bool = True
    description: str | None = None

    def exists(self) -> bool:
        return self.path.exists()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "path": str(self.path),
            "kind": self.kind,
            "optional": self.optional,
            "exists": self.exists(),
            "description": self.description,
        }


def list_products(cfg: dict[str, Any]) -> list[Product]:
    wd = resolve_work_dir(cfg)
    layout = ensure_work_layout(wd)

    products_root = layout.products
    qc = layout.qc
    rep = layout.report_legacy
    calibs = layout.calibs
    calib_legacy = layout.calib_legacy

    # v5.x products
    lin = products_root / "lin"
    sky = products_root / "sky"
    stack = products_root / "stack"
    spec = products_root / "spec"

    # wavesolution: needs disperser subdir (important for tests and real data)
    wsol = wavesol_dir(cfg)

    out: list[Product] = [
        # QC (canonical + legacy mirrors)
        Product("manifest", "qc", qc / "manifest.json", "json", optional=False, description="Reproducibility manifest"),
        Product("products_manifest", "qc", qc / "products_manifest.json", "json", optional=True, description="Products manifest (incl. per-exposure trees)"),
        Product("qc_json", "qc", qc / "qc_report.json", "json", optional=True, description="QC summary (machine-readable)"),
        Product("qc_html", "qc", qc / "index.html", "html", optional=True, description="QC report (human-readable)"),
        Product("timings", "qc", qc / "timings.json", "json", optional=True, description="Stage timings"),

        Product("linearize_qc", "linearize", qc / "linearize_qc.json", "json", optional=True, description="Linearize QC metrics (S/N, coverage, mask fractions)"),

        Product("manifest_legacy", "report", rep / "manifest.json", "json", optional=True),
        Product("products_manifest_legacy", "report", rep / "products_manifest.json", "json", optional=True),
        Product("qc_json_legacy", "report", rep / "qc_report.json", "json", optional=True),
        Product("qc_html_legacy", "report", rep / "index.html", "html", optional=True),
        Product("timings_legacy", "report", rep / "timings.json", "json", optional=True),

        Product("linearize_qc_legacy", "linearize", rep / "linearize_qc.json", "json", optional=True),

        # Calibrations (canonical + legacy)
        Product("superbias_fits", "superbias", calibs / "superbias.fits", "fits", optional=True),
        Product("superflat_fits", "superflat", calibs / "superflat.fits", "fits", optional=True),
        Product("superbias_fits_legacy", "superbias", calib_legacy / "superbias.fits", "fits", optional=True),
        Product("superflat_fits_legacy", "superflat", calib_legacy / "superflat.fits", "fits", optional=True),

        # Wavesolution key artifacts
        Product("lambda_map", "wavesol", wsol / "lambda_map.fits", "fits", optional=True),
        Product("wavesol_solution", "wavesol", wsol / "solution.json", "json", optional=True),

        # Core science (quicklook + canonical endpoints)
        Product("lin_preview_fits", "linearize", lin / "lin_preview.fits", "fits", optional=True),
        Product("lin_preview_png", "linearize", lin / "lin_preview.png", "png", optional=True),

        # Sky subtraction artifacts
        Product("sky_done", "sky", sky / "sky_sub_done.json", "json", optional=True, description="Sky stage summary (per exposure)")
        ,
        Product("qc_sky_json", "sky", sky / "qc_sky.json", "json", optional=True, description="Sky QC (residual metrics + diag paths)")
        ,
        Product("sky_preview_fits", "sky", sky / "preview.fits", "fits", optional=True),
        Product("sky_preview_png", "sky", sky / "preview.png", "png", optional=True),

        Product("stack2d_done", "stack2d", stack / "stack2d_done.json", "json", optional=True, description="Stacking summary")
        ,
        Product("stacked2d_fits", "stack2d", stack / "stacked2d.fits", "fits", optional=True),
        Product("coverage_png", "stack2d", stack / "coverage.png", "png", optional=True),

        Product("extract1d_done", "extract1d", spec / "extract1d_done.json", "json", optional=True, description="Extraction summary")
        ,
        Product("spec1d_fits", "extract1d", spec / "spec1d.fits", "fits", optional=True),
        Product("spec1d_png", "extract1d", spec / "spec1d.png", "png", optional=True),
    ]

    return out


def products_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    out: dict[str, list[Product]] = {}
    for p in products:
        out.setdefault(p.stage, []).append(p)
    return out
