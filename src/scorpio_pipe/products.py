"""Registry of pipeline products.

This module provides a small, explicit product registry used by QC/UI.
It intentionally does NOT try to be exhaustive for every intermediate file;
instead it lists stable, canonical artifacts users can expect.

v5.38.4 notes
-------------
- Stage outputs live at workspace root: ``work/NN_slug/...``
- QC outputs live in ``work/manifest/...`` and the HTML entry point is ``work/index.html``.
- Legacy directories (``work/qc``, ``work/report``, ``work/products``) are not used by default.

UI
--
- All stages now expose Parameters as Basic/Advanced tabs with scrollable content.
- FITS preview: more robust image-HDU discovery + better diagnostics; stage frame browsers also recognize *.fts.
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir
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

    def size(self) -> int | None:
        """File size in bytes (None if missing/unstatable)."""
        try:
            return int(self.path.stat().st_size) if self.path.exists() else None
        except Exception:
            return None

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

    manifest_dir = layout.manifest

    # Canonical stage dirs (work/NN_slug)
    superbias_stage = stage_dir(wd, "superbias")
    superflat_stage = stage_dir(wd, "superflat")
    flatfield_stage = stage_dir(wd, "flatfield")
    cosmics_stage = stage_dir(wd, "cosmics")
    lin_stage = stage_dir(wd, "linearize")
    sky_stage = stage_dir(wd, "skyrect")
    stack_stage = stage_dir(wd, "stack2d")
    spec_stage = stage_dir(wd, "extract1d")

    # wavesolution: needs disperser subdir (important for tests and real data)
    wsol = wavesol_dir(cfg)

    out: list[Product] = [
        # Manifest + QC (canonical)
        Product(
            "manifest",
            "manifest",
            manifest_dir / "manifest.json",
            "json",
            optional=False,
            description="Reproducibility manifest",
        ),
        Product(
            "products_manifest",
            "manifest",
            manifest_dir / "products_manifest.json",
            "json",
            optional=True,
            description="Products manifest (incl. per-exposure trees)",
        ),
        Product(
            "qc_json",
            "manifest",
            manifest_dir / "qc_report.json",
            "json",
            optional=True,
            description="QC summary (machine-readable)",
        ),
        Product(
            "qc_html",
            "manifest",
            Path(wd) / "index.html",
            "html",
            optional=True,
            description="QC report (human-readable)",
        ),
        Product(
            "timings",
            "manifest",
            manifest_dir / "timings.json",
            "json",
            optional=True,
            description="Stage timings",
        ),
        Product(
            "linearize_qc",
            "linearize",
            manifest_dir / "linearize_qc.json",
            "json",
            optional=True,
            description="Linearize QC metrics (S/N, coverage, mask fractions)",
        ),
        # Calibrations
        Product(
            "superbias_fits",
            "superbias",
            superbias_stage / "superbias.fits",
            "fits",
            optional=True,
        ),
        Product(
            "superflat_fits",
            "superflat",
            superflat_stage / "superflat.fits",
            "fits",
            optional=True,
        ),
        # Flatfield stage (done marker)
        Product(
            "flatfield_done",
            "flatfield",
            flatfield_stage / "flatfield_done.json",
            "json",
            optional=True,
        ),
        # Cosmics stage
        Product(
            "cosmics_summary",
            "cosmics",
            cosmics_stage / "summary.json",
            "json",
            optional=True,
        ),
        # SuperNeon + LineID preparation (wavesol dir)
        Product(
            "superneon_fits",
            "superneon",
            wsol / "superneon.fits",
            "fits",
            optional=True,
        ),
        Product(
            "superneon_png", "superneon", wsol / "superneon.png", "png", optional=True
        ),
        Product(
            "peaks_candidates_csv",
            "superneon",
            wsol / "peaks_candidates.csv",
            "csv",
            optional=True,
        ),
Product(
    "superneon_shifts_csv",
    "superneon",
    wsol / "superneon_shifts.csv",
    "csv",
    optional=True,
    description="Per-frame X shifts + basic QC for SuperNeon stacking",
),
Product(
    "superneon_shifts_json",
    "superneon",
    wsol / "superneon_shifts.json",
    "json",
    optional=True,
    description="Per-frame X shifts + QC (machine-readable)",
),
Product(
    "superneon_qc_json",
    "superneon",
    wsol / "superneon_qc.json",
    "json",
    optional=True,
    description="SuperNeon QC summary (SNR, shifts, saturation, clipping)",
),

        Product(
            "lineid_template_csv",
            "lineid_prepare",
            wsol / "manual_pairs_template.csv",
            "csv",
            optional=True,
        ),
        Product(
            "lineid_auto_csv",
            "lineid_prepare",
            wsol / "manual_pairs_auto.csv",
            "csv",
            optional=True,
        ),
        Product(
            "lineid_report_txt",
            "lineid_prepare",
            wsol / "lineid_report.txt",
            "txt",
            optional=True,
        ),
        Product(
            "hand_pairs_txt",
            "lineid",
            wsol / "hand_pairs.txt",
            "txt",
            optional=True,
            description="Manual line pairs (x_pix, lambda)",
        ),
        # Wavesolution key artifacts
        Product(
            "wavesolution_1d_png",
            "wavesol",
            wsol / "wavesolution_1d.png",
            "png",
            optional=True,
        ),
        Product(
            "wavesolution_1d_json",
            "wavesol",
            wsol / "wavesolution_1d.json",
            "json",
            optional=True,
        ),
        Product(
            "wavesolution_2d_json",
            "wavesol",
            wsol / "wavesolution_2d.json",
            "json",
            optional=True,
        ),
        Product(
            "lambda_map", "wavesol", wsol / "lambda_map.fits", "fits", optional=True
        ),
        Product(
            "wavelength_matrix_png",
            "wavesol",
            wsol / "wavelength_matrix.png",
            "png",
            optional=True,
        ),
        Product(
            "residuals_2d_png",
            "wavesol",
            wsol / "residuals_2d.png",
            "png",
            optional=True,
        ),
        Product(
            "residuals_2d_audit_png",
            "wavesol",
            wsol / "residuals_2d_audit.png",
            "png",
            optional=True,
        ),
        Product(
            "control_points_2d_csv",
            "wavesol",
            wsol / "control_points_2d.csv",
            "csv",
            optional=True,
        ),
        Product(
            "residuals_1d_csv",
            "wavesol",
            wsol / "residuals_1d.csv",
            "csv",
            optional=True,
        ),
        Product(
            "residuals_2d_csv",
            "wavesol",
            wsol / "residuals_2d.csv",
            "csv",
            optional=True,
        ),
        Product(
            "residuals_vs_lambda_png",
            "wavesol",
            wsol / "residuals_vs_lambda.png",
            "png",
            optional=True,
        ),
        Product(
            "residuals_vs_y_png",
            "wavesol",
            wsol / "residuals_vs_y.png",
            "png",
            optional=True,
        ),
        Product(
            "residuals_hist_png",
            "wavesol",
            wsol / "residuals_hist.png",
            "png",
            optional=True,
        ),
        Product(
            "wavesolution_report_txt",
            "wavesol",
            wsol / "wavesolution_report.txt",
            "txt",
            optional=True,
        ),
        # Core science (quicklook + canonical endpoints)
        Product(
            "lin_preview_fits",
            "linearize",
            lin_stage / "lin_preview.fits",
            "fits",
            optional=True,
        ),
        Product(
            "lin_preview_png",
            "linearize",
            lin_stage / "lin_preview.png",
            "png",
            optional=True,
        ),
        # Sky subtraction artifacts
        Product(
            "sky_done",
            "skyrect",
            sky_stage / "sky_sub_done.json",
            "json",
            optional=True,
            description="Sky stage summary (per exposure)",
        ),
        Product(
            "qc_sky_json",
            "skyrect",
            sky_stage / "qc_sky.json",
            "json",
            optional=True,
            description="Sky QC (residual metrics + diag paths)",
        ),
        Product(
            "sky_preview_fits",
            "skyrect",
            sky_stage / "preview.fits",
            "fits",
            optional=True,
        ),
        Product(
            "sky_preview_png",
            "skyrect",
            sky_stage / "preview.png",
            "png",
            optional=True,
        ),
        Product(
            "stack2d_done",
            "stack2d",
            stack_stage / "stack2d_done.json",
            "json",
            optional=True,
            description="Stacking summary",
        ),
        Product(
            "stacked2d_fits",
            "stack2d",
            stack_stage / "stacked2d.fits",
            "fits",
            optional=True,
        ),
        Product(
            "coverage_png",
            "stack2d",
            stack_stage / "coverage.png",
            "png",
            optional=True,
        ),
        Product(
            "extract1d_done",
            "extract1d",
            spec_stage / "extract1d_done.json",
            "json",
            optional=True,
            description="Extraction summary",
        ),
        Product(
            "spec1d_fits", "extract1d", spec_stage / "spec1d.fits", "fits", optional=True
        ),
        Product(
            "spec1d_png", "extract1d", spec_stage / "spec1d.png", "png", optional=True
        ),
    ]

    return out


def products_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    out: dict[str, list[Product]] = {}
    for p in products:
        out.setdefault(p.stage, []).append(p)
    return out


# Compatibility alias used by qc_report.py (older name).
def group_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    return products_by_stage(products)


_TASK_COMPLETION: dict[str, list[list[str]]] = {
    # Each inner list is an OR-group (any existing product from the group satisfies that requirement).
    "manifest": [["manifest"]],
    "superbias": [["superbias_fits"]],
    "superflat": [["superflat_fits"]],
    "flatfield": [["flatfield_done"]],
    "cosmics": [["cosmics_summary"]],
    "superneon": [["superneon_fits", "superneon_png"]],
    "lineid_prepare": [["lineid_template_csv"], ["lineid_auto_csv"]],
    "lineid": [["hand_pairs_txt"]],
    "wavesolution": [["lambda_map"], ["wavesolution_2d_json"]],
    "linearize": [["lin_preview_fits"]],
    "skyrect": [["sky_done"]],
    "stack2d": [["stacked2d_fits"]],
    "extract1d": [["spec1d_fits"]],
    "qc_report": [["qc_json"], ["qc_html"]],
}


def products_for_task(cfg: dict[str, Any], task: str) -> list[Product]:
    """Return products that represent completion of a task.

    Used by the GUI runner for skip logic and hashing.
    """

    t = (task or "").strip().lower()
    rules = _TASK_COMPLETION.get(t)
    if not rules:
        return []
    all_products = {p.key: p for p in list_products(cfg)}
    keys: set[str] = set(k for group in rules for k in group)
    return [all_products[k] for k in keys if k in all_products]


def task_is_complete(cfg: dict[str, Any], task: str) -> bool:
    """Return True if task outputs exist on disk."""
    t = (task or "").strip().lower()
    rules = _TASK_COMPLETION.get(t)
    if not rules:
        return False
    prods = {p.key: p for p in list_products(cfg)}
    for group in rules:
        ok = False
        for k in group:
            p = prods.get(k)
            if p is not None and p.exists():
                ok = True
                break
        if not ok:
            return False
    return True
