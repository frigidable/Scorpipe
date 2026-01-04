"""Stage contract registry (inputs/outputs/metrics).

This module provides a small, explicit "contract" for each pipeline stage.
The contract is used by tests/QC/UI to keep behavior stable as the code evolves.

Design goals
------------
* **Stable**: contracts change rarely and only with clear version bumps.
* **Concrete**: outputs are expressed as *product keys* from :mod:`scorpio_pipe.products`.
* **Non-invasive**: stage code does not need to depend on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scorpio_pipe.products import list_products


@dataclass(frozen=True)
class StageContract:
    stage: str
    title: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]  # product keys
    metrics: tuple[str, ...] = ()
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metrics": list(self.metrics),
            "notes": self.notes,
        }


# Contracts are keyed by *canonical* task names as used by the GUI runner.
CONTRACTS: dict[str, StageContract] = {
    "manifest": StageContract(
        stage="manifest",
        title="Manifest / provenance",
        inputs=("config.yaml", "raw frames referenced by config"),
        outputs=("manifest",),
        metrics=("provenance",),
    ),
    "superbias": StageContract(
        stage="superbias",
        title="Superbias",
        inputs=("raw/bias/*.fits",),
        outputs=("superbias_fits",),
        metrics=("superbias.n_inputs", "superbias.combine", "superbias.sigma_clip"),
    ),
    "superflat": StageContract(
        stage="superflat",
        title="Superflat",
        inputs=("raw/flat/*.fits", "superbias"),
        outputs=("superflat_fits",),
    ),
    "flatfield": StageContract(
        stage="flatfield",
        title="Flatfielding",
        inputs=("raw/obj/*.fits", "superflat", "superbias"),
        outputs=("flatfield_done",),
    ),
    "cosmics": StageContract(
        stage="cosmics",
        title="Cosmic ray cleaning",
        inputs=("raw/obj/*.fits", "superbias"),
        outputs=("cosmics_summary",),
        metrics=("cosmics.replaced_fraction", "cosmics.kind"),
    ),
    "superneon": StageContract(
        stage="superneon",
        title="SuperNeon (arc stack + peak candidates)",
        inputs=("raw/neon*.fits", "superbias (optional)"),
        outputs=("superneon_fits", "superneon_png", "peaks_candidates_csv"),
        metrics=("superneon.n_peaks", "superneon.snr"),
    ),
    "lineid_prepare": StageContract(
        stage="lineid_prepare",
        title="LineID preparation (templates/auto pairs)",
        inputs=("superneon_fits", "line atlas resources"),
        outputs=("lineid_template_csv", "lineid_auto_csv", "lineid_report_txt"),
    ),
    "wavesolution": StageContract(
        stage="wavesolution",
        title="Wavelength solution (1D + 2D lambda map)",
        inputs=("peaks_candidates_csv", "hand_pairs.txt (from LineID GUI or library)"),
        outputs=(
            "wavesolution_1d_png",
            "wavesolution_1d_json",
            "wavesolution_2d_json",
            "lambda_map",
            "rectification_model_json",
            "wave_done_json",
            "residuals_1d_csv",
            "residuals_2d_csv",
            "control_points_2d_csv",
            "residuals_2d_png",
            "residuals_2d_audit_png",
            "residuals_vs_lambda_png",
            "residuals_vs_y_png",
            "residuals_hist_png",
            "wavesolution_report_txt",
        ),
        metrics=(
            "wavesol.rms1d_A",
            "wavesol.wrms1d_A",
            "wavesol.sigma1d_mad_A",
            "wavesol.rms2d_A",
            "wavesol.wrms2d_A",
            "wavesol.sigma2d_mad_A",
            "wavesol.n_pairs_used",
            "wavesol.n_lines_used",
        ),
        notes="Manual line rejection must recompute RMS on the *same* residual definition as QC.",
    ),
    "linearize": StageContract(
        stage="linearize",
        title="Linearize (apply 2D lambda map)",
        inputs=("raw/obj/*.fits", "lambda_map.fits"),
        outputs=("lin_preview_fits", "lin_preview_png", "linearize_qc"),
    ),
    "sky": StageContract(
        stage="sky",
        title="Sky subtraction (per exposure)",
        inputs=(
            "05_cosmics/*_clean.fits (preferred) OR raw/obj/*.fits (RAW detector geometry)",
            "05_cosmics/*_mask.fits (optional)",
            "08_wavesol/lambda_map.fits (\u03bb(x,y) in RAW geometry)",
            "ROI/geometry settings (cfg.sky.geometry.*)",
        ),
        outputs=("sky_done", "qc_sky_json", "sky_preview_fits", "sky_preview_png"),
        metrics=("sky.rms_sky", "sky.flexure_shift_A"),
    ),
    "stack2d": StageContract(
        stage="stack2d",
        title="2D stacking",
        inputs=("per-exp sky-subtracted frames (*_skysub.fits)",),
        outputs=("stack2d_done", "stacked2d_fits", "coverage_png"),
    ),
    "extract1d": StageContract(
        stage="extract1d",
        title="1D extraction",
        inputs=(
            "11_stack/stack2d.fits (default) or 10_linearize/<stem>_skysub.fits (extract1d.input_mode=single_frame)",
        ),
        outputs=("extract_done", "trace_json", "spec1d_fits", "spec1d_png", "extract1d_done"),
    ),
    "qc_report": StageContract(
        stage="qc_report",
        title="QC report",
        inputs=("products registry", "stage state", "timings"),
        outputs=("qc_json", "qc_html", "timings"),
    ),
}


def list_stage_contracts() -> dict[str, StageContract]:
    return dict(CONTRACTS)


def validate_contracts(cfg: Mapping[str, Any]) -> None:
    """Validate contracts against the current product registry.

    Parameters
    ----------
    cfg
        Any object acceptable by :func:`scorpio_pipe.products.list_products`.
    """

    product_keys = {p.key for p in list_products(dict(cfg))}
    for st, c in CONTRACTS.items():
        if st != c.stage:
            raise ValueError(f"Contract key '{st}' != contract.stage '{c.stage}'")
        missing = [k for k in c.outputs if k not in product_keys]
        if missing:
            raise ValueError(
                f"Contract '{st}' refers to unknown product keys: {missing}"
            )
