"""UI-only metadata for parameters.

Single source of truth for compact, consistent GUI parameter rendering.

Keys are dotted config paths used by the UI (e.g. ``"wavesol.poly_deg_1d"``).
Tooltips are intentionally short (2–4 lines). Keep long explanations in docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ParamMeta:
    label: str
    tooltip: str
    units: Optional[str] = None
    typical: Optional[str] = None


# NOTE: Keep tooltips concrete and short. Default is appended by the UI.
PARAM_META: Dict[str, ParamMeta] = {
    # ------------------------------------------------------------------
    # Wavesolution (Wavelength Solution)
    # ------------------------------------------------------------------
    "wavesol.poly_deg_1d": ParamMeta(
        label="1D poly degree",
        tooltip=(
            "Degree of per-row 1D fit λ(x).\n"
            "Higher: more flexible • risk of overfit.\n"
            "Use small degrees unless residuals demand more."
        ),
        typical="3–6",
    ),
    "wavesol.poly_sigma_clip": ParamMeta(
        label="1D sigma clip",
        units="σ",
        tooltip=(
            "Sigma-clipping for 1D fit residuals.\n"
            "Lower: stricter • may over-clip.\n"
            "Higher: keeps more points • tolerates outliers."
        ),
        typical="2–5",
    ),
    "wavesol.poly_maxiter": ParamMeta(
        label="1D max iters",
        tooltip=(
            "Max clipping iterations for the 1D fit.\n"
            "More iters: safer convergence • slower.\n"
            "Stop early when the mask stabilizes."
        ),
        typical="5–15",
    ),
    "wavesol.blend_weight": ParamMeta(
        label="Blend weight",
        tooltip=(
            "Blend between 1D and 2D models (0..1).\n"
            "0: 1D only. 1: 2D only.\n"
            "Use mid-values when 2D helps but is imperfect."
        ),
        typical="0.3–0.8",
    ),
    "wavesol.model2d": ParamMeta(
        label="2D model",
        tooltip=(
            "2D λ(x,y) residual model family.\n"
            "Auto: choose best. Power: power basis.\n"
            "Cheb: separable Chebyshev (degX/degY)."
        ),
    ),
    "wavesol.power_deg": ParamMeta(
        label="Power degree",
        tooltip=(
            "Degree for the power-basis 2D model.\n"
            "Higher: more flexibility • risk of ringing.\n"
            "Keep low unless residual maps demand more."
        ),
        typical="1–4",
    ),
    "wavesol.power_sigma_clip": ParamMeta(
        label="Power sigma clip",
        units="σ",
        tooltip=(
            "Sigma-clipping for 2D power-model fit residuals.\n"
            "Lower: stricter rejection.\n"
            "Tune if the solution becomes unstable."
        ),
        typical="2–5",
    ),
    "wavesol.power_maxiter": ParamMeta(
        label="Power max iters",
        tooltip=(
            "Max clipping iterations for 2D power-model fit.\n"
            "More iters: safer • slower.\n"
            "Prefer moderate values."
        ),
        typical="5–15",
    ),
    "wavesol.cheb_degx": ParamMeta(
        label="Cheb deg X",
        tooltip=(
            "Chebyshev degree along dispersion (X).\n"
            "Higher: fits more structure • may overfit.\n"
            "Increase only if residual maps show trends."
        ),
        typical="3–8",
    ),
    "wavesol.cheb_degy": ParamMeta(
        label="Cheb deg Y",
        tooltip=(
            "Chebyshev degree along spatial axis (Y).\n"
            "Usually lower than degX for long-slit.\n"
            "Increase only if needed."
        ),
        typical="1–5",
    ),
    "wavesol.cheb_sigma_clip": ParamMeta(
        label="Cheb sigma clip",
        units="σ",
        tooltip=(
            "Sigma-clipping for 2D Chebyshev fit residuals.\n"
            "Lower: stricter. Higher: tolerates outliers.\n"
            "Tune if you see unstable solutions."
        ),
        typical="2–5",
    ),
    "wavesol.cheb_maxiter": ParamMeta(
        label="Cheb max iters",
        tooltip=(
            "Max clipping iterations for 2D Chebyshev fit.\n"
            "More iters: safer convergence • slower.\n"
            "Stop when the mask stabilizes."
        ),
        typical="5–15",
    ),
    "wavesol.edge_crop_x": ParamMeta(
        label="Edge crop X",
        units="px",
        tooltip=(
            "Ignore edge pixels in X during modeling.\n"
            "Helps when borders have defects/poor calibration.\n"
            "Increase if edges dominate residuals."
        ),
        typical="0–50",
    ),
    "wavesol.edge_crop_y": ParamMeta(
        label="Edge crop Y",
        units="px",
        tooltip=(
            "Ignore edge pixels in Y during modeling.\n"
            "Useful if slit edges are vignetted/noisy.\n"
            "Increase if trace is unstable near borders."
        ),
        typical="0–50",
    ),

    # Trace controls
    "wavesol.trace_template_hw": ParamMeta(
        label="Template half-width",
        units="px",
        tooltip=(
            "Half-width of the template window for trace matching.\n"
            "Larger: more robust • less local.\n"
            "Use a few × FWHM in pixels."
        ),
        typical="4–12",
    ),
    "wavesol.trace_avg_half": ParamMeta(
        label="Row avg half",
        units="px",
        tooltip=(
            "Half-window for averaging rows before tracing.\n"
            "Larger: smoother • may blur structure.\n"
            "Smaller: more detail • noisier."
        ),
        typical="0–2",
    ),
    "wavesol.trace_search_rad": ParamMeta(
        label="Search radius",
        units="px",
        tooltip=(
            "Search radius around predicted trace position.\n"
            "Larger: finds trace under drift • slower.\n"
            "Too large may jump to wrong features."
        ),
        typical="5–30",
    ),
    "wavesol.trace_y_step": ParamMeta(
        label="Trace Y step",
        units="px",
        tooltip=(
            "Step in Y between trace anchor points.\n"
            "Smaller: denser sampling • slower.\n"
            "Larger: faster • less detailed."
        ),
        typical="1–3",
    ),
    "wavesol.trace_amp_thresh": ParamMeta(
        label="Amp threshold",
        units="σ",
        tooltip=(
            "Min amplitude (in σ) to accept a trace point.\n"
            "Higher: fewer false points • may miss faint trace.\n"
            "Lower: more points • risk of noise hits."
        ),
        typical="3–10",
    ),
    "wavesol.trace_min_pts": ParamMeta(
        label="Min points",
        tooltip=(
            "Minimum accepted trace points.\n"
            "Too low: unstable. Too high: may fail on short slits.\n"
            "Tune for slit length and S/N."
        ),
        typical="20–200",
    ),
    "wavesol.trace_y0": ParamMeta(
        label="Trace y0",
        units="px",
        tooltip=(
            "Optional fixed starting y-position for tracing.\n"
            "Auto: estimate from data. Set value to lock start.\n"
            "Use when auto locks to a wrong feature."
        ),
        typical="auto",
    ),

    # ------------------------------------------------------------------
    # SuperNeon peak finding (stored under wavesol.* in config)
    # ------------------------------------------------------------------
    "superneon.bias_sub": ParamMeta(
        label="Bias subtraction",
        tooltip=(
            "Subtract superbias from NEON frames before stacking.\n"
            "Disable only if inputs are already bias-subtracted.\n"
            "Recommended ON for raw NEON frames."
        ),
    ),
    "wavesol.y_half": ParamMeta(
        label="Profile half-height",
        units="px",
        tooltip=(
            "Half-height in Y for building 1D slit-summed profile.\n"
            "Smaller: noisier. Larger: mixes gradients.\n"
            "Tune to seeing/binning and LSF width."
        ),
        typical="10–40",
    ),
    "wavesol.xshift_max_abs": ParamMeta(
        label="Max |x-shift|",
        units="px",
        tooltip=(
            "Max absolute X shift when aligning NEON frames.\n"
            "Increase if frames drift; keep small if stable.\n"
            "Too large may allow wrong alignment."
        ),
        typical="2–8",
    ),
    "wavesol.peak_snr": ParamMeta(
        label="Peak SNR",
        units="σ",
        tooltip=(
            "Peak detection threshold in robust σ units.\n"
            "Lower: more peaks • more false detections.\n"
            "Higher: cleaner list • may miss weak lines."
        ),
        typical="4–8",
    ),
    "wavesol.peak_prom_snr": ParamMeta(
        label="Prominence",
        units="σ",
        tooltip=(
            "Required peak prominence above local baseline (in σ).\n"
            "Higher: suppresses small bumps.\n"
            "Use to reduce noise-induced peaks."
        ),
        typical="3–7",
    ),
    "wavesol.peak_floor_snr": ParamMeta(
        label="Floor",
        units="σ",
        tooltip=(
            "Minimum peak height above local floor (in σ).\n"
            "Higher: fewer false peaks.\n"
            "Lower: keeps weak lines."
        ),
        typical="2–5",
    ),
    "wavesol.peak_distance": ParamMeta(
        label="Min distance",
        units="px",
        tooltip=(
            "Minimum distance between peaks along X.\n"
            "Smaller: allows blended lines.\n"
            "Larger: prevents double-detection of same line."
        ),
        typical="2–6",
    ),
    "wavesol.peak_autotune": ParamMeta(
        label="Auto-tune",
        tooltip=(
            "Auto-adjust Peak SNR if too few/many peaks found.\n"
            "Helpful for varying lamp intensity/exposure.\n"
            "Recommended ON for real data."
        ),
    ),
    # Composite/pseudo labels (for paired controls)
    "wavesol.peak_target_range": ParamMeta(
        label="Target peaks",
        units="peaks",
        tooltip=(
            "Desired peak count range for auto-tune.\n"
            "(0,0) disables peak-count control.\n"
            "Use wide bounds; narrow bounds may oscillate."
        ),
    ),
    "wavesol.peak_snr_bounds": ParamMeta(
        label="Auto-tune SNR bounds",
        units="σ",
        tooltip=(
            "Bounds on Peak SNR during auto-tune.\n"
            "min: how low it may relax; max: how high it may boost.\n"
            "Widen if auto-tune hits limits."
        ),
        typical="min 2–4, max 10–20",
    ),
    "wavesol.peak_snr_relax_boost": ParamMeta(
        label="Relax/Boost",
        tooltip=(
            "Multipliers for decreasing/increasing Peak SNR in auto-tune.\n"
            "Relax < 1 lowers the threshold; Boost > 1 raises it.\n"
            "Use gentle steps to avoid oscillations."
        ),
        typical="relax 0.8–0.9, boost 1.1–1.3",
    ),
    # Underlying paired fields (for per-field tooltips/defaults)
    "wavesol.peak_target_min": ParamMeta(
        label="Target min",
        units="peaks",
        tooltip=(
            "Lower bound on desired number of detected peaks.\n"
            "Used only when auto-tune is enabled.\n"
            "Set 0 to disable bound."
        ),
    ),
    "wavesol.peak_target_max": ParamMeta(
        label="Target max",
        units="peaks",
        tooltip=(
            "Upper bound on desired number of detected peaks.\n"
            "Used only when auto-tune is enabled.\n"
            "Set 0 to disable bound."
        ),
    ),
    "wavesol.noise.baseline_bin_size": ParamMeta(
        label="Baseline bin",
        units="px",
        tooltip=(
            "Bin size along X for estimating baseline/noise.\n"
            "Larger: smoother baseline • less local detail.\n"
            "Tune for grism dispersion and line density."
        ),
        typical="20–80",
    ),
    "wavesol.noise.baseline_quantile": ParamMeta(
        label="Baseline quantile",
        tooltip=(
            "Quantile (0..1) used as baseline level in each bin.\n"
            "Lower: closer to floor. Higher: closer to median.\n"
            "Use to avoid bright lines biasing baseline."
        ),
        typical="0.2–0.5",
    ),
    "wavesol.noise.baseline_smooth_bins": ParamMeta(
        label="Baseline smooth",
        units="bins",
        tooltip=(
            "Smoothing width for baseline in bin units.\n"
            "Higher: smoother baseline • may wash broad trends.\n"
            "Set 0 for no smoothing."
        ),
        typical="2–8",
    ),
    "wavesol.noise.empty_quantile": ParamMeta(
        label="Empty quantile",
        tooltip=(
            "Quantile (0..1) for estimating empty/noise-only level.\n"
            "Lower: more conservative (closer to floor).\n"
            "Tune if noise estimate is biased."
        ),
        typical="0.05–0.2",
    ),
    "wavesol.noise.clip": ParamMeta(
        label="Noise clip",
        units="σ",
        tooltip=(
            "Sigma-clipping when estimating robust σ.\n"
            "0 disables clipping.\n"
            "Use if outliers bias the noise model."
        ),
        typical="2–5",
    ),
    "wavesol.noise.n_iter": ParamMeta(
        label="Noise iters",
        tooltip=(
            "Iterations for sigma-clipping in noise estimation.\n"
            "More: safer • slower.\n"
            "Stop when mask stabilizes."
        ),
        typical="2–8",
    ),
    "wavesol.gauss_half_win": ParamMeta(
        label="Gaussian half-window",
        units="px",
        tooltip=(
            "Half-window (px) for Gaussian peak refinement.\n"
            "Larger: more context • can include neighbors.\n"
            "Smaller: sharper • may be noisy."
        ),
        typical="3–8",
    ),
    "wavesol.peak_snr_min": ParamMeta(
        label="SNR min",
        units="σ",
        tooltip=(
            "Lower bound for Peak SNR during auto-tune.\n"
            "Prevents threshold from going too low.\n"
            "Increase if too many false peaks appear."
        ),
    ),
    "wavesol.peak_snr_max": ParamMeta(
        label="SNR max",
        units="σ",
        tooltip=(
            "Upper bound for Peak SNR during auto-tune.\n"
            "Prevents threshold from going too high.\n"
            "Increase if weak lines are missed."
        ),
    ),
    "wavesol.peak_snr_relax": ParamMeta(
        label="Relax",
        tooltip=(
            "Multiply Peak SNR by this factor to relax threshold (< 1).\n"
            "Smaller: faster relaxation • more risk of false peaks.\n"
            "Use gentle steps."
        ),
    ),
    "wavesol.peak_snr_boost": ParamMeta(
        label="Boost",
        tooltip=(
            "Multiply Peak SNR by this factor to increase threshold (> 1).\n"
            "Larger: faster suppression of false peaks.\n"
            "Use gentle steps to avoid overshooting."
        ),
    ),
    "wavesol.peak_autotune_max_tries": ParamMeta(
        label="Max tries",
        tooltip=(
            "Max auto-tune attempts per stack.\n"
            "More tries: more robust • slower.\n"
            "Increase only if auto-tune frequently fails."
        ),
        typical="6–15",
    ),

    # ------------------------------------------------------------------
    # Linearize
    # ------------------------------------------------------------------
    "linearize.enabled": ParamMeta(
        label="Enable",
        tooltip=(
            "Enable linearization (rectification + resampling).\n"
            "Disable only for debugging/custom workflows.\n"
            "Normally kept ON."
        ),
    ),
    "linearize.dlambda_A": ParamMeta(
        label="Δλ",
        units="Å",
        tooltip=(
            "Resampling step for rectified wavelength grid.\n"
            "Smaller: finer grid • slower. Larger: faster • more interpolation.\n"
            "Auto uses instrument solution/resolution."
        ),
        typical="auto or 0.5–2",
    ),
    "linearize.lambda_min_A": ParamMeta(
        label="λ min",
        units="Å",
        tooltip=(
            "Lower limit of output wavelength range.\n"
            "Auto uses grism limits + solution coverage.\n"
            "Set manually for consistent cuts across frames."
        ),
        typical="auto",
    ),
    "linearize.lambda_max_A": ParamMeta(
        label="λ max",
        units="Å",
        tooltip=(
            "Upper limit of output wavelength range.\n"
            "Auto uses grism limits + solution coverage.\n"
            "Set manually for consistent cuts across frames."
        ),
        typical="auto",
    ),
    "linearize.y_crop_top": ParamMeta(
        label="Crop top",
        units="px",
        tooltip=(
            "Crop rows from the top (spatial axis).\n"
            "Use to exclude vignetted/noisy slit edges.\n"
            "0 means no crop."
        ),
        typical="0–50",
    ),
    "linearize.y_crop_bottom": ParamMeta(
        label="Crop bottom",
        units="px",
        tooltip=(
            "Crop rows from the bottom (spatial axis).\n"
            "Use to exclude vignetted/noisy slit edges.\n"
            "0 means no crop."
        ),
        typical="0–50",
    ),
    "linearize.save_png": ParamMeta(
        label="Save PNG",
        tooltip=(
            "Write quick-look PNG diagnostics for rectified frames.\n"
            "Useful for QA; adds a little I/O overhead.\n"
            "Recommended ON while tuning."
        ),
    ),
    "linearize.save_per_frame": ParamMeta(
        label="Save per frame",
        tooltip=(
            "Save rectified outputs for each input frame.\n"
            "Useful for debugging; increases disk usage.\n"
            "Recommended OFF for routine reductions."
        ),
    ),
}


def get_param_meta(cfg_path: str) -> Optional[ParamMeta]:
    return PARAM_META.get(cfg_path)
