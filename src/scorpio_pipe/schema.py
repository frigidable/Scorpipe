"""Pydantic schema for config.yaml (v4+).

The pipeline continues to operate on a plain dict/YAML config for backward
compatibility. This schema is used by validation/doctor to catch common issues
and warn about likely typos.

Notes
-----
- We intentionally allow extra keys (forward compatibility).
- `find_unknown_keys()` provides user-facing warnings about typos.
- `schema_validate()` returns a small report object (ok/errors/warnings).
"""


from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------- report objects ----------------------------


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    message: str
    hint: str = ""


@dataclass(frozen=True)
class SchemaReport:
    ok: bool
    errors: List[SchemaIssue]
    warnings: List[SchemaIssue]


# ------------------------------ pydantic ------------------------------


class QCBlock(BaseModel):
    """QC configuration.

    thresholds: numeric values that override auto thresholds.
    auto: whether to compute sensible defaults when thresholds are absent.
    """

    model_config = ConfigDict(extra="allow")

    thresholds: Dict[str, float] = Field(default_factory=dict)
    auto: bool = True


class WavesolBlock(BaseModel):
    """Wave-solution configuration.

    This section is used by multiple stages:
      - superneon: profile extraction, X-alignment, robust noise model, peak detection
      - lineid GUI: amplitude thresholding controls, atlas/line-list paths
      - wavesolution: 1D poly + 2D model (power/chebyshev) and tracing/fit params

    We allow extra keys (forward compatibility) but keep common keys here to
    avoid noisy "unknown key" warnings.
    """

    model_config = ConfigDict(extra="allow")

    disperser: Optional[str] = None
    slit: Optional[str] = None
    binning: Optional[str] = None

    neon_lines_csv: str = "neon_lines.csv"
    atlas_pdf: str = "HeNeAr_atlas.pdf"

    # superneon: profile extraction
    profile_y: Optional[Any] = None  # tuple[int,int] or list[int]
    y_half: int = 20

    # superneon: alignment
    xshift_max_abs: int = 6

    # superneon: robust noise/baseline model for peak detection
    noise: Dict[str, Any] = Field(default_factory=dict)

    # peak detection (in units of robust sigma)
    peak_snr: float = 5.0
    peak_prom_snr: float = 4.0
    peak_floor_snr: float = 3.0
    peak_distance: int = 3
    gauss_half_win: int = 4

    # optional explicit thresholds (ADU); if not set -> auto from sigma
    peak_min_amp: Optional[float] = None
    peak_prominence: Optional[float] = None

    # autotune peak threshold if too few/many peaks are found
    peak_autotune: bool = True
    peak_target_min: int = 0
    peak_target_max: int = 0
    peak_snr_min: float = 2.5
    peak_snr_max: float = 12.0
    peak_snr_relax: float = 0.85
    peak_snr_boost: float = 1.15
    peak_autotune_max_tries: int = 10

    # lineid GUI: amplitude cutoff controls
    gui_min_amp_sigma_k: float = 5.0
    gui_min_amp: Optional[float] = None

    # wavesolution: hand pairs
    hand_pairs_path: Optional[str] = None

    # 1D dispersion
    poly_deg_1d: int = 4
    blend_weight: float = 0.3
    poly_sigma_clip: float = 3.0
    poly_maxiter: int = 10

    # 2D tracing and fit
    model2d: str = "auto"  # auto|power|cheb
    edge_crop_x: int = 12
    edge_crop_y: int = 12

    trace_y0: Optional[int] = None
    trace_template_hw: int = 6
    trace_avg_half: int = 3
    trace_search_rad: int = 12
    trace_y_step: int = 1
    trace_amp_thresh: float = 20.0
    trace_min_pts: int = 120

    power_deg: int = 5
    power_sigma_clip: float = 3.0
    power_maxiter: int = 10

    cheb_degx: int = 5
    cheb_degy: int = 3
    cheb_sigma_clip: float = 3.0
    cheb_maxiter: int = 10

    rejected_lines_A: List[float] = Field(default_factory=list)

    qc: QCBlock = Field(default_factory=QCBlock)


class CalibBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    superbias_path: Optional[str] = None
    superflat_path: Optional[str] = None

    bias_combine: str = "median"  # median|mean
    bias_sigma_clip: float = 0.0


class SuperneonBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    bias_sub: bool = True


class CosmicsBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    # auto | stack_mad | two_frame_diff | laplacian
    method: str = "stack_mad"
    # Global detection threshold (see method-specific notes).
    k: float = 9.0
    bias_subtract: bool = True
    save_png: bool = True
    save_mask_fits: bool = True
    apply_to: List[str] = Field(default_factory=lambda: ["obj", "sky"])  # obj|sky|sunsky|neon

    # --- Common tuning knobs ---
    # Binary mask dilation radius (pixels). 0 disables.
    dilate: int = 1

    # --- stack_mad tuning ---
    # Thresholding uses |x-med| / (mad_scale*MAD) > k.
    mad_scale: float = 1.0
    # Optional floor for MAD (prevents pathological over-masking on flat pixels).
    min_mad: float = 0.0
    # Optional cap for per-frame masked fraction (0..1). None disables.
    max_frac_per_frame: Optional[float] = None
    # Per-method override for dilate (None -> use `dilate`).
    stack_dilate: Optional[int] = None

    # --- two_frame_diff tuning ---
    # Local mean radius for |diff| (used in local threshold term).
    local_r: int = 2
    two_diff_local_r: Optional[int] = None
    # Global threshold factor: max(k2_min, k2_scale*k) * sigma(diff)
    two_diff_k2_scale: float = 0.8
    two_diff_k2_min: float = 5.0
    # Local threshold: thr_local_a*loc + thr_local_b*sigma
    two_diff_thr_local_a: float = 4.0
    two_diff_thr_local_b: float = 2.5
    two_diff_dilate: Optional[int] = None

    # --- laplacian tuning ---
    lap_local_r: Optional[int] = None
    # Laplacian threshold: max(lap_k_min, lap_k_scale*k) * sigma(lap)
    lap_k_scale: float = 0.8
    lap_k_min: float = 5.0
    lap_dilate: Optional[int] = None


class FlatfieldBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    method: str = "median"
    norm: str = "median"  # median|mean
    bias_subtract: bool = True
    save_png: bool = True
    apply_to: List[str] = Field(default_factory=lambda: ["obj", "sky", "sunsky"])  # obj|sky|sunsky|neon


class LinearizeBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    # Wavelength grid (Angstrom). If dlambda_A is None -> auto from lambda_map.
    dlambda_A: Optional[float] = None
    lambda_min_A: Optional[float] = None
    lambda_max_A: Optional[float] = None

    # How to build the common wavelength grid when min/max are not given:
    #   - "intersection": robust intersection across Y (recommended default)
    #   - "percentile": robust global min/max percentiles
    #   - "union": robust union (wider; may create large no-coverage zones)
    grid_mode: str = "intersection"
    # Robust percentiles used by grid_mode (in percent, 0..100)
    grid_lo_pct: float = 1.0
    grid_hi_pct: float = 99.0
    # For intersection mode: use high percentile of per-row minima and low percentile of per-row maxima.
    grid_intersection_min_pct: float = 95.0
    grid_intersection_max_pct: float = 5.0

    # Optional crop in Y before producing outputs (pixels).
    y_crop_top: int = 0
    y_crop_bottom: int = 0

    # Real long-slit workflow needs per-exposure rectification products.
    # ``per_exposure`` is the canonical key; ``save_per_frame`` is kept for
    # older configs/UI and treated as an alias by the implementation.
    per_exposure: bool = True
    save_per_frame: bool = False  # deprecated alias (kept for compatibility)
    save_png: bool = True
    fill_value: float = float("nan")
    # Produce a quick-look stacked linearized frame (for ROI selection / QC).
    # This is NOT used for the final scientific stacking (stack2d does that).
    stack_preview: bool = True

    # Mask propagation policy (uint16 bitmask). "or" is recommended.
    mask_combine: str = "or"  # or|nearest


class SkyBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    method: str = "kelson"

    # Advanced (v5.12+): per-exposure sky subtraction and optional stacking.
    per_exposure: bool = True
    # Stack rectified per-exposure sky-subtracted frames into a combined product.
    # In the GUI this is presented as a checkbox near "Run Sky".
    stack_after: bool = True
    # Persist per-exposure sky model by default (needed for reproducibility & QC).
    save_per_exp_model: bool = True
    # Save a quick-look 1D sky spectrum (mean over sky rows). Useful for QC.
    save_spectrum_1d: bool = True

    # P1: optional flexure (global Δλ) correction per exposure using sky lines (subpixel shift on λ-grid).
    flexure_enabled: bool = False
    flexure_max_shift_pix: int = 5
    flexure: Optional[Dict[str, Any]] = None  # advanced options container

    # Region of interest (pixel indices in the *linearized* frame)
    roi: Dict[str, Any] = Field(default_factory=dict)
    # If ROI is not provided in config and GUI is available, allow interactive ROI selection.
    roi_interactive: bool = False

    # QC: wavelength zones where residuals are reported separately (Angstrom on linear WCS).
    critical_windows_A: List[List[float]] = Field(default_factory=lambda: [[6800.0, 6900.0]])

    # Kelson-like 1D B-spline fit to sky spectrum (in Angstrom)
    bsp_degree: int = 3
    bsp_step_A: float = 3.0
    sigma_clip: float = 3.0
    maxiter: int = 6

    # Simple spatial variation: sky(y,λ) ≈ scale(y)*S(λ) + offset(y)
    use_spatial_scale: bool = True
    spatial_poly_deg: int = 1
    scale_smooth_y: int = 41

    save_sky_model: bool = True
    save_png: bool = True


class Stack2DBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    sigma_clip: float = 4.0
    maxiter: int = 3
    chunk_rows: int = 128

    # P1: optional y-alignment before stacking (subpixel shift on y-grid).
    y_align_enabled: bool = False
    y_align_max_shift_pix: int = 10
    y_align: Optional[Dict[str, Any]] = None  # advanced options container

    save_png: bool = True


class Extract1DBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    # boxcar = aperture sum around trace; optimal = Horne-style (advanced).
    method: str = "boxcar"  # boxcar|optimal|sum|mean (sum/mean kept for compat)
    # Aperture half-width in pixels around the trace.
    aperture_half_width: int = 6
    # Trace estimation (centroid in bins along λ)
    trace_bin_A: float = 60.0
    trace_smooth_deg: int = 3
    # For optimal extraction: build a single profile template from a λ-range.
    optimal_profile_half_width: int = 12
    optimal_sigma_clip: float = 5.0
    save_png: bool = True


class FramesBlock(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    bias: List[str] = Field(default_factory=list)
    flat: List[str] = Field(default_factory=list)
    neon: List[str] = Field(default_factory=list)
    obj: List[str] = Field(default_factory=list)
    sky: List[str] = Field(default_factory=list)
    sunsky: List[str] = Field(default_factory=list)

    # stored under frames.__setup__ in YAML
    setup: Dict[str, Any] = Field(default_factory=dict, alias="__setup__")

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        # Allow a single string where a list is expected (common user typo).
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for k in ("bias", "flat", "neon", "obj", "sky", "sunsky"):
            v = out.get(k)
            if isinstance(v, str):
                out[k] = [v]
        return out


class ConfigSchema(BaseModel):
    """Schema for the *resolved* config dict (after load_config).

    load_config() injects a few computed fields (config_dir, project_root, ...),
    so the schema includes them as optional.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    work_dir: str
    data_dir: str

    frames: FramesBlock = Field(default_factory=FramesBlock)
    calib: CalibBlock = Field(default_factory=CalibBlock)
    cosmics: CosmicsBlock = Field(default_factory=CosmicsBlock)
    flatfield: FlatfieldBlock = Field(default_factory=FlatfieldBlock)
    wavesol: WavesolBlock = Field(default_factory=WavesolBlock)
    superneon: SuperneonBlock = Field(default_factory=SuperneonBlock)
    linearize: LinearizeBlock = Field(default_factory=LinearizeBlock)
    sky: SkyBlock = Field(default_factory=SkyBlock)
    stack2d: Stack2DBlock = Field(default_factory=Stack2DBlock)
    extract1d: Extract1DBlock = Field(default_factory=Extract1DBlock)

    profiles: Optional[Dict[str, Any]] = None

    # computed / meta
    config_path: Optional[str] = None
    config_dir: Optional[str] = None
    project_root: Optional[str] = None
    work_dir_abs: Optional[str] = None
    setup: Optional[Dict[str, Any]] = None
    _profiles_applied: Optional[List[str]] = None


# ----------------------- typo/unknown key support -----------------------


_TOP_KEYS = {
    "work_dir",
    "data_dir",
    "frames",
    "calib",
    "cosmics",
    "flatfield",
    "wavesol",
    "superneon",
    "linearize",
    "sky",
    "stack2d",
    "extract1d",
    "qc",
    "profiles",
    "config_path",
    "config_dir",
    "project_root",
    "work_dir_abs",
    "setup",
    "_profiles_applied",
}

_FRAMES_KEYS = {"bias", "flat", "neon", "obj", "sky", "sunsky", "__setup__"}

_WAVESOL_KEYS = {
    "disperser",
    "slit",
    "binning",
    "neon_lines_csv",
    "atlas_pdf",
    "profile_y",
    "y_half",
    "xshift_max_abs",
    "noise",
    "peak_snr",
    "peak_prom_snr",
    "peak_floor_snr",
    "peak_distance",
    "gauss_half_win",
    "peak_min_amp",
    "peak_prominence",
    "peak_autotune",
    "peak_target_min",
    "peak_target_max",
    "peak_snr_min",
    "peak_snr_max",
    "peak_snr_relax",
    "peak_snr_boost",
    "peak_autotune_max_tries",
    "gui_min_amp_sigma_k",
    "gui_min_amp",
    "poly_deg_1d",
    "blend_weight",
    "poly_sigma_clip",
    "poly_maxiter",
    "hand_pairs_path",
    "model2d",
    "edge_crop_x",
    "edge_crop_y",
    "trace_y0",
    "trace_template_hw",
    "trace_avg_half",
    "trace_search_rad",
    "trace_y_step",
    "trace_amp_thresh",
    "trace_min_pts",
    "power_deg",
    "power_sigma_clip",
    "power_maxiter",
    "cheb_degx",
    "cheb_degy",
    "cheb_sigma_clip",
    "cheb_maxiter",
    "rejected_lines_A",
    "ignore_lines_A",
    "qc",
}

_CALIB_KEYS = {"superbias_path", "superflat_path", "bias_combine", "bias_sigma_clip"}

_SUPERNEON_KEYS = {"bias_sub"}

_COSMICS_KEYS = {
    "enabled",
    "method",
    "k",
    "bias_subtract",
    "save_png",
    "save_mask_fits",
    "apply_to",
    # common
    "dilate",
    # stack_mad
    "mad_scale",
    "min_mad",
    "max_frac_per_frame",
    "stack_dilate",
    # two_frame_diff
    "local_r",
    "two_diff_local_r",
    "two_diff_k2_scale",
    "two_diff_k2_min",
    "two_diff_thr_local_a",
    "two_diff_thr_local_b",
    "two_diff_dilate",
    # laplacian
    "lap_local_r",
    "lap_k_scale",
    "lap_k_min",
    "lap_dilate",
}

_FLATFIELD_KEYS = {"enabled", "method", "norm", "bias_subtract", "save_png", "apply_to"}

_QC_KEYS = {"thresholds", "auto"}


def find_unknown_keys(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return unknown keys grouped by section.

    Keys injected by load_config are considered known.
    """

    unknown: Dict[str, List[str]] = {}

    top_unknown = sorted(k for k in cfg.keys() if str(k) not in _TOP_KEYS)
    if top_unknown:
        unknown["top"] = top_unknown

    frames = cfg.get("frames")
    if isinstance(frames, dict):
        u = sorted(k for k in frames.keys() if str(k) not in _FRAMES_KEYS)
        if u:
            unknown["frames"] = u

    w = cfg.get("wavesol")
    if isinstance(w, dict):
        u = sorted(k for k in w.keys() if str(k) not in _WAVESOL_KEYS)
        if u:
            unknown["wavesol"] = u
        qc = w.get("qc")
        if isinstance(qc, dict):
            u2 = sorted(k for k in qc.keys() if str(k) not in _QC_KEYS)
            if u2:
                unknown["wavesol.qc"] = u2

    c = cfg.get("calib")
    if isinstance(c, dict):
        u = sorted(k for k in c.keys() if str(k) not in _CALIB_KEYS)
        if u:
            unknown["calib"] = u

    sn = cfg.get("superneon")
    if isinstance(sn, dict):
        u = sorted(k for k in sn.keys() if str(k) not in _SUPERNEON_KEYS)
        if u:
            unknown["superneon"] = u

    cm = cfg.get("cosmics")
    if isinstance(cm, dict):
        u = sorted(k for k in cm.keys() if str(k) not in _COSMICS_KEYS)
        if u:
            unknown["cosmics"] = u

    ff = cfg.get("flatfield")
    if isinstance(ff, dict):
        u = sorted(k for k in ff.keys() if str(k) not in _FLATFIELD_KEYS)
        if u:
            unknown["flatfield"] = u

    return unknown


def schema_validate(cfg: Dict[str, Any]) -> SchemaReport:
    """Validate config dict against the pydantic schema."""

    try:
        ConfigSchema.model_validate(cfg)
        unknown = find_unknown_keys(cfg)
        if unknown:
            items: List[str] = []
            for sec, keys in unknown.items():
                for k in keys:
                    items.append(f"{sec}: {k}")
            msg = "Unknown config keys (typos are treated as errors):\n" + "\n".join(items)
            return SchemaReport(
                ok=False,
                errors=[SchemaIssue(code="UNKNOWN_KEYS", message=msg, hint="Remove/rename unknown keys")],
                warnings=[],
            )
        return SchemaReport(ok=True, errors=[], warnings=[])
    except Exception as e:
        # Keep it human-readable; detailed trace is not useful for users.
        msg = str(e)
        if len(msg) > 2000:
            msg = msg[:2000] + "…"
        return SchemaReport(
            ok=False,
            errors=[SchemaIssue(code="SCHEMA", message=msg, hint="Check config types/sections")],
            warnings=[],
        )
