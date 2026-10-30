from __future__ import annotations

"""Shared extraction facade.

P1.2 keeps science and standard extraction on the same trusted mechanics without
forcing both stages to import each other's entrypoints.

The current science implementation still lives in ``stages.extract1d``.  This
module exposes the reusable primitives through a stable extraction namespace so
new calibration-domain stages can share the same core behavior.
"""

from pathlib import Path
from typing import Any

from scorpio_pipe.stages.extract1d import (
    _boxcar_extract as boxcar_extract,
    _estimate_sky_residual as estimate_sky_residual,
    _linear_wave_axis as linear_wave_axis,
    _read_mef as read_mef,
    _safe_json as safe_json,
    _write_quicklook_png as write_quicklook_png,
    _write_spec1d_fits as write_spec1d_fits,
)
from scorpio_pipe.stages.linearize import (
    _compute_output_grid,
    _get_linearize_grid_plan,
    _process_linearize_frame,
)


def rectify_with_lambda_map(
    *,
    cfg: dict[str, Any],
    in_path: str | Path,
    lambda_map,
    wave0: float,
    dw: float,
    nlam: int,
    unit: str = "Angstrom",
    frame_signature_key: str = "standard_prepare",
    lambda_map_key: str = "lambda_map",
    y0: int = 0,
    y1: int | None = None,
) -> dict[str, Any]:
    """Rectify a single frame using the science linearization kernel.

    The return payload matches the internal dict produced by
    :func:`scorpio_pipe.stages.linearize._process_linearize_frame`.
    """

    import numpy as np

    lam_map = np.asarray(lambda_map, dtype=np.float64)
    ny_full = int(lam_map.shape[0])
    y0i = max(0, int(y0))
    y1i = ny_full if y1 is None else min(ny_full, int(y1))
    if y1i <= y0i:
        raise ValueError("Invalid y crop for standard rectification")
    lam_use = lam_map[y0i:y1i, :]
    wave_edges = float(wave0) + float(dw) * np.arange(int(nlam) + 1, dtype=np.float64)
    plan, _ = _get_linearize_grid_plan(
        lam_use,
        edges_out=wave_edges,
        frame_signature_key=str(frame_signature_key),
        lambda_map_key=str(lambda_map_key),
        wave0=float(wave0),
        dw=float(dw),
        nlam=int(nlam),
    )
    lcfg = (cfg.get("linearize") if isinstance(cfg.get("linearize"), dict) else {}) or {}
    stcfg = (cfg.get("stack2d") if isinstance(cfg.get("stack2d"), dict) else {}) or {}
    ecfg = (cfg.get("extract1d") if isinstance(cfg.get("extract1d"), dict) else {}) or {}
    mode = str(lcfg.get("bunit_mode", lcfg.get("output_bunit_mode", "adu_bin"))).strip().lower()
    density_aliases = {
        "adu/angstrom", "adu_per_angstrom", "adu_per_a", "adu/a",
        "per_angstrom", "density", "adu/nm", "adu_per_nm", "per_nm",
        "adu_per_unit",
    }
    scale_to_bin = mode not in density_aliases
    science_normalize = bool(lcfg.get("normalize_exptime", True))
    extract_input_mode = str(ecfg.get("input_mode", "stack2d")).strip().lower()
    if extract_input_mode in {"stack2d", "stack"}:
        # The science spectrum is a coadd, potentially normalized by Stack2D
        # even when Linearize was configured to retain original counts.
        science_normalize |= bool(stcfg.get("normalize_exptime", True))

    return _process_linearize_frame(
        {
            "path": str(Path(in_path)),
            "lam_map": lam_use,
            "linearize_grid_plan": plan,
            "ny": int(lam_use.shape[0]),
            "nlam": int(nlam),
            "y0": int(y0i),
            "y1": int(y1i),
            "mask_dir": str(Path(in_path).parent),
            "lcfg": dict((cfg.get("linearize") if isinstance(cfg.get("linearize"), dict) else {}) or {}),
            "cfg": dict(cfg),
            "gain_override_f": None,
            "rdnoise_override_f": None,
            # Match the science SPEC1D input units before standard-star
            # sensitivity fitting; mismatched count/rate or bin/density
            # normalization biases the derived absolute response.
            "scale_to_bin": bool(scale_to_bin),
            "normalize_exptime": bool(science_normalize),
            "wave0": float(wave0),
            "dw": float(dw),
            "unit": str(unit),
        }
    )


__all__ = [
    "boxcar_extract",
    "estimate_sky_residual",
    "linear_wave_axis",
    "read_mef",
    "rectify_with_lambda_map",
    "safe_json",
    "write_quicklook_png",
    "write_spec1d_fits",
    "_compute_output_grid",
]
