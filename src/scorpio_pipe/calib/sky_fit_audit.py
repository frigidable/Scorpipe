"""Read-only fidelity audit of an ACTUAL frozen Kelson sky fit and FITS products.

Checks model reconstruction on the native CCD, the recorded LSQR residual,
and FITS-to-NPZ identity. Does not modify science arrays or certify the
uncertainty of a data-dependent sky fit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.sparse import diags, hstack

from scorpio_pipe.calib.sky_fit_capture import _REQUIRED, _VERSION
from scorpio_pipe.stages.sky_sub import _bspline_design_matrix


def _required_hdu(hdul: fits.HDUList, name: str) -> np.ndarray:
    if name not in hdul or hdul[name].data is None:
        raise ValueError(f"Sky FITS product missing {name} image")
    return hdul[name].data


def audit_frozen_kelson_products(
    frozen_fit_path: Path,
    skymodel_fits: Path,
    skysub_fits: Path,
    *,
    model_rtol: float = 1e-5,
    model_atol: float = 2e-4,
    residual_rtol: float = 1e-2,
    chunk_rows: int = 32,
) -> dict[str, Any]:
    """Check a captured fit against the two actual sky FITS products.

    The LSQR residual norm is evaluated on the actual FINAL clipping set.
    A successful result is an archival/data-fidelity check, NOT an
    endorsement of LSQR numerical rank or conditional astrophysical VAR.
    """
    fitpath = Path(frozen_fit_path)
    modelpath = Path(skymodel_fits)
    subpath = Path(skysub_fits)
    if (not np.isfinite(model_rtol) or model_rtol < 0
            or not np.isfinite(model_atol) or model_atol < 0
            or not np.isfinite(residual_rtol) or residual_rtol < 0
            or int(chunk_rows) < 1):
        raise ValueError("Audit tolerances and chunk_rows must be finite and valid")

    with np.load(fitpath, allow_pickle=False) as data:
        if set(data.files) != _REQUIRED:
            raise ValueError("Frozen Kelson archive schema mismatch")
        if int(data["version"]) != _VERSION:
            raise ValueError("Unsupported frozen Kelson archive version")
        if str(data["status"].item()) != "FROZEN_CONDITIONAL":
            raise ValueError("Frozen Kelson archive is not conditional fit")
        lam = np.asarray(data["lambda_map"], dtype=np.float64)
        y_rows = np.asarray(data["y_norm_rows"], dtype=np.float64)
        index = np.asarray(data["fit_indices"])
        var = np.asarray(data["fit_variance"], dtype=np.float64)
        knots = np.asarray(data["knots"], dtype=np.float64)
        degree = int(data["degree"])
        y_order = int(data["y_order"])
        c0 = np.asarray(data["coeff0"], dtype=np.float64)
        c1 = np.asarray(data["coeff1"], dtype=np.float64)
        recorded_residual = float(data["lsqr_residual"])
        solver_stop = int(data["lsqr_stop"])
        solver_iters = int(data["lsqr_iterations"])
        solver_limit = int(data["lsqr_iter_limit"])
        solver_condition = float(data["lsqr_condition"])
    if (lam.ndim != 2 or not all(s > 0 for s in lam.shape)
            or not np.isfinite(lam).all()):
        raise ValueError("Frozen native wavelength map is invalid")
    ny, nx = lam.shape
    if (y_rows.shape != (ny,) or not np.isfinite(y_rows).all()
            or index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
            or index.size == 0 or len(np.unique(index)) != index.size
            or np.any(index < 0) or np.any(index >= lam.size)
            or var.shape != index.shape or not np.isfinite(var).all()
            or np.any(var <= 0)):
        raise ValueError("Frozen selected-pixel geometry or variance invalid")
    if (degree < 1 or y_order not in (0, 1) or knots.ndim != 1
            or not np.isfinite(knots).all()
            or knots.size < degree + 2
            or np.any(np.diff(knots) < 0)):
        raise ValueError("Frozen B-spline geometry invalid")
    n_basis = knots.size - degree - 1
    if (c0.shape != (n_basis,) or not np.isfinite(c0).all()
            or c1.shape != ((n_basis,) if y_order else (0,))
            or not np.isfinite(c1).all()):
        raise ValueError("Frozen sky model coefficient geometry invalid")
    if (solver_stop not in range(8) or solver_limit <= 0
            or solver_iters < 0 or solver_iters > solver_limit
            or not np.isfinite(recorded_residual) or recorded_residual < 0
            or not np.isfinite(solver_condition) or solver_condition < 0):
        raise ValueError("Frozen LSQR diagnostics invalid")

    max_abs = 0.0
    sumsq = 0.0
    total_px = 0
    max_ratio = 0.0
    with fits.open(modelpath, memmap=True) as fm, fits.open(subpath, memmap=True) as fs:
        model = _required_hdu(fm, "SCI")
        residual = _required_hdu(fs, "SCI")
        if model.shape != lam.shape or residual.shape != lam.shape:
            raise ValueError("FITS and frozen model CCD geometries differ")
        for hdu in (fm[0], fs[0]):
            hdr = hdu.header
            if (str(hdr.get("SKYMD") or "").lower() != "kelson_raw"
                    or int(hdr.get("SKYFRZV") or 0) != _VERSION
                    or str(hdr.get("SKYFRZQ") or "") != "FIT_ONLY"
                    or str(hdr.get("SKYFRZP") or "") != fitpath.name):
                raise ValueError("FITS/NPZ frozen-fit identity cards disagree")
        for i in range(0, ny, int(chunk_rows)):
            j = min(ny, i + int(chunk_rows))
            lchunk = lam[i:j].ravel()
            B = _bspline_design_matrix(lchunk, knots, degree)
            predicted = np.asarray(B @ c0, dtype=np.float64).reshape(j-i, nx)
            if y_order:
                predicted += (
                    np.asarray(B @ c1).reshape(j-i, nx)
                    * y_rows[i:j, None]
                )
            expected = np.asarray(model[i:j], dtype=np.float64)
            if not np.isfinite(expected).all() or not np.isfinite(predicted).all():
                raise ValueError("Nonfinite FITS/model values")
            diff = np.abs(predicted - expected)
            allowed = model_atol + model_rtol * np.abs(expected)
            if np.any(diff > allowed):
                raise ValueError("Frozen Kelson coefficients do not reproduce skymodel FITS")
            max_abs = max(max_abs, float(np.max(diff)))
            sumsq += float(np.sum(diff * diff, dtype=np.float64))
            total_px += diff.size
            max_ratio = max(max_ratio, float(np.max(diff / np.maximum(allowed, 1e-30))))

        # The actual FITS subtracted-sky pixels are the output LSQR fit
        # residuals on the saved FINAL fitted samples.
        rr = np.asarray(residual.ravel()[index], dtype=np.float64)
        if not np.isfinite(rr).all():
            raise ValueError("Nonfinite sky FITS residual on final fit pixels")
        measured_residual = float(np.linalg.norm(rr / np.sqrt(var)))
        if not np.isclose(
            measured_residual, recorded_residual,
            rtol=residual_rtol, atol=model_atol * np.sqrt(index.size),
        ):
            raise ValueError("Actual sky residual is inconsistent with archived LSQR")

    return {
        "status": "MATCH",
        "model_max_abs": max_abs,
        "model_rms": float(np.sqrt(sumsq / total_px)),
        "model_max_allowed_fraction": max_ratio,
        "lsqr_reported_residual_norm": recorded_residual,
        "lsqr_measured_residual_norm": measured_residual,
        "lsqr_residual_relative_difference": (
            float(abs(measured_residual-recorded_residual)
                  / max(recorded_residual, np.finfo(float).tiny))
        ),
        "lsqr_stop_code": solver_stop,
        "lsqr_iterations": solver_iters,
        "lsqr_condition_estimate": solver_condition,
        "lsqr_convergence_reported": solver_stop in (1, 2),
        "n_selected_fit_pixels": int(index.size),
        "n_native_pixels": int(ny * nx),
        "science_uncertainty_qualified": False,
    }
