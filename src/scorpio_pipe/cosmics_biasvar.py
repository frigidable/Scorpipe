"""Typed MasterBias variance sidecar for primary-only Cosmics CLEAN images.

Cosmics preserves its legacy primary-image FITS contract. When it actually
subtracts an *indexed, readout-matched* MasterBias, the unmodified MasterBias
variance map is persisted in an adjacent typed sidecar. Downstream stages
must add this ADU**2 term exactly once to an independently estimated photon
plus CCD read-noise variance, before conversion to electron units.

This sidecar carries the *existing upstream declared* MasterBias VAR. Its
statistical interpretation must be audited independently (scatter of input
bias frames versus uncertainty of their combined master estimator).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic_fits import atomic_write_fits


def write_clean_biasvar_sidecar(
    clean_path: Path,
    *,
    var_adu2: np.ndarray,
    group_id: str,
) -> Path:
    path = Path(clean_path)
    if not path.stem.endswith("_clean") or not group_id:
        raise ValueError("MasterBias VAR sidecar needs a CLEAN image and group id")
    var = np.asarray(var_adu2, dtype=float)
    if var.ndim != 2 or not np.all(np.isfinite(var)) or np.any(var < 0):
        raise ValueError("MasterBias VAR must be a finite nonnegative 2D image")
    dest = path.with_name(path.stem + "_biasvar.fits")
    header = fits.Header()
    header["BUNIT"] = ("ADU^2", "MasterBias variance unit")
    header["SB_GID"] = (group_id, "Exact MasterBias readout group")
    header["BVERS"] = (1, "Bias variance sidecar schema version")
    atomic_write_fits(dest, var.astype(np.float32), header=header, overwrite=True)
    return dest


def read_clean_biasvar_sidecar(
    clean_path: Path,
    hdr: fits.Header,
    sci_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Read a linked MasterBias VAR; corrupt or missing links fail closed."""
    ref = str(hdr.get("BVARREF") or "").strip()
    if not ref:
        return None
    path = Path(clean_path)
    expected = path.with_name(path.stem + "_biasvar.fits")
    if not path.stem.endswith("_clean") or ref != expected.name:
        raise ValueError(f"Untrusted or inconsistent MasterBias BVARREF: {ref}")
    if not bool(hdr.get("BIASSUB", False)):
        raise ValueError("MasterBias VAR link requires BIASSUB=True")
    from scorpio_pipe.units_model import UnitModel, infer_unit_model

    if infer_unit_model(hdr, default=UnitModel.ADU) != UnitModel.ADU:
        raise ValueError("Pending MasterBias VAR requires ADU science units")
    gid = str(hdr.get("SB_GID") or "").strip()
    if not gid:
        raise ValueError("Linked MasterBias VAR requires SB_GID")
    if bool(hdr.get("BVARINC", False)):
        raise ValueError("MasterBias VAR already included; extra addition is unsafe")
    if not expected.is_file():
        raise ValueError(f"Missing linked MasterBias VAR: {expected}")
    with fits.open(expected, memmap=False) as hdul:
        if len(hdul) != 1 or hdul[0].data is None:
            raise ValueError(f"Invalid linked MasterBias VAR layout: {expected}")
        side = hdul[0].header
        if str(side.get("BUNIT") or "").strip().upper() != "ADU^2":
            raise ValueError(f"Linked MasterBias VAR has invalid units: {expected}")
        if str(side.get("SB_GID") or "").strip() != gid:
            raise ValueError(f"Linked MasterBias VAR group mismatch: {expected}")
        if int(side.get("BVERS") or 0) != 1:
            raise ValueError(f"Linked MasterBias VAR schema mismatch: {expected}")
        arr = np.asarray(hdul[0].data)
        if arr.shape != tuple(sci_shape):
            raise ValueError(
                f"MasterBias VAR shape {arr.shape} != science {sci_shape}"
            )
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("Linked MasterBias VAR is not floating-point")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("Linked MasterBias VAR contains unphysical values")
        return np.asarray(arr, dtype=np.float32)


def validate_bias_var_extra(
    baseline_var_adu2: np.ndarray,
    bias_var_adu2: np.ndarray,
    sci_shape: tuple[int, ...],
) -> np.ndarray:
    """Include one pending MasterBias term in the sky variance estimate.

    Only use this when input VAR was missing and independently estimated from
    *bias-subtracted* science counts. The resulting variance remains in ADU^2.
    """
    base = np.asarray(baseline_var_adu2, dtype=float)
    bias = np.asarray(bias_var_adu2, dtype=float)
    if base.shape != tuple(sci_shape) or bias.shape != tuple(sci_shape):
        raise ValueError("MasterBias VAR / sky variance geometry mismatch")
    if (not np.all(np.isfinite(base)) or not np.all(np.isfinite(bias))
            or np.any(base < 0) or np.any(bias < 0)):
        raise ValueError("MasterBias VAR / sky variance is unphysical")
    return np.asarray(base + bias, dtype=np.float32)
