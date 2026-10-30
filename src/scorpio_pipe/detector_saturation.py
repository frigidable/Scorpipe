"""Detector saturation is a property of raw CCD readout, not sky residuals.

Only evaluate a threshold on identifiable, unprocessed, ADU-valued samples.
Otherwise preserve upstream SATURATED bits and record that the decision could
not be established.  Header SATURATE is used only if SATUNIT declares ADU;
full-well electron limits are a distinct physical calibration.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from astropy.io import fits

from scorpio_pipe.maskbits import SATURATED


def _saturation_threshold_adu(
    hdr: fits.Header, cfg: Mapping,
) -> tuple[float | None, float, str]:
    det = cfg.get("detector")
    det = det if isinstance(det, Mapping) else {}
    legacy = cfg.get("linearize")
    legacy = legacy if isinstance(legacy, Mapping) else {}
    for params, prefix in ((det, "detector"), (legacy, "linearize")):
        for key in ("saturation_adu", "sat_adu", "saturation_level_adu"):
            if key not in params or params[key] is None:
                continue
            try:
                threshold = float(params[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{prefix}.{key} must be a positive finite ADU threshold") from exc
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"{prefix}.{key} must be a positive finite ADU threshold")
            break
        else:
            continue
        source = f"{prefix}.{key}"
        break
    else:
        source = ""
        threshold = None
        if hdr.get("SATADU") is not None:
            source = "SATADU"
            threshold = float(hdr["SATADU"])
        elif str(hdr.get("SATUNIT") or "").strip().upper() == "ADU":
            for key in ("SATURATE", "SATLEVEL", "SATURATION"):
                if hdr.get(key) is not None:
                    source = key
                    threshold = float(hdr[key])
                    break
        if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
            raise ValueError(f"Invalid detector saturation threshold in header {source}")
    margin = det.get("saturation_margin_adu", legacy.get("saturation_margin_adu", 0.0))
    try:
        margin = float(margin if margin is not None else 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("saturation_margin_adu must be nonnegative and finite") from exc
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("saturation_margin_adu must be nonnegative and finite")
    if threshold is not None and margin >= threshold:
        raise ValueError("saturation_margin_adu cannot equal or exceed threshold")
    return threshold, float(margin), source


def _is_identifiably_raw_adu(hdr: fits.Header) -> bool:
    unit = str(hdr.get("BUNIT") or "").strip().upper()
    if unit != "ADU":
        return False
    # Bias subtraction, flat correction, and pixel replacement all destroy
    # the direct comparison with the original detector ADC readout.
    for keyword in ("BIASSUB", "FLATCOR", "COSMCLEA", "SKYSUB"):
        if bool(hdr.get(keyword, False)):
            return False
    if str(hdr.get("SKYMD") or "").strip():
        return False
    ctype = str(hdr.get("CTYPE1") or "").strip().upper()
    if ctype.startswith(("WAVE", "AWAV")) or hdr.get("SCORP_DL") is not None:
        return False
    return True


def carry_detector_saturation(
    sci: np.ndarray, mask: np.ndarray | None, hdr: fits.Header, cfg: Mapping,
) -> tuple[np.ndarray, fits.Header]:
    """OR trustworthy raw-ADU saturation evidence into a uint16 DQ mask.

    SATQ=RAW_EVAL: threshold evaluated on identifiable raw ADU input;
    SATQ=PROP: SATURATED bit received from upstream but no raw evaluation here;
    SATQ=UNKNOWN: saturation not independently qualified.  A previous
    RAW_EVAL stamp survives transformed frames, even if no pixels saturated.
    """
    arr = np.asarray(sci)
    if arr.ndim != 2:
        raise ValueError("Detector saturation input must be a 2D CCD image")
    out = np.zeros(arr.shape, dtype=np.uint16) if mask is None else np.asarray(mask, dtype=np.uint16).copy()
    if out.shape != arr.shape:
        raise ValueError("Detector saturation MASK shape mismatch")
    h = fits.Header(hdr)
    det = cfg.get("detector")
    det = det if isinstance(det, Mapping) else {}
    lcfg = cfg.get("linearize")
    lcfg = lcfg if isinstance(lcfg, Mapping) else {}
    enabled = bool(det.get("mask_saturation", lcfg.get("mask_saturation", True)))
    if enabled and _is_identifiably_raw_adu(h):
        threshold, margin, source = _saturation_threshold_adu(h, cfg)
        if threshold is not None:
            out[np.isfinite(arr) & (arr >= (threshold - margin))] |= SATURATED
            h["SATQ"] = ("RAW_EVAL", "Raw ADU threshold checked before correction")
            h["SATADU"] = (float(threshold), "Raw detector threshold [ADU]")
            h["SATMARG"] = (float(margin), "Detector threshold margin [ADU]")
            h["SATSRC"] = (str(source)[:68], "Saturation threshold provenance")
            return out, h
    if str(h.get("SATQ") or "").strip().upper() != "RAW_EVAL":
        h["SATQ"] = (
            "PROP" if np.any((out & SATURATED) != 0) else "UNKNOWN",
            "Saturation upstream evidence only, or unqualified",
        )
    return out, h
