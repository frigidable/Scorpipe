"""Read-only audit of actual pre-SkySub variance components and bias groups.

The upstream FlatField MEF may contain electron-unit VBIAS, VREST,
signed BSENS (electron/ADU), and a shared MasterBias BVARADU (ADU^2).
SkySub verifies those planes at input but deliberately drops them from
its output. This diagnostic reopens SHA256-bound input files; it does
not reconstruct noise components from post-sky VAR.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.calib.sky_fit_lineage import audit_frozen_kelson_fit_lineage


_COMPONENT_UNITS = {
    "VBIAS": "e-^2",
    "VREST": "e-^2",
    "BSENS": "e-/ADU",
    "BVARADU": "ADU^2",
}
_COMPONENT_NAMES = frozenset(_COMPONENT_UNITS)
# Deliberately no claim that VREST is uncorrelated between exposures:
# MasterFlat and other upstream calibrations can also be shared.


def _component_source(path: Path, *, max_pixels: int) -> tuple[dict[str, Any], np.ndarray | None]:
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        names = [str(h.header.get("EXTNAME", "")).upper().strip()
                 for h in hdul[1:]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source FITS extension names")
        detected = set(names) & _COMPONENT_NAMES
        present = "VCMPVER" in hdr
        group = str(hdr.get("SB_GID") or "").strip()
        if not present:
            if detected or str(hdr.get("VCMPGID") or "").strip():
                raise ValueError("Unversioned MasterBias component planes cannot be admitted")
            return ({
                "source_status": "HOLD_NO_DECLARED_FLATFIELD_COMPONENTS",
                "masterbias_group": group or None,
                "shared_bias_decomposition_validated": False,
                "pre_sky_variance_components_recovered": False,
            }, None)
        if (type(hdr.get("VCMPVER")) is not int or hdr["VCMPVER"] != 1
                or str(hdr.get("VCMPQ") or "").strip() != "LOCAL"
                or not bool(hdr.get("BIASSUB", False))
                or not bool(hdr.get("FLATCOR", False))
                or not bool(hdr.get("BVARINC", False))
                or str(hdr.get("BUNIT") or "").strip() != "e-"
                or str(hdr.get("SCORPUM") or "").strip().upper() != "ELECTRON"
                or not group or group != str(hdr.get("VCMPGID") or "").strip()):
            raise ValueError("Unqualified source variance unit, MasterBias group, or calibration state")
        if detected != _COMPONENT_NAMES:
            raise ValueError("Incomplete declared FlatField variance component maps")
        if not all(k in names for k in ("SCI", "VAR")):
            raise ValueError("Declared component source missing SCI or VAR plane")
        sci_hdu = hdul[names.index("SCI")+1]
        var_hdu = hdul[names.index("VAR")+1]
        shape = sci_hdu.data.shape
        if len(shape) != 2 or min(shape) <= 0 or int(np.prod(shape)) > max_pixels:
            raise ValueError("Pre-sky component CCD geometry exceeds explicit pixel limit")
        if var_hdu.data.shape != shape:
            raise ValueError("Pre-sky variance source shape mismatch")
        planes: dict[str, np.ndarray] = {}
        for k in _COMPONENT_UNITS:
            hdu = hdul[names.index(k)+1]
            if hdu.data is None or hdu.data.shape != shape:
                raise ValueError(f"{k} has wrong CCD component geometry")
            if str(hdu.header.get("BUNIT") or "").strip() != _COMPONENT_UNITS[k]:
                raise ValueError(f"{k} has missing or wrong physical BUNIT")
            planes[k] = np.asarray(hdu.data, dtype=np.float64)
        var = np.asarray(var_hdu.data, dtype=np.float64)
        vb, vr, bs, bm = (planes[k] for k in ("VBIAS", "VREST", "BSENS", "BVARADU"))
        if any(not np.isfinite(x).all() for x in (var, vb, vr, bs, bm)):
            raise ValueError("Nonfinite declared MasterBias variance components")
        if any(np.any(x < 0) for x in (var, vb, vr, bm)):
            raise ValueError("Negative declared variance component")
        if not np.allclose(vb + vr, var, rtol=3e-5, atol=3e-5):
            raise ValueError("VBIAS + VREST does not reproduce source VAR")
        if not np.allclose(vb, bs*bs*bm, rtol=3e-5, atol=3e-5):
            raise ValueError("Signed BSENS and BVARADU do not reproduce source VBIAS")
        return ({
            "source_status": "VERIFIED_FLATFIELD_COMPONENTS_LOCAL_ONLY",
            "masterbias_group": group,
            "source_ccd_shape": list(shape),
            "n_native_ccd_pixels": int(var.size),
            "shared_bias_decomposition_validated": True,
            "pre_sky_variance_components_recovered": True,
            "component_units": dict(_COMPONENT_UNITS),
            "residual_vrest_may_include_shared_flat_calibration": True,
        }, bm.copy())


def audit_sky_masterbias_group_sources(
    lineage_manifests: Sequence[Path],
    *,
    max_pixels_per_exposure: int = 8_388_608,
    max_exposures: int = 64,
) -> dict[str, Any]:
    """Audit same-run source FITS component decomposition for frozen SkySub.

    No hypothetical MasterBias response is created: source BSENS is signed
    and must be physically linked to original FlatField/CCD data. Group
    members must have the SAME MasterBias variance image and CCD shape.

    The result is a provenance report, never a science-qualified covariance
    or a claim of independence for VREST / sky-selection uncertainty.
    """
    if (type(max_pixels_per_exposure) is not int or max_pixels_per_exposure <= 0
            or type(max_exposures) is not int or max_exposures <= 0):
        raise ValueError("Invalid explicit source audit exposure/pixel budget")
    if (isinstance(lineage_manifests, (str, bytes, Path))
            or not isinstance(lineage_manifests, Sequence)
            or not (1 <= len(lineage_manifests) <= max_exposures)):
        raise ValueError("Source audit requires a bounded, explicit manifest sequence")
    entries = []
    used_sources: set[Path] = set()
    groups: dict[str, np.ndarray] = {}
    members: dict[str, list[int]] = {}
    held = 0
    for idx, item in enumerate(lineage_manifests):
        # Verifies full-file SHA256 and actual Kelson→science FITS replay
        # before looking at the source variance components.
        verified = audit_frozen_kelson_fit_lineage(
            Path(item), check_fit_against_products=True
        )
        source = Path(verified["selected_sky_input"]).resolve(strict=True)
        if source in used_sources:
            raise ValueError("Repeated SkySub source cannot be independent exposures")
        used_sources.add(source)
        evidence, bmap = _component_source(
            source, max_pixels=max_pixels_per_exposure
        )
        evidence["exposure_index"] = idx
        evidence["source_fits"] = str(source)
        evidence["frozen_sky_fit_replayed"] = True
        entries.append(evidence)
        if bmap is None:
            held += 1
            continue
        gid = evidence["masterbias_group"]
        if gid in groups:
            prev = groups[gid]
            if prev.shape != bmap.shape:
                raise ValueError("Shared MasterBias group has incompatible CCD geometry")
            # Equal group ID alone is insufficient for common calibration:
            # reject if its actual variance field differs across members.
            if not np.array_equal(prev, bmap):
                raise ValueError("Shared MasterBias group has inconsistent BVARADU maps")
        else:
            groups[gid] = bmap
        members.setdefault(gid, []).append(idx)
    return {
        "status": ("HOLD_PRE_SKY_VARIANCE_COMPONENTS_PRESENT_BUT_NOT_PROPAGATED"
                   if held == 0 else "HOLD_INCOMPLETE_PRE_SKY_VARIANCE_LINEAGE"),
        "file_and_fit_lineage_verified": True,
        "n_exposures": len(entries),
        "n_sources_without_verified_component_planes": held,
        "source_entries": entries,
        "masterbias_group_members": members,
        "shared_group_bias_variance_maps_consistent": True if groups else None,
        "unbound_provenance": [
            "actual_native_detector_noise_response_and_units_per_exposure",
            "signed_MasterBias_operator_through_selected_sky_and_COUNT_V1",
            "VREST_shared_MasterFlat_and_other_cross_exposure_correlations",
            "frozen_selection_and_adaptive_extraction_uncertainty",
            "effective_science_LSF",
        ],
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
