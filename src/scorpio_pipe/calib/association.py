"""Calibration-domain association policy.

This module now hosts two related responsibilities:

1. Stable lazy wrappers for dataset-manifest builders.
2. Standard-star / telluric association logic used by the upcoming
   calibration stages.

The logic is intentionally lightweight and machine-readable.  It answers a
question the reduction core previously could not answer formally:

*Is the matched standard usable only for response shape, or also for telluric
and/or absolute scaling?*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from scorpio_pipe.metadata import FrameMeta


class StandardCompatibilityStatus(str, Enum):
    """Compatibility class for a science/standard pairing."""

    INCOMPATIBLE = "INCOMPATIBLE"
    RELATIVE_RESPONSE_ONLY = "RELATIVE_RESPONSE_ONLY"
    RELATIVE_TELLURIC_ONLY = "RELATIVE_TELLURIC_ONLY"
    ABSOLUTE_APPROX = "ABSOLUTE_APPROX"
    ABSOLUTE_TRUSTED = "ABSOLUTE_TRUSTED"


@dataclass(frozen=True)
class AssociationThresholds:
    """Thresholds controlling downgrade decisions."""

    slit_width_tol_arcsec: float = 0.15
    max_time_delta_hours_trusted: float = 2.0
    max_time_delta_hours_approx: float = 6.0
    max_airmass_delta_trusted: float = 0.15
    max_airmass_delta_approx: float = 0.35


@dataclass(frozen=True)
class StandardAssociationReport:
    """Machine-readable report for standard-star compatibility."""

    status: StandardCompatibilityStatus
    standard_role: str
    is_shape_compatible: bool
    is_absolute_compatible: bool
    downgrade_reasons: tuple[str, ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status.value),
            "standard_role": str(self.standard_role),
            "is_shape_compatible": bool(self.is_shape_compatible),
            "is_absolute_compatible": bool(self.is_absolute_compatible),
            "downgrade_reasons": list(self.downgrade_reasons),
            "checks": dict(self.checks),
        }


def _time_delta_hours(a: FrameMeta, b: FrameMeta) -> float | None:
    try:
        dt = abs((a.date_time_utc - b.date_time_utc).total_seconds()) / 3600.0
        return float(dt)
    except Exception:
        return None


def _airmass_delta(a: FrameMeta, b: FrameMeta) -> float | None:
    try:
        if a.airmass is None or b.airmass is None:
            return None
        da = abs(float(a.airmass) - float(b.airmass))
        return float(da) if isfinite(da) else None
    except Exception:
        return None


def _compatible_wavelength_coverage(a: FrameMeta, b: FrameMeta) -> bool:
    sa = str(a.sperange or "").strip()
    sb = str(b.sperange or "").strip()
    if not sa or not sb:
        return True
    return sa == sb


def _meta_quality(meta: FrameMeta, field: str) -> str:
    try:
        return str(meta.provenance_class(field))
    except Exception:
        return "missing"


def _shape_checks(science: FrameMeta, standard: FrameMeta) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "science_disperser": science.disperser,
        "standard_disperser": standard.disperser,
        "science_mode": science.mode,
        "standard_mode": standard.mode,
        "science_binning": f"{science.binning_x}x{science.binning_y}",
        "standard_binning": f"{standard.binning_x}x{standard.binning_y}",
        "science_shape": [int(science.naxis1), int(science.naxis2)],
        "standard_shape": [int(standard.naxis1), int(standard.naxis2)],
        "science_sperange": science.sperange,
        "standard_sperange": standard.sperange,
    }

    if str(science.mode).strip().lower() != str(standard.mode).strip().lower():
        reasons.append("mode_mismatch")
    if str(science.disperser).strip() != str(standard.disperser).strip():
        reasons.append("disperser_mismatch")
    if (int(science.binning_x), int(science.binning_y)) != (int(standard.binning_x), int(standard.binning_y)):
        reasons.append("binning_mismatch")
    if (int(science.naxis1), int(science.naxis2)) != (int(standard.naxis1), int(standard.naxis2)):
        reasons.append("detector_geometry_mismatch")
    if not _compatible_wavelength_coverage(science, standard):
        reasons.append("wavelength_coverage_mismatch")

    return (len(reasons) == 0), reasons, checks


def assess_standard_association(
    science: FrameMeta,
    standard: FrameMeta,
    *,
    standard_role: str = "standard",
    thresholds: AssociationThresholds | None = None,
) -> StandardAssociationReport:
    """Assess whether *standard* is suitable for *science*.

    Parameters
    ----------
    standard_role:
        ``"standard"`` or ``"telluric_standard"``.  Telluric standards are
        allowed to land at ``RELATIVE_TELLURIC_ONLY`` rather than plain response.
    """

    th = thresholds or AssociationThresholds()
    role = str(standard_role or "standard").strip().lower()

    is_shape_ok, reasons, checks = _shape_checks(science, standard)
    if not is_shape_ok:
        return StandardAssociationReport(
            status=StandardCompatibilityStatus.INCOMPATIBLE,
            standard_role=role,
            is_shape_compatible=False,
            is_absolute_compatible=False,
            downgrade_reasons=tuple(reasons),
            checks=checks,
        )

    slit_delta = None
    try:
        slit_delta = abs(float(science.slit_width_arcsec) - float(standard.slit_width_arcsec))
    except Exception:
        slit_delta = None
    time_delta_h = _time_delta_hours(science, standard)
    airmass_delta = _airmass_delta(science, standard)

    checks.update(
        {
            "science_slit_width_arcsec": science.slit_width_arcsec,
            "standard_slit_width_arcsec": standard.slit_width_arcsec,
            "slit_width_delta_arcsec": slit_delta,
            "time_delta_hours": time_delta_h,
            "airmass_delta": airmass_delta,
            "metadata_quality": {
                "science_airmass": _meta_quality(science, "airmass"),
                "standard_airmass": _meta_quality(standard, "airmass"),
                "science_site_name": _meta_quality(science, "site_name"),
                "standard_site_name": _meta_quality(standard, "site_name"),
                "science_observatory": _meta_quality(science, "observatory"),
                "standard_observatory": _meta_quality(standard, "observatory"),
            },
        }
    )

    absolute_reasons: list[str] = []
    approx_reasons: list[str] = []

    if slit_delta is None:
        absolute_reasons.append("slit_width_unknown")
        approx_reasons.append("slit_width_unknown")
    elif slit_delta > th.slit_width_tol_arcsec:
        absolute_reasons.append("slit_width_mismatch")
        approx_reasons.append("slit_width_mismatch")

    if time_delta_h is None:
        absolute_reasons.append("time_mismatch_unknown")
    elif time_delta_h > th.max_time_delta_hours_trusted:
        absolute_reasons.append("time_mismatch")
        if time_delta_h > th.max_time_delta_hours_approx:
            approx_reasons.append("time_mismatch")

    if airmass_delta is None:
        absolute_reasons.append("airmass_mismatch_unknown")
    elif airmass_delta > th.max_airmass_delta_trusted:
        absolute_reasons.append("airmass_mismatch")
        if airmass_delta > th.max_airmass_delta_approx:
            approx_reasons.append("airmass_mismatch")

    if _meta_quality(science, "airmass") not in {"header", "derived", "untracked"} or _meta_quality(standard, "airmass") not in {"header", "derived", "untracked"}:
        absolute_reasons.append("airmass_provenance_degraded")
    if _meta_quality(science, "site_name") not in {"header", "derived", "untracked"} or _meta_quality(standard, "site_name") not in {"header", "derived", "untracked"}:
        absolute_reasons.append("site_context_degraded")
    if _meta_quality(science, "observatory") not in {"header", "derived", "untracked"} or _meta_quality(standard, "observatory") not in {"header", "derived", "untracked"}:
        absolute_reasons.append("observatory_context_degraded")

    if not absolute_reasons:
        status = StandardCompatibilityStatus.ABSOLUTE_TRUSTED
        is_abs = True
    elif not approx_reasons and role != "telluric_standard":
        status = StandardCompatibilityStatus.ABSOLUTE_APPROX
        is_abs = True
        reasons.extend(sorted(set(absolute_reasons)))
    else:
        status = (
            StandardCompatibilityStatus.RELATIVE_TELLURIC_ONLY
            if role == "telluric_standard"
            else StandardCompatibilityStatus.RELATIVE_RESPONSE_ONLY
        )
        is_abs = False
        reasons.extend(sorted(set(absolute_reasons + approx_reasons)))

    return StandardAssociationReport(
        status=status,
        standard_role=role,
        is_shape_compatible=True,
        is_absolute_compatible=is_abs,
        downgrade_reasons=tuple(dict.fromkeys(reasons)),
        checks=checks,
    )


def build_dataset_manifest(*args: Any, **kwargs: Any):
    """Lazy wrapper for backward compatibility."""

    from scorpio_pipe.dataset.builder import build_dataset_manifest as _impl

    return _impl(*args, **kwargs)


def build_dataset_manifest_from_records(*args: Any, **kwargs: Any):
    """Lazy wrapper for backward compatibility."""

    from scorpio_pipe.dataset.builder import build_dataset_manifest_from_records as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "AssociationThresholds",
    "StandardCompatibilityStatus",
    "StandardAssociationReport",
    "assess_standard_association",
    "build_dataset_manifest",
    "build_dataset_manifest_from_records",
]
