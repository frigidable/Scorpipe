"""Calibration semantics for flux-calibrated products (P0.7)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class CalibrationStatus(str, Enum):
    RAW = "RAW"
    RELATIVE = "RELATIVE"
    RELATIVE_TELLURIC = "RELATIVE_TELLURIC"
    ABSOLUTE_APPROX = "ABSOLUTE_APPROX"
    ABSOLUTE_TRUSTED = "ABSOLUTE_TRUSTED"


class CalibrationProductKind(str, Enum):
    SPEC1D = "spec1d"
    MEF2D = "mef2d"


_ALLOWED = {s.value for s in CalibrationStatus}
_ALLOWED_BY_KIND: dict[CalibrationProductKind, tuple[str, ...]] = {
    CalibrationProductKind.SPEC1D: (
        CalibrationStatus.RELATIVE.value,
        CalibrationStatus.RELATIVE_TELLURIC.value,
        CalibrationStatus.ABSOLUTE_APPROX.value,
        CalibrationStatus.ABSOLUTE_TRUSTED.value,
    ),
    CalibrationProductKind.MEF2D: (
        CalibrationStatus.RELATIVE.value,
        CalibrationStatus.RELATIVE_TELLURIC.value,
        CalibrationStatus.ABSOLUTE_APPROX.value,
    ),
}


def normalize_calibration_status(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s not in _ALLOWED:
        raise ValueError(f"Unknown calibration status: {value!r}")
    return s


def calibration_statuses_for_product_kind(product_kind: str | CalibrationProductKind) -> tuple[str, ...]:
    try:
        pk = product_kind if isinstance(product_kind, CalibrationProductKind) else CalibrationProductKind(str(product_kind))
    except Exception as e:
        raise ValueError(f"Unknown calibrated product kind: {product_kind!r}") from e
    return tuple(_ALLOWED_BY_KIND[pk])


def telluric_semantics_required(status: str) -> bool:
    st = normalize_calibration_status(status)
    return st in {
        CalibrationStatus.RELATIVE_TELLURIC.value,
        CalibrationStatus.ABSOLUTE_APPROX.value,
        CalibrationStatus.ABSOLUTE_TRUSTED.value,
    }


def validate_calibration_claim(status: str, *, product_kind: str | CalibrationProductKind) -> str:
    st = normalize_calibration_status(status)
    if st == CalibrationStatus.RAW.value:
        raise ValueError("Calibrated products must not declare CALSTAT=RAW")
    allowed = calibration_statuses_for_product_kind(product_kind)
    if st not in allowed:
        raise ValueError(f"CALSTAT={st!r} is not allowed for product kind {str(product_kind)!r}; allowed={allowed}")
    return st


def coerce_calibration_status(status: str, *, product_kind: str | CalibrationProductKind) -> str:
    st = normalize_calibration_status(status)
    try:
        pk = product_kind if isinstance(product_kind, CalibrationProductKind) else CalibrationProductKind(str(product_kind))
    except Exception:
        pk = product_kind
    allowed = calibration_statuses_for_product_kind(product_kind)
    if st in allowed:
        return st
    if pk == CalibrationProductKind.MEF2D and st == CalibrationStatus.ABSOLUTE_TRUSTED.value:
        return CalibrationStatus.ABSOLUTE_APPROX.value
    return allowed[0]


def calibration_semantics_summary(status: str, *, product_kind: str | CalibrationProductKind) -> dict[str, Any]:
    st = validate_calibration_claim(status, product_kind=product_kind)
    return {
        "status": st,
        "product_kind": (product_kind.value if isinstance(product_kind, CalibrationProductKind) else str(product_kind)),
        "telluric_semantics_required": bool(telluric_semantics_required(st)),
        "allowed_statuses": list(calibration_statuses_for_product_kind(product_kind)),
        "absolute_claim": st in {CalibrationStatus.ABSOLUTE_APPROX.value, CalibrationStatus.ABSOLUTE_TRUSTED.value},
    }


def stamp_calibration_semantics(
    hdr: Any,
    *,
    status: str,
    flux_unit: str,
    reference_context_id: str,
    sensfunc_ref: str | None = None,
    telluric_ref: str | None = None,
) -> Any:
    st = normalize_calibration_status(status)
    hdr["CALSTAT"] = (st, "Calibration status")
    hdr["FLUXUNIT"] = (str(flux_unit), "Flux unit")
    hdr["REFCTID"] = (str(reference_context_id), "Reference context id")
    if sensfunc_ref:
        hdr["SENSFUNC"] = (str(sensfunc_ref), "Applied sensitivity function ref")
    if telluric_ref:
        hdr["TELLURIC"] = (str(telluric_ref), "Applied telluric ref")
    return hdr


__all__ = [
    "CalibrationStatus",
    "CalibrationProductKind",
    "normalize_calibration_status",
    "calibration_statuses_for_product_kind",
    "telluric_semantics_required",
    "validate_calibration_claim",
    "coerce_calibration_status",
    "calibration_semantics_summary",
    "stamp_calibration_semantics",
]
