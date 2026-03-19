"""QC flag helpers.

Contract (P2)
-------------
Stages emit a compact list of flags into their stage-local ``done.json``.
Each flag **must** include:

- ``code``: stable machine-readable identifier
- ``severity``: one of ``INFO``, ``WARN``, ``ERROR``
- ``message``: short human-readable summary
- ``hint``: actionable suggestion (may be empty)

Extra keys are allowed (e.g. ``stage``, ``path``, ``stem``) to help the UI,
reports, or debugging.

This module provides:

- :func:`make_flag` for stage code
- :func:`normalize_severity` / :func:`normalize_flag` for robust IO
- :func:`max_severity` for aggregation
- operational failure taxonomy helpers used by the LONG-parity benchmark layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


# Public severities (ordered).
_SEV_ORDER: dict[str, int] = {
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3,
}

# Backward compatible aliases seen in older payloads.
_SEV_ALIASES: dict[str, str] = {
    "OK": "INFO",
    "WARNING": "WARN",
    "BAD": "ERROR",
    "FAIL": "ERROR",
    "FATAL": "ERROR",
    "CRIT": "ERROR",
    "CRITICAL": "ERROR",
}


@dataclass(frozen=True)
class OperationalFailureSpec:
    code: str
    title: str
    domain: str
    default_severity: str
    hint: str
    aliases: tuple[str, ...] = ()


_OPERATIONAL_FAILURES: tuple[OperationalFailureSpec, ...] = (
    OperationalFailureSpec(
        code="WAVE_GEOMETRY_WEAK",
        title="Wavelength geometry is weak or poorly constrained",
        domain="wavesolution",
        default_severity="WARN",
        hint="Inspect line density, rejected lines, and geometry priors before trusting the λ-map edges.",
        aliases=("WAVESOL_2D_RMS", "WAVESOL_2D_P95"),
    ),
    OperationalFailureSpec(
        code="WAVE_LINESET_POOR",
        title="Arc line set is too weak or ambiguous",
        domain="wavesolution",
        default_severity="WARN",
        hint="Switch to a curated per-grism line preset or review rejected arc lines manually.",
        aliases=("LAMP_UNKNOWN", "LAMP_CONTRACT_FAILED"),
    ),
    OperationalFailureSpec(
        code="SKY_WINDOWS_CONTAMINATED",
        title="Sky windows are contaminated or too sparse",
        domain="sky",
        default_severity="WARN",
        hint="Inspect upper/lower sky stripes and tighten object masking or redefine sky windows.",
        aliases=("NO_SKY_WINDOWS", "SKY_COVERAGE_TOO_LOW", "SKY_COVERAGE_LOW", "TOO_FEW_SKY_PIXELS"),
    ),
    OperationalFailureSpec(
        code="SKY_MODEL_UNSTABLE",
        title="Sky model fit is unstable",
        domain="sky",
        default_severity="ERROR",
        hint="Check flexure estimation, skyline residuals, and sky-template compatibility before using the result scientifically.",
        aliases=("SKY_SUB_FAILED", "FLEXURE_FAILED", "FLEXURE_UNCERTAIN", "SKY_TEMPLATE_INCOMPATIBLE", "SKY_TEMPLATE_READ_FAILED", "SKY_REFINE_INPUT_MISSING"),
    ),
    OperationalFailureSpec(
        code="SKY_RESIDUALS_STRONG",
        title="Residual skyline structure remains strong after subtraction",
        domain="sky",
        default_severity="WARN",
        hint="Inspect skyline windows and consider residual-cleanup/refine settings before extraction.",
        aliases=("OBJECT_EATING_RISK", "SKY_REFINE_NO_GAIN", "SKY_REFINE_OBJECT_RISK"),
    ),
    OperationalFailureSpec(
        code="SLIT_METADATA_MISMATCH",
        title="Slit metadata are inconsistent with the data",
        domain="calibration",
        default_severity="WARN",
        hint="Split incompatible setups and verify slit position / rotation / readout metadata before reusing calibrations.",
        aliases=("CALIB_ROT_MISMATCH", "CALIB_SLITPOS_MISMATCH", "CALIB_READOUT_DIFF"),
    ),
    OperationalFailureSpec(
        code="EXTRACT_TRACE_UNCERTAIN",
        title="Object trace or extraction aperture is uncertain",
        domain="extraction",
        default_severity="WARN",
        hint="Review the spatial profile, compare fixed vs optimal extraction, and verify contamination risk.",
        aliases=("NO_OBJECT_TRACE", "TRACE_UNCERTAIN", "EXTRACT_APERTURE_UNCERTAIN"),
    ),
    OperationalFailureSpec(
        code="PUBLICATION_PRODUCT_DEGRADED",
        title="Publication-ready product is degraded",
        domain="publication_ready",
        default_severity="WARN",
        hint="Do not export the product as science-ready until the upstream QC issue is understood.",
        aliases=("PUBLICATION_READY_DEGRADED", "UPSTREAM_SKY_PASSTHROUGH"),
    ),
)

_OPERATIONAL_BY_CODE: dict[str, OperationalFailureSpec] = {spec.code: spec for spec in _OPERATIONAL_FAILURES}
_OPERATIONAL_ALIAS_TO_CODE: dict[str, str] = {
    alias: spec.code
    for spec in _OPERATIONAL_FAILURES
    for alias in spec.aliases
}


def normalize_severity(sev: str | None) -> str:
    """Normalize a severity string to INFO/WARN/ERROR."""

    s = (sev or "").strip().upper()
    if not s:
        return "INFO"
    if s in _SEV_ORDER:
        return s
    if s in _SEV_ALIASES:
        return _SEV_ALIASES[s]
    # Unknown values should not break consumers.
    return "INFO"


def get_operational_failure_spec(code: str | None) -> OperationalFailureSpec | None:
    k = str(code or '').strip().upper()
    if not k:
        return None
    return _OPERATIONAL_BY_CODE.get(k)


def infer_operational_failure_code(code: str | None) -> str | None:
    k = str(code or '').strip().upper()
    if not k:
        return None
    if k in _OPERATIONAL_BY_CODE:
        return k
    return _OPERATIONAL_ALIAS_TO_CODE.get(k)


def list_operational_failure_specs() -> tuple[OperationalFailureSpec, ...]:
    return _OPERATIONAL_FAILURES


def annotate_operational_flag(flag: Any) -> dict[str, Any] | None:
    d = normalize_flag(flag)
    if d is None:
        return None
    op_code = infer_operational_failure_code(str(d.get('code') or ''))
    if not op_code:
        return d
    spec = get_operational_failure_spec(op_code)
    if spec is None:
        return d
    d = dict(d)
    d['operational_code'] = spec.code
    d['operational_title'] = spec.title
    d['operational_domain'] = spec.domain
    if not str(d.get('hint') or '').strip() and spec.hint:
        d['hint'] = spec.hint
    return d


def make_flag(
    code: str,
    severity: str,
    message: str,
    hint: str | None = "",
    **extra: Any,
) -> dict[str, Any]:
    """Create a QC flag dict in the canonical format."""

    d: dict[str, Any] = {
        "code": str(code or "").strip(),
        "severity": normalize_severity(severity),
        "message": str(message or "").strip(),
        "hint": str(hint or "").strip(),
    }
    for k, v in (extra or {}).items():
        if v is None:
            continue
        d[str(k)] = v
    return d


def make_operational_flag(
    code: str,
    message: str | None = None,
    severity: str | None = None,
    hint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    spec = get_operational_failure_spec(code)
    if spec is None:
        return make_flag(code, severity or 'WARN', message or '', hint=hint or '', **extra)
    return annotate_operational_flag(
        make_flag(
            spec.code,
            severity or spec.default_severity,
            message or spec.title,
            hint=hint or spec.hint,
            **extra,
        )
    ) or {}


def normalize_flag(flag: Any) -> dict[str, Any] | None:
    """Normalize an arbitrary flag-like object to the canonical dict."""

    if flag is None:
        return None

    if isinstance(flag, str):
        code = flag.strip()
        if not code:
            return None
        return make_flag(code, "INFO", "")

    if not isinstance(flag, dict):
        return None

    d = dict(flag)
    code = str(d.get("code") or "").strip()
    if not code:
        return None

    sev = normalize_severity(str(d.get("severity") or ""))
    msg = str(d.get("message") or "").strip()
    hint = str(d.get("hint") or "").strip()

    d["code"] = code
    d["severity"] = sev
    d["message"] = msg
    d["hint"] = hint
    return d


def coerce_flags(flags: Any) -> list[dict[str, Any]]:
    """Coerce a flags field to a list of canonical flag dicts."""

    out: list[dict[str, Any]] = []
    if flags is None:
        return out

    if isinstance(flags, dict):
        f = annotate_operational_flag(flags)
        return [f] if f else []

    if isinstance(flags, (list, tuple)):
        for x in flags:
            f = annotate_operational_flag(x)
            if f:
                out.append(f)
        return out

    f = annotate_operational_flag(flags)
    return [f] if f else []


def max_severity(flags: Iterable[dict[str, Any]] | None) -> str:
    """Return the maximum severity among flags (INFO/WARN/ERROR)."""

    best = "INFO"
    for f in flags or []:
        sev = normalize_severity(str((f or {}).get("severity") or ""))
        if _SEV_ORDER.get(sev, 0) > _SEV_ORDER.get(best, 0):
            best = sev
    return best
