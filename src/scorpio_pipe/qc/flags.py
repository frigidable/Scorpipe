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
"""

from __future__ import annotations

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
    # Keep extra fields at top-level for readability in JSON.
    for k, v in (extra or {}).items():
        if v is None:
            continue
        d[str(k)] = v
    return d


def normalize_flag(flag: Any) -> dict[str, Any] | None:
    """Normalize an arbitrary flag-like object to the canonical dict."""

    if flag is None:
        return None

    # Allow legacy "CODE" string flags.
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
        f = normalize_flag(flags)
        return [f] if f else []

    if isinstance(flags, (list, tuple)):
        for x in flags:
            f = normalize_flag(x)
            if f:
                out.append(f)
        return out

    f = normalize_flag(flags)
    return [f] if f else []


def max_severity(flags: Iterable[dict[str, Any]] | None) -> str:
    """Return the maximum severity among flags (INFO/WARN/ERROR)."""

    best = "INFO"
    for f in flags or []:
        sev = normalize_severity(str((f or {}).get("severity") or ""))
        if _SEV_ORDER.get(sev, 0) > _SEV_ORDER.get(best, 0):
            best = sev
    return best
