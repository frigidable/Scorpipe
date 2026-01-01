"""Small helpers for stage-level QC flags.

Stages should emit a compact list of flags into their ``done.json``. The
runner can then gate downstream stages without having to re-run heavy QC.
"""

from __future__ import annotations

from typing import Any, Iterable


def make_flag(code: str, severity: str, message: str, **meta: Any) -> dict[str, Any]:
    d = {"code": str(code), "severity": str(severity).upper(), "message": str(message)}
    for k, v in meta.items():
        d[k] = v
    return d


def max_severity(flags: Iterable[dict[str, Any]] | None) -> str:
    order = {"OK": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}
    best = "OK"
    for f in flags or []:
        s = str((f or {}).get("severity") or "OK").upper()
        if s not in order:
            s = "INFO"
        if order[s] > order[best]:
            best = s
    return best
