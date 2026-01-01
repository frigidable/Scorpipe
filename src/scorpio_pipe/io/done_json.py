"""Stage done.json helper.

P2 contract
-----------
- Every stage must write ``done.json`` even if it fails.
- The payload must contain a minimal, stable skeleton for reproducibility.
- QC flags must use the canonical schema from :mod:`scorpio_pipe.qc.flags`.

This helper is intentionally generic so stages with richer reports can
still embed extra fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.qc.flags import coerce_flags, max_severity
from scorpio_pipe.version import PIPELINE_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_done_json(
    stage: str,
    *,
    stage_dir: str | Path,
    status: str,
    inputs: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    flags: Sequence[Any] | None = None,
    qc: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
    version: str | None = None,
    extra: Mapping[str, Any] | None = None,
    legacy_paths: Sequence[str | Path] | None = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> dict[str, Any]:
    """Write a stage ``done.json`` in a consistent format.

    Parameters
    ----------
    stage
        Canonical stage key (e.g. "linearize").
    stage_dir
        Directory where outputs are written.
    status
        One of: "ok", "warn", "fail".
    legacy_paths
        Optional additional JSON paths to mirror the same payload for backward
        compatibility.
    """

    out_dir = Path(stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "stage": str(stage),
        "status": str(status),
        "version": str(version or PIPELINE_VERSION),
        "created_utc": str(created_utc or utc_now_iso()),
        "inputs": dict(inputs or {}),
        "params": dict(params or {}),
        "outputs": dict(outputs or {}),
        "metrics": dict(metrics or {}),
    }

    if error is not None:
        payload["error"] = dict(error)

    # Merge QC info.
    qc_payload: dict[str, Any] = {}
    if isinstance(qc, Mapping):
        qc_payload.update(dict(qc))

    # Flags may be provided either in the top-level ``flags`` parameter or
    # inside qc['flags']; we merge them and publish in two places:
    #   - payload['flags'] (stable, easy to grep)
    #   - payload['qc']['flags'] (rich QC section)
    # This is deliberately redundant to keep legacy callers and P2 tests
    # happy.
    fl_top = coerce_flags(flags)
    fl_qc = coerce_flags(qc_payload.get("flags"))
    merged: list[dict[str, Any]] = []
    if fl_top or fl_qc:
        seen: set[tuple[str, str]] = set()

        def _add_all(items: list[dict[str, Any]]) -> None:
            for it in items:
                code = str(it.get("code", ""))
                sev = str(it.get("severity", ""))
                key = (code, sev)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)

        _add_all(fl_top)
        _add_all(fl_qc)

    # Always publish a top-level flags list for a stable contract.
    payload["flags"] = merged

    if merged:
        qc_payload["flags"] = merged
        qc_payload["max_severity"] = max_severity(merged)

    if qc_payload:
        payload["qc"] = qc_payload

    if extra:
        # Do not overwrite the stable keys above.
        for k, v in extra.items():
            if k in payload:
                continue
            payload[k] = v

    # Canonical path
    atomic_write_json(out_dir / "done.json", payload, indent=int(indent), ensure_ascii=bool(ensure_ascii))

    # Optional mirrors
    for p in legacy_paths or []:
        try:
            atomic_write_json(Path(p), payload, indent=int(indent), ensure_ascii=bool(ensure_ascii))
        except Exception:
            pass

    return payload
