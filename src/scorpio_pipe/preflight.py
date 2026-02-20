from __future__ import annotations

"""Preflight scan and quarantine (P0-A1).

This module performs an early, conservative scan of a raw-night directory and
produces machine-readable artifacts for the GUI and downstream stages.

Artifacts (relative to run_root)
-------------------------------
manifest/night_profile.json
    Aggregated, conservative summary of the night (instrument/mode/grism/
    binning/window/shape/readout/lamp-hints/Z-stats).

manifest/preflight_report.json
    Overall status: OK/WARN/ERROR + aggregated reasons.

quarantine/<id>.json
    Per-frame issue ticket(s) for frames that are excluded or have issues.

manifest/quarantine_index.json
    Index of all quarantine tickets.

Determinism
-----------
These JSON files are *deterministic*: for the same inputs (data_dir contents +
exclude policy + code version), re-running preflight produces byte-identical
JSON.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from scorpio_pipe.exclude_policy import resolve_exclude_set, resolve_data_dir
from scorpio_pipe.metadata import HeaderContractError, FrameMeta, parse_frame_meta
from scorpio_pipe.version import PIPELINE_VERSION


_FITS_SUFFIXES: tuple[str, ...] = (".fits", ".fit", ".fts", ".fz")


SEVERITY_ORDER: dict[str, int] = {"INFO": 0, "WARN": 1, "ERROR": 2}


def _sev_max(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0) else b


def _stable_id(rel_path: str) -> str:
    # Stable across runs and independent of scan ordering.
    h = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"F{h}"


def _as_rel_str(p: Path, base: Path) -> str:
    try:
        return p.relative_to(base).as_posix()
    except Exception:
        return p.as_posix()


def _json_dumps(obj: Any) -> str:
    # NOTE: sort_keys makes dict ordering deterministic.
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(obj), encoding="utf-8")
    return path


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    a = float(sorted_vals[lo])
    b = float(sorted_vals[hi])
    t = pos - lo
    return a + (b - a) * t


# -----------------------------------------------------------------------------
# Reason codes (stable; used by UI)
# -----------------------------------------------------------------------------

RC_EXCLUDED_BY_POLICY = "EXCLUDED_BY_POLICY"  # project_manifest.yaml / config exclude
RC_FITS_OPEN_ERROR = "FITS_OPEN_ERROR"
RC_HEADER_CONTRACT_ERROR = "HEADER_CONTRACT_ERROR"
RC_META_PARSE_ERROR = "META_PARSE_ERROR"
RC_META_MISSING_OPTIONAL = "META_MISSING_OPTIONAL"
RC_META_FALLBACK_USED = "META_FALLBACK_USED"


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
        "context": dict(context or {}),
    }


def _iter_fits_paths(data_dir: Path, *, recursive: bool = True) -> list[Path]:
    paths: list[Path] = []
    if recursive:
        for suf in _FITS_SUFFIXES:
            paths.extend(sorted(data_dir.rglob(f"*{suf}")))
    else:
        for suf in _FITS_SUFFIXES:
            paths.extend(sorted(data_dir.glob(f"*{suf}")))
    # Deterministic ordering independent of filesystem and absolute base path.
    return sorted(paths, key=lambda p: _as_rel_str(p, data_dir))


def _read_header(path: Path) -> Mapping[str, Any]:
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:
        raise ImportError(
            "Astropy is required for preflight FITS scanning. Install with: pip install astropy"
        ) from e
    return fits.getheader(path, 0, memmap=False)


def _read_frame_meta(hdr: Mapping[str, Any]) -> FrameMeta:
    # Strict required keys: quarantine on violation.
    return parse_frame_meta(hdr, strict=True)


def _read_frame_signature(hdr: Mapping[str, Any], meta: FrameMeta | None) -> Any:
    fallback_shape = None
    if meta is not None:
        try:
            fallback_shape = (int(meta.naxis2), int(meta.naxis1))
        except Exception:
            fallback_shape = None
    try:
        from scorpio_pipe.frame_signature import FrameSignature
    except Exception as e:
        raise ImportError(
            "FrameSignature is unavailable. This usually means required dependencies are missing."
        ) from e
    # FrameSignature expects an Astropy Header, but it only accesses mapping keys.
    # Be defensive: if the input type is not a real Header, convert to dict.
    try:
        return FrameSignature.from_header(hdr, fallback_shape=fallback_shape)  # type: ignore[arg-type]
    except Exception:
        return FrameSignature.from_header(dict(hdr), fallback_shape=fallback_shape)  # type: ignore[arg-type]


def run_preflight(
    cfg: dict[str, Any],
    run_root: str | Path,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    """Run preflight scan and write artifacts.

    Returns a small, JSON-serializable summary.
    """

    run_root_p = Path(run_root).expanduser().resolve()
    man_dir = run_root_p / "manifest"
    quar_dir = run_root_p / "quarantine"
    man_dir.mkdir(parents=True, exist_ok=True)
    quar_dir.mkdir(parents=True, exist_ok=True)

    # Avoid stale tickets on reruns: keep quarantine directory content aligned
    # with the current scan results.
    try:
        for old in sorted(quar_dir.glob("*.json")):
            old.unlink(missing_ok=True)
    except Exception:
        pass

    data_dir_p = resolve_data_dir(cfg)
    if data_dir_p is None:
        raise FileNotFoundError("preflight: could not resolve data_dir from config")
    data_dir_p = data_dir_p.expanduser().resolve()
    if not data_dir_p.exists() or not data_dir_p.is_dir():
        raise FileNotFoundError(f"preflight: data_dir does not exist or is not a directory: {data_dir_p}")

    ex = resolve_exclude_set(cfg, data_dir=data_dir_p)
    excluded_abs = set(ex.excluded_abs or set())

    # Night-profile aggregators (deterministic: we keep sorted counters).
    from collections import Counter

    c_instr: Counter[str] = Counter()
    c_mode: Counter[str] = Counter()
    c_grism: Counter[str] = Counter()
    c_bin: Counter[str] = Counter()
    c_shape: Counter[str] = Counter()
    c_window: Counter[str] = Counter()
    c_readout: Counter[str] = Counter()
    c_lamp: Counter[str] = Counter()
    z_vals: list[float] = []

    quarantine_entries: list[dict[str, Any]] = []
    issue_counts: Counter[tuple[str, str]] = Counter()  # (severity, code)

    paths = _iter_fits_paths(data_dir_p, recursive=recursive)
    n_total = int(len(paths))

    for p in paths:
        p_abs = p.expanduser().resolve()
        rel = _as_rel_str(p_abs, data_dir_p)
        fid = _stable_id(rel)

        issues: list[dict[str, Any]] = []
        excluded = str(p_abs) in excluded_abs
        if excluded:
            issues.append(
                _issue(
                    RC_EXCLUDED_BY_POLICY,
                    "INFO",
                    "Excluded by policy (project_manifest.yaml/config exclude).",
                    context={"path": rel, "exclude_summary": dict(ex.summary or {})},
                )
            )

        hdr: Mapping[str, Any] | None = None
        meta: FrameMeta | None = None
        sig: Any | None = None
        if not excluded:
            try:
                hdr = _read_header(p_abs)
            except Exception as e:
                issues.append(
                    _issue(
                        RC_FITS_OPEN_ERROR,
                        "ERROR",
                        "Failed to open FITS header.",
                        context={"path": rel, "error": str(e)},
                    )
                )
            if hdr is not None:
                try:
                    meta = _read_frame_meta(hdr)
                except HeaderContractError as e:
                    ctx: dict[str, Any] = {"path": rel}
                    if getattr(e, "missing_keys", None):
                        ctx["missing_keys"] = list(getattr(e, "missing_keys"))
                    if getattr(e, "instrument", None):
                        ctx["instrument"] = str(getattr(e, "instrument"))
                    issues.append(
                        _issue(
                            RC_HEADER_CONTRACT_ERROR,
                            "ERROR",
                            str(e),
                            context=ctx,
                        )
                    )
                except Exception as e:
                    issues.append(
                        _issue(
                            RC_META_PARSE_ERROR,
                            "ERROR",
                            "Unexpected error while parsing metadata.",
                            context={"path": rel, "error": str(e)},
                        )
                    )
            if hdr is not None and meta is not None:
                # Record soft problems as WARN (not quarantining by themselves).
                if getattr(meta, "meta_missing_optional", None):
                    issues.append(
                        _issue(
                            RC_META_MISSING_OPTIONAL,
                            "WARN",
                            "Some optional metadata keys are missing.",
                            context={"path": rel, "missing": list(meta.meta_missing_optional)},
                        )
                    )
                if getattr(meta, "meta_fallback_used", None):
                    issues.append(
                        _issue(
                            RC_META_FALLBACK_USED,
                            "WARN",
                            "Metadata fallback values were applied.",
                            context={"path": rel, "fallback": dict(meta.meta_fallback_used)},
                        )
                    )

                # Night profile counters (only for non-excluded frames with valid meta).
                c_instr[str(meta.instrument or "")] += 1
                c_mode[str(meta.mode or "")] += 1
                c_grism[str(meta.disperser or "")] += 1
                c_bin[str(meta.binning_key or "")] += 1
                c_shape[f"{int(meta.naxis2)}x{int(meta.naxis1)}"] += 1
                ro = meta.readout_key
                c_readout[f"{ro.node}|rate={ro.rate:g}|gain={ro.gain:g}"] += 1
                lamp = str(getattr(meta, "lamp_type", "") or "")
                lamp_src = str(getattr(meta, "lamp_source", "") or "")
                if lamp:
                    c_lamp[f"{lamp} ({lamp_src})" if lamp_src else lamp] += 1
                try:
                    sig = _read_frame_signature(hdr, meta)
                    c_window[str(sig.window or "")] += 1
                except Exception:
                    sig = None

                z = _safe_float(hdr.get("Z") if hasattr(hdr, "get") else None)
                if z is None:
                    # Some headers use ZD or ZDIST.
                    z = _safe_float(
                        (hdr.get("ZD") if hasattr(hdr, "get") else None)
                        or (hdr.get("ZDIST") if hasattr(hdr, "get") else None)
                    )
                if z is not None:
                    z_vals.append(float(z))

        # Collect quarantine tickets for any frame with issues.
        if issues:
            sev: str | None = None
            for it in issues:
                sev = _sev_max(sev, str(it.get("severity")))
                issue_counts[(str(it.get("severity")), str(it.get("code")))] += 1
            sev = sev or "INFO"

            ticket = {
                "id": fid,
                "path": rel,
                "severity": sev,
                # Prefer ERROR first in ticket view.
                "issues": sorted(
                    issues,
                    key=lambda x: (
                        -SEVERITY_ORDER.get(str(x.get("severity")), 0),
                        str(x.get("code")),
                    ),
                ),
            }

            ticket_path = quar_dir / f"{fid}.json"
            _write_json(ticket_path, ticket)

            quarantine_entries.append(
                {
                    "id": fid,
                    "path": rel,
                    "severity": sev,
                    "codes": sorted({str(i.get("code")) for i in issues}),
                    "ticket": f"quarantine/{fid}.json",
                }
            )

    # Deterministic ordering for outputs.
    quarantine_entries.sort(key=lambda e: (str(e.get("severity")), str(e.get("path")), str(e.get("id"))))

    # Z-stats
    z_sorted = sorted(z_vals)
    z_stats = {
        "n": int(len(z_sorted)),
        "min": _quantile(z_sorted, 0.0),
        "p10": _quantile(z_sorted, 0.10),
        "median": median(z_sorted) if z_sorted else None,
        "p90": _quantile(z_sorted, 0.90),
        "max": _quantile(z_sorted, 1.0),
    }
    # Round floats for stability.
    for k in list(z_stats.keys()):
        if isinstance(z_stats[k], float):
            z_stats[k] = round(float(z_stats[k]), 6)

    def _counter_list(c: "Counter[str]") -> list[dict[str, Any]]:
        return [
            {"value": k, "n": int(v)}
            for k, v in sorted(c.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            if str(k) != ""
        ]

    night_profile = {
        "schema": "scorpio-pipe.night-profile.v1",
        "pipeline": {"version": str(PIPELINE_VERSION)},
        "data_dir": str(data_dir_p),
        "exclude_summary": dict(ex.summary or {}),
        "summary": {
            "n_total_fits": n_total,
            "n_quarantine": int(len(quarantine_entries)),
            "n_excluded": int(sum(1 for e in quarantine_entries if RC_EXCLUDED_BY_POLICY in (e.get("codes") or []))),
        },
        "instrument": _counter_list(c_instr),
        "mode": _counter_list(c_mode),
        "grism": _counter_list(c_grism),
        "binning": _counter_list(c_bin),
        "shape": _counter_list(c_shape),
        "window": _counter_list(c_window),
        "readout": _counter_list(c_readout),
        "lamp_hints": _counter_list(c_lamp),
        "z_stats": z_stats,
    }

    # Preflight report
    any_error = any(e.get("severity") == "ERROR" for e in quarantine_entries)
    any_warn = any(e.get("severity") == "WARN" for e in quarantine_entries)
    status = "ERROR" if any_error else ("WARN" if any_warn else "OK")

    # Aggregate reasons by (severity, code)
    reasons: list[dict[str, Any]] = []
    for (sev, code), n in sorted(
        issue_counts.items(),
        key=lambda kv: (-SEVERITY_ORDER.get(kv[0][0], 0), -int(kv[1]), str(kv[0][1])),
    ):
        # sample up to 5 paths for this code
        sample: list[str] = []
        for ent in quarantine_entries:
            if code in (ent.get("codes") or []):
                sample.append(str(ent.get("path")))
            if len(sample) >= 5:
                break
        reasons.append({"severity": sev, "code": code, "n": int(n), "sample_paths": sample})

    preflight_report = {
        "schema": "scorpio-pipe.preflight-report.v1",
        "status": status,
        "data_dir": str(data_dir_p),
        "summary": {
            "n_total_fits": n_total,
            "n_quarantine": int(len(quarantine_entries)),
            "n_reasons": int(len(reasons)),
        },
        "reasons": reasons,
    }

    quarantine_index = {
        "schema": "scorpio-pipe.quarantine-index.v1",
        "data_dir": str(data_dir_p),
        "entries": quarantine_entries,
    }

    _write_json(man_dir / "night_profile.json", night_profile)
    _write_json(man_dir / "preflight_report.json", preflight_report)
    _write_json(man_dir / "quarantine_index.json", quarantine_index)

    return {
        "status": status,
        "n_total_fits": n_total,
        "n_quarantine": int(len(quarantine_entries)),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def ensure_preflight(
    cfg: dict[str, Any],
    run_root: str | Path,
    *,
    recursive: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure preflight artifacts exist and return a small summary.

    This is the *single preflight entry point* used by the engine and tasks.
    It is intentionally conservative:
    - If artifacts exist and are readable, it reuses them (unless ``force=True``).
    - Otherwise, it runs :func:`run_preflight`.
    - If preflight fails unexpectedly, it writes a deterministic ERROR report
      and returns ``{"status": "ERROR", ...}``.
    """

    run_root_p = Path(run_root).expanduser().resolve()
    man_dir = run_root_p / "manifest"
    report_p = man_dir / "preflight_report.json"
    night_p = man_dir / "night_profile.json"
    qidx_p = man_dir / "quarantine_index.json"

    if not force and report_p.exists() and night_p.exists() and qidx_p.exists():
        rep = _read_json(report_p) or {}
        status = str(rep.get("status") or "").upper()
        if status in {"OK", "WARN", "ERROR"}:
            # Best-effort: also return the most important counters.
            summ = rep.get("summary") if isinstance(rep.get("summary"), dict) else {}
            return {
                "status": status,
                "n_total_fits": int((summ or {}).get("n_total_fits") or 0),
                "n_quarantine": int((summ or {}).get("n_quarantine") or 0),
                "reused": True,
            }

    try:
        res = run_preflight(cfg, run_root_p, recursive=recursive)
        out = dict(res)
        out["reused"] = False
        # Normalise status for callers.
        out["status"] = str(out.get("status") or "").upper()
        return out
    except Exception as e:
        # Best-effort: still write a deterministic ERROR report.
        try:
            man_dir.mkdir(parents=True, exist_ok=True)
            data_dir_p = None
            try:
                data_dir_p = resolve_data_dir(cfg)
            except Exception:
                data_dir_p = None
            payload = {
                "schema": "scorpio-pipe.preflight-report.v1",
                "status": "ERROR",
                "data_dir": str(data_dir_p) if data_dir_p is not None else None,
                "summary": {"error": str(e)},
                "reasons": [
                    {
                        "severity": "ERROR",
                        "code": "PREFLIGHT_FAILED",
                        "n": 1,
                        "sample_paths": [],
                    }
                ],
            }
            report_p.write_text(_json_dumps(payload), encoding="utf-8")
        except Exception:
            pass
        return {
            "status": "ERROR",
            "n_total_fits": 0,
            "n_quarantine": 0,
            "reused": False,
            "error": str(e),
        }


__all__ = [
    "run_preflight",
    "ensure_preflight",
    # reason codes
    "RC_EXCLUDED_BY_POLICY",
    "RC_FITS_OPEN_ERROR",
    "RC_HEADER_CONTRACT_ERROR",
    "RC_META_PARSE_ERROR",
    "RC_META_MISSING_OPTIONAL",
    "RC_META_FALLBACK_USED",
]
