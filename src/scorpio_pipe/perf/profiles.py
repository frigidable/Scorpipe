from __future__ import annotations

import copy
from typing import Any, Mapping

PROFILE_SCHEMA = "scorpio_pipe.performance-profile.v1"
_VALID_MODES = {"fast", "full"}


def normalize_performance_mode(mode: Any) -> str:
    s = str(mode or "").strip().lower()
    if not s:
        return "full"
    return s


def _deepcopy_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return copy.deepcopy(dict(cfg))
    except Exception:
        return dict(cfg)


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    cur = parent.get(key)
    if not isinstance(cur, dict):
        cur = {}
        parent[key] = cur
    return cur


def _get_path(cfg: Mapping[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in str(path).split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = str(path).split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = _ensure_mapping(cur, part)
    cur[parts[-1]] = value


def _as_int(value: Any) -> int | None:
    try:
        if value in (None, "", False):
            return None
        iv = int(value)
    except Exception:
        return None
    return iv


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, "", False):
            return None
        return float(value)
    except Exception:
        return None


def _apply_min(cfg: dict[str, Any], *, path: str, limit: int | float, rationale: str, changes: list[dict[str, Any]]) -> None:
    cur = _get_path(cfg, path)
    cur_num = _as_float(cur)
    new_val = limit if cur_num is None else min(cur_num, float(limit))
    if cur_num is not None and abs(float(new_val) - float(cur_num)) < 1e-12:
        return
    final: Any = int(new_val) if isinstance(limit, int) and float(new_val).is_integer() else float(new_val)
    _set_path(cfg, path, final)
    changes.append(
        {
            "path": str(path),
            "strategy": "min",
            "before": cur,
            "after": final,
            "rationale": str(rationale),
        }
    )


def _apply_max(cfg: dict[str, Any], *, path: str, floor: int | float, rationale: str, changes: list[dict[str, Any]]) -> None:
    cur = _get_path(cfg, path)
    cur_num = _as_float(cur)
    new_val = floor if cur_num is None else max(cur_num, float(floor))
    if cur_num is not None and abs(float(new_val) - float(cur_num)) < 1e-12:
        return
    final: Any = int(new_val) if isinstance(floor, int) and float(new_val).is_integer() else float(new_val)
    _set_path(cfg, path, final)
    changes.append(
        {
            "path": str(path),
            "strategy": "max",
            "before": cur,
            "after": final,
            "rationale": str(rationale),
        }
    )


def get_performance_profile_summary(cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(cfg, Mapping):
        return None
    runtime = cfg.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    prof = runtime.get("performance_profile")
    return dict(prof) if isinstance(prof, Mapping) else None


def compact_performance_profile_summary(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = cfg.get("runtime") if isinstance(cfg, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    mode = normalize_performance_mode(runtime.get("performance_mode"))
    prof = get_performance_profile_summary(cfg) or {}
    overrides = prof.get("overrides") if isinstance(prof.get("overrides"), list) else []
    return {
        "performance_mode": mode,
        "preset": str(prof.get("preset") or ("quick_inspection" if mode == "fast" else "scientific_maximum")),
        "n_overrides": int(len(overrides)),
        "scientific_intent": str(
            prof.get("scientific_intent")
            or (
                "rapid_look" if mode == "fast" else "scientific_maximum"
            )
        ),
    }


def apply_performance_profile(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return config with resolved fast/full performance profile.

    The applied profile is stored in ``runtime.performance_profile`` so it is
    persisted into effective_config provenance and visible to downstream QC.
    The function is idempotent for already-applied configs.
    """

    out = _deepcopy_cfg(cfg)
    runtime = _ensure_mapping(out, "runtime")
    mode = normalize_performance_mode(runtime.get("performance_mode"))
    if mode not in _VALID_MODES:
        return out

    existing = runtime.get("performance_profile")
    if isinstance(existing, Mapping):
        if str(existing.get("schema") or "") == PROFILE_SCHEMA and str(existing.get("applied_mode") or "") == mode:
            runtime["performance_mode"] = mode
            return out

    changes: list[dict[str, Any]] = []

    if mode == "fast":
        _apply_max(
            out,
            path="wavesol.trace_y_step",
            floor=2,
            rationale="Trace every 2 detector rows in fast mode to reduce Wave2D work.",
            changes=changes,
        )
        _apply_min(
            out,
            path="wavesol.noise.n_iter",
            limit=2,
            rationale="Use fewer robust-noise refinement iterations in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="wavesol.poly_maxiter",
            limit=5,
            rationale="Reduce 1D dispersion sigma-clip iterations in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="wavesol.power_maxiter",
            limit=5,
            rationale="Reduce 2D power-model sigma-clip iterations in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="wavesol.cheb_maxiter",
            limit=5,
            rationale="Reduce 2D Chebyshev sigma-clip iterations in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="sky.kelson_raw.max_iter",
            limit=2,
            rationale="Use fewer Kelson robust-fit iterations in fast mode.",
            changes=changes,
        )
        _apply_max(
            out,
            path="sky.kelson_raw.knot_step_A",
            floor=1.5,
            rationale="Use a coarser spectral knot spacing for the fast sky model.",
            changes=changes,
        )
        _apply_min(
            out,
            path="sky.kelson_raw.flexure_n_samples",
            limit=50000,
            rationale="Subsample flexure estimation more aggressively in fast mode.",
            changes=changes,
        )
        _apply_max(
            out,
            path="sky.kelson_raw.delta_step_A",
            floor=0.4,
            rationale="Use a coarser delta-lambda search grid in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="sky.sky_scale_raw.max_iter",
            limit=2,
            rationale="Use fewer robust-fit iterations in sky-scale fallback for fast mode.",
            changes=changes,
        )
        _apply_max(
            out,
            path="sky.sky_scale_raw.knot_step_A",
            floor=1.5,
            rationale="Use a coarser spline spacing in sky-scale fallback for fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="linearize.stack_maxiters",
            limit=1,
            rationale="Reduce robust linearize-stack refinement passes in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="stack2d.robust_iter",
            limit=1,
            rationale="Reduce robust stack2d refinement passes in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="stack2d.var_floor_sample_rows",
            limit=48,
            rationale="Use fewer sampled sky rows for var-floor estimation in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="stack2d.var_floor_sample_cols",
            limit=128,
            rationale="Use fewer sampled columns for var-floor estimation in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="cosmics.la_niter",
            limit=2,
            rationale="Run fewer LA-Cosmic iterations in fast mode.",
            changes=changes,
        )
        _apply_min(
            out,
            path="cosmics.lap_niter",
            limit=1,
            rationale="Run fewer Laplacian refinement iterations in fast mode.",
            changes=changes,
        )
        _apply_max(
            out,
            path="wavesol.geometry_prior.block_height",
            floor=32,
            rationale="Use thicker row blocks in geometry_prior for faster nightly triage.",
            changes=changes,
        )
        _apply_min(
            out,
            path="slit_sanity.max_candidate_frames",
            limit=4,
            rationale="Inspect fewer candidate frames for slit sanity in fast mode.",
            changes=changes,
        )
        _apply_max(
            out,
            path="sky_refine.detect_snr",
            floor=5.0,
            rationale="Require stronger skyline residual detections in fast mode to reduce cleanup overhead.",
            changes=changes,
        )
        _apply_min(
            out,
            path="sky_refine.top_n_summary",
            limit=6,
            rationale="Keep a smaller observer summary package in fast mode.",
            changes=changes,
        )

    runtime["performance_mode"] = mode
    runtime["performance_profile"] = {
        "schema": PROFILE_SCHEMA,
        "requested_mode": mode,
        "applied_mode": mode,
        "preset": "quick_inspection" if mode == "fast" else "scientific_maximum",
        "scientific_intent": "rapid_look" if mode == "fast" else "scientific_maximum",
        "note": (
            "Fast mode trades some robustness/precision for turnaround; rerun with runtime.performance_mode=full for final science outputs."
            if mode == "fast"
            else "Full mode keeps the scientific/default maxima configured in the YAML."
        ),
        "overrides": changes,
        "n_overrides": int(len(changes)),
    }
    return out
