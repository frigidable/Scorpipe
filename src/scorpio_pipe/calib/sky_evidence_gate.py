"""Fail-closed SYNTHETIC evidence register for Scorpipe SkySub.

This checks an ensemble of distinct mathematical/synthetic QA scenarios,
NOT one observation or one instrument readout group. A green result never
authorizes release of scientifically calibrated VAR or removal of flags.
No I/O or modification of production reduction is performed.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import sqrt, isfinite
from typing import Any

import numpy as np


_REQUIRED = (
    "fidelity",
    "lsqr",
    "svd",
    "frozen_mc",
    "reclipped_mc",
    "stable_clipping",
    "changed_clipping",
    "changed_reclipping",
    "condition_sweep",
)


def _num(value: Any, field: str, *, lower: float = 0.) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field}: finite numeric value required")
    x = float(value)
    if not isfinite(x) or x < lower:
        raise ValueError(f"{field}: nonfinite or below lower bound")
    return x


def _integer(value: Any, field: str, *, lower: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < lower):
        raise ValueError(f"{field}: integer >= {lower} required")
    return int(value)


def _false_qualification(item: Mapping[str, Any]) -> None:
    if item.get("science_uncertainty_qualified") is not False:
        raise ValueError("Unqualified synthetic science status required")


def _covariance(item: Mapping[str, Any], field: str) -> np.ndarray:
    arr = np.asarray(item.get(field), dtype=float)
    if (arr.shape != (2, 2) or not np.isfinite(arr).all()
            or not np.allclose(arr, arr.T, rtol=1e-8, atol=1e-12)
            or np.linalg.eigvalsh(arr)[0] <= 0):
        raise ValueError(f"{field}: finite symmetric positive-definite 2x2 required")
    return arr


def _coverage_pair(item: Mapping[str, Any], field: str) -> np.ndarray:
    arr = np.asarray(item.get(field), dtype=float)
    if (arr.shape != (2,) or not np.isfinite(arr).all()
            or np.any(arr < 0) or np.any(arr > 1)):
        raise ValueError(f"{field}: two coverage fractions in [0,1] required")
    return arr


def _check(name: str, item: Mapping[str, Any]) -> None:
    """Independent evidence consistency checks, NOT authentication of reports."""
    _false_qualification(item)

    if name == "fidelity":
        if (item.get("status") != "MATCH" or
                _num(item.get("model_max_allowed_fraction"), "model fidelity") > 1. or
                _num(item.get("lsqr_residual_relative_difference"), "LSQR residual") > .01 or
                _integer(item.get("n_selected_fit_pixels"), "selected pixels", lower=1) < 1):
            raise ValueError("FITS/NPZ fidelity not confirmed")
        return

    if name == "lsqr":
        if (item.get("status") != "MATCH"
                or item.get("scope") != "CONDITIONAL_FINAL_SKY_MASK_ONLY"
                or item.get("fidelity_audit") != "MATCH"
                or item.get("all_lsqr_stops_nominal") is not True
                or _num(item.get("max_relative_l2_difference"), "LSQR residual response") > .05
                or _num(item.get("max_model_response_relative_l2_difference"), "LSQR sky-model response") > .05
                or _num(item.get("replay_model_relative_difference"), "LSQR replay") > .01):
            raise ValueError("LSQR local derivative and replay not sufficiently confirmed")
        return

    if name == "svd":
        p = _integer(item.get("n_coefficients"), "model coefficients", lower=1)
        rank = _integer(item.get("effective_rank"), "effective rank", lower=0)
        if (item.get("status") != "STABLE"
                or rank != p
                or _num(item.get("condition_weighted_design"), "weighted design condition") > 1e5
                or any(_num(item.get(key), key) > 1e-6 for key in (
                    "max_forward_relative_error",
                    "max_model_relative_error",
                    "max_adjoint_relative_error",
                    "max_duality_relative_error",
                ))):
            raise ValueError("Independent SVD numerical oracle did not qualify")
        return

    if name in ("frozen_mc", "reclipped_mc"):
        is_reclipped = name == "reclipped_mc"
        min_n = 200 if is_reclipped else 1000
        n = _integer(item.get("n_realizations"), "MC realizations", lower=min_n)
        scope = ("SYNTHETIC_PRODUCTION_KELSON_RECLIPPED"
                 if is_reclipped else "SYNTHETIC_GAUSSIAN_FIXED_KELSON_WLS")
        if item.get("status") != "MC_CONSISTENT" or item.get("scope") != scope:
            raise ValueError(f"{name}: MC status or scope not qualified")
        pred = _covariance(item, "analytic_covariance")
        empirical = _covariance(item, "empirical_covariance")
        coverage95 = _coverage_pair(item, "coverage_95percent")
        frac_error = np.max(np.abs(np.diag(empirical)/np.diag(pred) - 1.))
        max_frac_error = max(.10 if is_reclipped else .05, 5*sqrt(2./(n-1)))
        max_coverage_error = max(.03 if is_reclipped else .02, 5*sqrt(.95*.05/n))
        if (frac_error > max_frac_error
                or np.max(np.abs(coverage95 - .95)) > max_coverage_error):
            raise ValueError(f"{name}: recomputed sampling variance/coverage disagreement")
        if is_reclipped:
            if (item.get("conditional_mc_consistent") is not True
                    or _num(item.get("selection_change_fraction_any_exposure"), "selection fraction") != 0
                    or item.get("n_selection_changed_per_exposure") != [0, 0]
                    or item.get("n_nonconverged_per_exposure") != [0, 0]):
                raise ValueError("Re-clipping MC must retain baseline mask and converge")
        return

    if name in ("stable_clipping", "changed_clipping"):
        if item.get("scope") != "SYNTHETIC_FIXED_GEOMETRY_REFIT_CLIPPING":
            raise ValueError("Clipping diagnostic scope mismatch")
        _integer(item.get("n_probes"), "clip probes", lower=1)
        probes = item.get("probes")
        if not isinstance(probes, list) or len(probes) != item["n_probes"]:
            raise ValueError("Invalid clipping probe list")
        if name == "stable_clipping":
            if (item.get("status") != "FROZEN_STABLE" or
                    any(p.get("status") != "FROZEN_STABLE" or
                        p.get("plus_selection_changes") != 0 or
                        p.get("minus_selection_changes") != 0
                        for p in probes)):
                raise ValueError("Stable sample and model derivative not established")
        elif (item.get("status") != "SELECTION_CHANGED"
              or not any(p.get("status") == "SELECTION_CHANGED" and
                         (p.get("plus_selection_changes", 0)
                          + p.get("minus_selection_changes", 0)) > 0
                         for p in probes)):
            raise ValueError("Sensitivity to synthetic clipping boundary not detected")
        return

    if name == "changed_reclipping":
        if (item.get("status") != "SELECTION_VARIABLE"
                or item.get("scope") != "SYNTHETIC_PRODUCTION_KELSON_RECLIPPED"
                or _num(item.get("selection_change_fraction_any_exposure"),
                        "re-clipped change fraction") <= 0
                or not any(x > 0 for x in item.get("n_selection_changed_per_exposure", []))):
            raise ValueError("Monte Carlo selection variability not detected")
        return

    if name == "condition_sweep":
        if item.get("status") != "REPORTED" or item.get("scope") != "SYNTHETIC_NEAR_COLLINEAR_WLS_SVD_MC":
            raise ValueError("Conditioning sweep scope or status mismatch")
        trials = item.get("trials")
        if not isinstance(trials, list) or len(trials) < 2:
            raise ValueError("At least two condition sweep points required")
        statuses = [t.get("status") for t in trials]
        if (not all(t.get("reference_mc_consistent") is True for t in trials)
                or "NUMERIC_SAFE" not in statuses
                or "NUMERIC_BLOCKED" not in statuses
                or any(s not in {"NUMERIC_SAFE", "NUMERIC_BLOCKED"} for s in statuses)):
            raise ValueError("Condition sweep must distinguish stable and blocked regimes")
        for trial in trials:
            cond = np.asarray(trial.get("weighted_design_conditions"), dtype=float)
            if cond.shape != (2,) or not np.isfinite(cond).all() or np.any(cond < 1.):
                raise ValueError("Invalid per-exposure conditions")
            if trial["status"] == "NUMERIC_SAFE" and (
                any(x != "STABLE" for x in trial.get("svd_gate_statuses", []))
            ):
                raise ValueError("Numerically safe regime contradicts SVD gates")
            if trial["status"] == "NUMERIC_BLOCKED" and (
                all(x == "STABLE" for x in trial.get("svd_gate_statuses", []))
            ):
                raise ValueError("Numerically blocked regime without failed gate")
        return

    raise ValueError(f"Unsupported evidence category: {name}")


def assess_synthetic_sky_evidence(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit cross-scenario synthetic diagnostics without elevating VAR claims.

    The input is a COLLECTION OF DISTINCT SYNTHETIC SCENARIOS, not a
    single-science-frame evidence bundle. Different members must never be
    interpreted as being co-registered observations or sharing a noise model.

    Validates numeric relationships in addition to checking statuses.
    Submitted results are NOT authenticated: there is no cryptographic
    attestation that the reports were generated by a particular commit.
    """
    if not isinstance(reports, Mapping):
        raise ValueError("Evidence register must be a mapping")
    unknown = sorted(set(reports) - set(_REQUIRED))
    if unknown:
        raise ValueError(f"Unknown evidence categories: {unknown}")

    checks: dict[str, dict[str, str]] = {}
    for name in _REQUIRED:
        item = reports.get(name)
        if item is None:
            checks[name] = {"status": "MISSING", "reason": "No synthetic report supplied"}
            continue
        if not isinstance(item, Mapping):
            checks[name] = {"status": "INVALID", "reason": "Report is not a mapping"}
            continue
        try:
            _check(name, item)
        except (TypeError, ValueError, KeyError, IndexError, OverflowError,
                np.linalg.LinAlgError) as exc:
            checks[name] = {
                "status": "INVALID",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        else:
            checks[name] = {"status": "SUPPORTED", "reason": "Synthetic scenario contract met"}

    all_passed = all(x["status"] == "SUPPORTED" for x in checks.values())
    any_invalid = any(x["status"] == "INVALID" for x in checks.values())
    status = ("SYNTHETIC_CONTRACTS_SUPPORTED" if all_passed else
              ("EVIDENCE_INVALID" if any_invalid else "EVIDENCE_INCOMPLETE"))
    return {
        "schema": "SKYSUB_SYNTHETIC_EVIDENCE_REGISTER_V1",
        "status": status,
        "scope": "CROSS_SCENARIO_SYNTHETIC_REGRESSION_ONLY",
        "scenarios": checks,
        "n_supported": sum(x["status"] == "SUPPORTED" for x in checks.values()),
        "n_required": len(_REQUIRED),
        "all_synthetic_contracts_supported": all_passed,
        "observation_specific_qualification": "NOT_ASSESSED",
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
        "override_science_status_supported": False,
    }
