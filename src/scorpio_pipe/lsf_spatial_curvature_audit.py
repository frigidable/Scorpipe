"""Non-publishing two-lamp check for wavelength-dependent slit LSF curvature.

One row per original ARC, laboratory PAIR, y-bin and profile hypothesis.
This checks whether a shared spatial polynomial is defensible across lines.
The model is diagnostic-only and never writes FITS or authorizes sky shifts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FWHM_PAIRS_A = (
    (4181.88, 4190.71), (4333.56, 4345.17),
    (5037.75, 5047.74), (5330.78, 5341.60),
)


def _as_finite_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid profile field " + field) from exc
    if not np.isfinite(number):
        raise ValueError("Nonfinite profile field " + field)
    return number


def audit_neon_spatial_curvature(
    measurements: Sequence[Mapping[str, Any]],
    v18_candidate: Mapping[str, Any],
    *,
    arc_ids: tuple[str, str] = ("arc06", "arc13"),
    half_window_px: int = 16,
    continuum_degree: int = 1,
    center_y_pix: float = 506.0,
    min_common_bins: int = 16,
) -> dict[str, Any]:
    """Measure *separate* per-line curvature and compare to v18 global model.

    Both ARC repeats must show the effect. Curvature coefficients are in
    the dimensionless coordinate v=(y-515.5)/515.5. They are diagnostics,
    NOT independent formal uncertainties (shared wavelength map).
    """
    if (len(arc_ids) != 2 or len(set(arc_ids)) != 2
            or not all(isinstance(a, str) and a.strip() for a in arc_ids)):
        raise ValueError("Two distinct original-ARC identities are required")
    if (half_window_px < 10 or continuum_degree not in (1, 2)
            or not 0 <= center_y_pix <= 1031 or min_common_bins < 8):
        raise ValueError("Invalid original-ARC model audit configuration")
    if (v18_candidate.get("status") != "DIAGNOSTIC_CANDIDATE"
            or v18_candidate.get("science_lsf_validated") is not False
            or v18_candidate.get("auto_replace_production_lsf") is not False):
        raise ValueError("Only a proven diagnostic-only v18 baseline is permitted")
    coeff = np.asarray(v18_candidate.get("coefficients", []), dtype=float)
    if coeff.shape != (5,) or not np.all(np.isfinite(coeff)):
        raise ValueError("v18 coefficient vector must have five finite terms")
    rows: dict[tuple[int, str], dict[float, float]] = {}
    n_accepted = 0
    n_input = 0
    for record in measurements:
        n_input += 1
        try:
            status = str(record["status"])
            arc = str(record["arc_id"])
            pair_idx = int(record["pair"])
            y = _as_finite_float(record["y_pix"], "y_pix")
            window = int(record["half_window_px"])
            degree = int(record["continuum_degree"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed original-ARC hypothesis record") from exc
        if (arc not in arc_ids or pair_idx not in range(len(FWHM_PAIRS_A))
                or not 0 <= y <= 1031):
            raise ValueError("Unrecognized original ARC or pair or y")
        if status not in {"DIAGNOSTIC_FIT", "REJECT"}:
            raise ValueError("Unrecognized original-ARC fit status")
        if status != "DIAGNOSTIC_FIT" or window != half_window_px or degree != continuum_degree:
            continue
        fwhm = _as_finite_float(record["fwhm_A"], "fwhm_A")
        if not 0.5 < fwhm < 50.0:
            raise ValueError("Unphysical accepted original-ARC FWHM")
        samples = rows.setdefault((pair_idx, arc), {})
        if y in samples:
            raise ValueError("Duplicate ARC pair/spatial bin/hypothesis")
        samples[y] = fwhm
        n_accepted += 1

    result = []
    for idx, laboratory in enumerate(FWHM_PAIRS_A):
        first, second = (rows.get((idx,arc), {}) for arc in arc_ids)
        ys = np.asarray(sorted(set(first).intersection(second)), dtype=float)
        if ys.size < min_common_bins:
            result.append({
                "pair_air_A": list(laboratory), "status": "REJECT",
                "reason": "insufficient_common_original_arc_spatial_bins",
                "n_common_bins": int(ys.size),
            })
            continue
        spatial = (ys - 515.5) / 515.5
        if np.ptp(spatial) < 0.3:
            result.append({
                "pair_air_A": list(laboratory), "status": "REJECT",
                "reason": "insufficient_shared_spatial_leverage",
                "n_common_bins": int(ys.size),
            })
            continue
        widths = {
            arc_ids[0]: np.asarray([first[y] for y in ys]),
            arc_ids[1]: np.asarray([second[y] for y in ys]),
        }
        polynomial = {
            arc: np.polyfit(spatial, values, 2)
            for arc,values in widths.items()
        }
        mid = float(np.mean(laboratory))
        u = (mid-4600.0)/1000.0
        v0 = (center_y_pix-515.5)/515.5
        baseline_at_center = float(np.dot(coeff, [1.0, u, v0, v0*v0, u*v0]))
        pred_at_y = (coeff[0]+coeff[1]*u
                     +(coeff[2]+coeff[4]*u)*spatial
                     +coeff[3]*spatial*spatial)
        center_observations = {
            arc: float(np.polyval(poly, v0))
            for arc,poly in polynomial.items()
        }
        repeat = widths[arc_ids[1]]-widths[arc_ids[0]]
        mismatch = 0.5*(widths[arc_ids[0]]+widths[arc_ids[1]])-pred_at_y
        center_mask = np.abs(ys-center_y_pix) <= 100.0
        outer_mask = np.abs(ys-center_y_pix) >= 280.0
        result.append({
            "pair_air_A": list(laboratory),
            "status": "DIAGNOSTIC_ONLY",
            "n_common_bins": int(ys.size),
            "spatial_range_pix": [float(ys.min()), float(ys.max())],
            "fwhm_quadratic_coeff_v2_by_arc": {
                arc: float(poly[0]) for arc,poly in polynomial.items()
            },
            "fwhm_quadratic_coeff_v1_by_arc": {
                arc: float(poly[1]) for arc,poly in polynomial.items()
            },
            "shared_v18_quadratic_coeff_v2": float(coeff[3]),
            "arc_difference_median_abs_A": float(np.median(np.abs(repeat))),
            "arc_difference_p90_abs_A": float(np.percentile(np.abs(repeat),90)),
            "v18_model_center_fwhm_A": baseline_at_center,
            "independent_lamp_quadratic_fit_center_fwhm_A": center_observations,
            "center_zone_v18_residual_median_A": (
                float(np.median(mismatch[center_mask])) if np.any(center_mask) else None
            ),
            "outer_zone_v18_residual_median_A": (
                float(np.median(mismatch[outer_mask])) if np.any(outer_mask) else None
            ),
            "curvature_sign_matches_v18_for_both_arcs": bool(
                all(np.sign(poly[0]) == np.sign(coeff[3])
                    for poly in polynomial.values())
            ),
            "model_fit_uncertainty": "NOT_CALIBRATED",
        })
    return {
        "status": "DIAGNOSTIC_ONLY",
        "method": "independent_2arc_pair_y_quadratic_vs_shared_v18_y2",
        "n_input_hypotheses": n_input,
        "n_selected_hypotheses": n_accepted,
        "n_pairs_with_sufficient_repeatable_y": sum(
            row["status"] == "DIAGNOSTIC_ONLY" for row in result
        ),
        "pair_diagnostics": result,
        "shared_map_caution": (
            "Both original lamps share the _95 wavelength map and may share "
            "systematic LSF model errors; repeatability is not absolute accuracy."
        ),
        "uncertainty_status": "NOT_A_STATISTICAL_CONFIDENCE_INTERVAL",
        "science_lsf_validated": False,
        "auto_replace_production_lsf": False,
        "safe_for_stellar_kinematics": False,
    }
