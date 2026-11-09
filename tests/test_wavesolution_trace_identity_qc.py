"""Wave2D must reject line switches and biased/overclipped whole-line fits.

Synthetic regressions capture the actual N2146sat VPHG1200B failure mode:
bright neighbouring arc lines are ~10-13 pixels apart and the old search
radius of 12 pixels allowed independent traces to merge silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.stages.wavesolution import trace_lines_2d_cc
from scorpio_pipe.wave_trace_qc import (
    evaluate_solution_2d,
    leave_one_line_out_cv,
    trace_identity_qc,
)


def _neighbouring_arc_lines():
    h, w = 90, 170
    y = np.arange(h, dtype=float)
    x = np.arange(w, dtype=float)[None, :]
    shift = 0.026 * (y - 45.0)
    centre1 = 52.0 + shift
    centre2 = 63.0 + shift
    image = (
        3.0
        + 140.0 * np.exp(-0.5 * ((x-centre1[:,None])/1.1)**2)
        + 480.0 * np.exp(-0.5 * ((x-centre2[:,None])/1.1)**2)
    )
    return image, centre1, centre2


def test_bright_neighbour_cannot_capture_other_line_with_12px_search():
    image, a, b = _neighbouring_arc_lines()
    x, y, lam, score = trace_lines_2d_cc(
        image,
        lambda_list=[4181.88, 4190.71],
        x_guesses=[52.0, 63.0],
        y0=45, search_rad=12, max_step_px=3.0, amp_thresh=20.0,
    )
    assert x.size >= 2 * 85
    assert np.all(np.isfinite(score) & (score > 0))
    for wav, truth in [(4181.88, a), (4190.71, b)]:
        m = np.isclose(lam, wav)
        assert m.sum() >= 85
        expected = np.interp(y[m], np.arange(90), truth)
        assert np.max(np.abs(x[m] - expected)) < 1.0
        seed = np.argmin(np.abs(y[m] - 45))
        # Seed and adjacent rows must use correlation power, not raw amplitude.
        adj = np.argmin(np.abs(y[m] - 46))
        s = score[m]
        assert 0.1 < s[seed] / s[adj] < 10.0
    report = trace_identity_qc(
        x, y, lam, np.array([52., 63.]), np.array([4181.88, 4190.71])
    )
    assert report["status"] == "PASS"
    assert report["adjacent_pairs"][0]["collision_fraction"] == 0.0


def test_trace_identity_rejects_neighbour_switches():
    y = np.arange(40, dtype=float)
    x1 = 52.0 + 0.02 * y
    x2 = 63.0 + 0.02 * y
    switched = x2.copy()
    switched[3:] = x1[3:] + 0.05
    x = np.r_[x1, switched]
    yy = np.r_[y, y]
    lam = np.r_[np.full(y.size, 4181.88), np.full(y.size, 4190.71)]
    report = trace_identity_qc(
        x, yy, lam, np.array([52., 63.]), np.array([4181.88,4190.71])
    )
    assert report["status"] == "REJECT"
    assert report["adjacent_pairs"][0]["collision_fraction"] > 0.9
    assert any("Trace collision" in issue for issue in report["issues"])


def test_trace_identity_rejects_unresolved_manual_seeds():
    y = np.arange(40, dtype=float)
    report = trace_identity_qc(
        np.r_[np.full(40, 20.), np.full(40, 22.)],
        np.r_[y,y],
        np.r_[np.full(40, 4000.), np.full(40, 4001.)],
        np.array([20.,22.]), np.array([4000.,4001.]),
    )
    assert report["status"] == "REJECT"
    assert any("Unresolved manual anchors" in x for x in report["issues"])


def _synthetic_2d_line_set():
    rng = np.random.default_rng(721)
    lines = 3900.0 + 45.0*np.arange(19)
    y = np.arange(60, dtype=float)
    yy = np.tile(y, len(lines))
    ll = np.repeat(lines, len(y))
    x0 = ((ll - 3900.0)/0.87
          + 0.035*(yy-30.0)
          + 0.0006*(yy-30.0)**2)
    xx = x0 + rng.normal(0.0, 0.035, x0.size)
    residual = 0.045*np.sin(yy/8.0)
    used = np.ones_like(xx, dtype=bool)
    return xx, yy, ll, residual, used


def test_good_traces_pass_per_line_and_whole_line_cv():
    x, y, lam, residuals, used = _synthetic_2d_line_set()
    report = evaluate_solution_2d(
        x,y,lam,residuals,used,dispersion_A_per_px=0.87
    )
    assert report["status"] == "PASS", report["issues"]
    assert report["rms_px"] < 0.1
    assert report["line_level_cv_rms_px"] < 0.3
    assert report["line_level_cv"]["n_independent_lines"] == 19


def test_bad_line_is_fatal_even_if_global_rms_is_diluted():
    x, y, lam, residuals, used = _synthetic_2d_line_set()
    residuals = residuals.copy()
    residuals[lam == np.unique(lam)[8]] += 5.0
    report = evaluate_solution_2d(
        x,y,lam,residuals,used,dispersion_A_per_px=0.87
    )
    assert report["status"] == "REJECT"
    assert any("median 2D bias" in problem for problem in report["issues"])


def test_whole_line_sigma_clipping_cannot_hang_stage_success():
    x, y, lam, residuals, used = _synthetic_2d_line_set()
    used = used.copy()
    used[lam == np.unique(lam)[4]] = False
    report = evaluate_solution_2d(
        x,y,lam,residuals,used,dispersion_A_per_px=0.87
    )
    assert report["status"] == "REJECT"
    assert any("retained 0.0%" in problem for problem in report["issues"])


def test_cv_rejects_rank_deficient_line_geometry():
    y = np.tile(np.arange(12,dtype=float), 8)
    lam = np.repeat(4000.0 + 40.0*np.arange(8), 12)
    x = y.copy()   # No independent spectral coordinate!
    with pytest.raises(ValueError, match="rank deficient"):
        leave_one_line_out_cv(x,y,lam)


def test_missing_manual_line_is_not_silently_ignored():
    y = np.arange(20, dtype=float)
    report = trace_identity_qc(
        np.r_[np.full(20, 50.), np.full(20, 65.)],
        np.r_[y,y],
        np.r_[np.full(20, 4000.), np.full(20, 4100.)],
        np.array([50., 65., 80.]),
        np.array([4000., 4100., 4200.]),
    )
    assert report["status"] == "REJECT"
    assert any("4200.000 A has no usable" in s for s in report["issues"])


def test_failed_science_2d_gate_precedes_lambda_map_fits_writer():
    from pathlib import Path
    from scorpio_pipe.stages import wavesolution as ws

    source = Path(ws.__file__).read_text(encoding="utf-8")
    routine = source[source.index("def build_wavesolution("):]
    assert routine.index('if trace_qc["status"] != "PASS"') < routine.index(
        "robust_polyfit_2d_power("
    )
    assert routine.index('if science_qc["status"] != "PASS"') < routine.index(
        "atomic_write_fits(\n        lambda_map_fits"
    )
    assert "SCIENCE_2D_QC FAILED" in routine
    assert "TRACE_IDENTITY_QC FAILED" in routine


def test_line_preserving_recovery_has_independent_cv_and_keeps_all_anchors():
    """19 laboratory anchors are independent; spatial rows are not."""
    from scorpio_pipe.stages.wavesolution import (
        robust_polyfit_2d_cheb, polyval2d_cheb,
    )

    anchors = np.array([
        [400.176298, 3888.65], [473.380668, 3948.98],
        [588.619268, 4044.42], [751.382565, 4181.88],
        [761.594648, 4190.71], [841.297918, 4259.36],
        [927.492923, 4333.56], [940.576356, 4345.17],
        [1084.128231, 4471.48], [1355.173277, 4713.00],
        [1523.345819, 4863.96], [1586.839180, 4921.93],
        [1690.314344, 5015.68], [1714.459619, 5037.75],
        [1725.428199, 5047.74], [1761.235524, 5080.38],
        [1916.558676, 5221.80], [2035.670268, 5330.78],
        [2047.665238, 5341.60],
    ])
    ys = np.arange(16.0, 1020.0, 20.0)
    xx = np.tile(anchors[:, 0], ys.size)
    yy = np.repeat(ys, anchors.shape[0])
    ll = np.tile(anchors[:, 1], ys.size)
    # Smooth SCORPIO-like curvature; each spectral line remains distinct.
    x = xx + (0.02 * (yy-516.0) + 0.00003*(yy-516.0)**2) * (
        1.0 + 0.00005*(xx - 1100.0)
    )
    weights = np.ones_like(ll)
    for line in np.unique(ll):
        indices = ll == line
        weights[indices] = 1.0 / np.sqrt(indices.sum())

    coeff, meta, used = robust_polyfit_2d_cheb(
        x, yy, ll, 3, 2, weights=weights, maxiter=0,
    )
    assert np.all(used), "No manually identified line may disappear"
    delta = polyval2d_cheb(x, yy, coeff, meta) - ll
    report = evaluate_solution_2d(
        x, yy, ll, delta, used, dispersion_A_per_px=0.87,
    )
    assert report["status"] == "PASS", report["issues"]
    assert report["line_level_cv_rms_px"] < 1.25
    assert all(row["used_fraction"] == 1.0 for row in report["per_line"])


def test_automatic_fallback_does_not_override_forced_model_or_hide_failure():
    from pathlib import Path
    from scorpio_pipe.stages import wavesolution as ws

    source = Path(ws.__file__).read_text(encoding="utf-8")
    routine = source[source.index("def build_wavesolution("):]
    assert "model_selection_diagnostic" in routine
    assert "maxiter=0,  # no line may disappear through global clipping" in routine
    assert 'model2d not in ("cheb", "chebyshev", "pow", "power", "poly")' in routine
    assert 'fallback_review["status"] == "PASS"' in routine
    assert 'model_selection_diagnostic["fallback_selected"] = True' in routine
    assert 'if science_qc["status"] != "PASS"' in routine
    assert routine.index("wavesolution_model_selection.json") < routine.index(
        "wavesolution_2d.json"
    )
