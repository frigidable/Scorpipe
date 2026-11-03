"""Independent grouped MasterBias covariance oracle and provenance tests."""
import numpy as np
import pytest
from scorpio_pipe.calib.bias_covariance import bias_group_diagnostics, fixed_weight_group_variance


@pytest.mark.parametrize("n", [1, 2, 7, 20])
def test_shared_masterbias_error_does_not_decay_with_number_of_exposures(n):
    weights = np.ones((n, 1))
    v_ind, v_cal, v_total = fixed_weight_group_variance(
        weights, 9. * weights, weights, ["g"] * n, {"g": 4.}
    )
    np.testing.assert_allclose(v_ind, 9. / n)
    np.testing.assert_allclose(v_cal, 4.)
    np.testing.assert_allclose(v_total, 9. / n + 4.)


def test_unequal_weights_two_groups_and_sensitivities():
    w = np.array([1., 2., 3., 4.])[:, None]
    v = np.array([2., 4., 6., 8.])[:, None]
    sensitivity = np.array([1., .5, 1.5, .25])[:, None]
    vi, vb, total = fixed_weight_group_variance(
        w, v, sensitivity, ["a", "b", "a", "b"], {"a": 3., "b": 5.}
    )
    expected_ind = (2. + 16. + 54. + 128.) / 100.
    expected_shared = 3. * ((1. + 4.5) / 10.) ** 2 + 5. * ((1. + 1.) / 10.) ** 2
    np.testing.assert_allclose([vi.item(), vb.item(), total.item()],
                               [expected_ind, expected_shared, expected_ind + expected_shared])


def test_monte_carlo_group_correlation_reference():
    rng = np.random.default_rng(117031)
    n = 20000
    errors = rng.normal(size=(n, 4)) * np.sqrt([9., 4., 9., 16.])
    group_a = rng.normal(size=(n, 1)) * np.sqrt(3.)
    group_b = rng.normal(size=(n, 1)) * np.sqrt(2.)
    obs = errors + np.concatenate([group_a, group_b, group_a, group_b], axis=1)
    w = np.array([1., 2., 1., .5])
    emp = np.var(obs @ w / w.sum(), ddof=1)
    _, _, theoretical = fixed_weight_group_variance(
        w[:, None], np.array([9., 4., 9., 16.])[:, None],
        np.ones((4, 1)), ["a", "b", "a", "b"], {"a": 3., "b": 2.}
    )
    np.testing.assert_allclose(emp, theoretical.item(), rtol=.05)


def test_zero_shared_uncertainty_is_independent_limit():
    w = np.array([[1.], [2.]])
    v = np.array([[9.], [4.]])
    vi, vb, total = fixed_weight_group_variance(
        w, v, np.ones_like(v), ["g", "g"], {"g": 0.}
    )
    np.testing.assert_allclose(vb, 0.)
    np.testing.assert_allclose(total, vi)


def test_zero_weight_nan_does_not_poison_fixed_weight_calculation():
    vi, vb, total = fixed_weight_group_variance(
        np.array([[1.], [0.]]), np.array([[3.], [np.nan]]),
        np.array([[1.], [np.nan]]), ["g", "g"], {"g": 2.}
    )
    np.testing.assert_allclose([vi.item(), vb.item(), total.item()], [3., 2., 5.])


@pytest.mark.parametrize("problem", ["missing_id", "bad_weight", "bad_variance"])
def test_invalid_common_calibration_input_fails_closed(problem):
    w, v = np.ones((2, 1)), np.ones((2, 1))
    ids = ["g", "g"]
    if problem == "missing_id":
        ids[0] = ""
    elif problem == "bad_weight":
        w[0, 0] = -1.
    else:
        v[0, 0] = -1.
    with pytest.raises(ValueError):
        fixed_weight_group_variance(w, v, np.ones((2, 1)), ids, {"g": 1.})


def test_bias_group_audit_shared_distinct_unknown():
    record = {"bias_group_id": "g", "bias_var_included": True}
    common = bias_group_diagnostics([record, record])
    assert common["status"] == "SHARED"
    assert common["shared_group_ids"] == ["g"]
    assert common["unique_group"] == "g"
    assert common["all_bias_var_included"]
    assert not common["variance_qualified"]
    distinct = bias_group_diagnostics([record, {"bias_group_id": "h"}])
    assert distinct["status"] == "DISTINCT"
    assert distinct["unique_group"] is None
    missing = bias_group_diagnostics([record, {}])
    assert missing["status"] == "UNKNOWN"
    assert missing["n_missing_group"] == 1


def test_extract1d_keeps_shared_bias_uncertainty_status(tmp_path):
    from scorpio_pipe.stages.extract1d import run_extract1d
    from tests.test_extract_stacked_rejection_semantics import _write_stack

    work = tmp_path / "work"
    input_path = tmp_path / "stack2d.fits"
    _write_stack(input_path, verified=True)
    with fits.open(input_path, mode="update") as hdul:
        h = hdul[0].header
        h["STKVRALG"] = "W2VAR"
        h["STKVARQ"] = "COND_ONLY"
        h["STKXCVQ"] = "UNMODEL"
        h["SBCOVQ"] = "UNMODEL"
        h["SBGSTAT"] = "SHARED"
        h["SBGRPN"] = 1
        hdul.flush()
    config = {
        "work_dir": str(work),
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        "extract1d": {
            "input_mode": "stack2d", "method": "boxcar", "save_png": False
        },
    }
    result = run_extract1d(config, in_fits=input_path, out_dir=work / "extract")
    with fits.open(result["spec1d_fits"]) as hdul:
        h = hdul[0].header
        assert h["STKVRALG"] == "W2VAR"
        assert h["STKVARQ"] == "COND_ONLY"
        assert h["STKXCVQ"] == "UNMODEL"
        assert h["SBCOVQ"] == "UNMODEL"
        assert h["SBGSTAT"] == "SHARED"
        assert h["SBGRPN"] == 1
