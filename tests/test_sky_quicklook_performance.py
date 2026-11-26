"""Regression tests for display-only masking and redundant sky WLS."""
import numpy as np
import pytest

from scorpio_pipe.io.quicklook import write_quicklook_png
from scorpio_pipe.stages import sky_sub


@pytest.mark.parametrize("method", ["linear", "asinh"])
def test_quicklook_fatal_pixels_are_neutral(tmp_path, method):
    from matplotlib import image
    a = np.arange(100, dtype=float).reshape(10, 10)
    a[3, 4] = 1e6
    a[4, 5] = -1e6
    m = np.zeros(a.shape, dtype=np.uint16)
    m[3, 4] = 4
    m[4, 5] = 4
    before = a.copy()
    path = tmp_path / "preview.png"
    meta = write_quicklook_png(a, path, mask=m, method=method)
    rgb = image.imread(path)
    assert np.allclose(rgb[9-3, 4, :3], .5, atol=.01)
    assert np.allclose(rgb[9-4, 5, :3], .5, atol=.01)
    assert meta["n_display_masked"] == 2
    assert meta["n_good"] == 98
    np.testing.assert_array_equal(a, before)


def test_converged_sky_fit_does_not_repeat_identical_lsqr(monkeypatch):
    calls = []
    original = sky_sub.lsqr
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sky_sub, "lsqr", counted)
    lam = np.tile(np.linspace(4000, 4020, 101), 12)
    y = np.repeat(np.linspace(-.8, .8, 12), 101)
    data = 10 + .1 * (lam - 4000) + y
    fit = sky_sub._fit_kelson_bspline(
        lam, y, data, np.ones_like(data), knot_step_A=2.,
        y_order=1, clip_sigma=4.5, max_iter=4, min_pixels=100,
        degree=3, rng=np.random.default_rng(1))
    assert len(calls) == 1
    assert fit["solver_diagnostics"]["reused_final_solve"]
    assert fit["clipped_frac"] == 0.
    assert fit["sigma_r_after"] < 1e-5
