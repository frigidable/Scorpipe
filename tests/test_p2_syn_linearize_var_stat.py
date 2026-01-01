import numpy as np


def _robust_sigma(x: np.ndarray) -> float:
    """Robust sigma estimator: 1.4826 * MAD."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _import_linearize_helpers():
    # Linearize helpers live in a module that depends on Astropy.
    import pytest

    pytest.importorskip("astropy")
    from scorpio_pipe.stages.linearize import (
        _rebin_row_cumulative,
        _rebin_row_var_weightsquared,
    )

    return _rebin_row_cumulative, _rebin_row_var_weightsquared


def test_p2_syn_linearize_white_noise_var_not_optimistic():
    """Linearization variance propagation should not be optimistic.

    We generate a synthetic 2D frame:
      - SCI = constant background + white noise
      - VAR = constant sigma^2
      - lambda(x,y) monotonic with small curvature

    After flux-conserving rebin + VAR propagation via Σ a^2 VAR, the normalized
    residuals r = (SCI_out - median)/sqrt(VAR_out) should have robust_sigma ~ 1.

    We allow a broad tolerance to account for edge effects and small numerical
    deviations.
    """

    _rebin_row_cumulative, _rebin_row_var_weightsquared = _import_linearize_helpers()

    rng = np.random.default_rng(12345)
    ny, nx = 64, 256

    bg = 100.0
    sigma = 5.0

    sci = bg + rng.normal(0.0, sigma, size=(ny, nx)).astype(float)
    var = np.full((ny, nx), sigma * sigma, dtype=float)

    x = np.arange(nx, dtype=float)[None, :]
    y = np.arange(ny, dtype=float)[:, None]

    # Monotonic λ(x,y) with mild curvature and gentle tilt in y.
    lam = 5000.0 + 2.0 * x + 0.002 * (x - (nx - 1) / 2.0) ** 2 + 0.4 * (y - (ny - 1) / 2.0) / ny

    # Common output edges: robust intersection over rows.
    lo = float(np.max(np.nanmin(lam, axis=1)))
    hi = float(np.min(np.nanmax(lam, axis=1)))
    dlam = float(np.median(np.diff(lam[ny // 2])))
    assert dlam > 0

    # Trim edges to avoid partial-coverage bins.
    lo += 2.0 * dlam
    hi -= 2.0 * dlam

    n = int(np.floor((hi - lo) / dlam))
    assert n > 32
    edges_out = lo + dlam * np.arange(n + 1, dtype=float)

    rs = []
    for j in range(ny):
        out, cov = _rebin_row_cumulative(sci[j], lam[j], edges_out)
        vout = _rebin_row_var_weightsquared(var[j], lam[j], edges_out)

        good = (cov > 0.99) & np.isfinite(out) & np.isfinite(vout) & (vout > 0)
        if good.sum() < 32:
            continue

        med = np.median(out[good])
        r = (out[good] - med) / np.sqrt(vout[good])
        rs.append(r)

    r_all = np.concatenate(rs)
    s = _robust_sigma(r_all)

    # Tolerances (configurable in principle): must not be substantially < 1.
    assert 0.85 <= s <= 1.30
