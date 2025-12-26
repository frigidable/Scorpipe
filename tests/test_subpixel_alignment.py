import numpy as np

from scorpio_pipe.shift_utils import xcorr_shift_subpix, shift2d_subpix_x, shift2d_subpix_x_var, shift2d_subpix_x_mask


def _shift_1d_linear(x: np.ndarray, shift: float, *, fill: float = 0.0) -> np.ndarray:
    """Shift signal so that out[i] = in[i - shift]."""
    n = x.size
    grid = np.arange(n, dtype=float)
    xin = grid - float(shift)
    out = np.interp(xin, grid, x, left=fill, right=fill)
    return out


def test_xcorr_shift_subpix_sign_and_accuracy() -> None:
    n = 2048
    grid = np.arange(n, dtype=float)
    ref = np.exp(-0.5 * ((grid - 900.0) / 7.5) ** 2)

    # Build `cur` as ref shifted to the RIGHT by +2.35 px.
    # To match ref, we need to shift cur LEFT, i.e. apply a negative shift.
    true = -2.35
    cur = _shift_1d_linear(ref, -true, fill=0.0)

    est = xcorr_shift_subpix(ref, cur, max_shift=8)
    assert abs(est.shift_pix - true) < 0.08
    assert np.isfinite(est.score)


def test_xcorr_shift_subpix_zero_shift() -> None:
    rng = np.random.default_rng(123)
    ref = rng.normal(0.0, 1.0, 1024)
    cur = ref.copy()
    est = xcorr_shift_subpix(ref, cur, max_shift=10)
    assert abs(est.shift_pix) < 1e-6


def _shift2d_linear_reference(a2d: np.ndarray, shifts: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    """Slow reference 2D shifter using np.interp per row."""
    a2d = np.asarray(a2d, dtype=float)
    ny, nx = a2d.shape
    shifts = np.asarray(shifts, dtype=float).reshape(-1)
    if shifts.size == 1:
        shifts = np.full(ny, float(shifts[0]))
    out = np.empty_like(a2d)
    grid = np.arange(nx, dtype=float)
    for y in range(ny):
        xin = grid - float(shifts[y])
        out[y] = np.interp(xin, grid, a2d[y], left=fill, right=fill)
    return out


def test_shift2d_subpix_x_matches_reference() -> None:
    ny, nx = 64, 512
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, (ny, nx))
    # smooth-ish structure
    x = np.arange(nx, dtype=float)
    a += 0.5 * np.exp(-0.5 * ((x - 250.0) / 9.0) ** 2)[None, :]

    shifts = rng.uniform(-2.5, 2.5, ny)
    ref = _shift2d_linear_reference(a, shifts, fill=0.0)
    out, filled = shift2d_subpix_x(a, shifts, fill=0.0)

    # Ignore filled pixels in the comparison
    good = ~filled
    assert np.allclose(out[good], ref[good], atol=1e-7, rtol=0.0)


def test_shift2d_subpix_x_var_kernel_propagation() -> None:
    ny, nx = 16, 256
    rng = np.random.default_rng(7)
    var = np.abs(rng.normal(0.0, 1.0, (ny, nx))) + 0.1
    shifts = rng.uniform(-1.8, 1.8, ny)

    # Monte-carlo check: propagate a noise realization and compare variance.
    nmc = 200
    acc = np.zeros((ny, nx), dtype=float)
    acc2 = np.zeros((ny, nx), dtype=float)
    for _ in range(nmc):
        noise = rng.normal(0.0, 1.0, (ny, nx)) * np.sqrt(var)
        x, _ = shift2d_subpix_x(noise, shifts, fill=0.0)
        acc += x
        acc2 += x * x
    mc_var = acc2 / nmc - (acc / nmc) ** 2

    out_var, filled = shift2d_subpix_x_var(var, shifts, fill=float('inf'))
    good = ~filled

    # MC is noisy; allow a loose tolerance.
    rel = np.median(np.abs(mc_var[good] - out_var[good]) / np.maximum(out_var[good], 1e-12))
    assert rel < 0.12


def test_shift2d_subpix_x_mask_or_and_fill() -> None:
    ny, nx = 8, 64
    m = np.zeros((ny, nx), dtype=np.uint16)
    m[:, 10] = 2
    m[:, 20] = 4
    shifts = np.linspace(-3.2, 3.2, ny)

    out, filled = shift2d_subpix_x_mask(m, shifts, no_cov=1)
    # Some pixels must be outside coverage
    assert filled.any()
    assert np.all((out[filled] & 1) == 1)
    # OR must preserve bits
    assert (out[:, 10] != 0).any() or (out[:, 20] != 0).any()


def test_shift2d_subpix_x_mask_accepts_no_coverage_bit_alias():
    m = np.zeros((8, 32), dtype=np.uint16)
    m[:, :2] = 4
    out, filled = shift2d_subpix_x_mask(m, 0.5, no_coverage_bit=8)
    assert out.shape == m.shape
    assert filled.shape == m.shape
    assert out.dtype == np.uint16
    # new no-coverage bit must appear at least somewhere
    assert (out & 8).any()
