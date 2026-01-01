import numpy as np


def _import_linearize_helpers():
    # The pipeline uses Astropy for FITS I/O. Some lightweight CI environments
    # may not include it; align with existing tests that conditionally skip.
    import pytest

    pytest.importorskip("astropy")
    from scorpio_pipe.stages.linearize import _rebin_row_var_weightsquared

    return _rebin_row_var_weightsquared


def test_rebin_row_var_weightsquared_simple_overlap():
    """Variance propagation should follow Σ a_k^2 VAR_k for independent pixels.

    We create a toy 1-D row with 3 input pixels and 2 output bins whose edges
    split the middle of the first two pixels and the middle of the last two pixels.
    """

    _rebin_row_var_weightsquared = _import_linearize_helpers()

    # Input pixel edges in wavelength units:
    # [-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]  -> centers [0, 1, 2]
    edges_in = np.array([-0.5, 0.5, 1.5, 2.5], dtype=float)
    lam_centers = 0.5 * (edges_in[:-1] + edges_in[1:])

    # Output bins: [0,1] and [1,2]
    edges_out = np.array([0.0, 1.0, 2.0], dtype=float)

    var_in = np.array([4.0, 4.0, 4.0], dtype=float)
    valid = np.array([True, True, True], dtype=bool)

    var_out = _rebin_row_var_weightsquared(var_in, lam_centers, edges_out, valid_mask=valid)

    # Each output bin overlaps two input pixels by 0.5 of the input width -> a_k = 0.5
    # VAR_out = 0.25*4 + 0.25*4 = 2.0
    assert np.allclose(var_out, np.array([2.0, 2.0]), atol=1e-12)


def test_rebin_row_var_weightsquared_identity_mapping():
    """If output bins match input bins, variance should be preserved."""

    _rebin_row_var_weightsquared = _import_linearize_helpers()

    edges_in = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    lam_centers = 0.5 * (edges_in[:-1] + edges_in[1:])
    edges_out = edges_in.copy()

    var_in = np.array([1.0, 2.0, 3.0], dtype=float)
    valid = np.array([True, True, True], dtype=bool)

    var_out = _rebin_row_var_weightsquared(var_in, lam_centers, edges_out, valid_mask=valid)
    assert np.allclose(var_out, var_in, atol=1e-12)
