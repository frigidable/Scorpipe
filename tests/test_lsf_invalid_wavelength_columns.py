"""Masked wavelength-map columns remain invalid without spurious LSF warnings."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from scorpio_pipe.stages.lsf_model import (
    _columnwise_finite_median,
    _approx_linear_grid_from_lambda_map,
)

fits = pytest.importorskip("astropy.io.fits")


def test_lsf_column_median_retains_invalid_columns_without_warnings() -> None:
    plane = np.array(
        [
            [np.nan, 5000.0, np.nan, 5006.0, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 5002.0, 5004.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 5006.0, 5008.0, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        med = _columnwise_finite_median(plane)

    assert recorded == []
    assert med.shape == (8,)
    assert np.isnan(med[0]) and np.all(np.isnan(med[4:]))
    assert np.allclose(med[1:4], [5001.0, 5005.0, 5007.0])


def test_lsf_column_median_entirely_invalid_is_explicit_nan() -> None:
    data = np.full((4, 16), np.nan)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        out = _columnwise_finite_median(data)
    assert recorded == []
    assert out.shape == (16,)
    assert np.isnan(out).all()
    assert _approx_linear_grid_from_lambda_map(
        data, fits.Header({"WAVEREF": "air"})
    ) is None


def test_lsf_linear_grid_uses_available_columns_only() -> None:
    y, x = 6, 32
    plane = np.broadcast_to(5000.0 + 2.0 * np.arange(x), (y, x)).copy()
    plane[:, :3] = np.nan
    plane[:, -4:] = np.nan
    plane[3, :] = np.nan
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        grid = _approx_linear_grid_from_lambda_map(
            plane, fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
        )
    assert recorded == []
    assert grid is not None
    assert grid.dlambda == pytest.approx(2.0, abs=1e-8)
    assert grid.lambda0 == pytest.approx(5000.0, abs=1e-8)
    assert grid.nlam == x
