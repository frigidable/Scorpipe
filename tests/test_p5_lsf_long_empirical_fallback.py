import pytest

pytest.importorskip('astropy.io.fits')

import numpy as np

from scorpio_pipe.stages.lsf_model import _manual_fallback_map


def test_lsf_manual_fallback_uses_long_empirical_prior_when_manual_table_missing():
    lam_map = np.full((4, 5), 6500.0, dtype=float)
    hdr = {'DISPERSE': 'VPHG1720', 'SLITWID': 1.0}

    out, meta = _manual_fallback_map(lam_map, hdr)
    assert np.isfinite(out).all()
    assert meta['source'] == 'long_empirical'
    assert meta['fwhm_A'] > 0
    assert meta['long_setup_prior']['setup'] == 'VPHG1720'
