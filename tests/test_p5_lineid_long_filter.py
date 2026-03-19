import numpy as np

from scorpio_pipe.long_setup_priors import filter_lab_lines_for_setup


def test_filter_lab_lines_for_setup_keeps_coverage_and_boosts_strong_lines():
    lam = np.array([3500.0, 3888.65, 4200.0, 5852.49, 7300.0, 9000.0], dtype=float)
    w = np.ones_like(lam)
    lam2, w2, meta = filter_lab_lines_for_setup(lam, w, setup_or_cfg='VPHG1200@540')

    assert meta['applied'] is True
    assert 3500.0 not in lam2
    assert 9000.0 not in lam2
    assert 3888.65 in lam2
    assert 5852.49 in lam2
    i = int(np.where(np.isclose(lam2, 5852.49))[0][0])
    assert w2[i] > 1.0
