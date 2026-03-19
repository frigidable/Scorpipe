from __future__ import annotations

import numpy as np

from scorpio_pipe.science.lines import DEFAULT_LINES, measure_lines


def test_measure_lines_propagates_stat_and_calibration_errors() -> None:
    wave = np.linspace(4800.0, 5100.0, 1200)
    continuum = np.full_like(wave, 1.0)
    line = 6.0 * np.exp(-0.5 * ((wave - 5006.84) / 1.5) ** 2)
    flux = continuum + line
    var = np.full_like(wave, 0.04)
    rel = np.full_like(wave, 0.05)
    tell = np.zeros_like(wave)
    absu = np.full_like(wave, 0.10)

    rows = measure_lines(wave, flux, var, calsys_rel=rel, calsys_tell=tell, calsys_abs=absu, lines=[s for s in DEFAULT_LINES if s.name == '[OIII]5007'])
    row = rows[0]

    assert row['detected'] is True
    assert abs(row['centroid'] - 5006.84) < 0.5
    assert row['err_stat'] > 0
    assert row['err_cal'] > 0
    assert row['err_total'] >= row['err_stat']
