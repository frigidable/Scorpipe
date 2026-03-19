from __future__ import annotations

import numpy as np

from scorpio_pipe.calib.common import air_to_vacuum, vacuum_to_air
from scorpio_pipe.calib.fluxcal import _status_from_safe_for
from scorpio_pipe.calib.telluric import _estimate_window_transmission


def test_air_vacuum_conversion_is_close_to_roundtrip() -> None:
    w = np.linspace(3800.0, 9000.0, 128)
    vac = air_to_vacuum(w)
    back = vacuum_to_air(vac)
    assert np.max(np.abs(back - w)) < 0.3


def test_telluric_window_estimation_is_conservative() -> None:
    wave = np.linspace(6800.0, 7000.0, 400)
    flux = np.ones_like(wave)
    flux[(wave > 6870) & (wave < 6925)] *= 0.7
    var = np.full_like(wave, 1e-4)
    trans, conf = _estimate_window_transmission(wave, flux, var, 6860.0, 6965.0)
    assert trans.min() >= 0.25
    assert trans.max() <= 1.15
    assert float(np.nanmedian(conf)) > 0.2


def test_status_assignment_reflects_safe_for_and_telluric() -> None:
    assert _status_from_safe_for('absolute_candidate', telluric_applied=True) == 'ABSOLUTE_TRUSTED'
    assert _status_from_safe_for('approx_absolute', telluric_applied=True) == 'ABSOLUTE_APPROX'
    assert _status_from_safe_for('response_only', telluric_applied=True) == 'RELATIVE_TELLURIC'
    assert _status_from_safe_for('response_only', telluric_applied=False) == 'RELATIVE'
