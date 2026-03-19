from __future__ import annotations

from scorpio_pipe.compare.longslit import LongComparisonBundle, compare_against_long


def test_compare_against_long_emits_numeric_deltas() -> None:
    a = LongComparisonBundle(
        source='scorpipe',
        wavelength_solution={'rms': 0.12},
        sky_subtraction={'sky_rms': 1.1},
        extraction={'snr': 25.0},
        response_calibration={'smoothness': 0.02},
        science_outputs={'n_detected_lines': 5},
    )
    b = LongComparisonBundle(
        source='long',
        wavelength_solution={'rms': 0.18},
        sky_subtraction={'sky_rms': 1.3},
        extraction={'snr': 21.0},
        response_calibration={'smoothness': 0.03},
        science_outputs={'n_detected_lines': 4},
    )
    rep = compare_against_long(a, b)
    assert rep['wavelength_solution']['rms']['delta'] == 0.12 - 0.18
    assert rep['science_outputs']['n_detected_lines']['delta'] == 1
