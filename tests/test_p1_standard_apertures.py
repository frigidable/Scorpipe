from __future__ import annotations

import numpy as np

from scorpio_pipe.extract.apertures import (
    choose_standard_half_width,
    enclosed_flux_curve,
    estimate_contamination_proxy,
)


def test_choose_standard_half_width_is_conservative_by_default() -> None:
    pol = choose_standard_half_width(4, ny=60, cfg={})

    assert pol["mode"] == "box_wide"
    assert pol["selected_half_width"] >= 8
    assert pol["selected_half_width"] > pol["base_half_width"]



def test_enclosed_flux_curve_grows_toward_plateau() -> None:
    ny, nlam = 41, 80
    y = np.arange(ny, dtype=float)[:, None]
    profile = np.exp(-0.5 * ((y - 20.0) / 2.0) ** 2)
    spectrum = 100.0 * (1.0 + 0.1 * np.sin(np.linspace(0, 3.14, nlam))[None, :])
    sci = profile * spectrum + 3.0
    var = np.ones_like(sci)
    y_center = np.full(nlam, 20.0)

    curve = enclosed_flux_curve(
        sci,
        var,
        None,
        y_center,
        sky_windows=[(0, 6), (34, 40)],
        half_widths=[2, 4, 8, 12],
    )

    fracs = curve["capture_fraction"]
    assert len(fracs) == 4
    assert fracs[0] < fracs[-1]
    assert fracs[-1] > 0.95



def test_estimate_contamination_proxy_detects_secondary_peak() -> None:
    ny, nlam = 61, 30
    y = np.arange(ny, dtype=float)[:, None]
    main = np.exp(-0.5 * ((y - 30.0) / 2.0) ** 2)
    secondary = 0.3 * np.exp(-0.5 * ((y - 44.0) / 2.0) ** 2)
    sci = (main + secondary) * np.ones((1, nlam), dtype=float) + 1.0

    rep = estimate_contamination_proxy(
        sci,
        y_center=30.0,
        sky_windows=[(0, 6), (54, 60)],
        selected_half_width=6,
    )

    assert rep["secondary_to_primary"] > 0.2
    assert rep["secondary_y"] is not None
