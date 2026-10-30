"""Cross-stage physical unit compatibility: standard -> sensfunc -> science."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.calib.fluxcal import apply_fluxcal_to_spec1d
from scorpio_pipe.extract.core import rectify_with_lambda_map


@pytest.mark.parametrize(
    "science_input_mode, stack_normalize, expected_unit, expected_count",
    [
        ("stack2d", True, "e-/s", 0.1),
        ("single_frame", True, "e-", 10.0),
    ],
)
def test_standard_rectification_matches_science_final_unit(
    tmp_path, science_input_mode, stack_normalize, expected_unit, expected_count,
):
    ny, nx = 4, 12
    lam = 5000.0 + 2.0 * np.arange(nx, dtype=float)
    lammap = np.broadcast_to(lam, (ny, nx)).copy()
    h = fits.Header()
    h["BUNIT"] = "e-"
    h["EXPTIME"] = 100.0
    h["GAIN"] = 2.0
    h["RDNOISE"] = 3.0
    input_path = tmp_path / "standard.fits"
    fits.PrimaryHDU(data=np.full((ny, nx), 20., dtype="f4"), header=h).writeto(input_path)
    cfg = {
        "linearize": {
            "normalize_exptime": False, "bunit_mode": "adu_bin",
            "post_sky_cleanup": "off", "mask_saturation": False,
        },
        "stack2d": {"normalize_exptime": stack_normalize},
        "extract1d": {"input_mode": science_input_mode},
    }
    rect = rectify_with_lambda_map(
        cfg=cfg, in_path=input_path, lambda_map=lammap,
        wave0=4999., dw=1., nlam=24,
    )
    assert rect["hdr"]["BUNIT"] == expected_unit
    assert rect["hdr"]["REBNORM"] == "COUNT_V1"
    np.testing.assert_allclose(rect["rect"], expected_count, rtol=1e-6, atol=1e-6)


def _flux_cal_inputs(tmp_path, science_unit="e-/s", science_rebnorm="COUNT_V1",
                     standard_unit="e-/s", standard_rebnorm="COUNT_V1",
                     science_step=1., standard_step=1.,
                     science_ref="air", standard_ref="air",
                     standard_step_metadata=True):
    wave = 5000. + science_step * np.arange(11, dtype=float)
    sens_wave = 5000. + standard_step * np.arange(11, dtype=float)
    hdr = fits.Header()
    hdr["BUNIT"] = science_unit
    if science_ref is not None:
        hdr["WAVEREF"] = science_ref
    hdr["WAVEUNIT"] = "Angstrom"
    if science_rebnorm is not None:
        hdr["REBNORM"] = science_rebnorm
    spectrum_path = tmp_path / "science_spec1d.fits"
    columns = [
        fits.Column(name="LAMBDA", format="D", array=wave),
        fits.Column(name="FLUX_TRACE", format="E", array=np.full(11, 10.)),
        fits.Column(name="VAR_TRACE", format="E", array=np.full(11, 4.)),
        fits.Column(name="MASK_TRACE", format="J", array=np.zeros(11, dtype="i4")),
    ]
    fits.HDUList([
        fits.PrimaryHDU(header=hdr),
        fits.BinTableHDU.from_columns(columns, name="SPEC_TRACE"),
    ]).writeto(spectrum_path)

    sens_header = fits.Header()
    if standard_ref is not None:
        sens_header["WAVEREF"] = standard_ref
    sens_header["INWUNIT"] = "Angstrom"
    if standard_step_metadata:
        sens_header["INSTEP"] = standard_step
    if standard_unit is not None:
        sens_header["INBUNIT"] = standard_unit
    if standard_rebnorm is not None:
        sens_header["INREBNM"] = standard_rebnorm
    sens_path = tmp_path / "sensfunc.fits"
    sens_columns = [
        fits.Column(name="WAVELENGTH", format="D", array=sens_wave),
        fits.Column(name="RESPONSE_ABS", format="E", array=np.full(11, 2.)),
        fits.Column(name="RESPONSE_UNC", format="E", array=np.full(11, 0.1)),
    ]
    fits.HDUList([
        fits.PrimaryHDU(header=sens_header),
        fits.BinTableHDU.from_columns(sens_columns, name="SENSFUNC"),
    ]).writeto(sens_path)
    return spectrum_path, sens_path


def test_fluxcal_matching_count_units_preserves_expected_response(tmp_path):
    science, response = _flux_cal_inputs(tmp_path)
    result = apply_fluxcal_to_spec1d(science, sensfunc_path=response)
    np.testing.assert_allclose(result["flux"], 5., rtol=0, atol=1e-8)
    np.testing.assert_allclose(result["var"], 1., rtol=0, atol=1e-8)


def test_fluxcal_rejects_different_standard_and_science_bunit(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, standard_unit="e-/Angstrom/s")
    with pytest.raises(ValueError, match="Flux calibration unit mismatch"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_rejects_mixed_legacy_and_countv1_response(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, science_rebnorm=None)
    with pytest.raises(ValueError, match="Flux calibration resampling mismatch"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_rejects_integrated_counts_on_mismatched_wavelength_bins(tmp_path):
    # Constant F_lambda=2. The standard produces 20 e-/s in 1 A bins,
    # the science frame produces 40 e-/s in 2 A bins.
    # Applying the 1 A response to the 2 A signal would falsely give 4.
    science, response = _flux_cal_inputs(
        tmp_path, science_step=2., standard_step=1.,
    )
    with pytest.raises(ValueError, match="wavelength-bin width mismatch"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_density_mode_does_not_require_matching_widths(tmp_path):
    science, response = _flux_cal_inputs(
        tmp_path, science_step=2., standard_step=1.,
        science_unit="e-/Angstrom/s", standard_unit="e-/Angstrom/s",
    )
    result = apply_fluxcal_to_spec1d(science, sensfunc_path=response)
    np.testing.assert_allclose(result["flux"][:6], 5., atol=1e-8)


def test_fluxcal_rejects_air_vacuum_mismatch(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, science_ref="vacuum", standard_ref="air")
    with pytest.raises(ValueError, match="wavelength medium mismatch"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_rejects_missing_explicit_science_medium(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, science_ref=None)
    with pytest.raises(ValueError, match="explicit standard/science wavelength medium"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_rejects_missing_standard_grid_width(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, standard_step_metadata=False)
    with pytest.raises(ValueError, match="missing verified standard wavelength-bin width"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)


def test_fluxcal_rejects_countv1_science_with_unqualified_legacy_response(tmp_path):
    science, response = _flux_cal_inputs(tmp_path, standard_rebnorm=None)
    with pytest.raises(ValueError, match="COUNT_V1 science requires"):
        apply_fluxcal_to_spec1d(science, sensfunc_path=response)
