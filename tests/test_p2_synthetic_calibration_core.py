from __future__ import annotations

import numpy as np
import pytest

from scorpio_pipe.airvac import air_to_vac, refractive_index_air_morton2000, vac_to_air
from scorpio_pipe.calib.common import (
    air_to_vacuum,
    convert_wavelength_convention,
    read_spec1d,
    vacuum_to_air,
)
from scorpio_pipe.calib.fluxcal import _status_from_safe_for
from scorpio_pipe.calib.telluric import _estimate_window_transmission


def test_air_vacuum_conversion_is_close_to_roundtrip() -> None:
    w = np.linspace(3800.0, 9000.0, 128)
    vac = air_to_vacuum(w)
    back = vacuum_to_air(vac)
    np.testing.assert_allclose(back, w, rtol=0, atol=1e-9)


# Independent 50-digit Decimal evaluation of the published VALD/Morton formula:
# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
# n(v) = 1 + .0000834254 + .02406147/(130-s^2)
#                         + .00015998/(38.9-s^2), s=10000/v (v in Angstrom).
# Vacuum->air values use v/n(v); air->vacuum values solve v=a*n(v) with
# Decimal arithmetic. These literals never call a production conversion helper.
_REFERENCE_WAVE = np.array([3800., 4000., 5000., 6000., 7000., 9000.])
_REFERENCE_VACUUM = np.array([
    3801.078913806741, 4001.131021148522, 5001.394848639170,
    6001.661887416033, 7001.930628786668, 9002.470810882271,
])
_REFERENCE_AIR = np.array([
    3798.921366387797, 3998.869274253268, 4998.605522013399,
    5998.338557925336, 6998.069891046067, 8997.529857722393,
])


@pytest.mark.parametrize('convert', [air_to_vac, air_to_vacuum])
def test_air_to_vacuum_absolute_reference(convert) -> None:
    # Detects the original factor-100 refractivity error, even if roundtrip passes.
    np.testing.assert_allclose(convert(_REFERENCE_WAVE), _REFERENCE_VACUUM, rtol=0, atol=0.001)


@pytest.mark.parametrize('convert', [vac_to_air, vacuum_to_air])
def test_vacuum_to_air_absolute_reference(convert) -> None:
    np.testing.assert_allclose(convert(_REFERENCE_WAVE), _REFERENCE_AIR, rtol=0, atol=0.001)


def test_refractive_index_absolute_reference() -> None:
    assert refractive_index_air_morton2000(5000.) == pytest.approx(1.0002789734017738, rel=0, abs=1e-14)


@pytest.mark.parametrize('convert', [air_to_vac, air_to_vacuum, vac_to_air, vacuum_to_air])
def test_conversion_scalar_array_shape_and_monotonicity(convert) -> None:
    wave = _REFERENCE_WAVE.reshape(2, 3)
    before = wave.copy()
    converted = convert(wave)
    assert converted.shape == wave.shape
    assert np.all(np.diff(converted.ravel()) > 0)
    np.testing.assert_allclose(converted.ravel(), [convert(float(w)) for w in wave.ravel()], rtol=0, atol=1e-10)
    assert np.ndim(convert(5000.)) == 0
    np.testing.assert_array_equal(wave, before)
    assert convert(np.array([])).shape == (0,)


def test_public_entrypoints_agree_and_roundtrip_both_directions() -> None:
    wave = np.linspace(2000., 100000., 300)
    np.testing.assert_array_equal(air_to_vac(wave), air_to_vacuum(wave))
    np.testing.assert_array_equal(vac_to_air(wave), vacuum_to_air(wave))
    np.testing.assert_allclose(vac_to_air(air_to_vac(wave)), wave, rtol=0, atol=1e-9)
    # Includes air wavelengths just below 2000 A, obtained from valid 2000 A vacuum.
    np.testing.assert_allclose(air_to_vac(vac_to_air(wave)), wave, rtol=0, atol=1e-9)


@pytest.mark.parametrize('from_ref,to_ref', [('air', 'air'), ('vacuum', 'vacuum'), ('vac', 'vacuum'), (' AIR ', 'air')])
def test_wavelength_convention_identity(from_ref, to_ref) -> None:
    wave = _REFERENCE_WAVE.copy()
    np.testing.assert_array_equal(convert_wavelength_convention(wave, from_ref=from_ref, to_ref=to_ref), wave)


@pytest.mark.parametrize('medium', ['air', 'vacuum'])
@pytest.mark.parametrize('invalid', [0., -5000., np.nan, np.inf, 5000.+1j])
def test_identity_rejects_invalid_values_without_silent_casting(medium, invalid) -> None:
    with pytest.raises(ValueError):
        convert_wavelength_convention(np.array([invalid]), from_ref=medium, to_ref=medium)


def test_explicit_wavelength_convention_conversion() -> None:
    np.testing.assert_allclose(
        convert_wavelength_convention(_REFERENCE_WAVE, from_ref='air', to_ref='vac'),
        _REFERENCE_VACUUM, rtol=0, atol=0.001,
    )
    np.testing.assert_allclose(
        convert_wavelength_convention(_REFERENCE_WAVE, from_ref='vacuum', to_ref='air'),
        _REFERENCE_AIR, rtol=0, atol=0.001,
    )


@pytest.mark.parametrize('convert', [air_to_vac, air_to_vacuum, vac_to_air, vacuum_to_air, refractive_index_air_morton2000])
@pytest.mark.parametrize('invalid', [0., -5000., 1999., np.nan, np.inf, -np.inf, 5000.+1j])
def test_conversion_rejects_invalid_physical_inputs(convert, invalid) -> None:
    with pytest.raises(ValueError):
        convert(invalid)
    with pytest.raises(ValueError):
        convert(np.array([5000., invalid]))


@pytest.mark.parametrize('from_ref,to_ref', [('unknown', 'air'), ('air', 'unknown'), ('unknown', 'unknown'), ('', 'air'), (None, 'vacuum'), ('air', None)])
def test_conversion_rejects_unknown_conventions(from_ref, to_ref) -> None:
    with pytest.raises(ValueError):
        convert_wavelength_convention(5000., from_ref=from_ref, to_ref=to_ref)


@pytest.mark.parametrize('kwargs', [{'max_iter': 0}, {'max_iter': -1}, {'max_iter': 1.5}, {'rtol': 0.}, {'rtol': -1.}, {'rtol': np.nan}, {'rtol': np.inf}])
def test_air_to_vacuum_rejects_invalid_solver_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        air_to_vac(5000., **kwargs)


def test_air_to_vacuum_reports_nonconvergence() -> None:
    with pytest.raises(RuntimeError, match='converg'):
        air_to_vac(5000., max_iter=1)


@pytest.mark.parametrize('medium', ['air', 'vacuum'])
def test_reference_conversion_preserves_fits_spectrum_and_metadata(tmp_path, medium) -> None:
    from astropy.io import fits

    path = tmp_path / 'standard_1d.fits'
    primary = fits.PrimaryHDU()
    primary.header['WAVEREF'] = medium
    columns = [fits.Column(name='LAMBDA', format='D', array=_REFERENCE_WAVE)]
    for name in ['FLUX_TRACE', 'VAR_TRACE', 'MASK_TRACE']:
        columns.append(fits.Column(name=name, format='D', array=np.ones(6)))
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns, name='SPEC_TRACE')]).writeto(path)
    before = path.read_bytes()
    spec = read_spec1d(path)
    # This is the sensfunc handoff: explicitly convert reference wavelengths to
    # the spectrum's declared medium, never the observed spectrum or its header.
    converted = convert_wavelength_convention(_REFERENCE_WAVE, from_ref='air', to_ref=spec['waveref'])
    expected = _REFERENCE_WAVE if medium == 'air' else _REFERENCE_VACUUM
    np.testing.assert_allclose(converted, expected, rtol=0, atol=0.001)
    np.testing.assert_array_equal(spec['wave'], _REFERENCE_WAVE)
    assert spec['header']['WAVEREF'] == medium
    assert path.read_bytes() == before


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
