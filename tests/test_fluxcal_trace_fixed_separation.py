"""Science regression for separate TRACE/FIXED flux calibration (Issue #45)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import scorpio_pipe.calib.fluxcal as fc
from scorpio_pipe.calib.common import read_spec1d
from scorpio_pipe.calib.fluxcal import apply_fluxcal_to_spec1d, write_fluxcal_products


def _fixture(tmp_path, *, include_fixed=True, telluric=True, preview=True):
    n = 12
    wave = 5000. + np.arange(n)
    trace_flux = np.linspace(12., 24., n)
    fixed_flux = np.linspace(45., 70., n)
    trace_var = np.linspace(3., 5., n)
    fixed_var = np.linspace(6., 11., n)
    trace_sky = np.linspace(.4, .8, n)
    fixed_sky = np.linspace(.7, 1.4, n)
    trace_mask = np.zeros(n, dtype=np.int32)
    fixed_mask = np.zeros(n, dtype=np.int32)
    trace_mask[1] = 2
    fixed_mask[2] = 8
    trace_npix = np.full(n, 3, dtype=np.int16)
    fixed_npix = np.full(n, 7, dtype=np.int16)
    hdr = fits.Header()
    hdr['BUNIT'] = 'e-/s'
    hdr['WAVEREF'] = 'air'
    hdr['WAVEUNIT'] = 'Angstrom'
    hdr['REBNORM'] = 'COUNT_V1'
    source = tmp_path / 'spec1d.fits'

    def mode(name, flux, var, sky, mask, npix):
        cols = [
            fits.Column(name='LAMBDA', format='D', array=wave),
            fits.Column(name=f'FLUX_{name}', format='E', array=flux),
            fits.Column(name=f'VAR_{name}', format='E', array=var),
            fits.Column(name=f'MASK_{name}', format='J', array=mask),
            fits.Column(name=f'SKY_{name}', format='E', array=sky),
            fits.Column(name=f'NPIX_{name}', format='I', array=npix),
        ]
        return fits.BinTableHDU.from_columns(cols, name=f'SPEC_{name}')

    hdus = [fits.PrimaryHDU(header=hdr),
            mode('TRACE', trace_flux, trace_var, trace_sky, trace_mask, trace_npix)]
    if include_fixed:
        hdus.append(mode('FIXED', fixed_flux, fixed_var, fixed_sky, fixed_mask, fixed_npix))
    if preview and include_fixed:
        for name, arr in (
            ('FLUX2', np.stack([trace_flux, fixed_flux])),
            ('VAR2', np.stack([trace_var, fixed_var])),
            ('MASK2', np.stack([trace_mask, fixed_mask])),
            ('SKY2', np.stack([trace_sky, fixed_sky])),
            ('NPIX2', np.stack([trace_npix, fixed_npix])),
        ):
            hdus.append(fits.ImageHDU(data=np.asarray(arr), name=name))
    fits.HDUList(hdus).writeto(source)

    response = 2. + .1 * np.arange(n)
    response_unc = np.full(n, .08)
    sens_hdr = fits.Header()
    sens_hdr['WAVEREF'] = 'air'
    sens_hdr['INWUNIT'] = 'Angstrom'
    sens_hdr['INSTEP'] = 1.
    sens_hdr['INBUNIT'] = 'e-/s'
    sens_hdr['INREBNM'] = 'COUNT_V1'
    sens = tmp_path / 'sens.fits'
    fits.HDUList([
        fits.PrimaryHDU(header=sens_hdr),
        fits.BinTableHDU.from_columns([
            fits.Column(name='WAVELENGTH', format='D', array=wave),
            fits.Column(name='RESPONSE_ABS', format='E', array=response),
            fits.Column(name='RESPONSE_UNC', format='E', array=response_unc),
        ], name='SENSFUNC'),
    ]).writeto(sens)
    trans = .80 + .01 * np.arange(n)
    tel = None
    if telluric:
        tel = tmp_path / 'telluric.fits'
        fits.HDUList([
            fits.PrimaryHDU(header=fits.Header({'WAVEREF': 'air'})),
            fits.BinTableHDU.from_columns([
                fits.Column(name='WAVELENGTH', format='D', array=wave),
                fits.Column(name='TRANSMISSION', format='E', array=trans),
                fits.Column(name='CONFIDENCE', format='E', array=np.full(n, .9)),
            ], name='TELLURIC'),
        ]).writeto(tel)

    stack = tmp_path / 'stack2d.fits'
    hh = fits.Header()
    hh['SCORP_L0'] = wave[0]
    hh['SCORP_DL'] = 1.
    hh['SCORP_NL'] = n
    hh['SCORP_LU'] = 'Angstrom'
    hh['WAVEREF'] = 'air'
    hh['REBNORM'] = 'COUNT_V1'
    hh['BUNIT'] = 'e-/s'
    fits.HDUList([
        fits.PrimaryHDU(header=hh),
        fits.ImageHDU(data=np.full((3, n), 20., dtype=np.float32), name='SCI'),
        fits.ImageHDU(data=np.full((3, n), 4., dtype=np.float32), name='VAR'),
        fits.ImageHDU(data=np.zeros((3, n), dtype=np.uint16), name='MASK'),
    ]).writeto(stack)
    expected = {
        'wave': wave, 'trace_flux': trace_flux, 'fixed_flux': fixed_flux,
        'trace_var': trace_var, 'fixed_var': fixed_var,
        'trace_sky': trace_sky, 'fixed_sky': fixed_sky,
        'trace_mask': trace_mask, 'fixed_mask': fixed_mask,
        'trace_npix': trace_npix, 'fixed_npix': fixed_npix,
        'factor': 1. / response / (trans if telluric else 1.),
        'stack_path': stack,
    }
    return source, sens, tel, expected


def _write(tmp_path, monkeypatch, src, sens, tel, expected, payload):
    monkeypatch.setattr(
        fc, 'PRODUCT_CATALOG',
        SimpleNamespace(best_path=lambda *args, **kwargs: expected['stack_path']),
    )
    return write_fluxcal_products(
        {'work_dir': str(tmp_path)}, tmp_path / 'out',
        spec_payload=payload, sensfunc_path=sens,
        telluric_path=tel, reference_context_id='ctx45',
        science_spec_path=src,
    )


@pytest.mark.parametrize('telluric', [True, False])
def test_calibrated_trace_and_fixed_are_distinct_in_tables_and_previews(tmp_path, monkeypatch, telluric):
    src, sens, tel, e = _fixture(tmp_path, telluric=telluric)
    result = apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)
    assert result['fixed'] is not None
    scale = e['factor']
    np.testing.assert_allclose(result['flux'], e['trace_flux'] * scale, rtol=3e-7)
    np.testing.assert_allclose(result['var'], e['trace_var'] * scale**2, rtol=3e-7)
    np.testing.assert_allclose(result['fixed']['flux'], e['fixed_flux'] * scale, rtol=3e-7)
    np.testing.assert_allclose(result['fixed']['var'], e['fixed_var'] * scale**2, rtol=3e-7)

    products = _write(tmp_path, monkeypatch, src, sens, tel, e, result)
    path = products['fluxcal_spec1d_fits']
    with fits.open(path) as hdul:
        ht = hdul['SPEC_TRACE'].data
        hf = hdul['SPEC_FIXED'].data
        assert hdul[0].header['EXTID'] == 'TRACE+FIXED'
        assert hdul[0].header['BUNIT'] == 'erg/s/cm2/Angstrom'
        for name, data, orig, var, sky, mask, npix in (
            ('TRACE', ht, e['trace_flux'], e['trace_var'], e['trace_sky'],
             e['trace_mask'], e['trace_npix']),
            ('FIXED', hf, e['fixed_flux'], e['fixed_var'], e['fixed_sky'],
             e['fixed_mask'], e['fixed_npix']),
        ):
            np.testing.assert_allclose(data[f'FLUX_{name}'], orig * scale, rtol=3e-7)
            np.testing.assert_allclose(data[f'VAR_{name}'], var * scale**2, rtol=3e-7)
            np.testing.assert_allclose(data[f'SKY_{name}'], sky * scale, rtol=3e-7)
            np.testing.assert_array_equal(data[f'MASK_{name}'], mask)
            np.testing.assert_array_equal(data[f'NPIX_{name}'], npix)
            assert hdul[f'SPEC_{name}'].header['FUNIT'] == 'erg/s/cm2/Angstrom'
            assert hdul[f'SPEC_{name}'].header['SKYUNIT'] == 'erg/s/cm2/Angstrom'
        assert not np.allclose(ht['FLUX_TRACE'], hf['FLUX_FIXED'])
        for ext, key in [('FLUX2', 'FLUX'), ('VAR2', 'VAR'), ('SKY2', 'SKY'),
                         ('MASK2', 'MASK'), ('NPIX2', 'NPIX')]:
            for i, name in enumerate(('TRACE', 'FIXED')):
                np.testing.assert_array_equal(
                    hdul[ext].data[i], hdul[f'SPEC_{name}'].data[f'{key}_{name}']
                )
    fixed_read = read_spec1d(path, mode='fixed')
    np.testing.assert_allclose(fixed_read['flux'], e['fixed_flux'] * scale, rtol=3e-7)
    with fits.open(products['fluxcal_2d_fits']) as hdul:
        np.testing.assert_allclose(hdul['SCI'].data[0], 20. * scale, rtol=3e-7)
        np.testing.assert_allclose(hdul['VAR'].data[0], 4. * scale**2, rtol=3e-7)


def test_trace_only_legacy_api_kept_but_writer_fails_closed(tmp_path, monkeypatch):
    src, sens, tel, e = _fixture(tmp_path, include_fixed=False)
    result = apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)
    assert result['fixed'] is None
    np.testing.assert_allclose(result['flux'], e['trace_flux'] * e['factor'], rtol=3e-7)
    with pytest.raises(ValueError, match='cannot publish SPEC_FIXED'):
        _write(tmp_path, monkeypatch, src, sens, tel, e, result)


def test_fixed_grid_mismatch_rejected_before_calibration(tmp_path):
    src, sens, tel, _ = _fixture(tmp_path)
    with fits.open(src, mode='update') as hdul:
        hdul['SPEC_FIXED'].data['LAMBDA'][3] += .1
    with pytest.raises(ValueError, match='TRACE/FIXED wavelength grid mismatch'):
        apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)


def test_corrupt_fixed_shape_or_preview_mask_fails_closed(tmp_path, monkeypatch):
    src, sens, tel, e = _fixture(tmp_path)
    result = apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)
    result['fixed']['var'] = np.ones(5)
    with pytest.raises(ValueError, match='Calibrated FIXED var length mismatch'):
        _write(tmp_path, monkeypatch, src, sens, tel, e, result)

    result = apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)
    with fits.open(src, mode='update') as hdul:
        hdul['MASK2'].data[1, 0] = 16
    with pytest.raises(ValueError, match='MASK2 preview disagrees'):
        _write(tmp_path, monkeypatch, src, sens, tel, e, result)


def test_existing_count_v1_fluxcal_rejects_incompatible_units_for_both_modes(tmp_path):
    src, sens, tel, _ = _fixture(tmp_path)
    with fits.open(sens, mode='update') as hdul:
        hdul[0].header['INBUNIT'] = 'e-/Angstrom/s'
    with pytest.raises(ValueError, match='Flux calibration unit mismatch'):
        apply_fluxcal_to_spec1d(src, sensfunc_path=sens, telluric_path=tel)
