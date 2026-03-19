from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.calib.common import ensure_refctx, plot_lines_png, read_spec1d, save_json
from scorpio_pipe.contracts.calibration import CalibrationProductKind, CalibrationStatus, coerce_calibration_status, stamp_calibration_semantics
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG


def _load_sensfunc(path: str | Path) -> dict[str, np.ndarray]:
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        tab = hdul[1].data
        hdr = hdul[0].header
        return {
            'wave': np.asarray(tab['WAVELENGTH'], dtype=float),
            'response_abs': np.asarray(tab['RESPONSE_ABS'], dtype=float),
            'response_unc': np.asarray(tab['RESPONSE_UNC'], dtype=float),
            'waveref': str(hdr.get('WAVEREF') or 'air').strip().lower(),
        }


def _load_telluric(path: str | Path) -> dict[str, np.ndarray]:
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        tab = hdul[1].data
        hdr = hdul[0].header
        return {
            'wave': np.asarray(tab['WAVELENGTH'], dtype=float),
            'trans': np.asarray(tab['TRANSMISSION'], dtype=float),
            'conf': np.asarray(tab['CONFIDENCE'], dtype=float),
            'waveref': str(hdr.get('WAVEREF') or 'air').strip().lower(),
        }


def _status_from_safe_for(safe_for: str | None, *, telluric_applied: bool) -> str:
    s = str(safe_for or '').strip().lower()
    if s == 'absolute_candidate':
        return CalibrationStatus.ABSOLUTE_TRUSTED.value
    if s in {'approx_absolute', 'telluric_or_relative_only'}:
        return CalibrationStatus.ABSOLUTE_APPROX.value if 'absolute' in s else CalibrationStatus.RELATIVE_TELLURIC.value
    if telluric_applied:
        return CalibrationStatus.RELATIVE_TELLURIC.value
    return CalibrationStatus.RELATIVE.value


def apply_fluxcal_to_spec1d(science_spec_path: str | Path, *, sensfunc_path: str | Path, telluric_path: str | Path | None = None, standard_qc: dict[str, Any] | None = None, sensfunc_qc: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = read_spec1d(science_spec_path, mode='trace')
    wave = spec['wave']
    sens = _load_sensfunc(sensfunc_path)
    resp = np.interp(wave, sens['wave'], sens['response_abs'], left=np.nan, right=np.nan)
    resp_unc = np.interp(wave, sens['wave'], sens['response_unc'], left=np.nanmedian(sens['response_unc']), right=np.nanmedian(sens['response_unc']))
    flux = spec['flux'] / np.clip(resp, 1e-30, np.inf)
    var = spec['var'] / np.clip(resp, 1e-30, np.inf) ** 2
    tell_applied = False
    tell_conf = np.zeros_like(wave)
    transmission = np.ones_like(wave)
    if telluric_path is not None and Path(telluric_path).exists():
        tel = _load_telluric(telluric_path)
        transmission = np.interp(wave, tel['wave'], tel['trans'], left=1.0, right=1.0)
        tell_conf = np.interp(wave, tel['wave'], tel['conf'], left=0.0, right=0.0)
        flux = flux / np.clip(transmission, 0.25, np.inf)
        var = var / np.clip(transmission, 0.25, np.inf) ** 2
        tell_applied = True
    safe_for = None
    if isinstance(standard_qc, dict):
        safe_for = standard_qc.get('safe_for')
    status = _status_from_safe_for(safe_for, telluric_applied=tell_applied)
    if status == CalibrationStatus.ABSOLUTE_TRUSTED.value and isinstance(sensfunc_qc, dict):
        if not bool(sensfunc_qc.get('extinction_applied')) or str(sensfunc_qc.get('airmass_provenance') or '').startswith('fallback:') or str(sensfunc_qc.get('airmass_provenance') or '') == 'missing':
            status = CalibrationStatus.ABSOLUTE_APPROX.value
    calsys_rel = np.clip(resp_unc / np.clip(np.abs(resp), 1e-30, np.inf), 0.0, 2.0)
    calsys_tell = np.clip((1.0 - tell_conf) * (1.0 - transmission), 0.0, 1.0)
    if status == CalibrationStatus.ABSOLUTE_TRUSTED.value:
        abs_floor = 0.05
    elif status == CalibrationStatus.ABSOLUTE_APPROX.value:
        abs_floor = 0.12
    else:
        abs_floor = 0.25
    calsys_abs = np.full_like(wave, abs_floor, dtype=float)
    calsys_total = np.sqrt(calsys_rel * calsys_rel + calsys_tell * calsys_tell + calsys_abs * calsys_abs)
    return {
        'wave': wave,
        'flux': flux,
        'var': var,
        'mask': np.asarray(spec['mask']),
        'sky': np.asarray(spec['sky']),
        'npix': np.asarray(spec['npix']),
        'header': spec['header'],
        'status': status,
        'transmission': transmission,
        'tell_conf': tell_conf,
        'calsys_rel': calsys_rel,
        'calsys_tell': calsys_tell,
        'calsys_abs': calsys_abs,
        'calsys_total': calsys_total,
        'flux_unit': 'erg/s/cm2/Angstrom',
        'telluric_applied': tell_applied,
    }


def write_fluxcal_products(cfg: dict[str, Any], out_dir: str | Path, *, spec_payload: dict[str, Any], sensfunc_path: str | Path, telluric_path: str | Path | None, reference_context_id: str, science_spec_path: str | Path, standard_qc: dict[str, Any] | None = None) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_spec = Path(science_spec_path)
    out_spec = out_dir / 'fluxcal_spec1d.fits'
    from astropy.io import fits
    with fits.open(in_spec, memmap=False) as hdul:
        hdr0 = hdul[0].header.copy()
        hdr0['BUNIT'] = spec_payload['flux_unit']
        hdr0['WAVEREF'] = str(hdr0.get('WAVEREF') or 'air').strip().lower()
        stamp_calibration_semantics(
            hdr0,
            status=spec_payload['status'],
            flux_unit=spec_payload['flux_unit'],
            reference_context_id=str(reference_context_id),
            sensfunc_ref=Path(sensfunc_path).name,
            telluric_ref=(Path(telluric_path).name if telluric_path is not None and spec_payload['telluric_applied'] else None),
        )
        # tables
        t = hdul['SPEC_TRACE'].data.copy()
        t['FLUX_TRACE'] = np.asarray(spec_payload['flux'], dtype=np.float32)
        t['VAR_TRACE'] = np.asarray(spec_payload['var'], dtype=np.float32)
        f = hdul['SPEC_FIXED'].data.copy()
        # For first-stable version keep FIXED identical to TRACE after calibration.
        if 'FLUX_FIXED' in f.names:
            f['FLUX_FIXED'] = np.asarray(spec_payload['flux'], dtype=np.float32)
        if 'VAR_FIXED' in f.names:
            f['VAR_FIXED'] = np.asarray(spec_payload['var'], dtype=np.float32)
        ht = fits.BinTableHDU(data=t, header=hdul['SPEC_TRACE'].header.copy(), name='SPEC_TRACE')
        hf = fits.BinTableHDU(data=f, header=hdul['SPEC_FIXED'].header.copy(), name='SPEC_FIXED')
        for hh in (ht.header, hf.header):
            hh['FUNIT'] = spec_payload['flux_unit']
        new_hdus = [fits.PrimaryHDU(header=hdr0), ht, hf]
        for ext in ('FLUX2','VAR2','MASK2','NPIX2','SKY2'):
            if ext in hdul:
                h = hdul[ext].copy()
                if ext == 'FLUX2':
                    arr = np.asarray(h.data, dtype=float)
                    arr[0, :] = spec_payload['flux']
                    arr[1, :] = spec_payload['flux']
                    h.data = arr.astype(np.float32)
                elif ext == 'VAR2':
                    arr = np.asarray(h.data, dtype=float)
                    arr[0, :] = spec_payload['var']
                    arr[1, :] = spec_payload['var']
                    h.data = arr.astype(np.float32)
                new_hdus.append(h)
        fits.HDUList(new_hdus).writeto(out_spec, overwrite=True)

    # 2D calibrated output: relative/telluric layers applied, semantic status communicates trust.
    stack_path = PRODUCT_CATALOG.best_path(cfg, 'stacked2d_fits', run_root=Path(cfg.get('work_dir') or '.'))
    out_2d = out_dir / 'fluxcal_2d.fits'
    from astropy.io import fits
    with fits.open(stack_path, memmap=False) as hdul:
        hdr0 = hdul[0].header.copy()
        sci = np.asarray(hdul['SCI'].data, dtype=float)
        var = np.asarray(hdul['VAR'].data, dtype=float)
        mask = np.asarray(hdul['MASK'].data)
        cov = np.asarray(hdul['COV'].data) if 'COV' in hdul else None
        from scorpio_pipe.io.mef import try_read_grid, write_sci_var_mask
        grid = try_read_grid(hdr0) or try_read_grid(hdul['SCI'].header)
        if grid is None:
            raise ValueError('stacked2d missing wavelength grid')
        wave = grid.lambda0 + grid.dlambda * np.arange(grid.nlam, dtype=float)
        sens = _load_sensfunc(sensfunc_path)
        resp = np.interp(wave, sens['wave'], sens['response_abs'], left=np.nan, right=np.nan)
        sci = sci / np.clip(resp[None, :], 1e-30, np.inf)
        var = var / np.clip(resp[None, :], 1e-30, np.inf) ** 2
        if telluric_path is not None and spec_payload['telluric_applied']:
            tel = _load_telluric(telluric_path)
            tr = np.interp(wave, tel['wave'], tel['trans'], left=1.0, right=1.0)
            sci = sci / np.clip(tr[None, :], 0.25, np.inf)
            var = var / np.clip(tr[None, :], 0.25, np.inf) ** 2
        hdr0['BUNIT'] = spec_payload['flux_unit']
        status_2d = coerce_calibration_status(spec_payload['status'], product_kind=CalibrationProductKind.MEF2D)
        stamp_calibration_semantics(
            hdr0,
            status=status_2d,
            flux_unit=spec_payload['flux_unit'],
            reference_context_id=str(reference_context_id),
            sensfunc_ref=Path(sensfunc_path).name,
            telluric_ref=(Path(telluric_path).name if telluric_path is not None and spec_payload['telluric_applied'] else None),
        )
        write_sci_var_mask(out_2d, sci, var=var, mask=mask, header=hdr0, grid=grid, cov=cov, validate=True)

    # calibration systematics (1D companion)
    calsys = out_dir / 'calibration_systematics.fits'
    from astropy.io import fits
    ph = fits.Header()
    ph['REFCTID'] = str(reference_context_id)
    ph['CALSTAT'] = str(spec_payload['status'])
    ph['WAVEREF'] = str(spec_payload['header'].get('WAVEREF') or 'air').strip().lower()
    ph['SYSUNIT'] = 'fraction'
    ph['SENSFUNC'] = Path(sensfunc_path).name
    if telluric_path is not None and spec_payload['telluric_applied']:
        ph['TELLURIC'] = Path(telluric_path).name
    from astropy.io import fits
    fits.HDUList([
        fits.PrimaryHDU(header=ph),
        fits.ImageHDU(data=np.asarray(spec_payload['wave'], dtype=np.float32), name='WAVE'),
        fits.ImageHDU(data=np.asarray(spec_payload['calsys_rel'], dtype=np.float32), name='CALSYS_REL'),
        fits.ImageHDU(data=np.asarray(spec_payload['calsys_tell'], dtype=np.float32), name='CALSYS_TELL'),
        fits.ImageHDU(data=np.asarray(spec_payload['calsys_abs'], dtype=np.float32), name='CALSYS_ABS'),
        fits.ImageHDU(data=np.asarray(spec_payload['calsys_total'], dtype=np.float32), name='CALSYS_TOTAL'),
    ]).writeto(calsys, overwrite=True)

    preview = plot_lines_png(out_dir / 'fluxcal_preview.png', [
        (spec_payload['wave'], spec_payload['flux'], 'calibrated flux'),
        (spec_payload['wave'], spec_payload['transmission'], 'telluric transmission'),
    ], title='Fluxcal apply')
    qc = save_json(out_dir / 'fluxcal_qc.json', {
        'stage': 'fluxcal_apply',
        'status': spec_payload['status'],
        'status_2d': coerce_calibration_status(spec_payload['status'], product_kind=CalibrationProductKind.MEF2D),
        'reference_context_id': str(reference_context_id),
        'safe_for': (standard_qc or {}).get('safe_for') if isinstance(standard_qc, dict) else None,
        'telluric_applied': bool(spec_payload['telluric_applied']),
        'systematics': {
            'calsys_rel_median': float(np.nanmedian(spec_payload['calsys_rel'])),
            'calsys_tell_median': float(np.nanmedian(spec_payload['calsys_tell'])),
            'calsys_abs_median': float(np.nanmedian(spec_payload['calsys_abs'])),
            'calsys_total_median': float(np.nanmedian(spec_payload['calsys_total'])),
        },
    })
    plan = save_json(out_dir / 'fluxcal_apply_plan.json', {'stage': 'fluxcal_apply', 'status': spec_payload['status']})
    return {
        'fluxcal_spec1d_fits': str(out_spec),
        'fluxcal_2d_fits': str(out_2d),
        'fluxcal_preview_png': str(preview),
        'fluxcal_qc_json': str(qc),
        'calibration_systematics_fits': str(calsys),
        'fluxcal_apply_plan_json': str(plan),
    }
