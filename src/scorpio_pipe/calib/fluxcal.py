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
            'waveref_declared': bool(str(hdr.get('WAVEREF') or '').strip()),
            'input_bunit': str(hdr.get('INBUNIT') or '').strip(),
            'input_rebnorm': str(hdr.get('INREBNM') or '').strip(),
            'input_waveunit': str(hdr.get('INWUNIT') or '').strip(),
            'input_wave_step': hdr.get('INSTEP'),
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
    # TRACE/FIXED are distinct physical extractions. Historical trace-only
    # inputs remain readable for the numerical calibration API, but may not
    # be published as a two-mode calibrated FITS.
    from astropy.io import fits
    with fits.open(science_spec_path, memmap=False) as hdul:
        has_fixed = 'SPEC_FIXED' in hdul
    fixed = read_spec1d(science_spec_path, mode='fixed') if has_fixed else None
    wave = np.asarray(spec['wave'], dtype=float)
    if fixed is not None:
        fixed_wave = np.asarray(fixed['wave'], dtype=float)
        if (fixed_wave.shape != wave.shape
                or not np.allclose(fixed_wave, wave, rtol=0.0, atol=1e-9)):
            raise ValueError("Flux calibration TRACE/FIXED wavelength grid mismatch")
        for key in ('flux', 'var', 'mask', 'sky', 'npix'):
            if np.asarray(fixed[key]).shape != wave.shape:
                raise ValueError(f"Flux calibration FIXED {key} shape mismatch")
    sens = _load_sensfunc(sensfunc_path)
    # Sensitivity functions include the physical input count/rate units of
    # the STANDARD. A response fitted in e-/Angstrom/s cannot calibrate
    # science e-/bin spectra without an explicit unit conversion.
    in_unit = str(sens.get('input_bunit') or '').strip()
    sci_unit = str(spec.get('bunit') or '').strip()
    if in_unit and in_unit.casefold().replace(' ', '') != sci_unit.casefold().replace(' ', ''):
        raise ValueError(
            f"Flux calibration unit mismatch: standard={in_unit!r}, science={sci_unit!r}; "
            "reprocess with matching count/rate and bin/density conventions."
        )
    std_norm = str(sens.get('input_rebnorm') or '').strip()
    sci_norm = str(spec['header'].get('REBNORM') or '').strip()
    if std_norm and std_norm != sci_norm:
        raise ValueError(
            f"Flux calibration resampling mismatch: standard={std_norm!r}, "
            f"science={sci_norm or 'LEGACY_UNQUALIFIED'!r}"
        )
    if sci_norm == 'COUNT_V1' and std_norm != 'COUNT_V1':
        raise ValueError(
            'Flux calibration resampling mismatch: COUNT_V1 science requires '
            'a sensitivity function with matching COUNT_V1 provenance'
        )
    if sci_norm == 'COUNT_V1':
        # Legacy read_spec1d() may assume air and ADU for missing metadata;
        # such implicit defaults are forbidden for physical COUNT_V1 fluxing.
        sci_hdr = spec['header']
        sci_ref = str(sci_hdr.get('WAVEREF') or '').strip().lower()
        std_ref = str(sens.get('waveref') or '').strip().lower()
        if (sci_ref not in {'air', 'vacuum'}
                or std_ref not in {'air', 'vacuum'}
                or not bool(sens.get('waveref_declared'))):
            raise ValueError('Flux calibration requires explicit standard/science wavelength medium')
        if sci_ref != std_ref:
            raise ValueError(
                f"Flux calibration wavelength medium mismatch: standard={std_ref!r}, science={sci_ref!r}"
            )
        sci_waveunit = str(sci_hdr.get('WAVEUNIT') or sci_hdr.get('CUNIT1') or '').strip()
        std_waveunit = str(sens.get('input_waveunit') or '').strip()
        if not sci_waveunit or not std_waveunit:
            raise ValueError('Flux calibration requires explicit standard/science wavelength units')
        if sci_waveunit.casefold() != std_waveunit.casefold():
            raise ValueError(
                f"Flux calibration wavelength unit mismatch: standard={std_waveunit!r}, science={sci_waveunit!r}"
            )
        dw = np.diff(np.asarray(wave, dtype=float))
        if (dw.size == 0 or not np.isfinite(dw).all() or (dw <= 0).any()
                or not np.allclose(dw, np.median(dw), rtol=1e-6, atol=1e-9)):
            raise ValueError('COUNT_V1 science requires strictly increasing, uniform wavelength bins')
        std_dw = sens.get('input_wave_step')
        if std_dw is None or not np.isfinite(float(std_dw)) or float(std_dw) <= 0:
            raise ValueError('COUNT_V1 sensitivity is missing verified standard wavelength-bin width')
        std_dw = float(std_dw)
        std_steps = np.diff(sens['wave'])
        if (not np.isfinite(sens['wave']).all() or std_steps.size == 0
                or not np.isfinite(std_steps).all()
                or (std_steps <= 0).any()
                or not np.allclose(std_steps, std_dw, rtol=1e-6, atol=1e-9)):
            raise ValueError('COUNT_V1 sensitivity has inconsistent wavelength grid/INSTEP')
        # Bin-integrated electron response is proportional to bin width.
        # For density outputs e-/wavelength, this extra factor is absent.
        from scorpio_pipe.contracts.units import parse_bunit
        science_unit = parse_bunit(sci_unit)
        is_density = any(
            token in sci_unit.casefold().replace(' ', '')
            for token in ('/angstrom', '/nm', '/nanometer', '/pix', '/pixel')
        )
        if science_unit.base not in {'electron', 'adu'}:
            raise ValueError(f"COUNT_V1 sensitivity unsupported SCI unit: {sci_unit!r}")
        if not is_density and not np.isclose(
            float(np.median(dw)), std_dw, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"Flux calibration wavelength-bin width mismatch: "
                f"standard={std_dw:.9g}, science={float(np.median(dw)):.9g} "
                f"({sci_waveunit}); use a common count/bin wavelength grid "
                "or explicitly transform the sensitivity function."
            )
    resp = np.interp(wave, sens['wave'], sens['response_abs'], left=np.nan, right=np.nan)
    resp_unc = np.interp(wave, sens['wave'], sens['response_unc'], left=np.nanmedian(sens['response_unc']), right=np.nanmedian(sens['response_unc']))
    flux = spec['flux'] / np.clip(resp, 1e-30, np.inf)
    var = spec['var'] / np.clip(resp, 1e-30, np.inf) ** 2
    tell_applied = False
    tell_conf = np.zeros_like(wave)
    transmission = np.ones_like(wave)
    if telluric_path is not None and Path(telluric_path).exists():
        tel = _load_telluric(telluric_path)
        if sci_norm == 'COUNT_V1':
            tel_ref = str(tel.get('waveref') or '').strip().lower()
            if tel_ref != sci_ref:
                raise ValueError(
                    f"Telluric wavelength medium mismatch: model={tel_ref!r}, science={sci_ref!r}"
                )
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
    # The response/transmission are shared deterministic calibrations of
    # two different extracted spectra. Preserve their independent content.
    # Response-fit covariance is separate from conditional detector VAR.
    transfer = 1.0 / np.clip(resp, 1e-30, np.inf)
    if tell_applied:
        transfer = transfer / np.clip(transmission, 0.25, np.inf)
    fixed_payload = None
    if fixed is not None:
        fixed_payload = {
            'flux': np.asarray(fixed['flux'], dtype=float) * transfer,
            'var': np.asarray(fixed['var'], dtype=float) * transfer ** 2,
            'mask': np.asarray(fixed['mask']).copy(),
            'sky': np.asarray(fixed['sky'], dtype=float) * transfer,
            'npix': np.asarray(fixed['npix']).copy(),
        }
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
        'sky': np.asarray(spec['sky'], dtype=float) * transfer,
        'npix': np.asarray(spec['npix']),
        'fixed': fixed_payload,
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
        # After physical flux calibration, data no longer have detector
        # electron/ADU units. Preserve the historical gain/noise cards but
        # remove the detector-only SCI unit-model label.
        if 'SCORPUM' in hdr0:
            del hdr0['SCORPUM']
        hdr0['WAVEREF'] = str(hdr0.get('WAVEREF') or 'air').strip().lower()
        stamp_calibration_semantics(
            hdr0,
            status=spec_payload['status'],
            flux_unit=spec_payload['flux_unit'],
            reference_context_id=str(reference_context_id),
            sensfunc_ref=Path(sensfunc_path).name,
            telluric_ref=(Path(telluric_path).name if telluric_path is not None and spec_payload['telluric_applied'] else None),
        )
        # Require independent, wavelength-aligned TRACE and FIXED arrays.
        # Never synthesize FIXED by copying TRACE.
        fixed = spec_payload.get('fixed')
        if not isinstance(fixed, dict) or 'SPEC_FIXED' not in hdul:
            raise ValueError(
                "Flux calibration cannot publish SPEC_FIXED without its own "
                "input spectrum; re-extract TRACE/FIXED from the 2D science data"
            )
        nlam = len(np.asarray(spec_payload['wave']))
        t = hdul['SPEC_TRACE'].data.copy()
        f = hdul['SPEC_FIXED'].data.copy()
        if (len(t) != nlam or len(f) != nlam
                or not np.allclose(np.asarray(t['LAMBDA'], dtype=float),
                                   np.asarray(f['LAMBDA'], dtype=float),
                                   rtol=0.0, atol=1e-9)):
            raise ValueError("Flux calibration output TRACE/FIXED grid mismatch")
        trace_out = spec_payload
        fixed_out = fixed
        for mode, table, output in (
            ('TRACE', t, trace_out), ('FIXED', f, fixed_out)
        ):
            for key, column in (
                ('flux', f'FLUX_{mode}'),
                ('var', f'VAR_{mode}'),
            ):
                if column not in table.names:
                    raise ValueError(f"Missing {column} column in extraction table")
                values = np.asarray(output[key], dtype=float)
                if values.shape != (nlam,):
                    raise ValueError(f"Calibrated {mode} {key} length mismatch")
                table[column] = values.astype(np.float32)
            # SKY is a per-pixel local residual, not the total extracted
            # aperture flux. It carries the same per-wavelength physical
            # unit conversion but retains its original spatial definition.
            sky_col = f'SKY_{mode}'
            if sky_col in table.names:
                table[sky_col] = np.asarray(output['sky'], dtype=np.float32)
            for key, column in (('mask', f'MASK_{mode}'),
                                ('npix', f'NPIX_{mode}')):
                if column in table.names and not np.array_equal(
                    table[column], np.asarray(output[key])
                ):
                    raise ValueError(f"Flux calibration must preserve {column}")
        ht = fits.BinTableHDU(data=t, header=hdul['SPEC_TRACE'].header.copy(), name='SPEC_TRACE')
        hf = fits.BinTableHDU(data=f, header=hdul['SPEC_FIXED'].header.copy(), name='SPEC_FIXED')
        for hh in (ht.header, hf.header):
            hh['FUNIT'] = spec_payload['flux_unit']
            hh['SKYUNIT'] = (spec_payload['flux_unit'], "Sky residual unit per sampled spatial pixel")
        hdr0['EXTID'] = ('TRACE+FIXED', "Both extraction modes separately calibrated")
        new_hdus = [fits.PrimaryHDU(header=hdr0), ht, hf]
        for ext in ('FLUX2', 'VAR2', 'MASK2', 'NPIX2', 'SKY2'):
            if ext not in hdul:
                continue
            h = hdul[ext].copy()
            arr = np.asarray(h.data)
            if arr.shape != (2, nlam):
                raise ValueError(f"Flux calibration preview {ext} shape mismatch")
            if ext in ('FLUX2', 'VAR2', 'SKY2'):
                key = {'FLUX2': 'flux', 'VAR2': 'var', 'SKY2': 'sky'}[ext]
                h.data = np.stack([
                    np.asarray(trace_out[key], dtype=np.float32),
                    np.asarray(fixed_out[key], dtype=np.float32),
                ])
                h.header['BUNIT'] = (
                    f"({spec_payload['flux_unit']})^2" if ext == 'VAR2'
                    else spec_payload['flux_unit']
                )
            elif ext == 'MASK2':
                if not np.array_equal(arr[0], np.asarray(t['MASK_TRACE'])) or not np.array_equal(
                    arr[1], np.asarray(f['MASK_FIXED'])
                ):
                    raise ValueError("MASK2 preview disagrees with extraction tables")
            elif ext == 'NPIX2':
                if not np.array_equal(arr[0], np.asarray(t['NPIX_TRACE'])) or not np.array_equal(
                    arr[1], np.asarray(f['NPIX_FIXED'])
                ):
                    raise ValueError("NPIX2 preview disagrees with extraction tables")
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
        if 'SCORPUM' in hdr0:
            del hdr0['SCORPUM']
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
