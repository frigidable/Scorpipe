from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.calib.common import (
    convert_wavelength_convention,
    ensure_refctx,
    find_standard_record,
    load_library_yaml,
    plot_lines_png,
    read_interval_csv,
    read_small_csv,
    read_spec1d,
    reference_entry,
    reference_metadata,
    reference_path,
    safe_json,
    save_json,
    write_bintable_fits,
)
from scorpio_pipe.instruments import parse_frame_meta

BALMER_WINDOWS_A = [
    (3830.0, 3915.0),
    (4090.0, 4125.0),
    (4320.0, 4370.0),
    (4840.0, 4890.0),
    (6540.0, 6595.0),
]


def _robust_spline_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    good = np.isfinite(xx) & np.isfinite(yy)
    if w is not None:
        ww = np.asarray(w, dtype=float)
        good &= np.isfinite(ww) & (ww > 0)
    else:
        ww = None
    xx = xx[good]
    yy = yy[good]
    if ww is not None:
        ww = ww[good]
    if xx.size < 8:
        return np.full_like(x, np.nanmedian(yy) if yy.size else 1.0, dtype=float), {"method": "constant"}
    order = np.argsort(xx)
    xx = xx[order]; yy = yy[order]; ww = ww[order] if ww is not None else None
    try:
        from scipy.interpolate import UnivariateSpline  # type: ignore
        s = max(1.0, float(xx.size)) * float(np.nanmedian(np.abs(yy - np.nanmedian(yy))) + 1e-6) ** 2
        spl = UnivariateSpline(xx, yy, w=ww, s=s)
        fit = np.asarray(spl(np.asarray(x, dtype=float)), dtype=float)
        return fit, {"method": "UnivariateSpline", "s": float(s)}
    except Exception:
        co = np.polyfit(xx, yy, deg=min(5, max(1, xx.size - 1)), w=ww)
        fit = np.polyval(co, np.asarray(x, dtype=float))
        return fit, {"method": "polyfit", "degree": int(len(co) - 1)}


def _interval_mask(wave: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(np.asarray(wave).shape, dtype=bool)
    for a, b in intervals:
        m |= (wave >= float(a)) & (wave <= float(b))
    return m


def _load_standard_reference(cfg: dict[str, Any], *, selected_name: str | None) -> tuple[dict[str, Any], Path, str, str]:
    lib_path = reference_path(cfg, exact='standards.library')
    if lib_path is None:
        raise FileNotFoundError('standards.library not found in reference context')
    lib = load_library_yaml(lib_path)
    key, rec = find_standard_record(lib, selected_name)
    if rec is None:
        raise FileNotFoundError('No standard entries in library.yaml')
    tbl_req = str(rec.get('table') or '')
    tbl_path = Path(tbl_req)
    if not tbl_path.is_absolute():
        # table path in library is package-relative.
        base = lib_path.parent.parent if lib_path.name == 'library.yaml' else lib_path.parent
        cand = (base / tbl_path.name) if '/' not in tbl_req else (lib_path.parent.parent / tbl_req)
        if cand.exists():
            tbl_path = cand.resolve()
        else:
            tbl_path = reference_path(cfg, prefix='standards.') or cand.resolve()
    return lib, tbl_path, str(rec.get('units') or 'f_lambda'), str(rec.get('wavelength_convention') or 'air')


def build_sensfunc(cfg: dict[str, Any], *, standard_spec_path: str | Path, standard_qc: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = read_spec1d(standard_spec_path, mode='trace')
    wave = spec['wave']
    flux = spec['flux']
    var = spec['var']
    mask = np.asarray(spec['mask'])
    hdr0 = spec['header']
    if str(hdr0.get('REBNORM') or '').strip() == 'COUNT_V1':
        if str(hdr0.get('WAVEREF') or '').strip().lower() not in {'air', 'vacuum'}:
            raise ValueError('COUNT_V1 standard requires explicitly declared WAVEREF air/vacuum')
        if str(hdr0.get('WAVEUNIT') or hdr0.get('CUNIT1') or '').strip() == '':
            raise ValueError('COUNT_V1 standard requires explicit wavelength-axis units')
    std_name = None
    raw_std_path = None
    if isinstance(standard_qc, dict):
        sel = standard_qc.get('selected_standard') or {}
        std_name = str(sel.get('object_name') or '') or None
        raw_std_path = str(sel.get('path') or '') or None
    raw_meta = None
    if raw_std_path:
        try:
            from astropy.io import fits
            raw_meta = parse_frame_meta(fits.getheader(raw_std_path, 0), strict=False)
        except Exception:
            raw_meta = None

    ctx = ensure_refctx(cfg)
    lib, ref_tbl_path, ref_units, ref_waveref = _load_standard_reference(cfg, selected_name=std_name)
    tbl = read_small_csv(ref_tbl_path)
    rw = np.asarray(tbl['wavelength_A'] if 'wavelength_A' in tbl else tbl.get('wavelength', []), dtype=float)
    rf = np.asarray(tbl['flux'] if 'flux' in tbl else tbl.get('f_lambda', []), dtype=float)
    if rw.size < 2 or rf.size != rw.size:
        raise ValueError('Invalid standard reference table')
    rw = convert_wavelength_convention(rw, from_ref=ref_waveref, to_ref=str(spec.get('waveref') or 'air'))
    ref_flux = np.interp(wave, rw, rf, left=np.nan, right=np.nan)

    ext_path = reference_path(cfg, prefix='extinction_curve.')
    extinction_meta = reference_metadata(cfg, prefix='extinction_curve.')
    if ext_path is not None and ext_path.exists():
        etbl = read_small_csv(ext_path)
        ew = np.asarray(etbl['wavelength_A'] if 'wavelength_A' in etbl else etbl.get('wavelength', []), dtype=float)
        ek = np.asarray(etbl['k_lambda'] if 'k_lambda' in etbl else etbl.get('extinction', []), dtype=float)
        extinction = np.interp(wave, ew, ek, left=np.nanmedian(ek) if ek.size else 0.0, right=np.nanmedian(ek) if ek.size else 0.0)
    else:
        extinction = np.zeros_like(wave)
    airmass = None
    airmass_prov = 'missing'
    if raw_meta is not None:
        try:
            airmass_prov = str(raw_meta.provenance_for('airmass'))
        except Exception:
            airmass_prov = 'missing'
        if raw_meta.airmass is not None and getattr(raw_meta, 'provenance_class', lambda _f: 'missing')('airmass') in {'header', 'derived'}:
            airmass = float(raw_meta.airmass)
    extinction_applied = bool(airmass is not None and ext_path is not None and ext_path.exists())
    ext_corr = 10.0 ** (0.4 * extinction * float(airmass)) if extinction_applied else np.ones_like(wave)

    tell_mask_path = reference_path(cfg, exact='telluric_mask.default')
    tell_mask_meta = reference_metadata(cfg, exact='telluric_mask.default')
    tell_intervals = [(a, b) for a, b, _ in read_interval_csv(tell_mask_path)] if tell_mask_path and tell_mask_path.exists() else []
    feature_mask = _interval_mask(wave, BALMER_WINDOWS_A)
    telluric_mask = _interval_mask(wave, tell_intervals)
    snr = np.abs(flux) / np.sqrt(np.clip(var, 1e-12, np.inf))
    low_sn = snr < float((((cfg.get('fluxcal') if isinstance(cfg.get('fluxcal'), dict) else {}) or {}).get('sensfunc_min_snr', 5.0) or 5.0))
    bad_mask = np.asarray(mask) != 0
    finite = np.isfinite(flux) & np.isfinite(var) & np.isfinite(ref_flux) & (ref_flux > 0) & (flux > 0)
    fit_mask = finite & (~feature_mask) & (~telluric_mask) & (~low_sn) & (~bad_mask)
    raw_ratio = (flux * ext_corr) / ref_flux
    frac_unc = np.sqrt(np.clip(var, 0, np.inf)) / np.clip(np.abs(flux), 1e-12, np.inf)
    yfit = np.log(np.clip(raw_ratio, 1e-30, np.inf))
    weights = 1.0 / np.clip(frac_unc, 1e-4, np.inf)
    model_log, fit_meta = _robust_spline_fit(wave[fit_mask], yfit[fit_mask], w=weights[fit_mask])
    response_abs = np.exp(model_log)
    clean = fit_mask & np.isfinite(response_abs) & (response_abs > 0)
    pivot = float(np.nanmedian(response_abs[clean])) if np.any(clean) else 1.0
    response_rel = response_abs / max(pivot, 1e-30)
    resid = yfit[fit_mask] - np.interp(wave[fit_mask], wave, model_log)
    sigma_ln = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))) * 1.4826) if resid.size else 0.15
    response_unc = np.abs(response_abs) * max(sigma_ln, 0.02)
    q = {
        'stage': 'sensfunc',
        'reference_context_id': str(ctx.get('context_id') or ''),
        'standard_name': std_name,
        'reference_table': str(ref_tbl_path),
        'reference_units': str(ref_units),
        'reference_waveref': str(ref_waveref),
        'airmass': (float(airmass) if airmass is not None else None),
        'fit_points': int(np.count_nonzero(fit_mask)),
        'airmass_provenance': str(airmass_prov),
        'extinction_applied': bool(extinction_applied),
        'extinction_reference': safe_json(extinction_meta),
        'telluric_mask_reference': safe_json(tell_mask_meta),
        'mask_fractions': {
            'feature_mask': float(np.mean(feature_mask)) if wave.size else 0.0,
            'telluric_mask': float(np.mean(telluric_mask)) if wave.size else 0.0,
            'low_sn': float(np.mean(low_sn)) if wave.size else 0.0,
            'bad_mask': float(np.mean(bad_mask)) if wave.size else 0.0,
        },
        'residual_sigma_ln': float(sigma_ln),
        'fit_meta': fit_meta,
    }
    return {
        'wave': wave,
        'raw_ratio': raw_ratio,
        'response_abs': response_abs,
        'response_rel': response_rel,
        'response_unc': response_unc,
        'fit_mask': fit_mask.astype(np.uint8),
        'feature_mask': feature_mask.astype(np.uint8),
        'telluric_mask': telluric_mask.astype(np.uint8),
        'low_sn_mask': low_sn.astype(np.uint8),
        'qc': q,
        'flux_unit': 'erg/s/cm2/Angstrom',
        'waveref': str(spec.get('waveref') or 'air'),
        'reference_context_id': str(ctx.get('context_id') or ''),
        'input_bunit': str(spec.get('bunit') or ''),
        'input_rebnorm': str(hdr0.get('REBNORM') or ''),
        'input_waveunit': str(hdr0.get('WAVEUNIT') or hdr0.get('CUNIT1') or '').strip(),
        'input_wave_step': (
            float(np.median(np.diff(wave)))
            if wave.size >= 2 and np.isfinite(np.diff(wave)).all()
            and (np.diff(wave) > 0).all()
            and np.allclose(np.diff(wave), np.median(np.diff(wave)), rtol=1e-6, atol=1e-9)
            else None
        ),
        'fit_meta': fit_meta,
        'airmass_provenance': str(airmass_prov),
        'extinction_applied': bool(extinction_applied),
        'reference_lineage': {
            'standard_table': safe_json(reference_metadata(cfg, prefix='standards.')),
            'extinction_curve': safe_json(extinction_meta),
            'telluric_mask': safe_json(tell_mask_meta),
        },
    }


def write_sensfunc_products(out_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from astropy.io import fits
    ph = fits.Header()
    ph['WAVEREF'] = str(payload.get('waveref') or 'air')
    ph['WAVEUNIT'] = 'Angstrom'
    # Response is only applicable to observations in the same detector
    # count/rate and bin/density units as the standard star.
    if str(payload.get('input_bunit') or '').strip():
        ph['INBUNIT'] = str(payload['input_bunit']).strip()
    if str(payload.get('input_rebnorm') or '').strip():
        ph['INREBNM'] = str(payload['input_rebnorm']).strip()
    if str(payload.get('input_waveunit') or '').strip():
        ph['INWUNIT'] = str(payload['input_waveunit']).strip()
    if payload.get('input_wave_step') is not None:
        ph['INSTEP'] = float(payload['input_wave_step'])
    if str(payload.get('input_rebnorm') or '').strip() == 'COUNT_V1':
        # Count/bin sensitivity scales with the spectral bin width. Do not
        # publish a modern response without an explicit, uniform grid.
        if 'INWUNIT' not in ph or 'INSTEP' not in ph:
            raise ValueError(
                'COUNT_V1 sensitivity requires explicit wavelength units '
                'and strictly increasing, uniform spectral bins'
            )
        if str(payload.get('waveref') or '').strip().lower() not in {'air', 'vacuum'}:
            raise ValueError('COUNT_V1 sensitivity requires declared air/vacuum medium')
    ph['REFCTID'] = str(payload.get('reference_context_id') or '')
    ph['RESPUNIT'] = 'count_per_flux'
    table_path = write_bintable_fits(
        out_dir / 'sensfunc_table.fits',
        primary_header=ph,
        extname='SENSFUNC',
        columns={
            'WAVELENGTH': payload['wave'],
            'RAW_RATIO': payload['raw_ratio'],
            'RESPONSE_ABS': payload['response_abs'],
            'RESPONSE_REL': payload['response_rel'],
            'RESPONSE_UNC': payload['response_unc'],
            'FIT_MASK': payload['fit_mask'],
            'FEATURE_MASK': payload['feature_mask'],
            'TELLURIC_MASK': payload['telluric_mask'],
            'LOWSN_MASK': payload['low_sn_mask'],
        },
    )
    model_path = save_json(out_dir / 'sensfunc_model.json', {
        'stage': 'sensfunc',
        'fit_meta': payload.get('fit_meta'),
        'flux_unit': payload.get('flux_unit'),
        'waveref': payload.get('waveref'),
        'reference_context_id': payload.get('reference_context_id'),
    })
    qc_path = save_json(out_dir / 'sensfunc_qc.json', payload.get('qc') or {})
    preview_path = plot_lines_png(out_dir / 'sensfunc_preview.png', [
        (payload['wave'], payload['raw_ratio'], 'raw ratio'),
        (payload['wave'], payload['response_abs'], 'smoothed response'),
    ], title='Sensfunc', vspans=[(float(a), float(b)) for a, b in BALMER_WINDOWS_A])
    plan_path = save_json(out_dir / 'sensfunc_plan.json', {'stage': 'sensfunc', 'status': 'ok', 'qc': safe_json(payload.get('qc') or {})})
    return {
        'sensfunc_table_fits': str(table_path),
        'sensfunc_model_json': str(model_path),
        'sensfunc_qc_json': str(qc_path),
        'sensfunc_preview_png': str(preview_path),
        'sensfunc_plan_json': str(plan_path),
    }
