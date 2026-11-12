from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.calib.common import (
    ensure_refctx,
    plot_lines_png,
    read_interval_csv,
    read_spec1d,
    reference_metadata,
    reference_path,
    save_json,
    write_bintable_fits,
)


def _estimate_window_transmission(wave: np.ndarray, flux: np.ndarray, var: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    m = (wave >= a) & (wave <= b)
    if not np.any(m):
        return np.array([], dtype=float), np.array([], dtype=float)
    pad = max(5.0, 0.08 * (b - a))
    left = (wave >= a - pad) & (wave < a)
    right = (wave > b) & (wave <= b + pad)
    use = left | right
    if np.count_nonzero(use) < 4:
        cont = np.full(np.count_nonzero(m), np.nanmedian(flux[m]) if np.any(m) else 1.0)
    else:
        cont = np.interp(wave[m], wave[use], flux[use], left=np.nanmedian(flux[use]), right=np.nanmedian(flux[use]))
    trans = np.clip(flux[m] / np.clip(cont, 1e-12, np.inf), 0.25, 1.15)
    snr = np.abs(cont) / np.sqrt(np.clip(var[m], 1e-12, np.inf))
    conf = np.clip((snr - 5.0) / 20.0, 0.0, 1.0)
    return trans, conf


def build_telluric_model(cfg: dict[str, Any], *, standard_spec_path: str | Path) -> dict[str, Any]:
    spec = read_spec1d(standard_spec_path, mode='trace')
    wave, flux, var = spec['wave'], spec['flux'], spec['var']
    ctx = ensure_refctx(cfg)
    mask_path = reference_path(cfg, exact='telluric_mask.default')
    tmpl_path = reference_path(cfg, exact='telluric_template.default')
    mask_meta = reference_metadata(cfg, exact='telluric_mask.default')
    tmpl_meta = reference_metadata(cfg, exact='telluric_template.default')
    intervals = read_interval_csv(mask_path) if mask_path and mask_path.exists() else []
    transmission = np.ones_like(wave, dtype=float)
    confidence = np.zeros_like(wave, dtype=float)
    model_mask = np.zeros_like(wave, dtype=np.uint8)
    for a, b, _lab in intervals:
        trans, conf = _estimate_window_transmission(wave, flux, var, float(a), float(b))
        m = (wave >= float(a)) & (wave <= float(b))
        if trans.size == np.count_nonzero(m):
            transmission[m] = trans
            confidence[m] = conf
            model_mask[m] = 1
    # A four-point default template joins the O2 B and A bands with an
    # unphysical deep absorption trough. Never interpret sparse interpolation
    # between distinct atmospheric bands as science transmission.
    # The present standard-derived estimates are diagnostic ONLY; they are
    # not exposure-specific T_sci(lambda, airmass, weather).
    # An explicitly supplied template is retained in provenance for later
    # qualified fitting, but not applied until a physically justified model
    # and validation contract are implemented.
    template_ignored = bool(tmpl_path and tmpl_path.exists())
    transmission[model_mask == 0] = 1.0
    confidence[model_mask == 0] = 0.0
    qc = {
        'stage': 'telluric_model',
        'reference_context_id': str(ctx.get('context_id') or ''),
        'n_windows': int(len(intervals)),
        'coverage_fraction': float(np.mean(model_mask > 0)) if wave.size else 0.0,
        'median_confidence': float(np.nanmedian(confidence)) if confidence.size else 0.0,
        'transmission_floor': float(np.nanmin(transmission)) if transmission.size else 1.0,
        'scientific_status': 'DIAGNOSTIC',
        'science_qualified': False,
        'template_applied': False,
        'template_ignored': bool(template_ignored),
        'outside_model_windows': 'UNITY',
        'reason': 'standard-derived transmission is not an airmass/time-matched science model',
        'reference_lineage': {
            'telluric_mask': mask_meta,
            'telluric_template': tmpl_meta,
        },
    }
    return {
        'wave': wave,
        'transmission': transmission,
        'confidence': confidence,
        'model_mask': model_mask,
        'qc': qc,
        'reference_context_id': str(ctx.get('context_id') or ''),
        'waveref': str(spec.get('waveref') or 'air'),
    }


def write_telluric_products(out_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from astropy.io import fits
    ph = fits.Header()
    ph['WAVEREF'] = str(payload.get('waveref') or 'air')
    ph['WAVEUNIT'] = 'Angstrom'
    ph['REFCTID'] = str(payload.get('reference_context_id') or '')
    ph['TELQUAL'] = 'DIAGNOSTIC'
    fits_path = write_bintable_fits(
        out_dir / 'telluric_model.fits',
        primary_header=ph,
        extname='TELLURIC_MODEL',
        columns={
            'WAVELENGTH': payload['wave'],
            'TRANSMISSION': payload['transmission'],
            'CONFIDENCE': payload['confidence'],
            'MODEL_MASK': payload['model_mask'],
        },
    )
    preview = plot_lines_png(out_dir / 'telluric_preview.png', [
        (payload['wave'], payload['transmission'], 'telluric transmission'),
        (payload['wave'], payload['confidence'], 'confidence'),
    ], title='Telluric model')
    qc = save_json(out_dir / 'telluric_qc.json', payload.get('qc') or {})
    plan = save_json(out_dir / 'telluric_model_plan.json', {'stage': 'telluric_model', 'status': 'warn', 'qc': payload.get('qc') or {}})
    return {
        'telluric_model_fits': str(fits_path),
        'telluric_preview_png': str(preview),
        'telluric_qc_json': str(qc),
        'telluric_model_plan_json': str(plan),
    }
