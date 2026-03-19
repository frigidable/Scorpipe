from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic_fits import atomic_write_fits
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.qc.flags import make_flag, make_operational_flag, max_severity


@dataclass
class GeometryPriorResult:
    geometry_dir: Path
    geometry_model_json: Path
    geometry_preview_fits: Path
    geometry_qc_json: Path
    repers_json: Path
    source: str
    trust_score: float
    fit_rms_px: float


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parabolic_subpix(v: np.ndarray, i: int) -> float:
    if i <= 0 or i >= (len(v) - 1):
        return 0.0
    y0, y1, y2 = float(v[i - 1]), float(v[i]), float(v[i + 1])
    den = y0 - 2.0 * y1 + y2
    if den == 0.0:
        return 0.0
    return 0.5 * (y0 - y2) / den


def _xcorr_shift(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> tuple[float, float]:
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    if ref.size != cur.size or ref.size < 16:
        return 0.0, 0.0
    ref = np.nan_to_num(ref - np.nanmedian(ref))
    cur = np.nan_to_num(cur - np.nanmedian(cur))
    corr = np.correlate(cur, ref, mode='full')
    sh = np.arange(-(ref.size - 1), ref.size, dtype=float)
    m = np.abs(sh) <= float(max_shift)
    if not np.any(m):
        return 0.0, 0.0
    corr_m = corr[m]
    sh_m = sh[m]
    i = int(np.argmax(corr_m))
    sub = _parabolic_subpix(corr_m, i)
    shift = float(sh_m[i] + sub)
    denom = float(np.sqrt(np.sum(ref**2) * np.sum(cur**2)))
    score = float(corr_m[i] / denom) if denom > 0 else 0.0
    return shift, score


def _robust_polyfit(y: np.ndarray, x: np.ndarray, deg: int, sigma_clip: float = 3.0, maxiter: int = 8) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    if int(np.sum(m)) < max(4, deg + 1):
        return np.array([0.0]), np.zeros_like(y, dtype=bool)
    used = m.copy()
    coeff = np.array([0.0])
    for _ in range(maxiter):
        coeff = np.polyfit(y[used], x[used], deg=min(deg, int(np.sum(used)) - 1))
        pred = np.polyval(coeff, y)
        resid = x - pred
        rr = resid[used]
        med = float(np.median(rr)) if rr.size else 0.0
        mad = float(np.median(np.abs(rr - med))) if rr.size else 0.0
        sig = 1.4826 * mad if mad > 0 else float(np.std(rr))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_used = m & (np.abs(resid - med) <= sigma_clip * sig)
        if np.array_equal(new_used, used):
            break
        used = new_used
    return coeff, used


def _load_external_repers(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    ys = np.asarray(raw.get('y_samples_pix', []), float)
    shifts = np.asarray(raw.get('x_shift_pix', []), float)
    coeff = np.asarray(raw.get('x_shift_poly_coeffs', []), float)
    if ys.size and shifts.size == ys.size:
        if coeff.size == 0:
            coeff, _ = _robust_polyfit(ys, shifts, deg=3)
        return ys, shifts, coeff
    return None


def build_geometry_prior(cfg: dict[str, Any], *, run_root: str | Path | None = None) -> GeometryPriorResult:
    work_dir = Path(run_root).expanduser().resolve() if run_root is not None else Path(str(cfg.get('work_dir') or '.')).expanduser().resolve()
    outdir = _ensure_dir(work_dir / '07_geometry')
    wavesol_out = wavesol_dir(cfg, run_root=work_dir)
    superneon = wavesol_out / 'superneon.fits'
    if not superneon.exists():
        raise FileNotFoundError(f'Missing superneon for geometry prior: {superneon}')

    geom_cfg = ((cfg.get('wavesol') or {}).get('geometry_prior') if isinstance(cfg.get('wavesol'), dict) else {}) or {}
    repers_in = str(geom_cfg.get('repers_json') or geom_cfg.get('repers_path') or '').strip()
    repers_payload: dict[str, Any] = {}

    img = np.asarray(fits.getdata(superneon, memmap=False), dtype=float)
    if img.ndim != 2:
        raise RuntimeError('geometry_prior expects a 2D superneon image')
    H, W = img.shape
    x_crop_frac = float(geom_cfg.get('x_crop_frac', 0.12))
    x1 = int(max(0, round(W * x_crop_frac)))
    x2 = int(min(W, round(W * (1.0 - x_crop_frac))))
    if x2 - x1 < 32:
        x1, x2 = 0, W
    block_h = int(geom_cfg.get('block_height', max(8, H // 24)))
    max_shift = int(geom_cfg.get('max_shift_px', max(8, W // 32)))
    fit_deg = int(geom_cfg.get('fit_deg', 3))
    ref_y = int(np.clip(int(geom_cfg.get('reference_row', H // 2)), 0, H - 1))
    ref_half = int(geom_cfg.get('reference_half_height', max(2, block_h // 2)))
    ref_prof = np.nanmean(img[max(0, ref_y - ref_half):min(H, ref_y + ref_half + 1), x1:x2], axis=0)

    y_centers: list[float] = []
    shifts: list[float] = []
    scores: list[float] = []
    for y in range(block_h // 2, H - block_h // 2, block_h):
        yy1 = max(0, y - block_h // 2)
        yy2 = min(H, y + block_h // 2 + 1)
        prof = np.nanmean(img[yy1:yy2, x1:x2], axis=0)
        sh, sc = _xcorr_shift(ref_prof, prof, max_shift=max_shift)
        y_centers.append(float(y))
        shifts.append(float(sh))
        scores.append(float(sc))

    ys = np.asarray(y_centers, float)
    xsh = np.asarray(shifts, float)
    sc = np.asarray(scores, float)
    coeff, used = _robust_polyfit(ys, xsh, deg=fit_deg, sigma_clip=float(geom_cfg.get('sigma_clip', 3.0)))
    pred = np.polyval(coeff, ys) if coeff.size else np.zeros_like(ys)
    resid = xsh - pred
    fit_rms = float(np.sqrt(np.mean((resid[used]) ** 2))) if np.any(used) else float('nan')
    max_abs = float(np.max(np.abs(pred))) if pred.size else 0.0
    med_score = float(np.median(sc[used])) if np.any(used) else 0.0
    coverage = float(np.sum(used) / max(1, ys.size))
    smooth = float(np.median(np.abs(np.diff(np.diff(pred))))) if pred.size >= 3 else 0.0
    trust = float(np.clip(0.45 * coverage + 0.35 * max(0.0, med_score) + 0.20 * np.exp(-max(0.0, fit_rms) / 1.5), 0.0, 1.0))
    source = 'neon_curvature'

    if repers_in:
        try:
            ext = Path(repers_in)
            if not ext.is_absolute():
                ext = (work_dir / ext).resolve()
            loaded = _load_external_repers(ext)
            if loaded is not None:
                eys, esh, ecoeff = loaded
                repers_payload = {'external_repers_path': str(ext), 'n_repers': int(eys.size)}
                if ecoeff.size:
                    coeff = ecoeff
                    source = 'external_repers'
                    pred = np.polyval(coeff, ys) if ys.size else np.array([], float)
                    resid = xsh - pred if ys.size else np.array([], float)
                    fit_rms = float(np.sqrt(np.mean((resid[used]) ** 2))) if np.any(used) else fit_rms
                    trust = max(trust, 0.85)
        except Exception:
            pass

    shift_full = np.polyval(coeff, np.arange(H, dtype=float)) if coeff.size else np.zeros(H, float)
    shift_map = np.repeat(np.asarray(shift_full, dtype=np.float32)[:, None], W, axis=1)

    hdr = fits.Header()
    hdr['BUNIT'] = ('pix', 'Predicted X shift from geometry prior')
    hdr['GEOMSRC'] = (source, 'Geometry prior source')
    hdr['GEOMTRST'] = (float(trust), 'Geometry prior trust score')
    hdr['GEOMRMS'] = (float(fit_rms) if np.isfinite(fit_rms) else -1.0, 'Row-shift fit RMS [pix]')
    hdr = add_provenance(hdr, cfg, stage='geometry_prior')
    geometry_preview = outdir / 'geometry_preview.fits'
    atomic_write_fits(geometry_preview, fits.PrimaryHDU(data=shift_map.astype('float32'), header=hdr))

    flags: list[dict[str, Any]] = []
    if coverage < 0.55:
        flags.append(make_operational_flag('WAVE_GEOMETRY_WEAK', severity='WARN', message='Geometry prior coverage across slit is low.', coverage=coverage))
    if np.isfinite(fit_rms) and fit_rms > 2.5:
        flags.append(make_operational_flag('WAVE_GEOMETRY_WEAK', severity='WARN', message='Geometry prior fit RMS is high.', fit_rms_px=float(fit_rms)))
    if max_abs > max(6.0, 0.02 * W):
        flags.append(make_flag('GEOMETRY_SHIFT_LARGE', 'WARN', 'Estimated line curvature is unusually large.', max_abs_shift_px=float(max_abs)))

    payload = {
        'stage': 'geometry_prior',
        'schema_version': 1,
        'source': source,
        'superneon_fits': str(superneon),
        'x_window_pix': [int(x1), int(x2)],
        'reference_row': int(ref_y),
        'block_height': int(block_h),
        'fit_deg': int(fit_deg),
        'x_shift_poly_coeffs': [float(c) for c in np.atleast_1d(coeff)],
        'y_samples_pix': [float(v) for v in ys.tolist()],
        'x_shift_samples_pix': [float(v) for v in xsh.tolist()],
        'scores': [float(v) for v in sc.tolist()],
        'used_samples_mask': [bool(v) for v in used.tolist()],
        'metrics': {
            'fit_rms_px': float(fit_rms) if np.isfinite(fit_rms) else None,
            'median_score': float(med_score),
            'coverage': float(coverage),
            'smoothness': float(smooth),
            'max_abs_shift_px': float(max_abs),
            'trust_score': float(trust),
        },
        'repers': repers_payload or None,
        'flags': flags,
    }
    model_json = outdir / 'geometry_model.json'
    model_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    repers_json = outdir / 'repers.json'
    repers_json.write_text(json.dumps({'source': source, 'y_pix': ys.tolist(), 'x_shift_pix': xsh.tolist(), 'x_shift_poly_coeffs': [float(c) for c in np.atleast_1d(coeff)]}, indent=2), encoding='utf-8')
    qc_json = outdir / 'geometry_qc.json'
    qc_json.write_text(json.dumps({'status': 'warn' if max_severity(flags) == 'WARN' else 'ok', 'flags': flags, 'trust_score': trust, 'fit_rms_px': float(fit_rms) if np.isfinite(fit_rms) else None}, indent=2, ensure_ascii=False), encoding='utf-8')
    # stage-local done marker
    (outdir / 'done.json').write_text(json.dumps({'stage': 'geometry_prior', 'status': 'warn' if max_severity(flags) == 'WARN' else 'ok', 'products': {'geometry_model_json': str(model_json), 'geometry_preview_fits': str(geometry_preview), 'geometry_qc_json': str(qc_json), 'repers_json': str(repers_json)}}, indent=2), encoding='utf-8')
    return GeometryPriorResult(
        geometry_dir=outdir,
        geometry_model_json=model_json,
        geometry_preview_fits=geometry_preview,
        geometry_qc_json=qc_json,
        repers_json=repers_json,
        source=source,
        trust_score=float(trust),
        fit_rms_px=float(fit_rms) if np.isfinite(fit_rms) else float('nan'),
    )
