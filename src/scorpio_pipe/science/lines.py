from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scorpio_pipe.science.kinematics import sigma_to_fwhm, sigma_to_velocity_dispersion, velocity_from_centroid


@dataclass(frozen=True)
class LineSpec:
    name: str
    rest_wavelength: float
    window_half_width: float
    continuum_left: tuple[float, float]
    continuum_right: tuple[float, float]


DEFAULT_LINES: tuple[LineSpec, ...] = (
    LineSpec('Hbeta', 4861.33, 14.0, (-45.0, -20.0), (20.0, 45.0)),
    LineSpec('[OIII]4959', 4958.91, 12.0, (-35.0, -15.0), (15.0, 35.0)),
    LineSpec('[OIII]5007', 5006.84, 12.0, (-35.0, -15.0), (15.0, 35.0)),
    LineSpec('[OI]6300', 6300.30, 12.0, (-35.0, -15.0), (15.0, 35.0)),
    LineSpec('Halpha', 6562.80, 18.0, (-60.0, -25.0), (25.0, 60.0)),
    LineSpec('[NII]6548', 6548.05, 12.0, (-45.0, -18.0), (18.0, 45.0)),
    LineSpec('[NII]6583', 6583.45, 12.0, (-45.0, -18.0), (18.0, 45.0)),
    LineSpec('[SII]6716', 6716.44, 12.0, (-40.0, -18.0), (18.0, 40.0)),
    LineSpec('[SII]6731', 6730.82, 12.0, (-40.0, -18.0), (18.0, 40.0)),
)


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(np.asarray(arrs[0], dtype=float), dtype=bool)
    for a in arrs:
        m &= np.isfinite(np.asarray(a, dtype=float))
    return m


def _robust_median(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return float(np.nanmedian(a))


def _continuum_model(wave: np.ndarray, flux: np.ndarray, *, left: tuple[float, float], right: tuple[float, float], center: float, valid: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if valid is not None:
        good &= np.asarray(valid, dtype=bool)
    ml = (wave >= center + float(left[0])) & (wave <= center + float(left[1])) & good
    mr = (wave >= center + float(right[0])) & (wave <= center + float(right[1])) & good
    xl = wave[ml]
    yl = flux[ml]
    xr = wave[mr]
    yr = flux[mr]
    xl = xl[np.isfinite(yl)]
    yl = yl[np.isfinite(yl)]
    xr = xr[np.isfinite(yr)]
    yr = yr[np.isfinite(yr)]
    if xl.size == 0 and xr.size == 0:
        cont = np.full_like(wave, np.nanmedian(flux[np.isfinite(flux)]) if np.any(np.isfinite(flux)) else 0.0)
        return cont, {'continuum_left': float('nan'), 'continuum_right': float('nan'), 'continuum_slope': 0.0}
    x_nodes = []
    y_nodes = []
    if xl.size:
        x_nodes.append(float(np.nanmedian(xl)))
        y_nodes.append(float(np.nanmedian(yl)))
    if xr.size:
        x_nodes.append(float(np.nanmedian(xr)))
        y_nodes.append(float(np.nanmedian(yr)))
    if len(x_nodes) == 1:
        cont = np.full_like(wave, y_nodes[0], dtype=float)
        return cont, {'continuum_left': y_nodes[0], 'continuum_right': y_nodes[0], 'continuum_slope': 0.0}
    slope = (y_nodes[1] - y_nodes[0]) / max(1e-12, x_nodes[1] - x_nodes[0])
    cont = y_nodes[0] + slope * (wave - x_nodes[0])
    return cont, {'continuum_left': y_nodes[0], 'continuum_right': y_nodes[1], 'continuum_slope': float(slope)}


def _rms_relative(values: Iterable[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(v, dtype=float) for v in values]
    if not arrs:
        return np.array([], dtype=float)
    out = np.zeros_like(arrs[0], dtype=float)
    for a in arrs:
        out += np.where(np.isfinite(a), a * a, 0.0)
    return np.sqrt(out)


# EDGE is informational partial coverage, not an automatic science rejection.
# All remaining canonical detector/sky-quality bits veto the affected sample.
from scorpio_pipe.contracts.maskbits import (
    NO_COVERAGE, BADPIX, COSMIC, SATURATED, USER, REJECTED,
    OUTSIDE_SLIT, INVALID_WAVELENGTH, SKYMODEL_FAIL,
)
_FATAL_LINE_BITS = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER |
                       REJECTED | OUTSIDE_SLIT | INVALID_WAVELENGTH | SKYMODEL_FAIL)


def measure_line(
    wave: np.ndarray,
    flux: np.ndarray,
    var: np.ndarray,
    *,
    calsys_rel: np.ndarray | None,
    calsys_tell: np.ndarray | None,
    calsys_abs: np.ndarray | None,
    spec: LineSpec,
    redshift: float = 0.0,
    mask: np.ndarray | None = None,
    detection_snr_min: float = 3.0,
) -> dict[str, Any]:
    """Diagnostic integrated line measurement (diagonal-noise approximation).

    This is not a deblender or a full covariance-aware confidence interval.
    A detection requires positive integrated *net* flux with a finite,
    strictly positive statistical error and sufficient signed S/N.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    var = np.asarray(var, dtype=float)
    if (wave.ndim != 1 or wave.shape != flux.shape or wave.shape != var.shape
            or wave.size < 3 or not np.isfinite(wave).all()
            or not np.all(np.diff(wave) > 0)):
        raise ValueError("Line measurement requires aligned, increasing finite wavelength coordinates")
    snr_min = float(detection_snr_min)
    if not math.isfinite(snr_min) or snr_min <= 0:
        raise ValueError("detection_snr_min must be positive and finite")
    if mask is None:
        quality = np.zeros_like(wave, dtype=np.uint32)
    else:
        quality = np.asarray(mask)
        if quality.shape != wave.shape or not np.issubdtype(quality.dtype, np.integer):
            raise ValueError("Line mask must be an integer array aligned with wavelength")
        if np.any(quality < 0):
            raise ValueError("Line mask contains negative quality flags")
        quality = quality.astype(np.uint32)
    good = ((quality & _FATAL_LINE_BITS) == 0) & np.isfinite(flux) & np.isfinite(var) & (var > 0)
    rel = np.zeros_like(wave) if calsys_rel is None else np.asarray(calsys_rel, dtype=float)
    tel = np.zeros_like(wave) if calsys_tell is None else np.asarray(calsys_tell, dtype=float)
    absu = np.zeros_like(wave) if calsys_abs is None else np.asarray(calsys_abs, dtype=float)
    center = float(spec.rest_wavelength) * (1.0 + float(redshift))
    in_line = (wave >= center - float(spec.window_half_width)) & (wave <= center + float(spec.window_half_width))
    if np.count_nonzero(in_line) < 3:
        return {'name': spec.name, 'detected': False, 'reason': 'insufficient_sampling'}
    in_good_line = in_line & good
    if np.count_nonzero(in_good_line) < 3:
        return {'name': spec.name, 'detected': False, 'reason': 'insufficient_unmasked_line_samples'}
    left = (wave >= center + spec.continuum_left[0]) & (wave <= center + spec.continuum_left[1]) & good
    right = (wave >= center + spec.continuum_right[0]) & (wave <= center + spec.continuum_right[1]) & good
    if np.count_nonzero(left) < 2 or np.count_nonzero(right) < 2:
        return {'name': spec.name, 'detected': False, 'reason': 'insufficient_unmasked_continuum'}

    # Compute widths on the original monotonic grid, *before* removing bad
    # samples. Otherwise a masked gap inherits spuriously enlarged bin widths.
    all_dl = np.gradient(wave)
    w = wave[in_good_line]
    f = flux[in_good_line]
    v = var[in_good_line]
    dl = all_dl[in_good_line]
    cont_full, cont_meta = _continuum_model(
        wave, flux, left=spec.continuum_left, right=spec.continuum_right,
        center=center, valid=good,
    )
    cont = cont_full[in_good_line]
    fin = _finite_mask(w, f, v, cont, dl) & (dl > 0)
    if np.count_nonzero(fin) < 3:
        return {'name': spec.name, 'detected': False, 'reason': 'nonfinite_window'}
    w, f, v, cont, dl = w[fin], f[fin], v[fin], cont[fin], dl[fin]
    net = f - cont
    flux_int = float(np.sum(net * dl))
    err_stat = float(np.sqrt(np.sum(v * dl * dl)))
    cal_frac = _rms_relative([rel[in_good_line][fin], tel[in_good_line][fin], absu[in_good_line][fin]])
    rel_med = float(np.nanmedian(rel[in_good_line][fin]))
    tell_med = float(np.nanmedian(tel[in_good_line][fin]))
    abs_med = float(np.nanmedian(absu[in_good_line][fin]))
    err_cal_rel = float(abs(flux_int) * rel_med)
    err_cal_tell = float(abs(flux_int) * tell_med)
    err_cal_abs = float(abs(flux_int) * abs_med)
    err_cal = float(abs(flux_int) * np.nanmedian(cal_frac)) if cal_frac.size else 0.0
    sn = float(flux_int / err_stat) if err_stat > 0 and np.isfinite(err_stat) else float("nan")
    detected = bool(flux_int > 0 and np.isfinite(sn) and sn >= snr_min)
    centroid = sigma = None
    if detected:
        pos = np.clip(net, 0.0, np.inf)
        weight = pos * dl
        weight_sum = float(np.sum(weight))
        if weight_sum > 0:
            centroid = float(np.sum(w * weight) / weight_sum)
            sigma_val = float(np.sqrt(max(0.0, np.sum((w - centroid) ** 2 * weight) / weight_sum)))
            sigma = sigma_val if np.isfinite(sigma_val) else None
    fwhm = sigma_to_fwhm(sigma)
    sigma_kms = sigma_to_velocity_dispersion(spec.rest_wavelength, sigma)
    velocity = velocity_from_centroid(spec.rest_wavelength, centroid) if centroid is not None else None
    cont_level = float(np.nanmedian(cont))
    ew = float(np.sum(net / cont * dl)) if np.isfinite(cont_level) and np.all(cont > 0) else float('nan')
    result = {
        'name': spec.name,
        'rest_wavelength': float(spec.rest_wavelength),
        'expected_wavelength': float(center),
        'centroid': centroid,
        'velocity_kms': velocity,
        'flux': flux_int,
        'equivalent_width': ew,
        'sigma': sigma,
        'sigma_kms': sigma_kms,
        'fwhm': fwhm,
        'local_continuum': cont_level,
        'err_stat': err_stat,
        'err_cal': err_cal,
        'err_cal_rel': err_cal_rel,
        'err_cal_tell': err_cal_tell,
        'err_cal_abs': err_cal_abs,
        'err_total': float(np.hypot(err_stat, err_cal)),
        'snr_stat': sn,
        'detection_snr_min': snr_min,
        'detected': detected,
        'n_samples': int(w.size),
        'n_excluded_window': int(np.count_nonzero(in_line & ~good)),
        **cont_meta,
    }
    if not detected:
        result['reason'] = 'below_detection_threshold'
    return result


def measure_lines(
    wave: np.ndarray,
    flux: np.ndarray,
    var: np.ndarray,
    *,
    calsys_rel: np.ndarray | None,
    calsys_tell: np.ndarray | None,
    calsys_abs: np.ndarray | None,
    lines: Iterable[LineSpec] = DEFAULT_LINES,
    redshift: float = 0.0,
    mask: np.ndarray | None = None,
    detection_snr_min: float = 3.0,
) -> list[dict[str, Any]]:
    return [
        measure_line(
            wave, flux, var,
            calsys_rel=calsys_rel, calsys_tell=calsys_tell, calsys_abs=calsys_abs,
            spec=spec, redshift=redshift, mask=mask,
            detection_snr_min=detection_snr_min,
        )
        for spec in lines
    ]


def write_lines_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8')
        return p
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def plot_line_measurements_png(path: str | Path, wave: np.ndarray, flux: np.ndarray, rows: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    ax.plot(np.asarray(wave, dtype=float), np.asarray(flux, dtype=float), lw=1.0)
    for row in rows:
        if not row.get('detected'):
            continue
        cen = row.get('centroid')
        if cen is None:
            continue
        x = float(cen)
        ax.axvline(x, alpha=0.35, linestyle='--')
        ax.text(x, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.0, str(row.get('name')), rotation=90, va='top', ha='right', fontsize=7)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Flux')
    ax.set_title('Measured science lines')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p
