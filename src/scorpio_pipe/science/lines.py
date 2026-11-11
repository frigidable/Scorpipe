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


def _continuum_model(wave: np.ndarray, flux: np.ndarray, *, left: tuple[float, float], right: tuple[float, float], center: float) -> tuple[np.ndarray, dict[str, float]]:
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ml = (wave >= center + float(left[0])) & (wave <= center + float(left[1]))
    mr = (wave >= center + float(right[0])) & (wave <= center + float(right[1]))
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
    coverage_fraction: np.ndarray | None = None,
    waveref: str = "air",
    detection_snr: float = 3.0,
) -> dict[str, Any]:
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    var = np.asarray(var, dtype=float)
    if flux.shape != wave.shape or var.shape != wave.shape or wave.ndim != 1:
        raise ValueError("Line measurement expects same-length 1D wave/flux/var")
    if (wave.size < 3 or not np.isfinite(wave).all()
            or not np.all(np.diff(wave) > 0)):
        raise ValueError(
            "Line measurement requires finite strictly increasing wavelength samples"
        )
    if mask is None:
        quality_ok = np.ones(wave.shape, dtype=bool)
    else:
        m_in = np.asarray(mask)
        if m_in.shape != wave.shape or m_in.dtype.kind not in "iu":
            raise ValueError("Line measurement MASK must be 1D integer bitmask")
        quality_ok = m_in == 0
    quality_ok &= np.isfinite(flux) & np.isfinite(var) & (var > 0)
    coverage = None
    if coverage_fraction is not None:
        coverage = np.asarray(coverage_fraction, dtype=float)
        if (coverage.shape != wave.shape or not np.all(np.isfinite(coverage))
                or np.any(coverage < 0) or np.any(coverage > 1)):
            raise ValueError("LineMeasure requires finite FRAC_USED in [0,1]")
    medium = str(waveref or "").strip().lower()
    if medium not in {"air", "vacuum"}:
        raise ValueError("Line measurement needs explicit air/vacuum WAVEREF")
    if not (np.isfinite(detection_snr) and float(detection_snr) >= 3.0):
        raise ValueError("Line detection threshold must be >= 3 sigma")
    from scorpio_pipe.calib.common import air_to_vacuum

    rest = float(spec.rest_wavelength)
    if medium == "vacuum":
        rest = float(air_to_vacuum(np.array([rest], dtype=float))[0])
    rel = np.zeros_like(wave, dtype=float) if calsys_rel is None else np.asarray(calsys_rel, dtype=float)
    tel = np.zeros_like(wave, dtype=float) if calsys_tell is None else np.asarray(calsys_tell, dtype=float)
    absu = np.zeros_like(wave, dtype=float) if calsys_abs is None else np.asarray(calsys_abs, dtype=float)
    center = rest * (1.0 + float(redshift))
    # Both independent continuum sidebands need observed, usable samples.
    # A one-sided or entirely absent continuum can invent a positive line
    # excess from baseline drift. The internal continuum helper retains its
    # historical diagnostic fallback, but that fallback cannot certify a
    # science-level emission detection or a formal flux upper limit.
    left = (
        (wave >= center + float(spec.continuum_left[0]))
        & (wave <= center + float(spec.continuum_left[1]))
        & quality_ok
    )
    right = (
        (wave >= center + float(spec.continuum_right[0]))
        & (wave <= center + float(spec.continuum_right[1]))
        & quality_ok
    )
    n_left = int(np.count_nonzero(left))
    n_right = int(np.count_nonzero(right))
    if n_left < 2 or n_right < 2:
        return {
            'name': spec.name,
            'detected': False,
            'reason': 'insufficient_unmasked_continuum',
            'n_continuum_left': n_left,
            'n_continuum_right': n_right,
            'upper_limit_3sigma': None,
        }
    # Masked/invalid pixels must not influence either continuum sidebands
    # or integrated emission-line flux, even when their SCI is finite.
    flux_for_continuum = np.where(quality_ok, flux, np.nan)
    m = (wave >= center - float(spec.window_half_width)) & (wave <= center + float(spec.window_half_width))
    if np.count_nonzero(m) < 3:
        return {'name': spec.name, 'detected': False, 'reason': 'insufficient_sampling'}
    # V1 extraction's MASK reports usable 1D data, but losing a spatial
    # contributor can still remove unknown line flux from a boxcar aperture.
    # A *partial* line window is a measurement, NOT an automatically
    # qualified total emission-line flux (and not a valid standard upper
    # limit). Keep the partial integral for diagnostics and fail detection.
    partial = bool(
        coverage is not None
        and (np.any(coverage[m] < 1.0 - 1e-6) or np.any(~quality_ok[m]))
    )
    w = wave[m]
    f = flux[m]
    v = var[m]
    cont_full, cont_meta = _continuum_model(
        wave, flux_for_continuum,
        left=spec.continuum_left, right=spec.continuum_right, center=center
    )
    cont = cont_full[m]
    net = f - cont
    dl = np.gradient(w) if w.size > 1 else np.ones_like(w)
    dl = np.asarray(dl, dtype=float)
    fin = _finite_mask(w, f, v, cont, dl) & quality_ok[m] & (dl > 0)
    if np.count_nonzero(fin) < 3:
        return {
            'name': spec.name, 'detected': False,
            'reason': 'insufficient_valid_unmasked_pixels',
            'n_unmasked': int(np.count_nonzero(fin)),
        }
    w = w[fin]; f = f[fin]; v = v[fin]; cont = cont[fin]; net = net[fin]; dl = dl[fin]
    flux_int = float(np.sum(net * dl))
    err_stat = float(np.sqrt(np.sum(v * dl * dl)))
    cal_frac = _rms_relative([rel[m][fin], tel[m][fin], absu[m][fin]])
    rel_med = float(np.nanmedian(np.asarray(rel[m][fin], dtype=float))) if np.any(np.isfinite(np.asarray(rel[m][fin], dtype=float))) else 0.0
    tell_med = float(np.nanmedian(np.asarray(tel[m][fin], dtype=float))) if np.any(np.isfinite(np.asarray(tel[m][fin], dtype=float))) else 0.0
    abs_med = float(np.nanmedian(np.asarray(absu[m][fin], dtype=float))) if np.any(np.isfinite(np.asarray(absu[m][fin], dtype=float))) else 0.0
    err_cal_rel = float(abs(flux_int) * rel_med)
    err_cal_tell = float(abs(flux_int) * tell_med)
    err_cal_abs = float(abs(flux_int) * abs_med)
    err_cal = float(abs(flux_int) * np.nanmedian(cal_frac)) if cal_frac.size else 0.0
    sn = float(flux_int / err_stat) if err_stat > 0 else float("nan")
    detected = bool(np.isfinite(sn) and flux_int > 0
                    and sn >= float(detection_snr) and not partial)
    pos = np.clip(net, 0.0, np.inf)
    weight_sum = float(np.sum(pos * dl))
    centroid = None
    sigma = None
    if detected and weight_sum > 0:
        centroid = float(np.sum(w * pos * dl) / weight_sum)
        sigma_val = float(np.sqrt(max(
            0.0,
            np.sum(((w - centroid) ** 2) * pos * dl) / weight_sum
        )))
        sigma = sigma_val if math.isfinite(sigma_val) else None
    fwhm = sigma_to_fwhm(sigma)
    # Observed line width includes instrumental broadening. Divide by
    # observed wavelength, not the rest-frame wavelength of a redshifted line.
    sigma_kms = sigma_to_velocity_dispersion(centroid, sigma) if centroid is not None else None
    velocity = velocity_from_centroid(rest, centroid) if centroid is not None else None
    cont_level = float(np.nanmedian(cont)) if np.any(np.isfinite(cont)) else float('nan')
    ew = float(np.sum((net / np.clip(cont, 1e-12, np.inf)) * dl)) if np.isfinite(cont_level) and abs(cont_level) > 1e-12 else float('nan')
    blend_warning = spec.name in {
        'Halpha', '[NII]6548', '[NII]6583',
    }
    return {
        'name': spec.name,
        'rest_wavelength': rest,
        'waveref': medium,
        'blend_warning': bool(blend_warning),
        'flux_kind': ('partial_window_integral_unqualified' if partial else
                      'window_integral_not_deblended' if blend_warning
                      else 'continuum_subtracted_window_integral'),
        'sigma_kms_kind': 'observed_including_instrument_LSF',
        'expected_wavelength': float(center),
        # A nondetection has no meaningful fitted line center; JSON null is
        # intentional and must not be coerced with float(None).
        'centroid': None if centroid is None else float(centroid),
        'velocity_kms': velocity,
        'flux': float(flux_int),
        'equivalent_width': float(ew),
        'sigma': sigma,
        'sigma_kms': sigma_kms,
        'fwhm': fwhm,
        'local_continuum': cont_level,
        'err_stat': float(err_stat),
        'err_cal': float(err_cal),
        'err_cal_rel': float(err_cal_rel),
        'err_cal_tell': float(err_cal_tell),
        'err_cal_abs': float(err_cal_abs),
        'err_total': float(np.hypot(err_stat, err_cal)),
        'snr_stat': float(sn),
        'detected': detected,
        'reason': 'partial_aperture_or_spectral_coverage' if partial else None,
        'coverage_complete': not partial if coverage is not None else None,
        'min_spatial_fraction': float(np.min(coverage[m])) if coverage is not None else None,
        'detection_snr_threshold': float(detection_snr),
        'upper_limit_3sigma': None if partial or detected else float(3.0 * err_stat),
        'n_samples': int(w.size),
        'n_masked_in_window': int(np.count_nonzero(~quality_ok[m])),
        'n_continuum_left': n_left,
        'n_continuum_right': n_right,
        **cont_meta,
    }


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
    coverage_fraction: np.ndarray | None = None,
    waveref: str = "air",
    detection_snr: float = 3.0,
) -> list[dict[str, Any]]:
    return [
        measure_line(
            wave,
            flux,
            var,
            calsys_rel=calsys_rel,
            calsys_tell=calsys_tell,
            calsys_abs=calsys_abs,
            spec=spec,
            redshift=redshift,
            mask=mask,
            coverage_fraction=coverage_fraction,
            waveref=waveref,
            detection_snr=detection_snr,
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
