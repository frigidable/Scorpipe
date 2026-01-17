from __future__ import annotations

"""Noise / variance helpers (P0-D: unit-policy + honest variance contract).

This module centralizes how the pipeline interprets CCD noise metadata.

Key concepts
------------
- **GAIN** is in e-/ADU.
- **READNOISE** is in electrons RMS (per pixel per read, per amplifier/node).

Policy (ReadNoisePolicy)
------------------------
We *must not* silently guess read-noise. We resolve it with an explicit priority:

1) override argument (rn_src=OVR)
2) header value (rn_src=HDR)
3) config database (rn_src=CONF)
4) estimate from bias pairs (rn_src=EST)
5) documentation / heuristic defaults (rn_src=DOC)
6) last-resort fallback (rn_src=FALL)

For GAIN we prefer header/metadata and use gain=1 only if not strict.

FITS keywords (<=8 chars)
-------------------------
Stages that call :func:`stamp_noise_keywords` will have, at minimum:
- GAIN    : gain used [e-/ADU]
- RDNOISE : read noise used [e-]
- RN_SRC  : OVR/HDR/CONF/EST/DOC/FALL
- NOISRC  : compact provenance string

The pipeline also keeps the longer SCORP-prefixed provenance keys
(SCORPGN/SCORPRN/SCORPNS) via :mod:`scorpio_pipe.units_model`.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from astropy.io import fits


# ----------------------------- small helpers -----------------------------

def _parse_float(hdr: fits.Header, keys: Iterable[str]) -> float | None:
    for k in keys:
        if k in hdr:
            try:
                v = float(hdr[k])
                if np.isfinite(v):
                    return v
            except Exception:
                continue
    return None


def robust_sigma(x: np.ndarray) -> float:
    """Robust sigma via MAD (Gaussian-consistent)."""

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return float(1.4826 * mad)


def estimate_readnoise_from_bias_stack(
    bias_stack_adu: np.ndarray,
    *,
    edge_frac: float = 0.05,
    max_pairs: int = 24,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate read-noise from a stack of bias frames (ADU units).

    Method
    ------
    Use frame pairs (b1-b2) and robust sigma on the difference:

        RN_ADU = sigma(diff) / sqrt(2)

    Parameters
    ----------
    bias_stack_adu:
        Array shaped (N, ny, nx) in ADU.
    edge_frac:
        Fraction of pixels to trim at each border before measuring sigma.
    max_pairs:
        Maximum number of pairs to use (limits runtime on huge stacks).

    Returns
    -------
    rn_adu, diag
        rn_adu is None if estimation is impossible.
    """

    diag: dict[str, Any] = {
        'method': 'pairdiff_mad',
        'edge_frac': float(edge_frac),
        'pairs_used': 0,
    }
    stack = np.asarray(bias_stack_adu)
    if stack.ndim != 3:
        return None, {**diag, 'error': 'bad_dim'}
    n, ny, nx = stack.shape
    if n < 2:
        return None, {**diag, 'error': 'too_few_frames'}

    # ROI: central region to avoid overscan/edges.
    y0 = int(round(ny * edge_frac))
    y1 = int(round(ny * (1.0 - edge_frac)))
    x0 = int(round(nx * edge_frac))
    x1 = int(round(nx * (1.0 - edge_frac)))
    y0 = max(0, min(ny - 1, y0))
    y1 = max(y0 + 1, min(ny, y1))
    x0 = max(0, min(nx - 1, x0))
    x1 = max(x0 + 1, min(nx, x1))
    diag['roi'] = [y0, y1, x0, x1]

    # Pair frames: (0,1), (2,3), ...
    n_pairs = min(n // 2, int(max_pairs))
    if n_pairs <= 0:
        return None, {**diag, 'error': 'no_pairs'}

    sigmas = []
    for i in range(n_pairs):
        b1 = stack[2 * i]
        b2 = stack[2 * i + 1]
        diff = (np.asarray(b1, dtype=float) - np.asarray(b2, dtype=float))[y0:y1, x0:x1]
        s = robust_sigma(diff)
        if np.isfinite(s) and s > 0:
            sigmas.append(float(s))

    diag['pairs_used'] = int(len(sigmas))
    if not sigmas:
        return None, {**diag, 'error': 'sigma_failed'}

    sigma_diff = float(np.median(sigmas))
    rn_adu = sigma_diff / float(np.sqrt(2.0))
    diag['sigma_diff_adu'] = float(sigma_diff)
    diag['rn_adu'] = float(rn_adu)
    return float(rn_adu), diag


# --------------------------- noise params model --------------------------


@dataclass(frozen=True)
class NoiseParams:
    gain_e_per_adu: float
    rdnoise_e: float
    source: str
    gain_src: str = 'UNK'
    rn_src: str = 'UNK'


def stamp_noise_keywords(hdr: fits.Header, params: NoiseParams, *, overwrite: bool = True) -> fits.Header:
    """Stamp compact noise provenance into FITS header (in-place)."""

    h = fits.Header(hdr)

    def _set(k: str, v: Any, c: str) -> None:
        try:
            if overwrite or (k not in h):
                h[k] = (v, c)
        except Exception:
            pass

    _set('GAIN', float(params.gain_e_per_adu), 'Gain used [e-/ADU]')
    _set('RDNOISE', float(params.rdnoise_e), 'Read noise used [e-]')
    _set('RN_SRC', str(params.rn_src)[:8], 'Read-noise source')
    _set('NOISRC', str(params.source)[:68], 'Noise provenance')
    return h


# ------------------------- config database match -------------------------


def _norm(s: Any) -> str:
    return str(s or '').strip().upper()


def _parse_binning(meta_or_hdr: Any) -> tuple[int, int] | None:
    # meta may be FrameMeta; hdr is fits.Header
    try:
        bx = int(getattr(meta_or_hdr, 'binning_x'))
        by = int(getattr(meta_or_hdr, 'binning_y'))
        return max(1, bx), max(1, by)
    except Exception:
        pass
    try:
        h = meta_or_hdr
        bx = int(h.get('CCDBIN1') or h.get('BINX') or 1)
        by = int(h.get('CCDBIN2') or h.get('BINY') or 1)
        return max(1, bx), max(1, by)
    except Exception:
        return None


def _match_readnoise_db(
    db: list[dict[str, Any]],
    *,
    instrument: str,
    detector: str,
    node: str,
    rate: float | None,
    binning: tuple[int, int] | None,
    roi: str,
) -> tuple[float | None, dict[str, Any] | None]:
    """Return (rdnoise_e, matched_row) from cfg.noise.readnoise."""

    best = None
    best_score = -1

    for row in db:
        if not isinstance(row, dict):
            continue

        score = 0

        def _eq_field(key: str, actual: str) -> bool:
            nonlocal score
            if key not in row or row[key] in (None, ''):
                return True
            want = _norm(row.get(key))
            if not want:
                return True
            if want != _norm(actual):
                return False
            score += 1
            return True

        if not _eq_field('instrument', instrument):
            continue
        if not _eq_field('detector', detector):
            continue
        if not _eq_field('node', node):
            continue

        # ROI token
        if row.get('roi') not in (None, ''):
            want = str(row.get('roi') or '')
            if want != str(roi or ''):
                continue
            score += 1

        # Binning
        if row.get('binning') not in (None, '') and binning is not None:
            want = str(row.get('binning') or '')
            try:
                if 'x' in want:
                    wb, hb = want.lower().split('x', 1)
                    if (int(wb), int(hb)) != (int(binning[0]), int(binning[1])):
                        continue
                score += 1
            except Exception:
                pass

        # Readout rate constraints
        if rate is not None and np.isfinite(rate):
            rmin = row.get('rate_min', None)
            rmax = row.get('rate_max', None)
            try:
                if rmin is not None and float(rate) < float(rmin):
                    continue
                if rmax is not None and float(rate) > float(rmax):
                    continue
                if (rmin is not None) or (rmax is not None):
                    score += 1
            except Exception:
                pass

        # Found a candidate
        val = row.get('rdnoise_e', row.get('read_noise_e'))
        try:
            rn = float(val)
        except Exception:
            continue
        if not np.isfinite(rn) or rn < 0:
            continue

        if score > best_score:
            best_score = score
            best = (float(rn), row)

    if best is None:
        return None, None
    return best[0], best[1]


# -------------------------- doc / heuristic RN --------------------------


def _default_rdnoise_sc2(rate_kpix_s: float | None) -> tuple[float, str]:
    """Heuristic defaults for SCORPIO-2 CCD261-84 (DOC fallback)."""
    if rate_kpix_s is None or not np.isfinite(rate_kpix_s):
        return 3.0, 'DOC(sc2:typical)'
    if rate_kpix_s <= 100:
        return 2.2, 'DOC(sc2:slow)'
    if rate_kpix_s <= 250:
        return 3.0, 'DOC(sc2:normal)'
    return 4.0, 'DOC(sc2:fast)'


def _doc_fallback_rdnoise(hdr: fits.Header, instrument_hint: str | None = None) -> tuple[float, str]:
    instr = _norm(instrument_hint or hdr.get('INSTRUME', ''))
    det = _norm(hdr.get('DETECTOR', ''))
    rate = _parse_float(hdr, ('RATE', 'READRATE', 'RATERD'))

    if ('SCORPIO' in instr and '2' in instr) or ('CCD261' in det):
        rn, tag = _default_rdnoise_sc2(rate)
        return float(rn), tag

    # SCORPIO-1 (EEV 42-40) typical range 1.8–4 e- across modes.
    if 'SCORPIO' in instr:
        return 3.0, 'DOC(sc1:typical)'

    return 5.0, 'FALL(typical)'


# ------------------------------ main API --------------------------------


def resolve_noise_params(
    hdr: fits.Header,
    *,
    cfg: dict | None = None,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    bias_rn_est_adu: float | None = None,
    bias_rn_est_e: float | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> NoiseParams:
    """Resolve gain/read-noise (in electrons) from overrides/header/cfg/bias."""

    h = fits.Header(hdr)

    # --- infer a few matching fields (best-effort) ---
    instrument = _norm(instrument_hint or h.get('INSTRUME', ''))
    detector = _norm(h.get('DETECTOR', ''))
    node = _norm(h.get('NODE', h.get('AMPL', '')))
    rate = _parse_float(h, ('RATE', 'READRATE', 'RATERD'))

    roi = ''
    try:
        roi = str(h.get('FSIGROI') or h.get('ROI') or '')
    except Exception:
        roi = ''

    binning = _parse_binning(h)

    # 1) Gain
    if gain_override is not None:
        gain = float(gain_override)
        gain_src = 'OVR'
    else:
        gain = _parse_float(h, ('GAIN', 'EGAIN', 'GAIN_E', 'GAINE', 'SBGAIN'))
        if gain is None or gain <= 0 or not np.isfinite(gain):
            if require_gain:
                raise ValueError(
                    'Missing CCD gain (e-/ADU). Provide a valid GAIN/EGAIN header card or set gain_override.'
                )
            gain = 1.0
            gain_src = 'DEF'
        else:
            gain_src = 'HDR'

    # 2) Read noise
    if rdnoise_override is not None:
        rn = float(rdnoise_override)
        rn_src = 'OVR'
        src_detail = 'override'
    else:
        rn_hdr = _parse_float(h, ('RDNOISE', 'READNOIS', 'RON', 'RNOISE'))
        if rn_hdr is not None and rn_hdr > 0 and np.isfinite(rn_hdr):
            rn = float(rn_hdr)
            rn_src = 'HDR'
            src_detail = 'header'
        else:
            # config database
            rn = None
            row = None
            try:
                db = ((cfg or {}).get('noise') or {}).get('readnoise')
                if isinstance(db, list) and db:
                    rn, row = _match_readnoise_db(
                        db,
                        instrument=instrument,
                        detector=detector,
                        node=node,
                        rate=rate,
                        binning=binning,
                        roi=roi,
                    )
            except Exception:
                rn = None
                row = None

            if rn is not None:
                rn_src = 'CONF'
                src_detail = 'cfg'
            else:
                # bias estimate
                rn_est_e = None
                if bias_rn_est_e is not None:
                    try:
                        v = float(bias_rn_est_e)
                        if np.isfinite(v) and v >= 0:
                            rn_est_e = v
                    except Exception:
                        rn_est_e = None
                if rn_est_e is None and bias_rn_est_adu is not None:
                    try:
                        v = float(bias_rn_est_adu)
                        if np.isfinite(v) and v >= 0:
                            rn_est_e = v * float(gain)
                    except Exception:
                        rn_est_e = None

                if rn_est_e is not None:
                    rn = float(rn_est_e)
                    rn_src = 'EST'
                    src_detail = 'bias_pairs'
                else:
                    # doc fallback
                    rn, tag = _doc_fallback_rdnoise(h, instrument_hint=instrument_hint)
                    if tag.startswith('DOC'):
                        rn_src = 'DOC'
                        src_detail = tag
                    else:
                        rn_src = 'FALL'
                        src_detail = tag

    # Sanity
    if not np.isfinite(gain) or gain <= 0:
        if require_gain:
            raise ValueError('Invalid gain (e-/ADU).')
        gain = 1.0
        gain_src = 'DEF'
    if not np.isfinite(rn) or rn < 0:
        rn = 5.0
        rn_src = 'FALL'
        src_detail = 'fallback'

    src = f"GAIN:{gain_src};RN:{rn_src};{src_detail}"
    return NoiseParams(
        gain_e_per_adu=float(gain),
        rdnoise_e=float(rn),
        source=str(src),
        gain_src=str(gain_src),
        rn_src=str(rn_src),
    )


def estimate_variance_adu2(
    data_adu: np.ndarray,
    hdr: fits.Header,
    *,
    cfg: dict | None = None,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    bias_rn_est_adu: float | None = None,
    bias_rn_est_e: float | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, NoiseParams]:
    """Estimate per-pixel variance in ADU^2 from Poisson+RN CCD model."""

    params = resolve_noise_params(
        hdr,
        cfg=cfg,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        bias_rn_est_adu=bias_rn_est_adu,
        bias_rn_est_e=bias_rn_est_e,
        instrument_hint=instrument_hint,
        require_gain=require_gain,
    )
    gain = float(params.gain_e_per_adu)
    rn = float(params.rdnoise_e)

    # electrons: e = sci*gain ; var_e = max(e,0) + rn^2
    e = np.maximum(np.asarray(data_adu, dtype=np.float64) * gain, 0.0)
    var_e = e + rn * rn
    var_adu2 = var_e / (gain * gain)
    return np.asarray(var_adu2, dtype=np.float32), params


def estimate_variance_e2(
    sci_e: np.ndarray,
    *,
    rdnoise_e: float,
) -> np.ndarray:
    """Estimate per-pixel variance in electrons^2 from SCI in electrons."""

    sci = np.asarray(sci_e, dtype=np.float64)
    rn2 = float(rdnoise_e) ** 2
    return (np.maximum(sci, 0.0) + rn2).astype(np.float32)


def estimate_variance_auto(
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    cfg: dict | None = None,
    gain_override: float | None = None,
    rdnoise_override: float | None = None,
    bias_rn_est_adu: float | None = None,
    bias_rn_est_e: float | None = None,
    unit_model: str | None = None,
    instrument_hint: str | None = None,
    require_gain: bool = False,
) -> tuple[np.ndarray, NoiseParams, str]:
    """Estimate variance for SCI, automatically handling ADU vs electrons."""

    from scorpio_pipe.units_model import infer_unit_model, UnitModel

    h = fits.Header(hdr)
    model = infer_unit_model(h, default=UnitModel.ADU) if unit_model is None else UnitModel(str(unit_model).upper())

    params = resolve_noise_params(
        h,
        cfg=cfg,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        bias_rn_est_adu=bias_rn_est_adu,
        bias_rn_est_e=bias_rn_est_e,
        instrument_hint=instrument_hint,
        require_gain=require_gain if str(model.value).upper() == 'ADU' else False,
    )

    if model == UnitModel.ELECTRON:
        var_e2 = estimate_variance_e2(sci, rdnoise_e=params.rdnoise_e)
        return var_e2, params, model.value

    var_adu2, _ = estimate_variance_adu2(
        sci,
        h,
        cfg=cfg,
        gain_override=gain_override,
        rdnoise_override=rdnoise_override,
        bias_rn_est_adu=bias_rn_est_adu,
        bias_rn_est_e=bias_rn_est_e,
        instrument_hint=instrument_hint,
        require_gain=require_gain,
    )
    return var_adu2, params, model.value
