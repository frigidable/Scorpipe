from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scorpio_pipe.long_setup_priors import get_long_setup_prior


def _norm_token(s: Any) -> str:
    return "".join(ch for ch in str(s or "").upper() if ch.isalnum())


def _deep_setdefault(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict):
            cur = dst.get(k)
            if not isinstance(cur, dict):
                cur = {}
                dst[k] = cur
            _deep_setdefault(cur, v)
        else:
            dst.setdefault(k, v)
    return dst


def _extract_setup(cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    frames = cfg.get('frames') if isinstance(cfg.get('frames'), dict) else {}
    setup = frames.get('__setup__') if isinstance(frames.get('__setup__'), dict) else {}
    inst = str((cfg.get('instrument') or {}).get('name') or setup.get('instrument') or '').strip()
    disp = str((cfg.get('instrument') or {}).get('disperser') or setup.get('disperser') or '').strip()
    slit = str((cfg.get('instrument') or {}).get('slit') or setup.get('slit') or '').strip()
    mode = str((cfg.get('instrument') or {}).get('mode') or setup.get('mode') or '').strip()
    return inst, disp, slit, mode


def _parse_slit_arcsec(slit: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(slit or ''))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


@dataclass(frozen=True)
class SetupPreset:
    key: str
    label: str
    instrument: str
    disperser: str
    slit: str
    mode: str
    has_curated_lines: bool
    geometry_mode: str
    line_prior_mode: str
    sky_refine_defaults: dict[str, Any]
    extract_defaults: dict[str, Any]
    reasons: tuple[str, ...] = ()
    long_setup_prior: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'label': self.label,
            'instrument': self.instrument,
            'disperser': self.disperser,
            'slit': self.slit,
            'mode': self.mode,
            'has_curated_lines': bool(self.has_curated_lines),
            'geometry_mode': self.geometry_mode,
            'line_prior_mode': self.line_prior_mode,
            'sky_refine_defaults': dict(self.sky_refine_defaults),
            'extract_defaults': dict(self.extract_defaults),
            'reasons': list(self.reasons),
            'long_setup_prior': dict(self.long_setup_prior) if isinstance(self.long_setup_prior, dict) else None,
        }


def infer_setup_preset(cfg: dict[str, Any]) -> SetupPreset:
    from scorpio_pipe.wavesol_presets import load_curated_line_preset

    inst, disp, slit, mode = _extract_setup(cfg)
    disp_norm = _norm_token(disp)
    inst_norm = _norm_token(inst)
    slit_arcsec = _parse_slit_arcsec(slit)
    curated = False
    try:
        curated = load_curated_line_preset(cfg, project_root=None, work_dir=None) is not None
    except Exception:
        curated = False

    prior = get_long_setup_prior(disp)
    disp_a_per_px = float(prior.median_dispersion_A_per_px) if prior is not None else None
    high_disp = (disp_a_per_px is not None and disp_a_per_px <= 0.65) or any(tag in disp_norm for tag in ('VPHG1800', 'VPHG2300'))
    medium_disp = (disp_a_per_px is not None and 0.65 < disp_a_per_px <= 1.10) or 'VPHG1200' in disp_norm
    skyline_rich = (prior is not None and prior.lambda_max_A >= 6500.0) or any(tag in disp_norm for tag in ('540', '590', '650', '780', 'RED'))

    reasons: list[str] = []
    if curated:
        reasons.append('curated LONG-derived line priors are available for this disperser')
    if prior is not None:
        reasons.append(
            f"LONG setup prior: {prior.lambda_min_A:.0f}–{prior.lambda_max_A:.0f} Å, "
            f"{prior.median_dispersion_A_per_px:.3f} Å/px"
        )
    if high_disp:
        reasons.append('high-dispersion setup benefits from aggressive skyline monitoring')
    elif medium_disp:
        reasons.append('medium-dispersion setup uses balanced skyline and wavelength defaults')
    if slit_arcsec is not None:
        reasons.append(f'slit width {slit_arcsec:g}" informs extraction policy')

    sky_defaults: dict[str, Any]
    if high_disp:
        sky_defaults = {
            'enabled': True,
            'detect_sigma': 2.2,
            'detect_snr': 3.2,
            'top_n_summary': 12,
            'strength': 1.0,
            'lsf_strength': 0.75,
        }
    elif medium_disp or skyline_rich:
        sky_defaults = {
            'enabled': True,
            'detect_sigma': 2.5,
            'detect_snr': 3.8,
            'top_n_summary': 10,
            'strength': 1.0,
            'lsf_strength': 0.70,
        }
    else:
        sky_defaults = {
            'enabled': True,
            'detect_sigma': 2.8,
            'detect_snr': 4.5,
            'top_n_summary': 8,
            'strength': 0.9,
            'lsf_strength': 0.60,
        }

    extract_policy = 'boxcar_extended'
    method = 'boxcar'
    local_sky = True
    if (slit_arcsec is not None and slit_arcsec <= 1.0 and (high_disp or curated)):
        extract_policy = 'optimal_point_source'
        method = 'optimal'
        local_sky = True
    elif high_disp and slit_arcsec is not None and slit_arcsec <= 1.2:
        extract_policy = 'optimal_conservative'
        method = 'optimal'
        local_sky = True
    elif mode.strip().lower() in {'spectra', 'longslit', 'long-slit'}:
        extract_policy = 'boxcar_science'
        method = 'boxcar'
        local_sky = True

    key_parts = [tok for tok in (inst_norm or 'UNKNOWN', disp_norm or 'UNKNOWN', _norm_token(slit) or 'NOSLIT') if tok]
    preset_key = '__'.join(key_parts)
    label = f'{inst or "SCORPIO"} {disp or "default"} / slit {slit or "unknown"}'
    return SetupPreset(
        key=preset_key,
        label=label,
        instrument=inst,
        disperser=disp,
        slit=slit,
        mode=mode,
        has_curated_lines=curated,
        geometry_mode='auto',
        line_prior_mode='curated_auto' if curated else 'lamp_default',
        sky_refine_defaults=sky_defaults,
        extract_defaults={
            'policy': extract_policy,
            'method': method,
            'local_sky_correction': local_sky,
        },
        reasons=tuple(reasons),
        long_setup_prior=prior.as_dict() if prior is not None else None,
    )


def apply_setup_preset_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with conservative setup-aware defaults filled in.

    This is intentionally additive: explicit user values win.  It is safe to call
    repeatedly because only missing keys are populated.
    """

    out = dict(cfg)
    preset = infer_setup_preset(out)
    long_prior = get_long_setup_prior(out)
    wavesol = out.get('wavesol') if isinstance(out.get('wavesol'), dict) else {}
    sky_refine = out.get('sky_refine') if isinstance(out.get('sky_refine'), dict) else {}
    extract1d = out.get('extract1d') if isinstance(out.get('extract1d'), dict) else {}
    linearize = out.get('linearize') if isinstance(out.get('linearize'), dict) else {}
    lsf_model = out.get('lsf_model') if isinstance(out.get('lsf_model'), dict) else {}
    out['wavesol'] = dict(wavesol)
    out['sky_refine'] = dict(sky_refine)
    out['extract1d'] = dict(extract1d)
    out['linearize'] = dict(linearize)
    out['lsf_model'] = dict(lsf_model)

    _deep_setdefault(out['wavesol'], {
        'setup_preset': preset.key,
        'line_prior_mode': preset.line_prior_mode,
        'geometry_prior': {'mode': preset.geometry_mode},
    })
    _deep_setdefault(out['sky_refine'], preset.sky_refine_defaults)
    _deep_setdefault(out['extract1d'], {
        'method': 'auto',
        'operational_policy': preset.extract_defaults.get('policy'),
    })
    if out['extract1d'].get('method') == 'auto':
        _deep_setdefault(out['extract1d'], {
            'local_sky_correction': bool(preset.extract_defaults.get('local_sky_correction', True)),
        })

    if long_prior is not None:
        _deep_setdefault(out['wavesol'], {
            'pair_tol_A': float(long_prior.suggested_pair_tol_A),
            'pair_top_lines': int(long_prior.suggested_pair_top_lines),
            'peak_target_min': int(long_prior.suggested_peak_target_min),
            'peak_target_max': int(long_prior.suggested_peak_target_max),
            'expected_coverage_A': [float(long_prior.lambda_min_A), float(long_prior.lambda_max_A)],
            'expected_dispersion_A_per_px': float(long_prior.median_dispersion_A_per_px),
        })
        _deep_setdefault(out['linearize'], {
            'lambda_min_A': float(long_prior.trimmed_lambda_min_A),
            'lambda_max_A': float(long_prior.trimmed_lambda_max_A),
        })
        _deep_setdefault(out['lsf_model'], {
            'expected_dispersion_A_per_px': float(long_prior.median_dispersion_A_per_px),
            'expected_fwhm_A_1arcsec': float(long_prior.empirical_resolution_fwhm_A_1arcsec),
        })

    meta = out.get('_observer_setup_preset') if isinstance(out.get('_observer_setup_preset'), dict) else {}
    meta.update(preset.as_dict())
    out['_observer_setup_preset'] = meta
    return out


def resolve_extract_defaults(cfg: dict[str, Any], ecfg: dict[str, Any]) -> dict[str, Any]:
    preset = infer_setup_preset(cfg)
    out = dict(ecfg or {})
    method = str(out.get('method', 'boxcar') or 'boxcar').strip().lower()
    if method == 'auto':
        out['method'] = str(preset.extract_defaults.get('method') or 'boxcar')
        if 'local_sky_correction' not in out:
            out['local_sky_correction'] = bool(preset.extract_defaults.get('local_sky_correction', True))
    out.setdefault('operational_policy', preset.extract_defaults.get('policy'))
    return out


__all__ = [
    'SetupPreset',
    'infer_setup_preset',
    'apply_setup_preset_defaults',
    'resolve_extract_defaults',
]
