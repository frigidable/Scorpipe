from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.qc.flags import annotate_operational_flag, infer_operational_failure_code

RECOVERY_LIBRARY: dict[str, dict[str, Any]] = {
    'WAVE_LINESET_POOR': {
        'title': 'Strengthen arc line identification',
        'why': 'The active arc line set is too sparse or ambiguous for a stable 2D solution.',
        'steps': [
            'Switch to the curated per-grism line preset chosen for this setup.',
            'Inspect line_id_overview.png and reject obviously blended / wrong IDs.',
            'If needed, add a small hand_pairs.txt seed around the cleanest lines and rerun wavesolution.',
        ],
        'config_patch': {'wavesol': {'line_prior_mode': 'curated_auto'}},
        'artifacts': ['line_id_overview_png', 'wavesolution_2d_json', 'residuals_2d_png'],
        'retry_tasks': ['lineid_prepare', 'wavesolution'],
    },
    'WAVE_GEOMETRY_WEAK': {
        'title': 'Re-anchor the geometric prior',
        'why': 'The slit curvature prior is weak or inconsistent with traced control points.',
        'steps': [
            'Inspect geometry_overlay.png and geometry_model.json for edge drift or poor row coverage.',
            'Rerun geometry_prior with the auto/neon mode, or provide 13-dots repers when available.',
            'If the arc is weak at the slit edges, crop the worst edges rather than trusting them silently.',
        ],
        'config_patch': {'wavesol': {'geometry_prior': {'mode': 'auto'}}},
        'artifacts': ['geometry_overlay_png', 'geometry_model_json', 'geometry_prior_used_json'],
        'retry_tasks': ['geometry_prior', 'wavesolution'],
    },
    'SKY_WINDOWS_CONTAMINATED': {
        'title': 'Rebuild sky windows away from the object',
        'why': 'The current sky stripes contain object light or another contaminant.',
        'steps': [
            'Inspect sky_observer_pack.md and the per-frame sky window CSV/JSON outputs.',
            'Move the sky stripes farther from the object or tighten the object ROI / protection mask.',
            'Then rerun sky and sky_refine instead of accepting contaminated training regions.',
        ],
        'config_patch': {'sky': {'roi': {'obj_y0': '<adjust>', 'obj_y1': '<adjust>', 'sky_top': '<adjust>', 'sky_bot': '<adjust>'}}},
        'artifacts': ['sky_observer_pack_md', 'sky_residual_windows_json', 'qc_sky_refine_json'],
        'retry_tasks': ['sky', 'sky_refine', 'linearize'],
    },
    'SKY_MODEL_UNSTABLE': {
        'title': 'Stabilize sky subtraction and residual cleanup',
        'why': 'The first-pass sky model is unstable or leaves structured skyline residuals.',
        'steps': [
            'Enable or keep sky_refine enabled and inspect the worst skyline windows first.',
            'Check whether the problem is wavelength mismatch, amplitude mismatch, or LSF mismatch.',
            'If the residuals come from only one stripe, suspect contamination rather than more aggressive fitting.',
        ],
        'config_patch': {'sky_refine': {'enabled': True}},
        'artifacts': ['sky_observer_pack_md', 'sky_residual_windows_csv', 'qc_sky_refine_json'],
        'retry_tasks': ['sky_refine', 'linearize'],
    },
    'SKY_RESIDUALS_STRONG': {
        'title': 'Target the strongest skyline residual windows',
        'why': 'Residual skyline peaks remain strong after the baseline subtraction.',
        'steps': [
            'Look at the top problematic windows summary and verify whether both sky stripes agree.',
            'Treat one-strip-only bursts as cosmic leakage or contamination, not as a global wavelength error.',
            'After fixing the cause, rerun sky_refine and propagate the refined raw products into linearize.',
        ],
        'config_patch': {'sky_refine': {'enabled': True, 'detect_sigma': 2.5}},
        'artifacts': ['sky_observer_pack_md', 'sky_residual_windows_json', 'qc_sky_refine_json'],
        'retry_tasks': ['sky_refine', 'linearize'],
    },
    'SLIT_METADATA_MISMATCH': {
        'title': 'Split or relabel mixed slit setups',
        'why': 'The effective slit profile disagrees with the metadata or multiple setups were mixed.',
        'steps': [
            'Inspect slit_profile.png and slit_consistency.md.',
            'Separate exposures by slit width / binning / ROI before continuing calibration-sensitive work.',
            'Treat any resolution-dependent product as degraded until the setup mismatch is removed.',
        ],
        'config_patch': {'frames': {'__setup__': {'slit': '<verify>'}}},
        'artifacts': ['slit_sanity_json'],
        'retry_tasks': ['slit_sanity', 'wavesolution', 'lsf_model'],
    },
    'EXTRACT_TRACE_UNCERTAIN': {
        'title': 'Make extraction more conservative',
        'why': 'The trace or aperture geometry is not trustworthy enough for blind extraction.',
        'steps': [
            'Inspect extraction_diagnostics.png and extraction_qc.json.',
            'Widen the object ROI or force a fixed center if the trace is fragmentary.',
            'When contamination risk is high, prefer a conservative aperture over a fragile optimal profile.',
        ],
        'config_patch': {'extract1d': {'method': 'boxcar', 'fixed_center_y': '<optional>'}},
        'artifacts': ['extraction_qc_json', 'extraction_diagnostics_png'],
        'retry_tasks': ['extract1d', 'publication_ready', 'kinematics_prep'],
    },
    'PUBLICATION_PRODUCT_DEGRADED': {
        'title': 'Regenerate publication products from the authoritative maps',
        'why': 'The exported science handoff products were built from degraded upstream inputs.',
        'steps': [
            'Check publication_provenance.json and ensure lambda_map / dismap come from the authoritative stage outputs.',
            'Do not pass publication products downstream if wavelength or LSF status is still degraded.',
            'Rerun publication_ready after the upstream wave / sky / extraction issue is fixed.',
        ],
        'config_patch': {},
        'artifacts': ['publication_provenance_json', 'publication_spectrum_fts', 'publication_wl_fts', 'publication_dismap_fts'],
        'retry_tasks': ['publication_ready'],
    },
}


def _read_json(path: Path | None) -> dict[str, Any]:
    import json
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_recovery_plan(
    cfg_or_path: Mapping[str, Any] | str | Path,
    *,
    qc_payload: dict[str, Any] | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.config import load_config_any
    from scorpio_pipe.operational_presets import infer_setup_preset

    if isinstance(cfg_or_path, Mapping):
        cfg = dict(cfg_or_path)
    else:
        cfg = load_config_any(cfg_or_path)
    rr = Path(run_root or cfg.get('work_dir') or '.').expanduser().resolve()

    if qc_payload is None:
        try:
            qc_path = PRODUCT_CATALOG.best_path(cfg, 'qc_json', run_root=rr)
        except Exception:
            qc_path = None
        qc_payload = _read_json(qc_path)

    qc_block = qc_payload.get('qc') if isinstance(qc_payload, dict) and isinstance(qc_payload.get('qc'), dict) else {}
    raw_failures = list(qc_block.get('operational_failures') or [])
    raw_flags = list(qc_block.get('flags') or [])

    seen: set[str] = set()
    codes: list[tuple[str, str | None]] = []
    for row in raw_failures:
        code = str((row or {}).get('code') or '').strip()
        stage = str((row or {}).get('stage') or '').strip() or None
        if code and code not in seen:
            seen.add(code)
            codes.append((code, stage))
    for row in raw_flags:
        ann = annotate_operational_flag(dict(row or {})) or dict(row or {})
        code = infer_operational_failure_code(str(ann.get('operational_code') or ann.get('code') or '').strip())
        stage = str(ann.get('stage') or '').strip() or None
        if code and code not in seen:
            seen.add(code)
            codes.append((code, stage))

    preset = infer_setup_preset(cfg)
    actions: list[dict[str, Any]] = []
    for code, stage in codes:
        spec = RECOVERY_LIBRARY.get(code)
        if not spec:
            continue
        inspect_paths: list[str] = []
        for key in spec.get('artifacts') or []:
            try:
                p = PRODUCT_CATALOG.best_path(cfg, str(key), run_root=rr)
            except Exception:
                p = None
            if p is not None and Path(p).exists():
                inspect_paths.append(str(Path(p)))
        actions.append({
            'code': code,
            'stage': stage,
            'title': spec.get('title') or code,
            'why': spec.get('why') or '',
            'steps': list(spec.get('steps') or []),
            'config_patch': spec.get('config_patch') or {},
            'retry_tasks': list(spec.get('retry_tasks') or []),
            'inspect_paths': inspect_paths,
        })

    summary = 'no urgent recovery actions'
    if actions:
        summary = f'{len(actions)} recovery path(s) suggested'
    return {
        'setup_preset': preset.as_dict(),
        'summary': summary,
        'actions': actions,
        'n_actions': int(len(actions)),
    }


__all__ = ['RECOVERY_LIBRARY', 'build_recovery_plan']
