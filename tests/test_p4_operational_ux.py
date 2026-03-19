from __future__ import annotations

import json
from pathlib import Path


def test_setup_presets_fill_conservative_defaults() -> None:
    from scorpio_pipe.operational_presets import apply_setup_preset_defaults, infer_setup_preset

    cfg = {
        'work_dir': '.',
        'frames': {'__setup__': {'instrument': 'SCORPIO-2', 'mode': 'Spectra', 'disperser': 'VPHG1800@590', 'slit': '1.0', 'binning': '1x1'}},
        'instrument': {'name': 'SCORPIO-2', 'disperser': 'VPHG1800@590', 'slit': '1.0', 'mode': 'Spectra'},
    }
    preset = infer_setup_preset(cfg)
    patched = apply_setup_preset_defaults(cfg)

    assert preset.geometry_mode == 'auto'
    assert patched['wavesol']['geometry_prior']['mode'] == 'auto'
    assert patched['extract1d']['method'] == 'auto'
    assert patched['sky_refine']['enabled'] is True
    assert patched['sky_refine']['detect_snr'] <= 3.8
    assert patched['_observer_setup_preset']['key'] == preset.key


def test_observer_dashboard_builds_recovery_and_writes_files(tmp_path: Path) -> None:
    from scorpio_pipe.observer_dashboard import write_observer_dashboard
    from scorpio_pipe.pipeline.public_api import build_observer_dashboard
    from scorpio_pipe.workspace_paths import stage_dir

    cfg = {
        'work_dir': str(tmp_path),
        'frames': {'__setup__': {'instrument': 'SCORPIO-2', 'mode': 'Spectra', 'disperser': 'VPHG1200@540', 'slit': '1.0', 'binning': '1x1'}},
        'instrument': {'name': 'SCORPIO-2', 'mode': 'Spectra', 'disperser': 'VPHG1200@540', 'slit': '1.0'},
    }

    gdir = stage_dir(tmp_path, 'geometry_prior')
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / 'geometry_model.json').write_text(json.dumps({'source': 'neon_curvature', 'metrics': {'trust_score': 0.81, 'fit_rms_px': 0.22}}), encoding='utf-8')
    (gdir / 'geometry_qc.json').write_text(json.dumps({'status': 'ok', 'max_severity': 'OK'}), encoding='utf-8')

    wdir = stage_dir(tmp_path, 'wavesolution')
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / 'wavesolution_2d.json').write_text(json.dumps({'kind': 'chebyshev', 'rms_A': 0.14, 'n_used': 17}), encoding='utf-8')
    (wdir / 'wave_done.json').write_text(json.dumps({'metrics': {'wave_trust_score': 0.77, 'wave_line_density': 1.3, 'rms2d_A': 0.14}}), encoding='utf-8')
    (wdir / 'line_id_overview.png').write_bytes(b'')
    (wdir / 'geometry_prior_used.json').write_text(json.dumps({'median_abs_resid_px': 0.4}), encoding='utf-8')

    sdir = stage_dir(tmp_path, 'sky_refine')
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / 'qc_sky_refine.json').write_text(json.dumps({'status': 'warn', 'n_windows_total': 3, 'aggregate_metrics': {'sky_sigma_gain_median': 0.12}}), encoding='utf-8')
    (sdir / 'sky_observer_pack.md').write_text('# sky\n', encoding='utf-8')

    edir = stage_dir(tmp_path, 'extract1d')
    edir.mkdir(parents=True, exist_ok=True)
    (edir / 'extraction_qc.json').write_text(json.dumps({'diagnostics': {'trace_stability_px': 0.08, 'contamination_risk': 0.12}}), encoding='utf-8')
    (edir / 'extraction_diagnostics.png').write_bytes(b'')

    qcdir = tmp_path / 'qc'
    qcdir.mkdir(parents=True, exist_ok=True)
    (qcdir / 'qc_report.json').write_text(json.dumps({'qc': {'operational_failures': [{'code': 'SKY_MODEL_UNSTABLE', 'stage': 'sky_refine'}], 'flags': []}}), encoding='utf-8')

    out = write_observer_dashboard(cfg, out_dir=tmp_path / 'observer_dashboard', run_root=tmp_path)
    assert (tmp_path / 'observer_dashboard' / 'observer_dashboard.json').exists()
    assert (tmp_path / 'observer_dashboard' / 'recovery_plan.json').exists()
    assert 'observer_dashboard_html' in out

    dash = build_observer_dashboard(cfg)
    assert isinstance(dash, dict)
    assert dash['trust_score']['overall'] is not None
    assert dash['recovery_plan']['n_actions'] >= 1
    assert dash['observer_dashboard'] is not None


def test_catalog_and_workflow_register_observer_dashboard() -> None:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCTS
    from scorpio_pipe.catalog.stages import CATALOG as STAGES
    from workflow.dodo import get_task_deps

    assert 'observer_dashboard_json' in set(PRODUCTS.keys())
    stage = STAGES.get_task('observer_dashboard')
    assert 'observer_dashboard_html' in set(stage.product_keys)
    deps = get_task_deps()
    assert 'observer_dashboard' in deps
    assert 'qc_report' in deps['observer_dashboard']
