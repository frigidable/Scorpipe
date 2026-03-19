from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.qc.flags import (
    annotate_operational_flag,
    infer_operational_failure_code,
    list_operational_failure_specs,
    make_operational_flag,
)
from scorpio_pipe.qc_report import _collect_stage_flags
from scorpio_pipe.workspace_paths import stage_dir


REQUIRED = {
    'WAVE_GEOMETRY_WEAK',
    'WAVE_LINESET_POOR',
    'SKY_WINDOWS_CONTAMINATED',
    'SKY_MODEL_UNSTABLE',
    'SKY_RESIDUALS_STRONG',
    'SLIT_METADATA_MISMATCH',
    'EXTRACT_TRACE_UNCERTAIN',
    'PUBLICATION_PRODUCT_DEGRADED',
}


def test_operational_failure_taxonomy_contains_required_codes() -> None:
    specs = {spec.code for spec in list_operational_failure_specs()}
    assert REQUIRED.issubset(specs)
    assert infer_operational_failure_code('CALIB_SLITPOS_MISMATCH') == 'SLIT_METADATA_MISMATCH'
    assert infer_operational_failure_code('NO_SKY_WINDOWS') == 'SKY_WINDOWS_CONTAMINATED'


def test_operational_flag_annotation_uses_aliases() -> None:
    fl = annotate_operational_flag({'code': 'CALIB_ROT_MISMATCH', 'severity': 'WARN', 'message': 'rotation mismatch'})
    assert fl is not None
    assert fl['operational_code'] == 'SLIT_METADATA_MISMATCH'
    explicit = make_operational_flag('SKY_MODEL_UNSTABLE')
    assert explicit['operational_code'] == 'SKY_MODEL_UNSTABLE'
    assert explicit['operational_domain'] == 'sky'


def test_qc_report_collect_stage_flags_enriches_operational_failures(tmp_path: Path) -> None:
    flat_done = stage_dir(tmp_path, 'flatfield')
    flat_done.mkdir(parents=True, exist_ok=True)
    (flat_done / 'done.json').write_text(json.dumps({
        'qc': {
            'flags': [
                {'code': 'CALIB_SLITPOS_MISMATCH', 'severity': 'WARN', 'message': 'slit mismatch'},
            ]
        }
    }), encoding='utf-8')
    flags = _collect_stage_flags(tmp_path)
    assert flags
    assert flags[0]['operational_code'] == 'SLIT_METADATA_MISMATCH'
