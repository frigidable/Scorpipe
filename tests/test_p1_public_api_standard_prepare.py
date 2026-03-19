from __future__ import annotations

import json
from pathlib import Path


def test_read_standard_prepare_summary_returns_ui_safe_dataclass(tmp_path: Path) -> None:
    from scorpio_pipe.pipeline.public_api import read_standard_prepare_summary
    from scorpio_pipe.workspace_paths import stage_dir

    cfg = {"work_dir": str(tmp_path), "frames": {}}
    sdir = stage_dir(tmp_path, "standard_prepare")
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "standard_qc.json").write_text(
        json.dumps(
            {
                "selected_standard": {
                    "path": "/data/std1.fits",
                    "role": "standard",
                    "object_name": "BD+28 4211",
                },
                "association": {
                    "status": "ABSOLUTE_TRUSTED",
                    "downgrade_reasons": [],
                },
                "safe_for": "absolute_candidate",
                "max_severity": "OK",
                "flags": [],
            }
        ),
        encoding="utf-8",
    )

    summary = read_standard_prepare_summary(cfg)

    assert summary is not None
    assert summary.selected is not None
    assert summary.selected.object_name == "BD+28 4211"
    assert summary.selected.association_status == "ABSOLUTE_TRUSTED"
    assert summary.selected.safe_for == "absolute_candidate"
    assert summary.max_severity == "OK"
