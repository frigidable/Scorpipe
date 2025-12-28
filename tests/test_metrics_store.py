from __future__ import annotations

from pathlib import Path

from scorpio_pipe.qc.metrics_store import load_metrics, metrics_path, update_after_stage
from scorpio_pipe.work_layout import ensure_work_layout


def test_metrics_json_created_and_updated(tmp_path: Path):
    work_dir = tmp_path / "work"
    ensure_work_layout(work_dir)
    cfg = {"config_dir": str(tmp_path), "work_dir": str(work_dir)}

    p = update_after_stage(cfg, stage="manifest", status="ok", stage_hash="deadbeef")
    assert p == metrics_path(work_dir)
    assert p.exists()

    js = load_metrics(work_dir)
    assert "stages" in js
    assert "manifest" in js["stages"]
    assert js["stages"]["manifest"]["status"] == "ok"
