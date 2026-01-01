from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scorpio_pipe.qc_report import build_qc_report
from scorpio_pipe.workspace_paths import stage_dir


def _write_done(stage_path: Path, *, codes: list[str]) -> None:
    stage_path.mkdir(parents=True, exist_ok=True)
    flags = [
        {
            "code": c,
            "severity": "WARN",
            "message": c,
            "hint": "",
        }
        for c in codes
    ]
    payload = {
        "stage": stage_path.name,
        "status": "ok",
        "qc": {"flags": flags, "max_severity": "WARN"},
    }
    (stage_path / "done.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_p2_qc_aggregator_collects_required_flags():
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_qc_") as td:
        work_dir = Path(td) / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # low coverage (legacy codes mapped to COVERAGE_LOW)
        _write_done(stage_dir(work_dir, "linearize"), codes=["QC_LINEARIZE_COVERAGE"])

        # absent sky windows
        _write_done(stage_dir(work_dir, "extract"), codes=["NO_SKY_WINDOWS"])

        # eta anomaly
        _write_done(stage_dir(work_dir, "stack"), codes=["ETA_ANOMALY"])

        # flexure uncertain
        _write_done(stage_dir(work_dir, "sky"), codes=["FLEXURE_UNCERTAIN"])

        cfg = {"work_dir": str(work_dir)}
        out = build_qc_report(cfg)

        qc_json = Path(out.json_path)
        assert qc_json.exists()
        payload = json.loads(qc_json.read_text(encoding="utf-8"))
        flags = payload.get("qc", {}).get("flags", [])
        codes = {f.get("code") for f in flags if isinstance(f, dict)}

        assert "COVERAGE_LOW" in codes
        assert "NO_SKY_WINDOWS" in codes
        assert "ETA_ANOMALY" in codes
        assert "FLEXURE_UNCERTAIN" in codes


def test_p2_qc_aggregator_not_noisy_when_no_stage_flags():
    with tempfile.TemporaryDirectory(prefix="scorpipe_pytest_p2_qc_empty_") as td:
        work_dir = Path(td) / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        cfg = {"work_dir": str(work_dir)}
        out = build_qc_report(cfg)
        payload = json.loads(Path(out.json_path).read_text(encoding="utf-8"))
        flags = payload.get("qc", {}).get("flags", [])
        assert isinstance(flags, list)
        assert len(flags) == 0
