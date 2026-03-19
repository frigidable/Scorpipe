from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.artifact_index import (
    artifact_index_cache_stamp,
    artifact_index_legacy_path,
    artifact_index_path,
    load_artifact_index,
    update_after_stage,
)


def test_perf_72_1_update_after_stage_writes_append_only_jsonl(tmp_path: Path) -> None:
    cfg = {"work_dir": str(tmp_path), "frames": {}, "manifest": {}}

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    p1 = manifest_dir / "timings.jsonl"
    p1.write_text('{"stage":"demo","seconds":1.0}\n', encoding="utf-8")

    out_path = update_after_stage(
        cfg,
        work_dir=tmp_path,
        stage="manifest",
        stage_products={"timings": "manifest/timings.jsonl"},
    )
    assert out_path == artifact_index_path(tmp_path)
    assert out_path.suffix == ".jsonl"
    assert out_path.exists()

    lines1 = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines1) == 2  # meta + first stage record
    assert json.loads(lines1[0])["kind"] == "meta"
    assert json.loads(lines1[1])["kind"] == "stage"

    p2 = manifest_dir / "timings_v2.jsonl"
    p2.write_text('{"stage":"demo","seconds":2.0}\n', encoding="utf-8")
    update_after_stage(
        cfg,
        work_dir=tmp_path,
        stage="manifest",
        stage_products={"timings": "manifest/timings_v2.jsonl"},
    )

    lines2 = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines2) == 3  # exactly one appended record
    idx = load_artifact_index(tmp_path)
    assert isinstance(idx, dict)
    assert idx["schema"] == 2
    assert idx["format"] == "jsonl"
    assert idx["stages"]["manifest"]["keyed"]["timings"] == "manifest/timings_v2.jsonl"
    assert artifact_index_legacy_path(tmp_path).exists() is False



def test_perf_72_1_bootstraps_from_legacy_json_once(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    legacy = artifact_index_legacy_path(tmp_path)
    legacy.write_text(
        json.dumps(
            {
                "schema": 1,
                "updated_at": "2026-03-09T12:00:00Z",
                "stages": {
                    "wavesolution": {
                        "files": ["08_wavesol/lambda_map.fits"],
                        "keyed": {"lambda_map": "08_wavesol/lambda_map.fits"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    (manifest_dir / "timings.jsonl").write_text('{"stage":"demo","seconds":1.0}\n', encoding="utf-8")
    cfg = {"work_dir": str(tmp_path), "frames": {}, "manifest": {}}
    out_path = update_after_stage(
        cfg,
        work_dir=tmp_path,
        stage="manifest",
        stage_products={"timings": "manifest/timings.jsonl"},
    )

    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3  # meta + legacy bootstrap stage + new stage

    idx = load_artifact_index(tmp_path)
    assert isinstance(idx, dict)
    assert "wavesolution" in idx["stages"]
    assert "manifest" in idx["stages"]
    assert idx["stages"]["wavesolution"]["keyed"]["lambda_map"] == "08_wavesol/lambda_map.fits"

    stamp_path, stamp_ns = artifact_index_cache_stamp(tmp_path)
    assert stamp_path == artifact_index_path(tmp_path)
    assert stamp_ns > 0
