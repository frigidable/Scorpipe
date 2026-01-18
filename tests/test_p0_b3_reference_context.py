from __future__ import annotations

import json
from pathlib import Path

import pytest


def _read_ctx(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_context_written_and_hash_changes(tmp_path: Path) -> None:
    from scorpio_pipe.refs.context import ensure_reference_context
    from scorpio_pipe.stage_state import compute_stage_hash

    run_dir = tmp_path / "run"
    refs = tmp_path / "refs"
    refs.mkdir(parents=True)

    # Two tracked references, supplied via resources_dir.
    (refs / "my_lines.csv").write_text("# waveref: air\n1,2,3\n", encoding="utf-8")
    (refs / "atlas.pdf").write_bytes(b"%PDF-1.4\n%dummy\n")

    cfg = {
        "config_dir": str(tmp_path),
        "work_dir": str(run_dir),
        "resources_dir": str(refs),
        "wavesol": {
            "lamp_type": "Ne",
            "linelist_csv": "my_lines.csv",
            "atlas_pdf": "atlas.pdf",
        },
    }

    ctx1 = ensure_reference_context(cfg, resources_dir=refs)
    ctx_path = run_dir / "manifest" / "reference_context.json"
    assert ctx_path.exists()
    ctx_file1 = _read_ctx(ctx_path)
    assert ctx_file1.get("context_id") == ctx1.get("context_id")

    h1 = compute_stage_hash(stage="wavesolution", stage_cfg={"x": 1}, reference_context_id=str(ctx1["context_id"]))

    # Modify a tracked reference content -> context_id must change -> stage hash changes.
    (refs / "my_lines.csv").write_text("# waveref: air\n1,2,3\n4,5,6\n", encoding="utf-8")
    ctx2 = ensure_reference_context(cfg, resources_dir=refs)
    assert str(ctx2["context_id"]) != str(ctx1["context_id"])
    h2 = compute_stage_hash(stage="wavesolution", stage_cfg={"x": 1}, reference_context_id=str(ctx2["context_id"]))
    assert h2 != h1


def test_reference_context_id_independent_of_absolute_paths(tmp_path: Path) -> None:
    from scorpio_pipe.refs.context import ensure_reference_context

    run_dir = tmp_path / "run"
    refs = tmp_path / "refs"
    refs.mkdir(parents=True)

    lines = refs / "same.csv"
    lines.write_text("# waveref: air\n1,2,3\n", encoding="utf-8")
    (refs / "atlas.pdf").write_bytes(b"%PDF-1.4\n%dummy\n")

    cfg_rel = {
        "config_dir": str(tmp_path),
        "work_dir": str(run_dir),
        "resources_dir": str(refs),
        "wavesol": {"lamp_type": "Ne", "linelist_csv": "same.csv", "atlas_pdf": "atlas.pdf"},
    }
    cfg_abs = {
        **cfg_rel,
        "wavesol": {"lamp_type": "Ne", "linelist_csv": str(lines), "atlas_pdf": "atlas.pdf"},
    }

    ctx_rel = ensure_reference_context(cfg_rel, resources_dir=refs)
    ctx_abs = ensure_reference_context(cfg_abs, resources_dir=refs)
    assert str(ctx_rel["context_id"]) == str(ctx_abs["context_id"])


def test_reference_context_missing_reference_raises(tmp_path: Path) -> None:
    from scorpio_pipe.refs.context import ensure_reference_context

    run_dir = tmp_path / "run"
    refs = tmp_path / "refs"
    refs.mkdir(parents=True)

    cfg = {
        "config_dir": str(tmp_path),
        "work_dir": str(run_dir),
        "resources_dir": str(refs),
        "wavesol": {"lamp_type": "Ne", "linelist_csv": "missing.csv", "atlas_pdf": "atlas.pdf"},
    }

    with pytest.raises(FileNotFoundError):
        ensure_reference_context(cfg, resources_dir=refs)
