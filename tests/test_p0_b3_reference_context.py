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


def test_reference_context_tracks_calibration_domain_references(tmp_path: Path) -> None:
    from scorpio_pipe.refs.context import build_reference_context

    cfg = {
        "config_dir": str(tmp_path),
        "work_dir": str(tmp_path / "run"),
        "wavesol": {"lamp_type": "Ne"},
        "fluxcal": {
            "standard_name": "BD+28 4211",
            "site_name": "BTA",
        },
    }

    ctx = build_reference_context(cfg)
    refs = {r["name"]: r for r in ctx.get("references", [])}

    assert "standards.library" in refs
    assert "standards.bd_28_4211.table" in refs
    assert "extinction_curve.bta" in refs
    assert "telluric_mask.default" in refs
    assert "telluric_template.default" in refs
    assert refs["standards.bd_28_4211.table"]["metadata"]["units"] == "f_lambda"
    assert refs["extinction_curve.bta"]["metadata"]["wavelength_convention"] == "air"


def test_reference_context_changes_when_standard_table_changes(tmp_path: Path) -> None:
    from scorpio_pipe.refs.context import ensure_reference_context

    run_dir = tmp_path / "run"
    refs = tmp_path / "refs"
    refs.mkdir(parents=True)
    (refs / "my_lines.csv").write_text("# waveref: air\n1,2,3\n", encoding="utf-8")
    (refs / "atlas.pdf").write_bytes(b"%PDF-1.4\n%dummy\n")
    (refs / "standards").mkdir()
    (refs / "extinction").mkdir()
    (refs / "telluric").mkdir()
    (refs / "standards" / "library.yaml").write_text("version: 1\n", encoding="utf-8")
    (refs / "standards" / "custom.csv").write_text(
        "# version: 1\n# units: f_lambda\n# wavelength_convention: air\n# sampling: tabulated\nwavelength_A,flux\n5000,1e-14\n",
        encoding="utf-8",
    )
    (refs / "extinction" / "bta.csv").write_text(
        "# version: 1\n# units: mag_per_airmass\n# wavelength_convention: air\n# sampling: tabulated\nwavelength_A,k_lambda\n5000,0.2\n",
        encoding="utf-8",
    )
    (refs / "telluric" / "default_mask.csv").write_text(
        "# version: 1\n# units: transmission_mask\n# wavelength_convention: air\n# sampling: interval_list\nlambda_min_A,lambda_max_A,label\n6860,6965,O2_B\n",
        encoding="utf-8",
    )
    (refs / "telluric" / "default_template.csv").write_text(
        "# version: 1\n# units: relative_transmission\n# wavelength_convention: air\n# sampling: tabulated\nwavelength_A,transmission\n6900,0.7\n",
        encoding="utf-8",
    )

    cfg = {
        "config_dir": str(tmp_path),
        "work_dir": str(run_dir),
        "resources_dir": str(refs),
        "wavesol": {"lamp_type": "Ne", "linelist_csv": "my_lines.csv", "atlas_pdf": "atlas.pdf"},
        "fluxcal": {
            "standard_name": "Custom Std",
            "standard_library": "standards/library.yaml",
            "standard_table": "standards/custom.csv",
            "site_name": "BTA",
            "extinction_curve": "extinction/bta.csv",
            "telluric_mask": "telluric/default_mask.csv",
            "telluric_template": "telluric/default_template.csv",
        },
    }

    ctx1 = ensure_reference_context(cfg, resources_dir=refs)
    (refs / "standards" / "custom.csv").write_text(
        "# version: 2\n# units: f_lambda\n# wavelength_convention: air\n# sampling: tabulated\nwavelength_A,flux\n5000,2e-14\n",
        encoding="utf-8",
    )
    ctx2 = ensure_reference_context(cfg, resources_dir=refs, overwrite=True)

    assert ctx1["context_id"] != ctx2["context_id"]
