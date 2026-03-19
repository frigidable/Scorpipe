from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scorpio_pipe.cli import cmd_prepare_long_case


def test_prepare_long_case_updates_registry_and_manifest(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    fixtures_dir = tmp_path / "fixtures"
    cases_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    registry = {
        "schema": "scorpio_pipe.long_parity_registry",
        "schema_version": 1,
        "minimum_real_cases_required": 1,
        "entries": [
            {
                "case_id": "skyline_rich_residuals",
                "status": "pending-private-data",
                "manifest": None,
                "category": "sky",
            }
        ],
    }
    registry_path = cases_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    sp_bundle = fixtures_dir / "scorpipe_bundle.json"
    lg_bundle = fixtures_dir / "long_bundle.json"
    sp_bundle.write_text(json.dumps({"source": "scorpipe"}), encoding="utf-8")
    lg_bundle.write_text(json.dumps({"source": "long"}), encoding="utf-8")

    args = Namespace(
        config=None,
        scorpipe_bundle=str(sp_bundle),
        long_bundle=str(lg_bundle),
        registry=str(registry_path),
        case_id="skyline_rich_residuals",
        title="Skyline stress case",
        category="sky",
        source_type="real_private",
        description="High skyline residual density.",
        observer_notes="Watch upper/lower strip asymmetry.",
        status="ready",
        output_manifest=None,
        scorpipe_output=None,
        no_registry_update=False,
    )

    rc = cmd_prepare_long_case(args)
    assert rc == 0

    manifest_path = cases_dir / "skyline_rich_residuals.case.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["case_id"] == "skyline_rich_residuals"
    assert manifest["metrics"]["extraction.snr"]["prefer"] == "higher"

    updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = next(x for x in updated_registry["entries"] if x["case_id"] == "skyline_rich_residuals")
    assert entry["status"] == "ready"
    assert entry["manifest"] == "skyline_rich_residuals.case.json"
