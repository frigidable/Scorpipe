from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scorpio_pipe.cli import cmd_certify_long_parity


def test_cmd_certify_long_parity_runtime_does_not_raise_on_markup_false(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    reports_dir = tmp_path / "reports"
    out_dir = tmp_path / "cert"
    cases_dir.mkdir(parents=True, exist_ok=True)

    registry = {
        "schema": "scorpio_pipe.long_parity_registry",
        "schema_version": 1,
        "minimum_real_cases_required": 1,
        "entries": [
            {
                "case_id": "pending_case",
                "status": "pending-private-data",
                "manifest": None,
                "category": "reference",
            }
        ],
    }
    registry_path = cases_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    args = Namespace(
        registry=str(registry_path),
        reports_root=str(reports_dir),
        golden_compare=None,
        known_limitations=None,
        observer_signoff=None,
        output=str(out_dir),
        fail_on_failures=False,
        fail_on_not_ready=False,
    )

    rc = cmd_certify_long_parity(args)
    assert rc == 0
    payload = json.loads((out_dir / "long_parity_certification.json").read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "INCOMPLETE"
    assert payload["summary"]["blocking_requirements"]
