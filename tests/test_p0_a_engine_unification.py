from __future__ import annotations

import os
import runpy
from pathlib import Path

import yaml


def test_workflow_dodo_is_engine_adapter() -> None:
    dodo_p = Path(__file__).resolve().parents[1] / "workflow" / "dodo.py"
    src = dodo_p.read_text(encoding="utf-8")

    assert "scorpio_pipe.pipeline.engine" in src
    # No alternative execution semantics: dodo must not call stages directly.
    assert "scorpio_pipe.stages" not in src



def test_workflow_manifest_task_delegates_to_engine(tmp_path: Path, monkeypatch) -> None:
    # Create a workspace path that satisfies strict run_validate rules (used only to build targets).
    run_root = tmp_path / "workspace" / "31_12_2025" / "testobj_VPHG1200@540_01"

    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "work_dir": str(run_root),
        "config_dir": str(tmp_path),
        "frames": {},
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    monkeypatch.setenv("CONFIG", str(cfg_path))

    dodo_p = Path(__file__).resolve().parents[1] / "workflow" / "dodo.py"
    ns = runpy.run_path(str(dodo_p))

    calls: list[dict[str, object]] = []

    def fake_run_sequence(cfg_or_path, task_names, *, resume=True, force=False, qc_override=False, config_path=None, cancel_token=None):
        calls.append(
            {
                "cfg_or_path": cfg_or_path,
                "task_names": list(task_names),
                "resume": resume,
                "force": force,
                "qc_override": qc_override,
                "config_path": config_path,
                "cancel_token": cancel_token,
            }
        )
        return {"status": "ok"}

    # Patch the adapter's bound reference, to avoid executing heavy stages in unit tests.
    # (runpy returns a globals dict, but patching __globals__ is the most robust.)
    ns["_engine_run"].__globals__["run_sequence"] = fake_run_sequence

    task = ns["task_manifest"]()
    assert str(task["targets"][0]).endswith("/manifest/done.json")

    for act in task["actions"]:
        act()

    assert len(calls) == 1
    assert calls[0]["task_names"] == ["manifest"]
    assert calls[0]["config_path"] == cfg_path


def test_stage_specs_cover_canonical_tasks() -> None:
    from scorpio_pipe.pipeline.engine import CANONICAL_TASKS, STAGE_SPECS

    # P0-A7: StageSpec coverage must be 100% for canonical tasks.
    assert set(CANONICAL_TASKS) == set(STAGE_SPECS), (
        "STAGE_SPECS must match CANONICAL_TASKS exactly (no missing/extra tasks)"
    )

    # For every canonical task, the execution contract must declare outputs.
    for t in CANONICAL_TASKS:
        spec = STAGE_SPECS[t]
        assert tuple(spec.validate_globs), f"StageSpec.validate_globs must be non-empty for {t!r}"

        kind = str(getattr(spec, "contract_kind", "none") or "none").strip().lower()
        assert kind in {"none", "mef", "spec1d", "lambda_map"}
