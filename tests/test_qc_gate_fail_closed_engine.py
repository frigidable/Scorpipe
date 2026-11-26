"""Engine must never execute a stage after QC checker failure.

A corrupt upstream QC report, inaccessible file or a bug in the checker is
not equivalent to a clean scientific state. Test through real run_sequence
preflight/planning on a reproducible Golden fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scorpio_pipe.pipeline import engine
from scorpio_pipe.qc import gate
from tests.golden.mini_night import make_mini_night


def test_engine_fails_closed_on_unexpected_qc_checker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mn = make_mini_night(tmp_path)
    executed: list[str] = []

    def broken_check(*args, **kwargs) -> None:
        raise OSError("Synthetic QC report I/O failure")

    def fake_callable(task: str):
        def fake_run(*args, **kwargs) -> None:
            executed.append(task)
        return fake_run

    monkeypatch.setattr(gate, "check_qc_gate", broken_check)
    monkeypatch.setattr(engine, "task_callable", fake_callable)

    with pytest.raises(RuntimeError, match="QC gate evaluation failed") as err:
        engine.run_sequence(
            mn.cfg_path, ["manifest"], resume=False, force=True,
            qc_override=False, preflight_override=False,
            config_path=mn.cfg_path,
        )

    assert "Synthetic QC report I/O failure" in str(err.value)
    assert isinstance(err.value.__cause__, OSError)
    assert executed == [], "No stage may execute after a QC checker failure"

    done = mn.work_dir / "manifest" / "done.json"
    assert done.is_file(), "The blocked result must be auditable"
    report = json.loads(done.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "QC gate evaluation failed" in json.dumps(report)
