"""QC gate must reveal real upstream blockers, including through context managers."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from scorpio_pipe.qc import gate


def _blocker() -> dict[str, str]:
    return {
        "stage": "linearize",
        "code": "SCIENCE_QC",
        "severity": "ERROR",
        "message": "Synthetic QC says stop",
    }


@contextmanager
def _context_that_preserves_exceptions():
    yield


def test_qc_gate_error_reraises_through_contextlib_with_original_details() -> None:
    error = gate.QCGateError(
        task="stack2d",
        blockers=[_blocker()],
        upstream_max_severity="ERROR",
    )
    assert "SCIENCE_QC" in str(error)
    assert "stack2d" in str(error)
    with pytest.raises(gate.QCGateError) as caught:
        with _context_that_preserves_exceptions():
            raise error
    assert caught.value is error
    assert caught.value.__traceback__ is not None
    assert caught.value.summary().count("SCIENCE_QC") == 1


def test_gate_still_blocks_error_and_ignores_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = tmp_path / "linearize_done.json"
    monkeypatch.setattr(gate, "_stage_key_order", lambda: ["linearize", "stack2d"])
    monkeypatch.setattr(
        gate, "_find_done_json",
        lambda cfg, sk: done if sk == "linearize" else None,
    )
    warning = {"code": "TEST_WARN", "severity": "WARN", "message": "Note"}
    error = _blocker()

    done.write_text(json.dumps({"qc": {"flags": [warning]}}), encoding="utf-8")
    gate.check_qc_gate({}, task="stack2d")

    done.write_text(json.dumps({"qc": {"flags": [warning, error]}}), encoding="utf-8")
    with pytest.raises(gate.QCGateError, match="SCIENCE_QC") as caught:
        with _context_that_preserves_exceptions():
            gate.check_qc_gate({}, task="stack2d")
    assert len(caught.value.blockers) == 1
    assert caught.value.blockers[0]["stage"] == "linearize"

    # Requesting an override is not sufficient without explicit configuration.
    with pytest.raises(gate.QCGateError):
        gate.check_qc_gate({}, task="stack2d", allow_override=True)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("{malformed-json", "Cannot read upstream QC report"),
        ("[]", "not a JSON object"),
        ('{"qc": "not-an-object"}', "QC field is not a JSON object"),
        ('{"qc": {"flags": "ERROR"}}', "QC flags are not a list"),
        ('{"qc": {"flags": [null]}}', "QC flags are not a list"),
        ('{"qc": {"flags": [{"code": "UNKNOWN"}]}}', "lacks a valid severity"),
        (
            '{"qc": {"flags": [{"code": "UNKNOWN", "severity": "BOGUS"}]}}',
            "Unrecognized upstream QC flag severity",
        ),
    ],
)
def test_gate_rejects_corrupt_upstream_qc_instead_of_assuming_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    expected: str,
) -> None:
    done = tmp_path / "linearize_done.json"
    done.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(gate, "_stage_key_order", lambda: ["linearize", "stack2d"])
    monkeypatch.setattr(
        gate,
        "_find_done_json",
        lambda cfg, sk: done if sk == "linearize" else None,
    )
    with pytest.raises(ValueError, match=expected):
        gate.check_qc_gate({}, task="stack2d")


def test_gate_allows_valid_upstream_done_without_qc_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = tmp_path / "linearize_done.json"
    done.write_text('{"status": "ok"}', encoding="utf-8")
    monkeypatch.setattr(gate, "_stage_key_order", lambda: ["linearize", "stack2d"])
    monkeypatch.setattr(
        gate,
        "_find_done_json",
        lambda cfg, sk: done if sk == "linearize" else None,
    )
    gate.check_qc_gate({}, task="stack2d")
