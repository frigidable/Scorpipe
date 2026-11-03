"""Navigator reads the *canonical* QC report namespace and stage severity.

Smoke historically compared the two representations, but Navigator looked for
root flags even though qc_report builds qc.flags; this made a scientifically
WARN run falsely appear as 0 WARN / OK in the UI. A completed stage can also
have nonfatal WARN flags despite done.status='ok'.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scorpio_pipe.navigator import (
    _report_flags,
    _stage_flags,
    _status_from_done,
    build_navigator,
)
from scorpio_pipe.workspace_paths import stage_dir


_FLAGS = [
    {
        "code": "FLEXURE_UNVERIFIED",
        "severity": "WARN",
        "message": "No independent sky-line template",
    },
    {
        "code": "SKY_MODEL_VAR_UNVERIFIED",
        "severity": "WARN",
        "message": "Sky-model covariance not estimated",
    },
]


def test_canonical_nested_qc_takes_precedence_over_stale_root_flags() -> None:
    report = {
        "qc": {"flags": _FLAGS},
        "flags": [{"code": "STALE", "severity": "ERROR"}],
    }
    assert _report_flags(report) == _FLAGS
    assert _stage_flags(report) == _FLAGS
    assert _report_flags({"flags": _FLAGS}) == _FLAGS
    assert _report_flags({"qc": {"flags": []}, "flags": _FLAGS}) == []


@pytest.mark.parametrize(
    ("status", "flags", "expected"),
    [
        ("ok", _FLAGS, "DONE_WARN"),
        ("ok", [{"code": "FAIL", "severity": "ERROR"}], "DONE_FAIL"),
        ("warn", [], "DONE_WARN"),
        ("error", [], "DONE_FAIL"),
        ("blocked", [], "DONE_FAIL"),
        ("ok", [], "DONE_OK"),
    ],
)
def test_stage_status_keeps_science_qc_severity(
    status: str, flags: list[dict], expected: str,
) -> None:
    assert _status_from_done({
        "status": status,
        "qc": {"max_severity": "WARN" if flags == _FLAGS else "OK", "flags": flags},
    }) == expected


def test_navigator_report_summary_and_stage_status_match_science_warn(
    tmp_path: Path,
) -> None:
    work = tmp_path / "run"
    report = work / "qc" / "qc_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({
            "qc": {"max_severity": "WARN", "flags": _FLAGS},
            "flags": [],  # Simulate historical misleading root key.
        }),
        encoding="utf-8",
    )
    sky = stage_dir(work, "sky")
    sky.mkdir(parents=True)
    (sky / "done.json").write_text(
        json.dumps({
            "status": "ok",
            "qc": {"max_severity": "WARN", "flags": _FLAGS},
        }),
        encoding="utf-8",
    )

    page = build_navigator(work)
    nav = json.loads((page.parent / "data.json").read_text(encoding="utf-8"))
    summary = nav["qc_summary"]
    assert summary["max_severity"] == "WARN"
    assert summary["n_warn"] == 2
    assert summary["n_error"] == 0
    assert {f["code"] for f in summary["flags"]} == {
        "FLEXURE_UNVERIFIED", "SKY_MODEL_VAR_UNVERIFIED",
    }
    sky_stage = next(s for s in nav["stages"] if s["key"] == "sky")
    assert sky_stage["status"] == "DONE_WARN"
    assert sky_stage["qc"]["max_severity"] == "WARN"
    assert sky_stage["qc"]["n_warn"] == 2


def test_bad_or_missing_qc_shape_does_not_crash_navigator_helpers() -> None:
    assert _report_flags({"qc": None}) == []
    assert _report_flags({"qc": {"flags": None}}) == []
    assert _stage_flags({"qc": None}) == []
    assert _status_from_done({"status": "ok", "qc": None}) == "DONE_OK"
