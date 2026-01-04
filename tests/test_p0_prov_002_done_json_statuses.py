import json

from scorpio_pipe.io.done_json import write_done_json


def test_done_json_accepts_warn_and_metrics(tmp_path):
    d = tmp_path / "stage"
    payload = write_done_json(
        stage_dir=d,
        stage="sky",
        status="warn",
        metrics={"a": 1, "b": 2},
        qc={"max_severity": "WARN"},
        extra={"note": "ok"},
    )

    p = d / "done.json"
    assert p.exists()
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["status"] == "warn"
    assert on_disk["metrics"]["a"] == 1
    assert on_disk["qc"]["max_severity"] == "WARN"
    assert on_disk["extra"]["note"] == "ok"
    assert payload["status"] == "warn"


def test_done_json_normalizes_skip_alias(tmp_path):
    d = tmp_path / "stage"
    write_done_json(stage_dir=d, stage="manifest", status="skip")
    on_disk = json.loads((d / "done.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "skipped"
