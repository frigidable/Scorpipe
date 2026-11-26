"""Engine QC-report entrypoint must call the builder with its real API.

QCReportOutput uses mapping keys "html"/"json"; older entrypoint code passed
an unsupported run_root= kwarg and incorrectly read the nonexistent "qc_html".
"""
from __future__ import annotations

import json
from pathlib import Path

from scorpio_pipe.pipeline.context import StageContext
from scorpio_pipe.pipeline.entry_qc_report import run
from scorpio_pipe.qc_report import QCReportOutput


def test_qc_entry_creates_real_html_and_json_in_canonical_run(tmp_path: Path) -> None:
    root = tmp_path / "16_12_2025" / "mini_night_VPHG1200@540_01"
    ctx = StageContext(
        cfg={"work_dir": str(root), "wavesol": {"disperser": "VPHG1200@540"}, "frames": {}},
        run_root=root,
        stage_key="qc_report",
        stage_dir=root / "qc",
    )
    result = run(ctx)
    assert result.status == "ok"
    html = root / "index.html"
    report = root / "qc" / "qc_report.json"
    assert html.exists()
    assert report.exists()
    assert (root / "manifest" / "qc_report.json").exists()
    assert str(html.resolve()) in result.artifacts
    assert str(report.resolve()) in result.artifacts
    assert result.products["qc_html"] == str(html.resolve())
    assert result.products["qc_json"] == str(report.resolve())
    assert "products" in json.loads(report.read_text(encoding="utf-8"))


def test_qc_entry_obeys_engine_root_not_stale_cfg_root(
    tmp_path: Path, monkeypatch,
) -> None:
    from scorpio_pipe import qc_report

    root = tmp_path / "actual_run"
    stale = tmp_path / "wrong_run"
    captured = {}

    def fake_builder(cfg: dict, *, out_dir: str | Path | None = None):
        captured["cfg"] = dict(cfg)
        captured["out_dir"] = Path(out_dir)
        qc_dir = Path(out_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
        html = Path(cfg["work_dir"]) / "index.html"
        html.write_text("QC HTML", encoding="utf-8")
        out_json = qc_dir / "qc_report.json"
        out_json.write_text('{"products":[]}', encoding="utf-8")
        return QCReportOutput(json=out_json, html=html)

    monkeypatch.setattr(qc_report, "build_qc_report", fake_builder)
    ctx = StageContext(
        cfg={"work_dir": str(stale), "frames": {}},
        run_root=root,
        stage_key="qc_report",
        stage_dir=root / "qc",
    )

    result = run(ctx)
    assert captured["cfg"]["work_dir"] == str(root.resolve())
    assert captured["out_dir"] == root.resolve() / "qc"
    assert not stale.exists()
    assert result.artifacts == [
        str(root.resolve() / "index.html"),
        str(root.resolve() / "qc" / "qc_report.json"),
    ]
