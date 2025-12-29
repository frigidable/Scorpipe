from __future__ import annotations

import json
from pathlib import Path


def test_products_registry_uses_disperser_subdir(tmp_path: Path) -> None:
    from scorpio_pipe.products import list_products

    cfg = {
        "work_dir": str(tmp_path),
        "wavesol": {"disperser": "VPHG1200@550"},
        "frames": {},
    }

    prods = list_products(cfg)
    # superneon should live under work_dir/wavesol/<slug>/
    p = next(x for x in prods if x.key == "superneon_png")
    assert "wavesol" in str(p.path)
    assert "VPHG1200_550" in str(p.path)


def test_qc_report_builds_with_minimal_inputs(tmp_path: Path) -> None:
    from scorpio_pipe.qc_report import build_qc_report

    cfg = {
        "work_dir": str(tmp_path),
        "wavesol": {"disperser": "default"},
        "frames": {},
    }

    out = build_qc_report(cfg)
    assert out["html"].exists()
    assert out["json"].exists()

    js = json.loads(out["json"].read_text(encoding="utf-8"))
    assert "products" in js and isinstance(js["products"], list)


def test_timings_append(tmp_path: Path) -> None:
    from scorpio_pipe.timings import append_timing, timings_file

    append_timing(work_dir=tmp_path, stage="demo", seconds=1.23)
    p = timings_file(work_dir=tmp_path)
    assert p.exists()
