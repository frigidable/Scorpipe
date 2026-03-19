from __future__ import annotations

import json
import zipfile
from pathlib import Path


def test_product_catalog_covers_all_stage_product_keys() -> None:
    from scorpio_pipe.catalog.stages import CATALOG as STAGE_CATALOG
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

    declared: set[str] = set()
    for t in STAGE_CATALOG.task_infos():
        declared.update({str(k) for k in (t.product_keys or ())})

    assert declared.issubset(set(PRODUCT_CATALOG.keys())), sorted(list(declared - set(PRODUCT_CATALOG.keys())))


def test_export_package_uses_product_catalog_and_wavesol_subdir(tmp_path: Path) -> None:
    """Regression: export must find disperser-specific wavesol products."""

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    # Minimal config that selects a disperser.
    cfg_yaml = (
        "work_dir: .\n"
        "frames:\n"
        "  __setup__:\n"
        "    disperser: VPHG1200B\n"
    )
    (work_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")

    # Create the minimal workspace layout expected by tools.
    (work_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (work_dir / "qc").mkdir(parents=True, exist_ok=True)
    (work_dir / "ui" / "navigator").mkdir(parents=True, exist_ok=True)

    # Place a wavesol product into a disperser subdir only.
    wsol_sub = work_dir / "08_wavesol" / "VPHG1200B"
    wsol_sub.mkdir(parents=True, exist_ok=True)
    (wsol_sub / "lambda_map.fits").write_bytes(b"FAKE")

    # Minimal QC + manifest files so the export has something to include.
    (work_dir / "qc" / "qc_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (work_dir / "manifest" / "manifest.json").write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    (work_dir / "manifest" / "products_manifest.json").write_text(json.dumps({"schema": "y"}), encoding="utf-8")
    (work_dir / "ui" / "navigator" / "index.html").write_text("<html></html>", encoding="utf-8")

    from scorpio_pipe.export_package import export_science_package

    out_zip = export_science_package(work_dir, overwrite=True)
    assert out_zip.exists()

    with zipfile.ZipFile(out_zip, "r") as zf:
        names = set(zf.namelist())

    # Must include disperser-subdir lambda_map (this fails if paths are hard-coded).
    assert "08_wavesol/VPHG1200B/lambda_map.fits" in names
    # And should include the QC JSON.
    assert "qc/qc_report.json" in names


def test_product_catalog_registers_calibration_domain_products() -> None:
    from scorpio_pipe.catalog.products import CATALOG

    expected = {
        "standard_1d_fits",
        "sensfunc_table_fits",
        "telluric_model_fits",
        "fluxcal_spec1d_fits",
        "fluxcal_2d_fits",
        "science_lines_json",
        "science_report_html",
    }

    assert expected.issubset(set(CATALOG.keys()))
