"""Source SHA inventory only: no sky measurement, no raw FITS reads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

FILE = Path(__file__).resolve().parents[1] / (
    "scripts/qa/sc1_sky_only_evidence_archive_inventory.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("sc1_sky_archive_inventory", FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources(tmp_path: Path) -> Path:
    root = tmp_path / "original_archive"
    d = root / "Scorpipe_LineID26d_multi_band"
    e = root / "Scorpipe_LineID26e_acquisition"
    f = root / "Scorpipe_LineID26f_offset"
    for folder in (d, e, f):
        folder.mkdir(parents=True)
    (d / "lineid26d_report.json").write_text(
        '{"status":"REVIEW_REQUIRED","sky_only_approved":false}',
        encoding="utf-8",
    )
    (e / "lineid26e_matches.csv").write_text(
        "point_a,point_b\n1,2\n", encoding="utf-8",
    )
    with zipfile.ZipFile(f / "lineid26f_native.npz", "w") as z:
        z.writestr("raw_counts.npy", b"synthetic metadata only")
    (d / "raw_s22800607.fts").write_bytes(b"Do not read or copy this FITS")
    (root / "unrelated.json").write_text("{}", encoding="utf-8")
    return root


def test_archives_hashed_and_annotated_without_sky_approval(tmp_path: Path) -> None:
    mod = _module()
    root = _sources(tmp_path)
    original = (root / "Scorpipe_LineID26d_multi_band" / "lineid26d_report.json")
    checksum = hashlib.sha256(original.read_bytes()).hexdigest()
    output = tmp_path / "independent_qa" / "inventory.json"
    result = mod.inventory([root], output)

    assert result["files_found"] == 3
    assert result["phases_found"] == ["26d", "26e", "26f"]
    assert result["phases_missing"] == ["26c"]
    assert result["science_hold"] is True
    assert result["source_free_sky_mask_approved"] is False
    assert result["certified_sky_rows"] == 0
    assert result["stages_executed"] is False
    assert result["original_fits_read"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["files_found"] == 3
    assert hashlib.sha256(original.read_bytes()).hexdigest() == checksum

    d = next(f for f in result["files"] if f["phase"] == "26d")
    assert d["sha256"] == checksum
    assert d["structure"]["json_root_keys"] == ["sky_only_approved", "status"]
    e = next(f for f in result["files"] if f["phase"] == "26e")
    assert e["structure"]["csv_first_record"] == ["point_a", "point_b"]
    f = next(f for f in result["files"] if f["phase"] == "26f")
    assert f["structure"]["archive_member_names"] == ["raw_counts.npy"]
    assert all(not item["path"].endswith(".fts") for item in result["files"])


def test_output_inside_source_root_is_rejected(tmp_path: Path) -> None:
    mod = _module()
    root = _sources(tmp_path)
    output = root / "lineid26d_inventory.json"
    with pytest.raises(ValueError, match="outside"):
        mod.inventory([root], output)
    assert not output.exists()


def test_existing_external_report_cannot_be_replaced(tmp_path: Path) -> None:
    mod = _module()
    root = _sources(tmp_path)
    output = tmp_path / "already.json"
    output.write_text("original untouched", encoding="utf-8")
    with pytest.raises(FileExistsError):
        mod.inventory([root], output)
    assert output.read_text(encoding="utf-8") == "original untouched"


def test_missing_or_unrelated_files_do_not_imply_approval(tmp_path: Path) -> None:
    mod = _module()
    root = tmp_path / "empty"
    root.mkdir()
    (root / "unrelated.json").write_text('{"SKY_ONLY":true}', encoding="utf-8")
    output = tmp_path / "report.json"
    report = mod.inventory([root], output)
    assert report["files_found"] == 0
    assert report["phases_missing"] == ["26c", "26d", "26e", "26f"]
    assert report["source_free_sky_mask_approved"] is False


def test_large_artifact_is_marked_unhashed(tmp_path: Path, monkeypatch) -> None:
    mod = _module()
    root = tmp_path / "source"
    root.mkdir()
    p = root / "lineid26d_export.zip"
    p.write_bytes(b"x" * 64)
    monkeypatch.setattr(mod, "MAX_SINGLE_BYTES", 20)
    output = tmp_path / "report.json"
    report = mod.inventory([root], output)
    assert report["files_found"] == 0
    assert len(report["files_skipped_large"]) == 1
    assert report["files_skipped_large"][0]["reason"].endswith("NO_SHA")
    assert report["source_free_sky_mask_approved"] is False
