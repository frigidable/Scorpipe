from __future__ import annotations

from pathlib import Path

from scorpio_pipe.refs.context import _extract_reference_metadata


def test_reference_metadata_carries_interpolation_and_unit_family(tmp_path: Path) -> None:
    p = tmp_path / "std.csv"
    p.write_text("# standard_name: BD+28 4211\n# version: 1\n# wavelength_convention: air\n# units: f_lambda\n", encoding="utf-8")
    meta = _extract_reference_metadata("standards.bd_28_4211.table", p)
    assert meta["reference_kind"] == "standard_table"
    assert meta["unit_family"] == "flux_density"
    assert meta["interpolation_assumption"] == "linear_in_wavelength"
    assert meta["checksum_algorithm"] == "sha256"
