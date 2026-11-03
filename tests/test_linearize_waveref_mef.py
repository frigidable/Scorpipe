"""Linearize must serialize the wavelength medium in every output MEF.

Previously the output header declared WAVEREF=air, but the optional WaveGrid
writer argument defaulted to UNKNOWN and overwrote it at serialization time.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages import linearize


@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_wavelength_mef_writer_preserves_header_reference(
    tmp_path: Path, medium: str,
) -> None:
    path = tmp_path / "rectified.fits"
    header = fits.Header({"WAVEREF": medium, "BUNIT": "e-/s"})
    sci = np.ones((8, 16), dtype=np.float32)
    var = np.ones_like(sci)
    mask = np.zeros_like(sci, dtype=np.uint16)

    # Omitted writer keyword must use the real header reference, not UNKNOWN.
    linearize._write_mef(
        path, sci, header, var, mask, None,
        wave0=5000.0, dw=2.0, unit="Angstrom",
    )
    with fits.open(path, memmap=False) as hdul:
        for hdu in (hdul[0], hdul["SCI"], hdul["VAR"], hdul["MASK"]):
            assert str(hdu.header["WAVEREF"]).lower() == medium
        assert hdul[0].header["CTYPE1"] == (
            "AWAV" if medium == "air" else "WAVE"
        )


def test_wavelength_mef_writer_rejects_conflicting_references(tmp_path: Path) -> None:
    path = tmp_path / "wrong.fits"
    sci = np.ones((8, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="Conflicting wavelength reference"):
        linearize._write_mef(
            path, sci, fits.Header({"WAVEREF": "air"}),
            sci, np.zeros_like(sci, dtype=np.uint16), None,
            wave0=5000.0, dw=1.0, wave_ref="vacuum",
        )
    assert not path.exists()


@pytest.mark.parametrize("ref", [None, "", "UNKNOWN"])
def test_wavelength_mef_writer_rejects_undefined_reference(
    tmp_path: Path, ref: str | None,
) -> None:
    path = tmp_path / "unlabelled.fits"
    header = fits.Header()
    if ref is not None:
        header["WAVEREF"] = ref
    sci = np.ones((8, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="WAVEREF=air or vacuum"):
        linearize._write_mef(
            path, sci, header, sci, np.zeros_like(sci, dtype=np.uint16),
            None, wave0=5000.0, dw=1.0,
        )
    assert not path.exists()


def test_every_linearize_stage_mef_writer_passes_wavelength_medium() -> None:
    """Guard against a new stage call silently using the writer default."""
    tree = ast.parse(Path(linearize.__file__).read_text(encoding="utf-8"))
    stage = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_linearize"
    )
    calls = [
        node for node in ast.walk(stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_mef"
    ]
    assert len(calls) == 4
    for call in calls:
        keywords = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert {"wave0", "dw", "unit", "wave_ref"} <= keywords, (
            f"_write_mef at line {call.lineno} lacks wavelength metadata"
        )
