"""A08: publication products must never imply detector/rectified alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.stages.publication_ready import (
    _validate_export_coordinate_provenance, run_publication_ready,
)
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.workspace_paths import stage_dir


def _fixtures(
    tmp_path: Path, *, shape_map=(11, 21), shape_stack=(11, 17),
    map_ref="air", lsf_ref="air", stack_ref="air",
):
    work = tmp_path / "work"
    wave_dir = wavesol_dir({"work_dir": str(work)}, run_root=work)
    wave_dir.mkdir(parents=True, exist_ok=True)
    stackdir = stage_dir(work, "stack2d")
    stackdir.mkdir(parents=True, exist_ok=True)
    lsfd = work / "lsf_model"
    lsfd.mkdir(parents=True, exist_ok=True)

    raw_wave = (
        5000.0
        + np.arange(shape_map[1])[None, :] * 0.9
        + np.arange(shape_map[0])[:, None] * 0.05
    ).astype(np.float32)
    raw_wave[:, 0] = np.nan
    lsf = np.full_like(raw_wave, 1.3)
    head = fits.Header({"WAVEUNIT": "Angstrom", "WAVEREF": map_ref})
    fits.PrimaryHDU(data=raw_wave, header=head).writeto(wave_dir / "lambda_map.fits")
    lsf_head = head.copy()
    lsf_head["WAVEREF"] = lsf_ref
    fits.PrimaryHDU(data=lsf, header=lsf_head).writeto(lsfd / "lsf_map.fits")

    grid = WaveGrid(5001.0, 1.0, shape_stack[1], unit="Angstrom", wave_ref=stack_ref)
    image = np.arange(np.prod(shape_stack), dtype=np.float32).reshape(shape_stack)
    write_sci_var_mask(
        stackdir / "stack2d.fits",
        image,
        var=np.ones_like(image),
        mask=np.zeros_like(image, dtype=np.uint16),
        header=fits.Header({"BUNIT": "adu/s"}),
        grid=grid,
    )
    return {"work_dir": str(work)}, image, raw_wave, lsf


@pytest.mark.parametrize("stack_shape", [(11, 17), (11, 21)])
def test_export_geom_is_distinct_even_when_detector_shapes_match(tmp_path, stack_shape):
    cfg, src, wl, lsf = _fixtures(tmp_path, shape_stack=stack_shape)
    result = run_publication_ready(cfg)
    align = result["pixel_alignment"]
    assert align["direct_pixelwise_comparison_allowed"] is False
    assert align["spectrum_detector_maps_coregistered"] is False
    assert align["same_shape_does_not_imply_alignment"] is True
    assert align["wl_dismap_coregistered"] is None
    assert align["detector_map_alignment_qualification"] == "NOT_INDEPENDENTLY_VERIFIED"
    assert result["coordinate_provenance"]["coordinate_declarations"] == "COMPLETE"
    assert result["science_qc_pass"] is False
    assert result["publication_ready"] is False
    work = Path(cfg["work_dir"])
    out = work / "publication_ready"
    for name, geom in [
        ("Spectrum.fts", "RECTIFIED"),
        ("wl.fts", "DETECTOR"),
        ("dismap.fts", "DETECTOR"),
    ]:
        with fits.open(out / name, memmap=False) as hdul:
            for hdu in (hdul[0], hdul["SCI"]):
                assert hdu.header["PIXGEOM"] == geom
                assert hdu.header["MAPALGN"] is False
                assert hdu.header["EXPORTOK"] is True
                assert hdu.header["PUBRDY"] is False
                assert hdu.header["SCIQPASS"] is False
    with fits.open(out / "Spectrum.fts", memmap=False) as hdul:
        np.testing.assert_array_equal(hdul["SCI"].data, src)
    with fits.open(out / "wl.fts", memmap=False) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, wl, equal_nan=True)
    with fits.open(out / "dismap.fts", memmap=False) as hdul:
        dis = hdul["SCI"].data
        assert np.isnan(dis[:, 0]).all()
        np.testing.assert_allclose(dis[:, 1:], lsf[:, 1:])
    manifest = json.loads((out / "publication_provenance.json").read_text("utf-8"))
    assert manifest["geometry_contract_version"] == 1
    assert manifest["coordinate_provenance"]["wave_ref"] == "air"
    assert manifest["pixel_alignment"] == align


@pytest.mark.parametrize("refs", [
    ("vacuum", "air", "air"),
    ("air", "vacuum", "air"),
    ("air", "air", "vacuum"),
])
def test_explicit_medium_disagreement_refused(tmp_path, refs):
    cfg, *_ = _fixtures(
        tmp_path, map_ref=refs[0], lsf_ref=refs[1], stack_ref=refs[2]
    )
    with pytest.raises(ValueError, match="disagree on air/vacuum"):
        run_publication_ready(cfg)


def test_incompatible_unit_or_internal_conflict_fails_closed():
    h1 = fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
    h2 = fits.Header({"WAVEREF": "air", "WAVEUNIT": "nm"})
    h3 = fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
    with pytest.raises(ValueError, match="disagree on wavelength unit"):
        _validate_export_coordinate_provenance(h1, h2, h3)
    h2["WAVEUNIT"] = "meter"
    with pytest.raises(ValueError, match="unsupported wavelength unit"):
        _validate_export_coordinate_provenance(h1, h2, h3)
    h2["WAVEUNIT"] = "Angstrom"
    h2["CUNIT1"] = "nm"
    with pytest.raises(ValueError, match="inconsistent WAVEUNIT and CUNIT1"):
        _validate_export_coordinate_provenance(h1, h2, h3)


def test_missing_metadata_is_explicitly_unqualified():
    h1 = fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
    h2 = fits.Header()
    h3 = fits.Header({"WAVEREF": "air", "WAVEUNIT": "Angstrom"})
    result = _validate_export_coordinate_provenance(h1, h2, h3)
    assert result["coordinate_declarations"] == "INCOMPLETE"
    assert result["wave_ref"] == "air"
    assert result["wave_unit"] == "Angstrom"
