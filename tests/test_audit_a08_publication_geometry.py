"""A08: publication FITS images must not imply detector/rectified alignment."""
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


def _fixtures(tmp_path: Path, *, shape_map=(11, 21), shape_stack=(11, 17),
              map_ref="air", lsf_ref="air", stack_ref="air"):
    work = tmp_path / "work"
    wave_dir = wavesol_dir({"work_dir": str(work)}, run_root=work)
    wave_dir.mkdir(parents=True, exist_ok=True)
    stackdir = stage_dir(work, "stack2d")
    stackdir.mkdir(parents=True, exist_ok=True)
    lsfd = work / "lsf_model"
    lsfd.mkdir(parents=True, exist_ok=True)
    raw_wave = 5000. + np.arange(shape_map[1])[None, :] * .9 + np.arange(shape_map[0])[:, None] * .05
    raw_wave = raw_wave.astype(np.float32)
    raw_wave[:, 0] = np.nan
    lsf = np.ones_like(raw_wave, dtype=np.float32) * 1.3
    wave_head = fits.Header()
    wave_head["WAVEUNIT"] = "Angstrom"
    wave_head["WAVEREF"] = map_ref
    fits.PrimaryHDU(data=raw_wave, header=wave_head).writeto(wave_dir / "lambda_map.fits")
    lsf_head = wave_head.copy()
    lsf_head["WAVEREF"] = lsf_ref
    fits.PrimaryHDU(data=lsf, header=lsf_head).writeto(lsfd / "lsf_map.fits")
    grid = WaveGrid(5001., 1., shape_stack[1], unit="Angstrom", wave_ref=stack_ref)
    data = np.arange(np.prod(shape_stack), dtype=np.float32).reshape(shape_stack)
    shdr = fits.Header()
    shdr["BUNIT"] = "adu/s"
    write_sci_var_mask(stackdir / "stack2d.fits", data, var=np.ones_like(data),
                       mask=np.zeros_like(data, dtype=np.uint16), header=shdr, grid=grid)
    return {"work_dir": str(work)}, data, raw_wave, lsf


@pytest.mark.parametrize("stack_shape", [(11, 17), (11, 21)])
def test_export_marks_distinct_geometries_even_when_shapes_match(tmp_path, stack_shape):
    cfg, source, wl, lsf = _fixtures(tmp_path, shape_stack=stack_shape)
    result = run_publication_ready(cfg)
    align = result["pixel_alignment"]
    assert align["direct_pixelwise_comparison_allowed"] is False
    assert align["spectrum_detector_maps_coregistered"] is False
    assert align["same_shape_does_not_imply_alignment"] is True
    assert align["wl_dismap_coregistered"] is True
    path = Path(cfg["work_dir"]) / "publication_ready"
    for name, frame in [
        ("Spectrum.fts", "RECTIFIED"),
        ("wl.fts", "DETECTOR"),
        ("dismap.fts", "DETECTOR"),
    ]:
        with fits.open(path / name) as h:
            assert h[0].header["PIXGEOM"] == frame
            assert h["SCI"].header["PIXGEOM"] == frame
            assert h[0].header["MAPALGN"] is False
            assert h["SCI"].header["MAPALGN"] is False
    with fits.open(path / "Spectrum.fts") as h:
        np.testing.assert_array_equal(h["SCI"].data, source)
    with fits.open(path / "wl.fts") as h:
        np.testing.assert_allclose(h["SCI"].data, wl, equal_nan=True)
    with fits.open(path / "dismap.fts") as h:
        d = h["SCI"].data
        assert np.isnan(d[:, 0]).all()
        np.testing.assert_allclose(d[:, 1:], lsf[:, 1:])
    sidecar = json.loads((path / "publication_provenance.json").read_text())
    assert sidecar["geometry_contract_version"] == 1
    assert sidecar["coordinate_provenance"]["wave_ref"] == "air"


def test_conflicting_detector_and_stack_medium_fails_closed(tmp_path):
    cfg, *_ = _fixtures(tmp_path, map_ref="vacuum", stack_ref="air")
    with pytest.raises(ValueError, match="disagree on air/vacuum"):
        run_publication_ready(cfg)


def test_conflicting_detector_lsf_medium_fails_closed(tmp_path):
    cfg, *_ = _fixtures(tmp_path, map_ref="air", lsf_ref="vacuum")
    with pytest.raises(ValueError, match="disagree on air/vacuum"):
        run_publication_ready(cfg)


def test_invalid_export_wavelength_unit_fails_closed():
    h1, h2, h3 = fits.Header(), fits.Header(), fits.Header()
    h1["WAVEREF"] = h2["WAVEREF"] = h3["WAVEREF"] = "air"
    h1["WAVEUNIT"] = "Angstrom"
    h2["WAVEUNIT"] = "nm"
    with pytest.raises(ValueError, match="disagree on wavelength unit"):
        _validate_export_coordinate_provenance(h1, h2, h3)
    h2["WAVEUNIT"] = "meter"
    with pytest.raises(ValueError, match="unsupported wavelength unit"):
        _validate_export_coordinate_provenance(h1, h2, h3)
