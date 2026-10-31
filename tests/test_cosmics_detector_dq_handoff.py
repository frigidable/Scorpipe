"""Raw ADC saturation survives every cosmics route and DQ handoff.

All synthetic inputs are explicitly RAW ADU.  No detector threshold is
evaluated on bias-subtracted or replaced pixels.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from scorpio_pipe.maskbits import COSMIC, SATURATED
from scorpio_pipe.stages.cosmics import (
    _apply_manual_masks,
    _two_frame_diff_clean,
    clean_cosmics,
)
from scorpio_pipe.stages.flatfield import _read_sci_var_mask
from scorpio_pipe.stages.sky_sub import (
    _prepare_sky_input_planes,
    _read_sky_input_planes,
    _try_load_mask_for_clean,
)


def _raw_frame(path, *, shape=(24, 24), saturated=True):
    img = np.full(shape, 20.0, dtype=np.float32)
    if saturated:
        img[8, 8] = 600.0
    header = fits.Header({"BUNIT": "ADU", "GAIN": 2.0, "RDNOISE": 3.0})
    fits.PrimaryHDU(data=img, header=header).writeto(path)
    return path


def _cfg(tmp_path, paths, method):
    return {
        "config_dir": str(tmp_path),
        "data_dir": str(tmp_path),
        "work_dir": str(tmp_path / "work"),
        "frames": {"obj": [str(p) for p in paths]},
        "detector": {"saturation_adu": 500.0},
        "cosmics": {
            "method": method, "apply_to": ["obj"], "bias_subtract": False,
            "save_png": False, "save_mask_fits": True, "preserve_manual": True,
            "k": 5.0, "dilate": 0, "stack_dilate": 0,
        },
    }


@pytest.mark.parametrize(
    "method,n",
    [("two_frame_diff", 2), ("stack_mad", 3),
     ("laplacian", 1), ("la_cosmic", 1)],
)
def test_cosmics_qualified_raw_saturation_survives_methods_and_sky(
    tmp_path, method, n,
):
    paths = [_raw_frame(tmp_path / f"raw{i}.fits") for i in range(n)]
    out = tmp_path / "work" / "05_cosmics"
    clean_cosmics(_cfg(tmp_path, paths, method), out_dir=out)
    for p in paths:
        clean = out / "obj" / "clean" / f"{p.stem}_clean.fits"
        mask_path = clean.with_name(clean.stem + "_mask.fits")
        assert clean.exists() and mask_path.exists()
        with fits.open(clean) as hdul:
            hdr = hdul[0].header
            assert hdr["SATQ"] == "RAW_EVAL"
            assert hdr["SATMSK"] == "DQ_V1"
        with fits.open(mask_path) as hdul:
            dq = hdul[0].data
            assert hdul[0].header["SATQ"] == "RAW_EVAL"
            assert hdul[0].header["SATMSK"] == "DQ_V1"
            assert dq.dtype == np.uint16
            assert dq[8, 8] & SATURATED
        # Flatfield consumes primary-only cleaned FITS via typed sidecar.
        sci_ff, _var_ff, dq_ff, h_ff = _read_sci_var_mask(clean)
        assert dq_ff[8, 8] & SATURATED
        assert h_ff["SATQ"] == "RAW_EVAL"
        # Direct sky path must also retain qualification, even without a
        # SCI-extension MASK (because this is a primary-only clean FITS).
        sci, var, in_mask, sky_hdr = _read_sky_input_planes(clean)
        cosmic_dq = _try_load_mask_for_clean(clean, clean_header=sky_hdr)
        assert in_mask is None and var is None
        _, _, output_mask, out_hdr = _prepare_sky_input_planes(
            sci, var, in_mask, cosmic_dq, sky_hdr,
            cfg=_cfg(tmp_path, paths, method), sky_cfg={},
        )
        assert out_hdr["SATQ"] == "RAW_EVAL"
        assert output_mask[8, 8] & SATURATED


def test_two_frame_detector_mask_uses_pre_bias_ADU(tmp_path):
    raw = [_raw_frame(tmp_path / f"b{i}.fits") for i in range(2)]
    sb = np.full((24, 24), 200.0, dtype=np.float32)
    out = tmp_path / "bias_output"
    _two_frame_diff_clean(
        raw, out_dir=out, superbias=sb, bias_subtract=True,
        k=5, save_png=False, save_mask_fits=True, cfg={
            "detector": {"saturation_adu": 500.0},
        },
    )
    mask = fits.getdata(out / "clean" / "b0_clean_mask.fits")
    hdr = fits.getheader(out / "clean" / "b0_clean.fits")
    assert hdr["BIASSUB"] and hdr["SATQ"] == "RAW_EVAL"
    assert mask[8, 8] & SATURATED
    # The science value is 400 ADU after bias subtraction, below threshold.
    assert fits.getdata(out / "clean" / "b0_clean.fits")[8, 8] < 500


def test_manual_mask_edit_keeps_detector_saturation(tmp_path):
    paths = [_raw_frame(tmp_path / f"m{i}.fits") for i in range(2)]
    out = tmp_path / "work" / "05_cosmics"
    clean_cosmics(_cfg(tmp_path, paths, "two_frame_diff"), out_dir=out)
    root = out / "obj"
    manual = np.zeros((24, 24), dtype=np.uint8)
    manual[12, 12] = 1
    fits.writeto(root / "manual_masks_fits" / "m0_manual_mask.fits",
                 manual, overwrite=True)
    _apply_manual_masks(root, replace_r=1, save_png=False)
    mask = fits.getdata(root / "clean" / "m0_clean_mask.fits")
    assert mask[8, 8] & SATURATED
    assert mask[12, 12] & COSMIC
    assert fits.getheader(root / "clean" / "m0_clean.fits")["SATQ"] == "RAW_EVAL"


@pytest.mark.parametrize("corrupt", ["missing", "bad_shape", "bad_marker"])
def test_qualified_cosmics_sidecar_fails_closed(tmp_path, corrupt):
    paths = [_raw_frame(tmp_path / f"x{i}.fits") for i in range(2)]
    out = tmp_path / "work" / "05_cosmics"
    clean_cosmics(_cfg(tmp_path, paths, "two_frame_diff"), out_dir=out)
    clean = out / "obj" / "clean" / "x0_clean.fits"
    sidecar = clean.with_name(clean.stem + "_mask.fits")
    if corrupt == "missing":
        sidecar.unlink()
    elif corrupt == "bad_shape":
        fits.writeto(sidecar, np.zeros((2, 2), dtype=np.uint16), overwrite=True)
    else:
        m = fits.getdata(sidecar)
        fits.writeto(sidecar, m, overwrite=True)
    with pytest.raises(ValueError, match="cosmics DQ|qualified"):
        _read_sci_var_mask(clean)
    with pytest.raises(ValueError, match="cosmics DQ|qualified"):
        _try_load_mask_for_clean(clean, clean_header=fits.getheader(clean))


def test_unqualified_processed_input_never_claims_raw_evaluation(tmp_path):
    raw = [_raw_frame(tmp_path / f"q{i}.fits") for i in range(2)]
    for p in raw:
        with fits.open(p, mode="update") as h:
            h[0].header["BIASSUB"] = True
    out = tmp_path / "work" / "05_cosmics"
    clean_cosmics(_cfg(tmp_path, raw, "two_frame_diff"), out_dir=out)
    hdr = fits.getheader(out / "obj" / "clean" / "q0_clean.fits")
    assert hdr["SATQ"] == "UNKNOWN"
    assert not np.any(
        fits.getdata(out / "obj" / "clean" / "q0_clean_mask.fits") & SATURATED
    )


def test_cosmics_flatfield_sky_saturation_handoff(tmp_path, monkeypatch):
    from scorpio_pipe.stages import flatfield
    from scorpio_pipe.stages.flatfield import apply_flat
    from scorpio_pipe.io.mef import read_sci_var_mask

    paths = [_raw_frame(tmp_path / f"f{i}.fits") for i in range(2)]
    out = tmp_path / "work" / "05_cosmics"
    cfg = _cfg(tmp_path, paths, "two_frame_diff")
    clean_cosmics(cfg, out_dir=out)
    clean = out / "obj" / "clean" / "f0_clean.fits"

    flat = tmp_path / "superflat.fits"
    fits.PrimaryHDU(
        data=np.ones((24, 24), dtype=np.float32),
        header=fits.Header({"BUNIT": "ADU", "GAIN": 2.0, "RDNOISE": 3.0}),
    ).writeto(flat)
    flat_out = tmp_path / "f0_flat.fits"
    # Compatibility belongs to calibration association and is orthogonal to
    # this detector DQ test; run the real flatfield science transformation.
    monkeypatch.setattr(
        flatfield, "ensure_compatible_calib", lambda *a, **k: None,
    )
    apply_flat(clean, flat, flat_out, cfg=cfg, do_bias_subtract=False)
    sci, var, dq, hdr = read_sci_var_mask(flat_out)
    assert hdr["BUNIT"] == "e-"
    assert hdr["SATQ"] == "RAW_EVAL"
    assert hdr["SATMSK"] == "DQ_V1"
    assert dq[8, 8] & SATURATED
    assert var.shape == sci.shape

    sky_sci, sky_var, sky_mask, sky_hdr = _read_sky_input_planes(flat_out)
    _, _, next_dq, next_hdr = _prepare_sky_input_planes(
        sky_sci, sky_var, sky_mask, None, sky_hdr,
        cfg=cfg, sky_cfg={},
    )
    assert next_hdr["SATQ"] == "RAW_EVAL"
    assert next_dq[8, 8] & SATURATED


def test_cosmics_detector_mask_is_cropped_with_science(tmp_path):
    a = _raw_frame(tmp_path / "ca.fits", shape=(24, 24))
    b = _raw_frame(tmp_path / "cb.fits", shape=(20, 22))
    out = tmp_path / "crop"
    _two_frame_diff_clean(
        [a, b], out_dir=out, superbias=None, bias_subtract=False,
        k=5.0, save_png=False, save_mask_fits=True,
        cfg={"detector": {"saturation_adu": 500.0}},
    )
    for name in ("ca", "cb"):
        clean = out / "clean" / f"{name}_clean.fits"
        dq = fits.getdata(out / "clean" / f"{name}_clean_mask.fits")
        assert fits.getdata(clean).shape == (20, 22)
        assert dq.shape == (20, 22)
        assert dq[8, 8] & SATURATED


def test_optional_legacy_cosmic_mask_disabled_keeps_required_cosmic_dq(tmp_path):
    a = _raw_frame(tmp_path / "noauto_a.fits", saturated=False)
    b = _raw_frame(tmp_path / "noauto_b.fits", saturated=False)
    with fits.open(a, mode="update") as hdul:
        hdul[0].data[10, 10] = 300.0  # isolated transient, below ADU saturation
    cfg = _cfg(tmp_path, [a, b], "two_frame_diff")
    cfg["cosmics"]["save_mask_fits"] = False
    cfg["cosmics"]["k"] = 3.0
    out = tmp_path / "work" / "05_cosmics"
    clean_cosmics(cfg, out_dir=out)
    dq = fits.getdata(out / "obj" / "clean" / "noauto_a_clean_mask.fits")
    assert dq[10, 10] & COSMIC
    assert not (dq[10, 10] & SATURATED)
