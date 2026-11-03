"""Astro-SCRAPPY must not silently erase a protected narrow emission line.

A CR classification on a protected line pixel is ambiguous: retain measured
SCI but mark BADPIX in mandatory DQ, not an unmasked interpolated value.
CCD gain/read noise must come from the actual exposure, not library defaults.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe import maskbits
from scorpio_pipe.stages import cosmics


def test_lacosmic_reconciler_restores_only_protected_detected_pixels() -> None:
    original = np.arange(20, dtype="f4").reshape(4, 5)
    altered = original.copy()
    altered[1, 2] = -999.0
    altered[2, 3] = -998.0
    crmask = np.zeros(original.shape, dtype=bool)
    crmask[1, 2] = crmask[2, 3] = True
    protected = np.zeros_like(crmask)
    protected[1, 2] = True

    clean, cosmic, ambiguous = cosmics._reconcile_lacosmic_line_protection(
        original, altered, crmask, protected,
    )
    assert clean[1, 2] == original[1, 2]
    assert not cosmic[1, 2]
    assert ambiguous[1, 2]
    assert clean[2, 3] == -998.0
    assert cosmic[2, 3]
    assert not ambiguous[2, 3]
    assert original[1, 2] != altered[1, 2]


def test_singleframe_lacosmic_preserves_line_and_propagates_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.ones((16, 24), dtype="f4") * 120.0
    raw[5, 5] = 820.0  # legitimate, protected narrow line
    raw[7, 7] = 4000.0  # unprotected impulsive cosmic event
    header = fits.Header({
        "BUNIT": "ADU",
        "INSTRUME": "SCORPIO-2",
        "BIASSUB": True,
        "GAIN": 0.62,
        "RDNOISE": 3.0,
        "EXPTIME": 300.0,
    })
    source = tmp_path / "obj.fits"
    fits.writeto(source, raw, header)
    recorded = {}

    def fake_detect(image, **kwargs):
        recorded.update(kwargs)
        mask = np.zeros(image.shape, dtype=bool)
        mask[5, 5] = True
        mask[7, 7] = True
        clean = np.asarray(image, dtype="f4").copy()
        clean[5, 5] = 100.0
        clean[7, 7] = 120.0
        return mask, clean

    monkeypatch.setitem(
        sys.modules, "astroscrappy",
        types.SimpleNamespace(detect_cosmics=fake_detect),
    )

    def fake_boxcar(image: np.ndarray, *, r: int) -> np.ndarray:
        image = np.asarray(image)
        result = image.copy()
        if r == 6:
            result[5, 5] -= 10.0
        elif r == 1:
            result -= 0.5
        return result

    monkeypatch.setattr(cosmics, "_boxcar_mean2d", fake_boxcar)
    monkeypatch.setattr(cosmics, "_robust_sigma_mad_1d", lambda x: 1.0)

    out = tmp_path / "cosmics"
    cosmics._single_frame_lacosmic_clean(
        source,
        out_dir=out, superbias=np.ones_like(raw) * 1000,
        bias_subtract=True,
        save_png=False, save_mask_fits=True,
        cfg={}, dilate=0, protect_lines=True,
    )
    cleanfile = out / "clean" / "obj_clean.fits"
    dqfile = out / "clean" / "obj_clean_mask.fits"
    with fits.open(cleanfile) as hdul:
        clean = hdul[0].data
        hdr = hdul[0].header
        assert clean[5, 5] == pytest.approx(820.0)
        assert clean[7, 7] == pytest.approx(120.0)
        assert hdr["COSM_GN"] == pytest.approx(0.62)
        assert hdr["COSM_RN"] == pytest.approx(3.0)
        assert hdr["COSM_N"] == 1
        assert hdr["COSM_RS"] == "HDR"
        assert hdr["BIASSUB"] is True  # no double subtraction of 1000 ADU
    dq = fits.getdata(dqfile)
    assert bool(int(dq[5, 5]) & int(maskbits.BADPIX))
    assert not bool(int(dq[5, 5]) & int(maskbits.COSMIC))
    assert bool(int(dq[7, 7]) & int(maskbits.COSMIC))
    assert recorded["gain"] == pytest.approx(0.62)
    assert recorded["readnoise"] == pytest.approx(3.0)


@pytest.mark.parametrize("protect", [None, np.ones((1, 1), dtype=bool)])
def test_lacosmic_reconciler_rejects_wrong_shapes_and_preserves_untouched(
    protect,
) -> None:
    original = np.ones((2, 3), dtype="f4")
    altered = original.copy()
    cr = np.zeros(original.shape, dtype=bool)
    if protect is None:
        cleaned, m, ambiguous = cosmics._reconcile_lacosmic_line_protection(
            original, altered, cr, protect,
        )
        assert np.array_equal(cleaned, original)
        assert not m.any()
        assert not ambiguous.any()
    else:
        with pytest.raises(ValueError, match="Line protection shape"):
            cosmics._reconcile_lacosmic_line_protection(
                original, altered, cr, protect,
            )


def test_lacosmic_rejects_electron_units_with_adu_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "already_electrons.fits"
    fits.writeto(
        source, np.ones((8, 12), dtype="f4"),
        fits.Header({
            "BUNIT": "e-", "INSTRUME": "SCORPIO-2",
            "GAIN": 0.62, "RDNOISE": 3.0, "BIASSUB": True,
        }),
    )
    monkeypatch.setitem(
        sys.modules, "astroscrappy",
        types.SimpleNamespace(detect_cosmics=lambda *a, **kw: None),
    )
    with pytest.raises(ValueError, match="detector-integrated ADU"):
        cosmics._single_frame_lacosmic_clean(
            source,
            out_dir=tmp_path / "cosmics",
            superbias=None,
            bias_subtract=False,
            save_png=False,
            save_mask_fits=False,
            cfg={},
        )
