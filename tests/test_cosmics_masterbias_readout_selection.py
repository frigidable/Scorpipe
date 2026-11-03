"""Cosmics must use the science readout's MasterBias, not the default alias.

When multiple CCD readout groups enter one multi-exposure rejection operation,
fail before writing a misleading 'BIASSUB=True' output. Already-debiased CCD
inputs must never have bias subtracted a second time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages import calib, cosmics


def _frame(
    path: Path,
    *,
    rate: float = 225.0,
    gain: float = 0.62,
    bias: float = 100.0,
    debiased: bool = False,
) -> Path:
    rng = np.random.default_rng(740)
    data = 120.0 + rng.normal(0.0, 1.0, (20, 28))
    if not debiased:
        data += bias
    h = fits.Header()
    h["BUNIT"] = "ADU"
    h["RATE"] = rate
    h["GAIN"] = gain
    h["RDNOISE"] = 3.0
    h["NODE"] = "E"
    h["BIASSUB"] = debiased
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, data.astype("f4"), header=h)
    return path


def _stub_bias_index(
    monkeypatch: pytest.MonkeyPatch,
    masters: dict[float, Path],
) -> None:
    monkeypatch.setattr(
        calib, "load_master_bias_index", lambda work_dir: {"groups": [
            {"gid": "readout_index_present"}
        ]},
    )

    def select(cfg, header, *, sci_shape=None, policy_override=None):
        assert policy_override == "strict"
        assert tuple(sci_shape) == (20, 28)
        gain = float(header["GAIN"])
        if gain not in masters:
            raise RuntimeError(f"no matching MasterBias readout gain {gain}")
        master = masters[gain]
        var = master.with_name(master.stem + "_var.fits")
        if not var.exists():
            fits.writeto(
                var, np.full((20, 28), 4.0, dtype="f4"),
                header=fits.Header({"BUNIT": "ADU^2"}),
            )
        return calib.BiasSelection(
            sci_path=master, var_path=var, dq_path=None,
            gid=f"ccd_gain_{gain}", degraded=False, reason="",
        )

    monkeypatch.setattr(calib, "resolve_master_bias", select)


def test_cosmics_uses_matching_masterbias_not_global_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _frame(tmp_path / "raw0.fits")
    b = _frame(tmp_path / "raw1.fits")
    master = tmp_path / "master_matching.fits"
    fits.writeto(master, np.full((20, 28), 100.0, dtype="f4"))
    _stub_bias_index(monkeypatch, {0.62: master})

    # Deliberately incorrect global alias: previously selected regardless of
    # science NODE/RATE/GAIN when cosmics.bias_subtract was enabled.
    monkeypatch.setattr(
        cosmics, "_load_superbias",
        lambda root: np.full((20, 28), 900.0, dtype="f4"),
    )
    root = tmp_path / "run"
    cfg = {
        "work_dir": str(root), "data_dir": str(tmp_path),
        "config_dir": str(tmp_path),
        "frames": {"obj": [str(a), str(b)]},
        "cosmics": {
            "method": "two_frame_diff", "apply_to": ["obj"],
            "bias_subtract": True, "save_png": False,
            "save_mask_fits": True, "preserve_manual": False,
            "k": 8.0, "dilate": 0,
        },
    }
    cosmics.clean_cosmics(cfg, out_dir=root / "05_cosmics")
    for p in (a, b):
        clean = root / "05_cosmics" / "obj" / "clean" / f"{p.stem}_clean.fits"
        assert clean.exists()
        hdr = fits.getheader(clean)
        assert hdr["BIASSUB"] is True
        assert hdr["SB_GID"] == "ccd_gain_0.62"
        assert hdr["BVARREF"] == f"{p.stem}_clean_biasvar.fits"
        assert hdr["BVARINC"] is False
        with fits.open(clean.with_name(hdr["BVARREF"])) as hdul:
            assert hdul[0].header["BUNIT"] == "ADU^2"
            assert np.all(hdul[0].data == 4.0)
        assert np.median(fits.getdata(clean)) == pytest.approx(120, abs=1.0)


def test_cosmics_fails_closed_on_mixed_masterbias_readouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _frame(tmp_path / "same.fits", gain=0.62)
    b = _frame(tmp_path / "other.fits", gain=3.5)
    master_a = tmp_path / "bias_a.fits"
    master_b = tmp_path / "bias_b.fits"
    for p in (master_a, master_b):
        fits.writeto(p, np.full((20, 28), 100.0, dtype="f4"))
    _stub_bias_index(monkeypatch, {0.62: master_a, 3.5: master_b})
    cfg = {
        "work_dir": str(tmp_path / "run"), "data_dir": str(tmp_path),
        "frames": {"obj": [str(a), str(b)]},
        "cosmics": {
            "method": "two_frame_diff", "apply_to": ["obj"],
            "bias_subtract": True, "save_png": False,
        },
    }
    with pytest.raises(ValueError, match="different MasterBias readout groups"):
        cosmics.clean_cosmics(cfg, out_dir=tmp_path / "run" / "05_cosmics")
    assert not (tmp_path / "run" / "05_cosmics" / "obj" / "clean").exists()


def test_cosmics_refuses_mixed_raw_and_already_debiased_exposures(
    tmp_path: Path,
) -> None:
    a = _frame(tmp_path / "raw.fits")
    b = _frame(tmp_path / "already.fits", debiased=True)
    with pytest.raises(ValueError, match="mixes BIASSUB=True and raw"):
        cosmics._resolve_cosmics_superbias_for_kind(
            {}, [a, b], tmp_path, np.full((20, 28), 100, dtype="f4")
        )


def test_cosmics_never_subtracts_twice_when_input_is_debiased(
    tmp_path: Path,
) -> None:
    a = _frame(tmp_path / "done0.fits", debiased=True)
    b = _frame(tmp_path / "done1.fits", debiased=True)
    assert cosmics._resolve_cosmics_superbias_for_kind(
        {}, [a, b], tmp_path, np.full((20, 28), 100, dtype="f4")
    ) is None
    out = tmp_path / "out"
    cosmics._two_frame_diff_clean(
        [a, b], out_dir=out,
        superbias=np.full((20, 28), 100, dtype="f4"),
        bias_subtract=True, k=8, save_png=False, save_mask_fits=True,
        cfg={}, dilate=0,
    )
    for p in (a, b):
        clean = out / "clean" / f"{p.stem}_clean.fits"
        assert fits.getheader(clean)["BIASSUB"] is True
        assert np.median(fits.getdata(clean)) == pytest.approx(120, abs=1.0)


def test_cosmics_refuses_recleaning_pending_bias_variance(
    tmp_path: Path,
) -> None:
    a = _frame(tmp_path / "linked.fits", debiased=True)
    with fits.open(a, mode="update") as hdul:
        hdul[0].header["BVARREF"] = "linked_clean_biasvar.fits"
        hdul[0].header["BVARINC"] = False
        hdul.flush()
    with pytest.raises(ValueError, match="pending MasterBias BVARREF"):
        cosmics._resolve_cosmics_superbias_for_kind(
            {}, [a], tmp_path, None
        )


def test_cosmics_rejects_wrong_masterbias_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _frame(tmp_path / "wrong.fits")
    master = tmp_path / "small_master.fits"
    fits.writeto(master, np.zeros((12, 13), dtype="f4"))
    _stub_bias_index(monkeypatch, {0.62: master})
    with pytest.raises(ValueError, match="geometry does not match"):
        cosmics._resolve_cosmics_superbias_for_kind(
            {}, [a], tmp_path, None
        )
