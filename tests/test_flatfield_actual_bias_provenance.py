"""A cosmics *request* to debias is not proof of a debiased CCD exposure.

Test the actual Flatfield entrypoint with a 1000 ADU electronic pedestal,
read noise, gain and MasterBias variance. A cleaned file may exist even when
Cosmics lacked a usable MasterBias. Passing its unremoved pedestal into the
Poisson variance estimator would drastically overestimate uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages import calib, flatfield
from scorpio_pipe.workspace_paths import stage_dir


@pytest.mark.parametrize("already_subtracted", [False, True])
def test_flatfield_obeys_actual_cleaned_frame_biassub_and_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_subtracted: bool,
) -> None:
    work = tmp_path / "run"
    work.mkdir()
    shape = (16, 24)
    bias_adu = 1000.0
    signal_adu = 120.0
    gain = 0.62
    rn_e = 3.0
    bias_var_adu2 = 4.0

    hdr = fits.Header()
    hdr["BUNIT"] = "ADU"
    hdr["GAIN"] = gain
    hdr["RDNOISE"] = rn_e
    hdr["BIASSUB"] = already_subtracted
    hdr["EXPTIME"] = 300.0

    raw = work / "object_01.fits"
    fits.writeto(raw, np.full(shape, bias_adu + signal_adu, dtype="f4"), hdr)

    clean = stage_dir(work, "cosmics") / "obj" / "clean" / "object_01_clean.fits"
    clean.parent.mkdir(parents=True)
    # Cosmics was configured to subtract bias, but may not have found a
    # compatible MasterBias. Only this actual FITS marker is authoritative.
    clean_signal = signal_adu if already_subtracted else bias_adu + signal_adu
    fits.writeto(
        clean,
        np.full(shape, clean_signal, dtype="f4"),
        hdr,
    )

    bias = work / "bias.fits"
    bias_var = work / "bias_var.fits"
    superflat = work / "superflat.fits"
    fits.writeto(bias, np.full(shape, bias_adu, dtype="f4"))
    fits.writeto(bias_var, np.full(shape, bias_var_adu2, dtype="f4"))
    fits.writeto(superflat, np.ones(shape, dtype="f4"))

    from scorpio_pipe.stages.calib import BiasSelection

    chosen = BiasSelection(
        sci_path=bias,
        var_path=bias_var,
        dq_path=None,
        gid="ccd_node_E",
        degraded=False,
        reason="",
        gain_e_per_adu=gain,
        rdnoise_e=rn_e,
    )
    lookup_count = [0]

    def select_bias(cfg: dict, h: fits.Header, *, sci_shape=None):
        lookup_count[0] += 1
        assert tuple(sci_shape) == shape
        assert float(h["GAIN"]) == gain
        return chosen

    monkeypatch.setattr(calib, "resolve_master_bias", select_bias)
    monkeypatch.setattr(calib, "_resolve_superbias_path", lambda cfg, root: bias)
    monkeypatch.setattr(calib, "_resolve_superflat_path", lambda cfg, root: superflat)
    monkeypatch.setattr(flatfield, "ensure_compatible_calib", lambda *a, **kw: None)

    cfg = {
        "work_dir": str(work),
        "frames": {"obj": [str(raw)], "flat": [str(superflat)]},
        "cosmics": {"bias_subtract": True},
        "flatfield": {
            "enabled": True,
            "use_manifest": False,
            "apply_to": ["obj"],
            "bias_subtract": True,
            "save_png": False,
        },
    }
    output_dir = stage_dir(work, "flatfield")
    done_path = flatfield.run_flatfield(cfg, out_dir=output_dir)
    done = json.loads(Path(done_path).read_text(encoding="utf-8"))
    assert done["status"] == "ok"
    assert done["n_outputs"] == 1
    output = output_dir / "obj" / "object_01_flat.fits"
    assert output.exists()

    with fits.open(output, memmap=False) as hdul:
        sci_e = np.asarray(hdul["SCI"].data, dtype=float)
        var_e2 = np.asarray(hdul["VAR"].data, dtype=float)
        assert bool(hdul[0].header["BIASSUB"]) is True
        assert np.allclose(sci_e, signal_adu * gain, rtol=0, atol=1e-4)
        # For already-subtracted CLEAN data without input VAR the pipeline
        # estimates its noise but has no provenance for the earlier bias VAR.
        expected_var = signal_adu * gain + rn_e**2
        if not already_subtracted:
            expected_var += bias_var_adu2 * gain**2
        assert np.allclose(var_e2, expected_var, rtol=2e-4, atol=2e-4)

    if already_subtracted:
        assert lookup_count[0] == 0, "MasterBias must not be subtracted twice"
    else:
        assert lookup_count[0] == 1, "Missing BIASSUB requires actual MasterBias"
        assert done["bias_used"]["ccd_node_E"] == 1
