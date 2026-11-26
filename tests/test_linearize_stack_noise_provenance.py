"""Linearize preview records noise-model provenance without inventing detector values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages.linearize import (
    _stamp_stacked_noise_provenance,
    _write_mef,
)


def _record(*, gain: float = 0.62, rn: float = 3.0, source: str = "header") -> dict:
    return {
        "gain_e_per_adu": gain,
        "rdnoise_e": rn,
        "noise_source": source,
        "path": "obj_01_skysub_raw.fits",
        "var_source": "input",
    }


def test_stack_noise_provenance_preserves_homogeneous_inputs_in_mef(
    tmp_path: Path,
) -> None:
    hdr = _stamp_stacked_noise_provenance(
        fits.Header(), [_record(), _record()]
    )
    assert hdr["GAIN"] == pytest.approx(0.62)
    assert hdr["RDNOISE"] == pytest.approx(3.0)
    assert hdr["NOISRC"] == "header"
    assert hdr["NOIAGG"] == "SAME"
    assert hdr["NOINEXP"] == 2
    assert hdr["NOIDET"] == "linearize_done.json"

    sci = np.ones((8, 16), dtype=np.float32)
    path = tmp_path / "preview.fits"
    _write_mef(
        path, sci, hdr, np.ones_like(sci), np.zeros_like(sci, dtype=np.uint16),
        np.full_like(sci, 2.0),
    )
    with fits.open(path, memmap=False) as hdul:
        assert hdr["NOISRC"] == hdul[0].header["NOISRC"]
        assert hdul[0].header["NOINEXP"] == 2
        assert "VAR" in hdul and "MASK" in hdul


def test_stack_noise_provenance_distinguishes_mixed_readout_models() -> None:
    hdr = _stamp_stacked_noise_provenance(
        fits.Header(),
        [
            _record(gain=0.6, rn=2.0, source="header"),
            _record(gain=1.0, rn=4.0, source="db"),
        ],
    )
    assert hdr["GAIN"] == pytest.approx(0.8)
    assert hdr["RDNOISE"] == pytest.approx(3.0)
    assert hdr["NOISRC"] == "PER_EXP_MEDIAN"
    assert hdr["NOIAGG"] == "MEDIAN"
    assert hdr["NOINEXP"] == 2


@pytest.mark.parametrize(
    "items",
    [
        [],
        [_record(gain=0.0)],
        [_record(rn=-1.0)],
        [_record(gain=float("nan"))],
        [_record(source="")],
        [{"gain_e_per_adu": 0.62, "rdnoise_e": 3.0}],
    ],
)
def test_stack_noise_provenance_rejects_unknown_or_invalid_inputs(
    items: list[dict],
) -> None:
    with pytest.raises(ValueError, match="noise provenance"):
        _stamp_stacked_noise_provenance(fits.Header(), items)
