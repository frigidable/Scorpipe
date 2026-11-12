"""Stack2D must not silently average counts (ADU/e-) with rates (/s)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages.stack2d import run_stack2d
from scorpio_pipe.workspace_paths import stage_dir
from tests.test_perf_40_4_stack2d_streaming import _write_linearized_input


def _inputs(tmp_path: Path):
    work = tmp_path / "night"
    linearize = stage_dir(work, "linearize")
    linearize.mkdir(parents=True)
    p0 = linearize / "counts_skysub.fits"
    p1 = linearize / "rate_skysub.fits"
    _write_linearized_input(p0, 1200.0, var_value=40000.0)
    _write_linearized_input(p1, 16.0, var_value=4.0)
    with fits.open(p0, mode="update") as h:
        h[0].header["BUNIT"] = "ADU"
        h[0].header["NORMEXP"] = False
        h.flush()
    with fits.open(p1, mode="update") as h:
        h[0].header["BUNIT"] = "ADU/s"
        h[0].header["NORMEXP"] = True
        h.flush()
    return work, p0, p1


def _cfg(work: Path, normalize: bool) -> dict:
    return {
        "work_dir": str(work),
        "stack2d": {
            "method": "invvar",
            "normalize_exptime": normalize,
            "robust_iter": 0,
            "eta": {"enabled": False},
            "var_floor_enabled": False,
            "save_png": False,
        },
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_stack2d_rejects_mixed_count_and_rate_if_not_normalized(
    tmp_path: Path, reverse: bool,
) -> None:
    work, counts, rate = _inputs(tmp_path)
    files = [rate, counts] if reverse else [counts, rate]
    with pytest.raises(ValueError, match="incompatible SCI dimensions"):
        run_stack2d(
            _cfg(work, False), inputs=files,
            out_dir=stage_dir(work, "stack2d"),
        )


def test_stack2d_converts_integrated_counts_and_variance_when_explicit(
    tmp_path: Path,
) -> None:
    work, counts, rate = _inputs(tmp_path)
    result = run_stack2d(
        _cfg(work, True), inputs=[counts, rate],
        out_dir=stage_dir(work, "stack2d"),
    )
    with fits.open(result["stack2d_fits"]) as h:
        assert h[0].header["BUNIT"] == "ADU/s"
        assert np.allclose(h["SCI"].data, 14.0, atol=1e-5)
        assert np.allclose(h["VAR"].data, 2.0, atol=1e-5)
        assert np.all(h["COV"].data == 2)


def test_stack2d_refuses_mislabeled_normalized_input(
    tmp_path: Path,
) -> None:
    work, counts, rate = _inputs(tmp_path)
    with fits.open(counts, mode="update") as h:
        h[0].header["NORMEXP"] = True
        h.flush()
    with pytest.raises(ValueError, match="NORMEXP=True"):
        run_stack2d(
            _cfg(work, True), inputs=[counts, rate],
            out_dir=stage_dir(work, "stack2d"),
        )
