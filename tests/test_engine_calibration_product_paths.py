"""Engine output validation must use stage directories, never the run root."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.contracts.validators import ProductContractError
from scorpio_pipe.pipeline.engine import _validate_task_products
from scorpio_pipe.workspace_paths import stage_dir
from scorpio_pipe.wavesol_paths import wavesol_dir


def _fits(path: Path, data: np.ndarray, header: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, data, header=fits.Header(header or {}), overwrite=True)


def _cfg(run_root: Path) -> dict:
    return {
        "work_dir": str(run_root),
        "data_dir": str(run_root / "raw"),
        "frames": {"__setup__": {"disperser": "VPHG1200@540"}},
    }


def test_superbias_triplet_is_checked_inside_stage_dir(tmp_path: Path) -> None:
    sd = stage_dir(tmp_path, "superbias")
    _fits(sd / "master_bias.fits", np.ones((8, 8), dtype=np.float32),
          {"NBIAS": 5, "SB_GID": "gid", "SB_METH": "median"})
    _fits(sd / "master_bias_var.fits", np.ones((8, 8), dtype=np.float32))
    _fits(sd / "master_bias_dq.fits", np.zeros((8, 8), dtype=np.uint16))
    _validate_task_products(_cfg(tmp_path), "superbias", tmp_path)

    # Misplaced root-level mirror must not satisfy the canonical stage contract.
    (sd / "master_bias_var.fits").unlink()
    _fits(tmp_path / "master_bias_var.fits", np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ProductContractError) as exc:
        _validate_task_products(_cfg(tmp_path), "superbias", tmp_path)
    assert exc.value.code == "bias_var_missing"


def test_superflat_contract_uses_stage_output_dir(tmp_path: Path) -> None:
    sd = stage_dir(tmp_path, "superflat")
    arr = np.ones((8, 8), dtype=np.float32)
    arr[0, 0] = 0.9
    arr[0, 1] = 1.1
    _fits(sd / "superflat.fits", arr, {"SB_FULL": "/test/master_bias.fits"})
    _validate_task_products(_cfg(tmp_path), "superflat", tmp_path)


def test_superneon_uses_disperser_aware_wavesol_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    sd = wavesol_dir(cfg, run_root=tmp_path)
    rng = np.random.default_rng(0)
    _fits(sd / "superneon.fits", rng.normal(size=(16, 32)).astype(np.float32), {"NPK": 10})
    _validate_task_products(cfg, "superneon", tmp_path)


def test_product_catalog_receives_run_root_not_its_parent(tmp_path: Path, monkeypatch) -> None:
    from scorpio_pipe.catalog.products import CATALOG
    observed = []

    def capture(_cfg, *, stage_key, product_keys, run_root):
        observed.append((stage_key, Path(run_root), tuple(product_keys)))

    monkeypatch.setattr(CATALOG, "validate_products", capture)
    # Mock only stage triplet verification to isolate the catalog root argument.
    from scorpio_pipe.contracts import validators
    monkeypatch.setattr(validators, "validate_calib_bias_triplet", lambda *_a, **_k: None)
    _validate_task_products(_cfg(tmp_path), "superbias", tmp_path)
    assert observed and observed[0][0] == "superbias"
    assert observed[0][1] == tmp_path
    assert observed[0][2]
