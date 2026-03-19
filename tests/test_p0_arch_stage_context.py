from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")


def _write_fits(p: Path, data: np.ndarray) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(p, data.astype(np.float32), overwrite=True)
    return p


def test_superbias_entrypoint_respects_engine_run_root(tmp_path: Path) -> None:
    """P0-ARCH-02.1: superbias must not derive work_dir on its own."""

    # Create synthetic bias frames (absolute paths).
    bias_dir = tmp_path / "input_bias"
    bias_paths = []
    rng = np.random.default_rng(123)
    for i in range(5):
        arr = rng.normal(loc=1000.0, scale=3.0, size=(32, 64)).astype(np.float32)
        bias_paths.append(_write_fits(bias_dir / f"b{i:02d}.fits", arr))

    # Config tries to point work_dir elsewhere, but engine passes run_root explicitly.
    cfg = {
        "work_dir": str(tmp_path / "SHOULD_NOT_BE_USED"),
        "config_dir": str(tmp_path),
        "frames": {"bias": [str(p) for p in bias_paths]},
        "calib": {"bias_combine": "median"},
    }

    from scorpio_pipe.pipeline.context import StageContext
    from scorpio_pipe.workspace_paths import stage_dir
    from scorpio_pipe.stages.entry_superbias import run

    run_root = tmp_path / "run"
    sd = stage_dir(run_root, "superbias")

    ctx = StageContext(cfg=cfg, run_root=run_root, stage_key="superbias", stage_dir=sd)
    res = run(ctx)

    # Outputs must be under engine run_root.
    assert (sd / "master_bias.fits").is_file()
    assert (sd / "master_bias_index.json").is_file()
    assert (sd / "superbias.fits").is_file()

    # Must NOT write into cfg.work_dir.
    wrong = stage_dir(Path(cfg["work_dir"]), "superbias")
    assert not (wrong / "master_bias.fits").exists()

    # StageResult must be serializable and contain products dict.
    d = res.as_dict()
    assert isinstance(d.get("products"), dict)
