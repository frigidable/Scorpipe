"""Old frame-scatter MasterBias products cannot be used as estimator VAR."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

fits = pytest.importorskip("astropy.io.fits")

from scorpio_pipe.stages.calib import build_superbias, resolve_master_bias
from tests.golden.mini_night import make_mini_night


def _build_real_master(tmp_path: Path):
    mn = make_mini_night(tmp_path)
    build_superbias(
        mn.cfg, run_root=mn.work_dir,
        stage_dir=mn.work_dir / "03_bias",
        allow_legacy_mirrors=False,
    )
    ip = mn.work_dir / "03_bias" / "master_bias_index.json"
    idx = json.loads(ip.read_text(encoding="utf-8"))
    hdr = fits.getheader(mn.cfg["frames"]["obj"][0])
    return mn, ip, idx, hdr


def _resolve(mn, hdr):
    return resolve_master_bias(
        mn.cfg, hdr, sci_shape=(64, 128), policy_override="strict",
    )


def test_new_masterbias_index_is_v3_and_usable(tmp_path: Path) -> None:
    mn, _, idx, hdr = _build_real_master(tmp_path)
    assert idx["schema_version"] == 3
    assert idx["schema"] == "scorpio_pipe.master_bias_index"
    assert idx["groups"][0]["variance"]["definition"] == (
        "uncertainty_of_combined_master_bias"
    )
    selected = _resolve(mn, hdr)
    assert selected.var_path is not None
    assert selected.var_path.exists()
    vh = fits.getheader(selected.var_path)
    assert vh["SB_VMOD"] == "GAUSSMED"
    assert int(vh["NBIAS"]) == 3


@pytest.mark.parametrize("legacy_schema", [1, 2, None])
def test_resolver_rejects_old_scatter_variance_index(
    tmp_path: Path, legacy_schema,
) -> None:
    mn, ip, idx, hdr = _build_real_master(tmp_path)
    if legacy_schema is None:
        idx.pop("schema_version")
    else:
        idx["schema_version"] = legacy_schema
    ip.write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Obsolete MasterBias variance"):
        _resolve(mn, hdr)


@pytest.mark.parametrize(
    "damage",
    [
        "definition", "model", "missing_var",
        "wrong_unit", "wrong_model_header",
        "wrong_group_header", "wrong_sample_count", "wrong_factor",
    ],
)
def test_resolver_rejects_bad_estimator_provenance(
    tmp_path: Path, damage: str,
) -> None:
    mn, ip, idx, hdr = _build_real_master(tmp_path)
    group = idx["groups"][0]
    var_path = mn.work_dir / group["files"]["var"]
    if damage == "definition":
        group["variance"]["definition"] = "single_frame_MAD2"
    elif damage == "model":
        group["variance"]["model"] = "UNKNOWN"
    elif damage == "missing_var":
        var_path.unlink()
    else:
        with fits.open(var_path, mode="update") as hdul:
            h = hdul[0].header
            if damage == "wrong_unit":
                h["BUNIT"] = "e-^2"
            elif damage == "wrong_model_header":
                h["SB_VMOD"] = "MEANSEM"
            elif damage == "wrong_group_header":
                h["SB_GID"] = "another_readout"
            elif damage == "wrong_sample_count":
                h["NBIAS"] = 30
            elif damage == "wrong_factor":
                h["SB_VFAC"] *= 3
            hdul.flush()
    ip.write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        _resolve(mn, hdr)
