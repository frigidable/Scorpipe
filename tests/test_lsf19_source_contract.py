"""Independent-ARC v19 reproducibility and strict original-FITS reader contracts."""
import ast
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"qa"/"lsf19_profile_probe.py"


def _load_module():
    spec=importlib.util.spec_from_file_location("lsf19_probe_testmodule",SCRIPT)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_three_original_source_sha256_are_complete_and_canonical():
    tree=ast.parse(SCRIPT.read_text(encoding="utf-8"))
    matches=[
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node,ast.Assign)
        and any(isinstance(x,ast.Name) and x.id=="ORIGINAL_SHA"
                for x in node.targets)
    ]
    assert len(matches)==1
    hashes=matches[0]
    assert set(hashes)=={"arc06","arc13","wavelength_map"}
    assert all(re.fullmatch(r"[0-9a-f]{64}",x) for x in hashes.values())
    assert hashes["wavelength_map"]==(
        "37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab"
    )
    assert len(set(hashes.values()))==3


def test_strict_readonly_fits_supports_raw_unsigned_detector(tmp_path):
    fits=pytest.importorskip("astropy.io.fits")
    reader=_load_module().read_frozen_primary
    data=np.full((1032,2080),32769,dtype=np.uint16)
    data[506,2001]=45555
    name=tmp_path/"test_original_unsigned.fits"
    fits.PrimaryHDU(data).writeto(name)
    arr,hdr=reader(name)
    assert arr.shape==data.shape
    assert hdr["BITPIX"]==16
    assert arr[0,0]==32769
    assert arr[506,2001]==45555
    assert np.all(np.isfinite(arr))


def test_strict_readonly_fits_supports_float_lambda_map(tmp_path):
    fits=pytest.importorskip("astropy.io.fits")
    reader=_load_module().read_frozen_primary
    data=np.full((1032,2080),5200.,dtype=np.float32)
    data[506,2012]=5336.19
    path=tmp_path/"lambda_map.fits"
    header=fits.Header()
    header["WAVEREF"]="air"
    fits.PrimaryHDU(data,header=header).writeto(path)
    got,hdr=reader(path)
    assert got.shape==data.shape
    assert got[506,2012]==pytest.approx(data[506,2012])
    assert hdr["WAVEREF"]=="air"
    assert hdr["BITPIX"]==-32
