"""Reproduce Setup selection against a mixed BTA night, without Qt."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "src" / "scorpio_pipe" / "ui" / "launcher_window.py"


def _real_helper(headers):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
              and x.name == "_representative_header_from_inspect_df")
    namespace = {
        "Any": object,
        "Path": Path,
        "get_header_fields": lambda path: headers[str(path)],
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace[fn.name]


def _science_tables():
    rows = [
        ("CSS0854", "VPHG1200B", "1.2", "1x2", "obj", "css.fts"),
        ("N2146sat", "VPHG1200B", "1.2", "2x2", "obj", "wrong_bin.fts"),
        ("N2146sat", "VPHG1200B", "1.2", "1x2", "neon", "neon.fts"),
        ("N2146sat", "VPHG1200R", "1.2", "1x2", "obj", "wrong_disp.fts"),
        ("N2146sat", "VPHG1200B", "1.2", "1x2", "obj", "s22800607.fts"),
    ]
    return pd.DataFrame([{
        "object": obj,
        "object_norm": "".join(c for c in obj.upper() if c.isalnum()),
        "disperser": disp, "slit": slit, "binning": binning,
        "kind": kind, "path": path,
    } for obj, disp, slit, binning, kind, path in rows])


def test_selected_setup_never_uses_first_object_neon_or_wrong_configuration():
    headers = {
        "css.fts": {"OBJECT": "CSS0854", "DISPERSE": "VPHG1200B"},
        "wrong_bin.fts": {"OBJECT": "N2146sat", "DISPERSE": "VPHG1200B"},
        "neon.fts": {"OBJECT": "N2146sat", "IMAGETYP": "neon"},
        "wrong_disp.fts": {"OBJECT": "N2146sat", "DISPERSE": "VPHG1200R"},
        "s22800607.fts": {
            "OBJECT": "N2146sat", "DISPERSE": "VPHG1200B",
            "DATE-OBS": "2024/06/12", "DATE": "2024-12-06T15:55:05",
        },
    }
    f = _real_helper(headers)
    result = f(
        _science_tables(), object_name="N2146sat", disperser="VPHG1200B",
        slit="1.2", binning="1x2",
    )
    assert result["OBJECT"] == "N2146sat"
    assert result["DISPERSE"] == "VPHG1200B"
    assert result["DATE"] == "2024-12-06T15:55:05"


def test_missing_selected_configuration_fails_closed():
    f = _real_helper({"css.fts": {"OBJECT": "CSS0854"}})
    assert f(_science_tables(), object_name="NOT_FOUND",
             disperser="VPHG1200B", binning="1x2") is None


def test_header_object_mismatch_is_rejected():
    headers = {"s22800607.fts": {"OBJECT": "another_galaxy"}}
    f = _real_helper(headers)
    with __import__("pytest").raises(ValueError, match="does not match"):
        f(_science_tables(), object_name="N2146sat",
          disperser="VPHG1200B", slit="1.2", binning="1x2")
