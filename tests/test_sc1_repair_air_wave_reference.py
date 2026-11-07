"""Regression tests for precise one-line air reference restoration."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "qa"
          / "sc1_repair_air_wave_reference.py")


def module():
    spec = importlib.util.spec_from_file_location("sc1_repair_air_ref", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_repair_selftest():
    module().selftest()


@pytest.mark.parametrize(
    "old,new",
    [(b"UNKNOWN", b"air"),
     (b"'UNKNOWN'", b"'air'"),
     (b'"UNKNOWN"', b'"air"'),
     (b"unknown", b"air")],
)
def test_only_project_wave_ref_changes_and_newline_preserved(old, new):
    prefix = (b"data_dir: D:/raw\r\nproject:\r\n"
              b"  barycentric_correction: OFF\r\n")
    original = (prefix + b"  wave_reference: " + old + b" # marker\r\n"
                + b"wavesol:\r\n  other_field: UNKNOWN\r\n")
    patched, mode = module().prepare_patch(original)
    assert mode == "PATCH_TO_AIR"
    assert patched == original.replace(b"wave_reference: " + old,
                                       b"wave_reference: " + new)
    assert module().prepare_patch(patched) == (patched, "ALREADY_AIR")


def test_unknown_other_than_project_is_not_modified():
    raw = b"project:\n  wave_reference: air\nfoo: UNKNOWN\n"
    assert module().prepare_patch(raw) == (raw, "ALREADY_AIR")


def test_ambiguous_or_unsupported_yaml_fails_closed():
    m = module()
    with pytest.raises(ValueError, match="Expected exactly one"):
        m.prepare_patch(b"project: {wave_reference: UNKNOWN}\n")
    with pytest.raises(ValueError, match="Refuse to edit"):
        m.prepare_patch(b"project:\n  wave_reference: vacuum\n")
    with pytest.raises(ValueError, match="Expected a project"):
        m.prepare_patch(b"wavesol:\n  wave_reference: UNKNOWN\n")


def test_utf8_bom_preserved():
    m = module()
    raw = b"\xef\xbb\xbfproject:\n  wave_reference: UNKNOWN\n"
    out, mode = m.prepare_patch(raw)
    assert mode == "PATCH_TO_AIR"
    assert out == b"\xef\xbb\xbfproject:\n  wave_reference: air\n"
