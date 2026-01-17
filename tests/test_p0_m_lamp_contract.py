from __future__ import annotations

from pathlib import Path

from scorpio_pipe.lamp_contract import (
    LAMP_HENEAR,
    LAMP_NE,
    LAMP_UNKNOWN,
    infer_lamp_from_header,
    resolve_lamp_type,
    resolve_linelist_csv_path,
)


def test_object_neon_is_weak_hint_defaults_to_henear() -> None:
    hdr = {"OBJECT": "NEON"}

    raw, lt = infer_lamp_from_header(hdr)
    assert raw == "NEON"
    # OBJECT=NEON is a label; should not be treated as lamp_type evidence.
    assert lt == LAMP_UNKNOWN

    cfg = {"frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}}, "wavesol": {}}
    res = resolve_lamp_type(cfg, hdr=hdr, instrument_hint="scorpio")
    assert res.lamp_type == LAMP_HENEAR
    assert res.source == "default"

    ll = resolve_linelist_csv_path(cfg, res.lamp_type)
    assert isinstance(ll, Path)
    assert ll.name.startswith("henear_lines.csv")


def test_config_override_wins() -> None:
    hdr = {"OBJECT": "NEON"}
    cfg = {
        "frames": {"__setup__": {"instrument": "scorpio", "mode": "longslit"}},
        "wavesol": {"lamp_type": "Ne"},
    }
    res = resolve_lamp_type(cfg, hdr=hdr, instrument_hint="scorpio")
    assert res.lamp_type == LAMP_NE
    assert res.source == "config"
