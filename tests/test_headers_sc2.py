from __future__ import annotations

from datetime import timezone

import pytest

from scorpio_pipe.instruments import HeaderContractError, parse_frame_meta


def _parse_header_text(txt: str) -> dict[str, object]:
    # Keep duplicated from test_headers_sc1.py on purpose: tests are intentionally
    # standalone.
    out: dict[str, object] = {}
    for raw in txt.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("HISTORY") or line.startswith("COMMENT"):
            continue
        if line.strip() == "END":
            break
        if "=" not in line:
            continue
        key, rest = line.split("=", 1)
        key = key.strip()
        # FITS uses " / " to start a comment. Do not split on bare '/'
        # because SCORPIO DATE-OBS uses 'YYYY/DD/MM'.
        val_part = rest
        if " /" in rest:
            val_part = rest.split(" /", 1)[0]
        val_part = val_part.strip()

        if not val_part:
            out[key] = ""
            continue
        if val_part.startswith("'"):
            end = val_part.find("'", 1)
            if end != -1:
                out[key] = val_part[1:end]
            else:
                out[key] = val_part.strip("'")
            continue
        if val_part in {"T", "F"}:
            out[key] = (val_part == "T")
            continue
        try:
            if any(ch in val_part for ch in (".", "E", "e")):
                out[key] = float(val_part)
            else:
                out[key] = int(val_part)
            continue
        except Exception:
            out[key] = val_part
    return out


SC2_BIAS_HEADER = """
SIMPLE  =                    T
NAXIS   =                    2
NAXIS1  =                 4112
NAXIS2  =                 1040
DATE-OBS= '2025/16/12'
TIME-OBS= '15:32:38.733'
INSTRUME= 'SCORPIO-2'
OBJECT  = 'bias 2x2 h n'
IMAGETYP= 'bias    '
RATE    =                185.0
GAIN    =                 0.62
NODE    = 'E       '
BINNING = '1x2     '
UT      = '12:32:23.74'
MODE    = 'Image   '
DISPERSE= '        '
SLITWID =              1.00019
SLITPOS =                514.2
END
"""


SC2_NEON_HEADER = """
SIMPLE  =                    T
NAXIS   =                    2
NAXIS1  =                 4112
NAXIS2  =                 1040
DATE-OBS= '2025/16/12'
TIME-OBS= '17:59:06.633'
INSTRUME= 'SCORPIO-2'
OBJECT  = 'Neon'
IMAGETYP= 'neon    '
RATE    =                185.0
GAIN    =                 0.62
NODE    = 'E       '
BINNING = '1x2     '
UT      = '14:59:02.89'
MODE    = 'Spectra '
DISPERSE= 'VPHG1200@540'
SLITWID =              1.00019
END
"""


def test_sc2_parses_slitwid_and_date_format():
    h = _parse_header_text(SC2_BIAS_HEADER)
    m = parse_frame_meta(h)
    assert m.instrument == "SCORPIO2"
    assert m.imagetyp == "bias"
    assert m.slit_width_key == "1.0"  # 1.00019 -> stable key
    assert m.disperser == ""  # imaging: empty disperser is expected
    assert (m.binning_x, m.binning_y) == (1, 2)
    assert m.readout_key.node == "E"
    assert m.readout_key.rate == pytest.approx(185.0)
    assert m.readout_key.gain == pytest.approx(0.62)
    assert m.date_time_utc.tzinfo == timezone.utc
    assert (m.date_time_utc.year, m.date_time_utc.month, m.date_time_utc.day) == (
        2025,
        12,
        16,
    )


def test_sc2_spectra_requires_disperser_and_keeps_it():
    h = _parse_header_text(SC2_NEON_HEADER)
    m = parse_frame_meta(h)
    assert m.instrument == "SCORPIO2"
    assert m.imagetyp == "neon"
    assert m.disperser == "VPHG1200@540"


def test_sc2_missing_rate_is_contract_error():
    h = _parse_header_text(SC2_NEON_HEADER)
    h.pop("RATE", None)
    with pytest.raises(HeaderContractError) as e:
        parse_frame_meta(h)
    assert "RATE" in str(e.value)
