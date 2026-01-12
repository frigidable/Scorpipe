from __future__ import annotations

from datetime import timezone

import pytest

from scorpio_pipe.instruments import HeaderContractError, parse_frame_meta


def _parse_header_text(txt: str) -> dict[str, object]:
    """Very small FITS-header text parser for tests.

    The repository tests must be runnable without astropy, so we parse the
    common `KEY = value / comment` layout ourselves.
    """

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
        # FITS uses " / " to start a comment. Do *not* split on bare '/'
        # because SCORPIO DATE-OBS uses 'YYYY/DD/MM'.
        val_part = rest
        if " /" in rest:
            # This is safe enough for our header samples.
            val_part = rest.split(" /", 1)[0]
        val_part = val_part.strip()

        if not val_part:
            out[key] = ""
            continue

        # quoted string
        if val_part.startswith("'"):
            # find closing quote
            end = val_part.find("'", 1)
            while end != -1 and end + 1 < len(val_part) and val_part[end + 1] == "'":
                # doubled quotes inside string
                end = val_part.find("'", end + 2)
            if end != -1:
                out[key] = val_part[1:end]
            else:
                out[key] = val_part.strip("'")
            continue

        # boolean
        if val_part in {"T", "F"}:
            out[key] = (val_part == "T")
            continue

        # number
        try:
            if any(ch in val_part for ch in (".", "E", "e")):
                out[key] = float(val_part)
            else:
                out[key] = int(val_part)
            continue
        except Exception:
            out[key] = val_part

    return out


SC1_BIAS_HEADER = """
SIMPLE  =                    T
NAXIS   =                    2
NAXIS1  =                 2080
NAXIS2  =                 1032
DATE-OBS= '2024/04/12'
TIME-OBS= '16:00:05.030'
INSTRUME= 'SCORPIO-1'
OBJECT  = 'bias 1x2 h n'
IMAGETYP= 'bias    '
RATE    =                145.0
GAIN    =                 0.90
NODE    = 'A       '
BINNING = '1x2     '
UT      = '13:00:04.12'
MODE    = 'Image   '
DISPERSE= '        '
SLITWID =
FILTERS = 'slit_0.5 V '
SLITPOS =                552.6
END
"""


SC1_SCI_HEADER = """
SIMPLE  =                    T
NAXIS   =                    2
NAXIS1  =                 2080
NAXIS2  =                 1032
DATE-OBS= '2024/04/12'
TIME-OBS= '21:02:03.048'
INSTRUME= 'SCORPIO-1'
OBJECT  = 'KKH30'
IMAGETYP= 'obj     '
RATE    =                145.0
GAIN    =                 0.90
NODE    = 'A       '
BINNING = '1x2     '
UT      = '18:02:01.94'
MODE    = 'Spectra '
DISPERSE= 'VPHG1200B'
SLITWID =
FILTERS = 'slit_1.2  '
SLITPOS =                552.4
END
"""


def test_sc1_parses_slit_from_filters_when_slitwid_empty():
    h = _parse_header_text(SC1_BIAS_HEADER)
    m = parse_frame_meta(h)
    assert m.instrument == "SCORPIO1"
    assert m.imagetyp == "bias"
    assert m.mode.strip().lower().startswith("image")
    assert m.disperser == ""  # imaging: empty is expected
    assert m.slit_width_key == "0.5"
    assert (m.binning_x, m.binning_y) == (1, 2)
    assert m.readout_key.node == "A"
    assert m.readout_key.rate == pytest.approx(145.0)
    assert m.readout_key.gain == pytest.approx(0.90)
    # DATE-OBS is YYYY/DD/MM -> 2024-12-04
    assert m.date_time_utc.tzinfo == timezone.utc
    assert (m.date_time_utc.year, m.date_time_utc.month, m.date_time_utc.day) == (
        2024,
        12,
        4,
    )


def test_sc1_science_parses_slit_from_filters_and_requires_disperser():
    h = _parse_header_text(SC1_SCI_HEADER)
    m = parse_frame_meta(h)
    assert m.instrument == "SCORPIO1"
    assert m.imagetyp == "obj"
    assert m.disperser == "VPHG1200B"
    assert m.slit_width_key == "1.2"


def test_sc1_missing_binning_is_contract_error():
    h = _parse_header_text(SC1_SCI_HEADER)
    h.pop("BINNING", None)
    with pytest.raises(HeaderContractError) as e:
        parse_frame_meta(h)
    assert "BINNING" in str(e.value)
