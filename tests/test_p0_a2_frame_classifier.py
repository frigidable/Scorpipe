from __future__ import annotations

"""P0-A2 tests: deterministic frame classification + long-slit guardrail.

We keep these tests runnable without astropy by parsing the provided FITS header
text samples (same style as test_headers_sc1/sc2).
"""

from scorpio_pipe.dataset import FrameClass, classify_frame, is_longslit_mode
from scorpio_pipe.instruments import parse_frame_meta


def _parse_header_text(txt: str) -> dict[str, object]:
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


def test_classifier_maps_imagetyp_to_frameclass_sc1_sc2():
    m1 = parse_frame_meta(_parse_header_text(SC1_BIAS_HEADER))
    assert classify_frame(m1) == FrameClass.BIAS
    assert is_longslit_mode(m1) is False

    m2 = parse_frame_meta(_parse_header_text(SC2_BIAS_HEADER))
    assert classify_frame(m2) == FrameClass.BIAS
    assert is_longslit_mode(m2) is False

    m3 = parse_frame_meta(_parse_header_text(SC2_NEON_HEADER))
    assert classify_frame(m3) == FrameClass.ARC
    assert is_longslit_mode(m3) is True

    m4 = parse_frame_meta(_parse_header_text(SC1_SCI_HEADER))
    assert classify_frame(m4) == FrameClass.SCIENCE
    assert is_longslit_mode(m4) is True


def test_longslit_filter_is_exact_on_mode_string():
    # Guardrail: long-slit branch uses MODE == Spectra only.
    m = parse_frame_meta(_parse_header_text(SC1_SCI_HEADER))
    assert is_longslit_mode(m) is True

    # Any other value is not long-slit.
    m2 = m.__class__(
        **{**m.__dict__, "mode": "Image"}  # type: ignore[attr-defined]
    )
    assert is_longslit_mode(m2) is False
