from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from astropy.io import fits

from scorpio_pipe.fits_utils import open_fits_smart


def _norm_token(s: str | None) -> str:
    if not s:
        return ""
    # keep letters/digits and a few separators
    return re.sub(r"[^0-9A-Za-z._+-]+", "", str(s).strip())


def _parse_binning_str(s: str) -> tuple[int, int] | None:
    ss = str(s).strip().lower().replace(" ", "")
    if not ss:
        return None
    # common: '2x2', '2,2', '2 2', '2;2', '2/2'
    m = re.match(r"^(\d+)[x,;/](\d+)$", ss)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)(\d+)$", ss)
    if m and len(ss) == 2:
        return int(m.group(1)), int(m.group(2))
    # '2 2'
    m = re.match(r"^(\d+)\s+(\d+)$", str(s).strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _binning_from_header(hdr: fits.Header) -> tuple[int, int] | None:
    # Most common keywords in FITS from observatories / IRAF:
    # CCDSUM: '1 1', CCDBIN1/2, BINX/BINY, XBINNING/YBINNING, BINNING.
    for k in ("BINX", "BINY"):
        if k in hdr:
            try:
                bx = int(hdr.get("BINX"))
                by = int(hdr.get("BINY"))
                if bx > 0 and by > 0:
                    return bx, by
            except Exception:
                pass

    for kx, ky in (("CCDBIN1", "CCDBIN2"), ("XBINNING", "YBINNING")):
        if (kx in hdr) or (ky in hdr):
            try:
                bx = int(hdr.get(kx, 0) or 0)
                by = int(hdr.get(ky, 0) or 0)
                if bx > 0 and by > 0:
                    return bx, by
            except Exception:
                pass

    for k in ("CCDSUM", "CCDBIN", "BINNING"):
        v = hdr.get(k)
        if v is None:
            continue
        parsed = _parse_binning_str(str(v))
        if parsed:
            bx, by = parsed
            if bx > 0 and by > 0:
                return bx, by

    return None


def _window_from_header(hdr: fits.Header) -> str:
    # CCDSEC/DATASEC/TRIMSEC are typical: '[x1:x2,y1:y2]'
    for k in ("CCDSEC", "DATASEC", "TRIMSEC", "DETSEC"):
        v = hdr.get(k)
        if v:
            return str(v).strip()
    return ""


def _readout_from_header(hdr: fits.Header) -> str:
    """Readout-mode token.

    For SCORPIO/SCORPIO-2, the most informative, stable knobs are typically:
    - NODE (output amplifier / node)
    - RATE (readout rate; often numeric in kpix/s)
    - GAIN (e-/ADU; numeric)

    We also keep common generic keywords (READMODE/READOUT/AMP...) for other instruments.
    The resulting token is a normalized, human-readable string that can be compared strictly.
    """
    parts: list[str] = []

    node = hdr.get("NODE")
    if node not in (None, ""):
        parts.append(f"node{_norm_token(str(node))}")

    rate = hdr.get("RATE") or hdr.get("READRATE") or hdr.get("RD_RATE")
    if rate not in (None, ""):
        try:
            r = float(rate)
            # Avoid ultra-sensitivity to harmless formatting noise
            parts.append(f"rate{r:.1f}")
        except Exception:
            parts.append(f"rate{_norm_token(str(rate))}")

    gain = hdr.get("GAIN")
    if gain not in (None, ""):
        try:
            g = float(gain)
            parts.append(f"gain{g:.4f}")
        except Exception:
            parts.append(f"gain{_norm_token(str(gain))}")

    for k in (
        "READOUT",
        "READMODE",
        "RDOUTMODE",
        "RD_MODE",
        "AMPLMODE",
        "AMP",
        "CCDAMP",
        "AMPL",
        "GAINMOD",
    ):
        v = hdr.get(k)
        if v is None or v == "":
            continue
        parts.append(f"{k.lower()}{_norm_token(str(v))}")
        break

    return "_".join(parts)



@dataclass(frozen=True)
class FrameSignature:
    """A strict, physically meaningful signature for calibration compatibility.

    Fields are designed to be comparable across frames:
    - shape: data array shape (ny, nx)
    - binning: CCD binning (bx, by), if known
    - window: ROI/window section string, if present
    - readout: readout/gain/amp mode token, if present
    """

    ny: int
    nx: int
    bx: int | None = None
    by: int | None = None
    window: str = ""
    readout: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def binning(self) -> str:
        if self.bx is None or self.by is None:
            return ""
        return f"{int(self.bx)}x{int(self.by)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": f"{self.ny}x{self.nx}",
            "binning": self.binning(),
            "window": self.window,
            "readout": self.readout,
        }

    def describe(self) -> str:
        b = self.binning() or "?"
        w = self.window or "?"
        r = self.readout or "?"
        return f"shape={self.ny}x{self.nx}, bin={b}, roi={w}, readout={r}"

    def diff(self, other: "FrameSignature") -> list[str]:
        diffs: list[str] = []

        # shape is always the primary gate, but treat unknown (0x0) as missing
        if (
            (self.ny > 0 and self.nx > 0 and other.ny > 0 and other.nx > 0)
            and (self.shape != other.shape)
        ):
            diffs.append(f"shape {self.ny}x{self.nx} != {other.ny}x{other.nx}")
        elif (self.ny == 0 or self.nx == 0) and (other.ny > 0 and other.nx > 0):
            diffs.append(f"shape unknown != {other.ny}x{other.nx}")
        elif (other.ny == 0 or other.nx == 0) and (self.ny > 0 and self.nx > 0):
            diffs.append(f"shape {self.ny}x{self.nx} != unknown")

        # binning: strict when either side provides a value
        self_has_bin = self.bx is not None and self.by is not None
        other_has_bin = other.bx is not None and other.by is not None
        if self_has_bin and other_has_bin and (self.bx, self.by) != (other.bx, other.by):
            diffs.append(f"binning {self.binning()} != {other.binning()}")
        elif self_has_bin != other_has_bin:
            diffs.append(f"binning {self.binning() or 'unknown'} != {other.binning() or 'unknown'}")

        # ROI/window: strict when either side provides a value
        if self.window and other.window and self.window != other.window:
            diffs.append(f"roi {self.window} != {other.window}")
        elif bool(self.window) != bool(other.window):
            diffs.append(f"roi {self.window or 'unknown'} != {other.window or 'unknown'}")

        # readout: strict when either side provides a value
        if self.readout and other.readout and self.readout != other.readout:
            diffs.append(f"readout {self.readout} != {other.readout}")
        elif bool(self.readout) != bool(other.readout):
            diffs.append(f"readout {self.readout or 'unknown'} != {other.readout or 'unknown'}")

        return diffs

    def is_compatible_with(self, other: "FrameSignature") -> bool:
        return len(self.diff(other)) == 0

    @classmethod
    def from_header(cls, hdr: fits.Header, *, fallback_shape: tuple[int, int] | None = None) -> "FrameSignature":
        # shape: prefer NAXIS2/NAXIS1; fallback provided
        ny = hdr.get("NAXIS2")
        nx = hdr.get("NAXIS1")
        if isinstance(ny, int) and isinstance(nx, int) and ny > 0 and nx > 0:
            shape = (ny, nx)
        elif fallback_shape is not None:
            shape = fallback_shape
        else:
            # as a last resort, try to infer from section strings (rare)
            shape = (0, 0)

        b = _binning_from_header(hdr)
        win = _window_from_header(hdr)
        ro = _readout_from_header(hdr)
        return cls(shape[0], shape[1], *(b or (None, None)), window=win, readout=ro)

    @classmethod
    def from_path(cls, path: str | Path) -> "FrameSignature":
        p = Path(path)
        with open_fits_smart(p, memmap="auto") as hdul:
            hdu = hdul[0]
            hdr = hdu.header
            # Use header-only shape if available, otherwise look at data.shape (may map)
            fallback_shape = None
            try:
                if hdu.data is not None:
                    fallback_shape = tuple(int(x) for x in hdu.data.shape)
            except Exception:
                fallback_shape = None
            return cls.from_header(hdr, fallback_shape=fallback_shape)

    @classmethod
    def from_setup(cls, setup: Mapping[str, Any]) -> "FrameSignature":
        # setup comes from inspect/autocfg: shape like '2048x4096', binning like '1x1', window string
        shape_s = str(setup.get("shape", "") or "")
        ny, nx = 0, 0
        if "x" in shape_s:
            try:
                a, b = shape_s.lower().split("x")
                ny, nx = int(a), int(b)
            except Exception:
                ny, nx = 0, 0

        b = _parse_binning_str(str(setup.get("binning", "") or ""))
        window = str(setup.get("window", "") or "").strip()
        readout = _norm_token(str(setup.get("readout", "") or ""))
        bx, by = (b if b else (None, None))
        return cls(ny, nx, bx, by, window=window, readout=readout)


def format_signature_mismatch(*, expected: FrameSignature, got: FrameSignature, path: Path) -> str:
    diffs = got.diff(expected)
    if not diffs:
        return f"{path.name}: OK ({got.describe()})"
    return f"{path.name}: " + "; ".join(diffs) + f" (got {got.describe()}, expected {expected.describe()})"
