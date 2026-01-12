from __future__ import annotations

"""Calibration compatibility checks.

A calibration frame must match the science configuration strongly enough to be
physically meaningful. We implement a strict key (INSTRUME, detector, mode,
disperser, binning, slit, rot, ...) and provide helpers for matching and
reporting.

This is designed to prevent "almost right" flats/neons from being applied
silently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits

from scorpio_pipe.instruments import HeaderContractError, parse_frame_meta


def _s(v: Any) -> str:
    return str(v).strip()


def _norm(v: Any) -> str:
    return _s(v).replace(" ", "").upper()


def _get_binning(h: fits.Header) -> str:
    for k in ("CCDBIN1", "CCDBIN", "BINNING", "XBINNING"):
        if k in h:
            return _norm(h.get(k))
    # common pattern: "2 2"
    x = h.get("XBIN", None)
    y = h.get("YBIN", None)
    if x is not None and y is not None:
        return f"{int(x)}x{int(y)}"
    return ""


@dataclass(frozen=True)
class CalibKey:
    instrume: str = ""
    detector: str = ""
    mode: str = ""
    disperse: str = ""
    binning: str = ""
    slitwid: str = ""
    slitmask: str = ""
    node: str = ""
    rot: str = ""

    @classmethod
    def from_header(cls, h: fits.Header) -> "CalibKey":
        # Prefer strict, instrument-aware parsing. If the header violates the
        # contract, we want an explicit failure rather than silent guessing.
        try:
            m = parse_frame_meta(h, strict=True)
            instrume = _norm(m.instrument_db_key)
            mode = _norm(m.mode)
            disperse = _norm(m.disperser)
            binning = _norm(m.binning_key)
            slitwid = _norm(m.slit_width_key)
            node = _norm(m.readout_key.node)
        except HeaderContractError:
            # Fallback for non-SCORPIO headers or legacy datasets.
            instrume = _norm(h.get("INSTRUME", h.get("TELESCOP", "")))
            mode = _norm(h.get("MODE", h.get("OBS_MODE", "")))
            disperse = _norm(h.get("DISPERSE", h.get("GRISM", h.get("GRATING", ""))))
            binning = _get_binning(h)
            slitwid = _norm(h.get("SLITWID", h.get("SLIT", "")))
            node = _norm(h.get("NODE", h.get("READMODE", "")))

        return cls(
            instrume=instrume,
            detector=_norm(h.get("DETECTOR", h.get("CCDNAME", h.get("CCD", "")))),
            mode=mode,
            disperse=disperse,
            binning=binning,
            slitwid=slitwid,
            slitmask=_norm(h.get("SLITMASK", "")),
            node=node,
            rot=_norm(h.get("ROTANGLE", h.get("PA", ""))),
        )

    def as_tuple(self) -> tuple[str, ...]:
        return (
            self.instrume,
            self.detector,
            self.mode,
            self.disperse,
            self.binning,
            self.slitwid,
            self.slitmask,
            self.node,
            self.rot,
        )

    def to_header_cards(self, prefix: str = "CK") -> dict[str, Any]:
        return {
            f"{prefix}INS": self.instrume,
            f"{prefix}DET": self.detector,
            f"{prefix}MOD": self.mode,
            f"{prefix}DSP": self.disperse,
            f"{prefix}BIN": self.binning,
            f"{prefix}SLW": self.slitwid,
            f"{prefix}SLM": self.slitmask,
            f"{prefix}NOD": self.node,
            f"{prefix}ROT": self.rot,
        }


def diff_keys(expected: CalibKey, actual: CalibKey) -> dict[str, tuple[str, str]]:
    diffs: dict[str, tuple[str, str]] = {}
    for field in expected.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        ev = getattr(expected, field)
        av = getattr(actual, field)
        if (ev or av) and (ev != av):
            diffs[field] = (ev, av)
    return diffs


def ensure_compatible_calib(
    sci_hdr: fits.Header,
    calib_path: str | Path,
    *,
    kind: str = "calibration",
    strict: bool = True,
    allow_readout_diff: bool = True,
) -> dict[str, Any]:
    """Check calibration header compatibility; raise on mismatch if strict.

    Returns a JSON-friendly dict with key info and diffs.
    """
    calib_path = Path(calib_path).expanduser()
    with fits.open(calib_path, memmap=False) as hdul:
        ch = fits.Header(hdul[0].header)
        # If MEF, prefer SCI header.
        try:
            if "SCI" in hdul:
                ch = fits.Header(hdul["SCI"].header)  # type: ignore[index]
        except Exception:
            pass

    k_sci = CalibKey.from_header(sci_hdr)
    k_cal = CalibKey.from_header(ch)
    diffs = diff_keys(k_sci, k_cal)

    meta = {
        "kind": str(kind),
        "calib_path": str(calib_path),
        "key_science": k_sci.as_tuple(),
        "key_calib": k_cal.as_tuple(),
        "diffs": diffs,
    }

    if diffs and strict:
        raise ValueError(
            f"Incompatible {kind}: {calib_path.name}. Differences: {diffs}"
        )

    # Readout/gain differences are not grounds for rejection, but are QC-relevant.
    if allow_readout_diff:
        g_sci = sci_hdr.get("GAIN", sci_hdr.get("EGAIN", None))
        g_cal = ch.get("GAIN", ch.get("EGAIN", None))
        rn_sci = sci_hdr.get("RDNOISE", sci_hdr.get("READNOISE", None))
        rn_cal = ch.get("RDNOISE", ch.get("READNOISE", None))
        if (g_sci is not None and g_cal is not None and str(g_sci) != str(g_cal)) or (
            rn_sci is not None and rn_cal is not None and str(rn_sci) != str(rn_cal)
        ):
            meta["warning"] = "CALIB_READOUT_DIFF"
    return meta
