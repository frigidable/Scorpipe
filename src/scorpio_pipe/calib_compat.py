from __future__ import annotations

"""Calibration compatibility checks.

Contract (P0-F)
--------------
Calibration association must be:

* **hard/strict** on anything that changes pixel geometry or the physical
  meaning of the spectrum (binning, shape/ROI, disperser, slit width, node).
* **soft/QC-only** on orientation and small positional details that are not
  intrinsically fatal for long-slit reduction (ROTANGLE, SLITPOS, readout
  gain/rate differences).

The goal is to avoid *false* fatal mismatches (e.g. science ROTANGLE differs
from flat ROTANGLE) while still surfacing diagnostics in QC/provenance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.instruments import HeaderContractError, parse_frame_meta
from scorpio_pipe.qc.flags import make_flag


HeaderLike = Mapping[str, Any]


def _s(v: Any) -> str:
    return str(v).strip()


def _norm(v: Any) -> str:
    return _s(v).replace(" ", "").upper()


def _get_binning(h: HeaderLike) -> str:
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
class CalibMustKey:
    instrume: str = ""
    detector: str = ""
    mode: str = ""
    disperse: str = ""
    binning: str = ""
    naxis1: str = ""
    naxis2: str = ""
    slitwid: str = ""
    slitmask: str = ""
    node: str = ""

    @classmethod
    def from_header(cls, h: HeaderLike) -> "CalibMustKey":
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
            naxis1 = str(int(m.naxis1))
            naxis2 = str(int(m.naxis2))
        except HeaderContractError:
            # Fallback for non-SCORPIO headers or legacy datasets.
            instrume = _norm(h.get("INSTRUME", h.get("TELESCOP", "")))
            mode = _norm(h.get("MODE", h.get("OBS_MODE", "")))
            disperse = _norm(h.get("DISPERSE", h.get("GRISM", h.get("GRATING", ""))))
            binning = _get_binning(h)
            slitwid = _norm(h.get("SLITWID", h.get("SLIT", "")))
            node = _norm(h.get("NODE", h.get("READMODE", "")))
            naxis1 = _s(h.get("NAXIS1", ""))
            naxis2 = _s(h.get("NAXIS2", ""))

        return cls(
            instrume=instrume,
            detector=_norm(h.get("DETECTOR", h.get("CCDNAME", h.get("CCD", "")))),
            mode=mode,
            disperse=disperse,
            binning=binning,
            naxis1=_norm(naxis1),
            naxis2=_norm(naxis2),
            slitwid=slitwid,
            slitmask=_norm(h.get("SLITMASK", "")),
            node=node,
        )

    def as_tuple(self) -> tuple[str, ...]:
        return (
            self.instrume,
            self.detector,
            self.mode,
            self.disperse,
            self.binning,
            self.naxis1,
            self.naxis2,
            self.slitwid,
            self.slitmask,
            self.node,
        )

    def to_header_cards(self, prefix: str = "CK") -> dict[str, Any]:
        return {
            f"{prefix}INS": self.instrume,
            f"{prefix}DET": self.detector,
            f"{prefix}MOD": self.mode,
            f"{prefix}DSP": self.disperse,
            f"{prefix}BIN": self.binning,
            f"{prefix}NX": self.naxis1,
            f"{prefix}NY": self.naxis2,
            f"{prefix}SLW": self.slitwid,
            f"{prefix}SLM": self.slitmask,
            f"{prefix}NOD": self.node,
        }


@dataclass(frozen=True)
class CalibQCKey:
    """QC-only fields.

    These are *not* part of the hard compatibility key.
    """

    rot: str = ""
    slitpos: float | None = None

    @classmethod
    def from_header(cls, h: HeaderLike) -> "CalibQCKey":
        rot = _norm(h.get("ROTANGLE", h.get("PA", "")))
        # Prefer instrument-aware SLITPOS.
        slitpos: float | None = None
        try:
            m = parse_frame_meta(h, strict=False)
            slitpos = m.slit_pos
        except Exception:
            sp = h.get("SLITPOS", None)
            if sp is not None and _s(sp):
                try:
                    slitpos = float(sp)
                except Exception:
                    slitpos = None
        return cls(rot=rot, slitpos=slitpos)


def diff_keys(expected: Any, actual: Any) -> dict[str, tuple[Any, Any]]:
    diffs: dict[str, tuple[str, str]] = {}
    for field in expected.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        ev = getattr(expected, field)
        av = getattr(actual, field)
        if (ev or av) and (ev != av):
            diffs[field] = (ev, av)
    return diffs


class CalibrationMismatchError(ValueError):
    """Raised when a calibration violates must-match constraints."""

    def __init__(self, message: str, *, diffs: dict[str, Any] | None = None):
        super().__init__(message)
        self.diffs = diffs or {}


def compare_compat_headers(
    sci_hdr: HeaderLike,
    calib_hdr: HeaderLike,
    *,
    kind: str = "calibration",
    strict: bool = True,
    allow_readout_diff: bool = True,
    stage_flags: list[dict[str, Any]] | None = None,
    slitpos_warn_tol: float | None = None,
) -> dict[str, Any]:
    """Compare two headers and return a JSON-friendly compatibility payload.

    This is the core logic used by :func:`ensure_compatible_calib`.
    It intentionally accepts lightweight mapping headers (no Astropy required).
    """

    k_sci = CalibMustKey.from_header(sci_hdr)
    k_cal = CalibMustKey.from_header(calib_hdr)
    must_diffs = diff_keys(k_sci, k_cal)

    q_sci = CalibQCKey.from_header(sci_hdr)
    q_cal = CalibQCKey.from_header(calib_hdr)

    qc_diffs: dict[str, Any] = {}
    if (q_sci.rot or q_cal.rot) and q_sci.rot != q_cal.rot:
        qc_diffs["rot"] = (q_sci.rot, q_cal.rot)
        if stage_flags is not None:
            stage_flags.append(
                make_flag(
                    "CALIB_ROT_MISMATCH",
                    "WARN",
                    f"{kind} ROTANGLE differs: science={q_sci.rot!r} calib={q_cal.rot!r}",
                    "This is usually safe for long-slit flats; verify trace/illumination in QA.",
                    science_rot=q_sci.rot,
                    calib_rot=q_cal.rot,
                )
            )

    # SLITPOS: warn on any measurable difference (or beyond tolerance if provided).
    if (q_sci.slitpos is not None) and (q_cal.slitpos is not None):
        d = abs(float(q_sci.slitpos) - float(q_cal.slitpos))
        tol = float(slitpos_warn_tol) if slitpos_warn_tol is not None else 0.0
        if d > tol:
            qc_diffs["slitpos"] = (q_sci.slitpos, q_cal.slitpos)
            if stage_flags is not None:
                stage_flags.append(
                    make_flag(
                        "CALIB_SLITPOS_MISMATCH",
                        "WARN",
                        f"{kind} SLITPOS differs: science={q_sci.slitpos} calib={q_cal.slitpos} (|Δ|={d:.3g})",
                        "If the shift is large, consider a dedicated flat at the same slit position.",
                        science_slitpos=q_sci.slitpos,
                        calib_slitpos=q_cal.slitpos,
                        delta_slitpos=float(d),
                    )
                )

    # Readout gain/rate/readnoise differences are QC-only.
    if allow_readout_diff:
        g_sci = sci_hdr.get("GAIN", sci_hdr.get("EGAIN", None))
        g_cal = calib_hdr.get("GAIN", calib_hdr.get("EGAIN", None))
        r_sci = sci_hdr.get("RATE", None)
        r_cal = calib_hdr.get("RATE", None)
        rn_sci = sci_hdr.get("RDNOISE", sci_hdr.get("READNOISE", None))
        rn_cal = calib_hdr.get("RDNOISE", calib_hdr.get("READNOISE", None))

        readout_mismatch = False
        if (g_sci is not None and g_cal is not None and str(g_sci) != str(g_cal)):
            readout_mismatch = True
        if (r_sci is not None and r_cal is not None and str(r_sci) != str(r_cal)):
            readout_mismatch = True
        if (rn_sci is not None and rn_cal is not None and str(rn_sci) != str(rn_cal)):
            readout_mismatch = True

        if readout_mismatch:
            qc_diffs["readout"] = {
                "gain": (g_sci, g_cal),
                "rate": (r_sci, r_cal),
                "rdnoise": (rn_sci, rn_cal),
            }
            if stage_flags is not None:
                stage_flags.append(
                    make_flag(
                        "CALIB_READOUT_DIFF",
                        "WARN",
                        f"{kind} readout parameters differ (GAIN/RATE/RDNOISE).",
                        "Allowed. Noise model uses science frame params; verify VAR/QA if SNR is critical.",
                        science_gain=g_sci,
                        calib_gain=g_cal,
                        science_rate=r_sci,
                        calib_rate=r_cal,
                        science_rdnoise=rn_sci,
                        calib_rdnoise=rn_cal,
                    )
                )

    meta: dict[str, Any] = {
        "kind": str(kind),
        "must_key_science": k_sci.as_tuple(),
        "must_key_calib": k_cal.as_tuple(),
        "must_diffs": must_diffs,
        "qc_diffs": qc_diffs,
    }

    if must_diffs and strict:
        # Provide actionable hints for the most common fatal mismatches.
        hints: list[str] = []
        if "binning" in must_diffs:
            hints.append("binning mismatch: you need a separate calibration with the same binning")
        if "disperse" in must_diffs:
            hints.append("disperser mismatch: use a calibration with the same grism/disperser")
        if "naxis1" in must_diffs or "naxis2" in must_diffs:
            hints.append("shape/ROI mismatch: use a calibration with the same detector format")
        if "node" in must_diffs:
            hints.append("NODE mismatch: use a calibration from the same amplifier/node")
        hint_txt = ("; ".join(hints)) if hints else "use a calibration with the same geometry"
        raise CalibrationMismatchError(
            f"Incompatible {kind}: must-match differences {must_diffs}. Hint: {hint_txt}.",
            diffs=must_diffs,
        )

    return meta


def ensure_compatible_calib(
    sci_hdr: HeaderLike,
    calib_path: str | Path | HeaderLike,
    *,
    kind: str = "calibration",
    strict: bool = True,
    allow_readout_diff: bool = True,
    stage_flags: list[dict[str, Any]] | None = None,
    slitpos_warn_tol: float | None = None,
) -> dict[str, Any]:
    """Check calibration header compatibility; raise on mismatch if strict.

    Returns a JSON-friendly dict with key info and diffs.
    """
    # If caller passes a mapping as `calib_path`, treat it as a header already.
    if isinstance(calib_path, Mapping):
        meta = compare_compat_headers(
            sci_hdr,
            calib_path,
            kind=kind,
            strict=strict,
            allow_readout_diff=allow_readout_diff,
            stage_flags=stage_flags,
            slitpos_warn_tol=slitpos_warn_tol,
        )
        meta["calib_path"] = None
        return meta

    cpath = Path(str(calib_path)).expanduser()

    # Astropy is required only when we need to read FITS headers from disk.
    try:
        from astropy.io import fits  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Astropy is required to read FITS headers for calibration compatibility checks. "
            "Install 'astropy>=6' (project dependency) or pass a header mapping instead."
        ) from e

    with fits.open(cpath, memmap=False) as hdul:  # type: ignore[attr-defined]
        ch: HeaderLike = dict(hdul[0].header)
        # If MEF, prefer SCI header.
        try:
            if "SCI" in hdul:
                ch = dict(hdul["SCI"].header)  # type: ignore[index]
        except Exception:
            pass

    meta = compare_compat_headers(
        sci_hdr,
        ch,
        kind=kind,
        strict=strict,
        allow_readout_diff=allow_readout_diff,
        stage_flags=stage_flags,
        slitpos_warn_tol=slitpos_warn_tol,
    )
    meta["calib_path"] = str(cpath)
    return meta
