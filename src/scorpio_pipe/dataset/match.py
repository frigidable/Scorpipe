from __future__ import annotations

"""Strict calibration matching (P0-B2).

This module contains the deterministic matching logic used by the dataset
manifest builder.

Hard (must-match)
-----------------
* instrument
* GeometryKey (naxis1,naxis2,bin_x,bin_y)
* ReadoutKey (node,rate,gain)
* for flat/arc: SpectroKey (mode='Spectra', disperser, slit_width)

Soft (best-of-valid)
--------------------
* minimal |Δt| to the science set mid-time
* prefer same SPERANGE when available
* prefer closest SLITPOS when available

P0-E/P0-G note (flats/arcs)
----------------------------
For some nights the calibration readout (gain/rate) can differ from science
readout (e.g. science Normal, arc/flat Fast or different gain). When enabled,
``allow_readout_diff`` relaxes readout matching for flats/arcs to keep *NODE*
strict but allow gain/rate mismatch, while still preferring same-readout frames
when available.

We keep *NODE* strict even in allow_readout_diff mode: different amps can carry
instrumental patterns that should not be silently mixed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence


Kind = Literal["bias", "flat", "arc"]


@dataclass(frozen=True)
class ReadoutKey:
    node: str
    rate: float
    gain: float


@dataclass(frozen=True)
class GeometryKey:
    naxis1: int
    naxis2: int
    bin_x: int
    bin_y: int


@dataclass(frozen=True)
class SpectroKey:
    mode: str
    disperser: str
    slit_width_key: str
    slit_width_arcsec: float


@dataclass(frozen=True)
class ConfigKey:
    instrument: str
    geometry: GeometryKey
    readout: ReadoutKey
    spectro: SpectroKey


@dataclass(frozen=True)
class Candidate:
    """Lightweight candidate bundle for matching."""

    calib_id: str
    kind: Kind
    date_time_utc: datetime
    instrument: str
    geometry: GeometryKey
    readout: ReadoutKey
    spectro: SpectroKey | None = None
    sperange: str | None = None
    slit_pos: float | None = None


def _norm_node(node: str) -> str:
    return (node or "").strip().upper()


def make_readout_key(node: str, rate: float, gain: float) -> ReadoutKey:
    # Floats can arrive as 0.899999...; quantize conservatively.
    return ReadoutKey(_norm_node(node), round(float(rate), 3), round(float(gain), 3))


def hard_compatible(
    science: ConfigKey,
    cand: Candidate,
    kind: Kind,
    *,
    allow_readout_diff: bool = False,
) -> bool:
    """Return True if candidate satisfies hard (must-match) constraints."""

    if cand.instrument != science.instrument:
        return False
    if cand.geometry != science.geometry:
        return False

    if allow_readout_diff and kind in ("flat", "arc"):
        # P0-E/P0-G: allow gain/rate mismatch for flats/arcs (still keep NODE strict).
        if _norm_node(cand.readout.node) != _norm_node(science.readout.node):
            return False
    else:
        # Bias is always strict by full readout key.
        if cand.readout != science.readout:
            return False

    if kind in ("flat", "arc"):
        if cand.spectro is None:
            return False
        # Must be Spectra branch for long-slit matching
        if (cand.spectro.mode or "").strip().lower() != "spectra":
            return False
        if cand.spectro.disperser != science.spectro.disperser:
            return False
        if cand.spectro.slit_width_key != science.spectro.slit_width_key:
            return False

    return True


def _abs_dt_s(t0: datetime, t1: datetime) -> float:
    return abs((t0 - t1).total_seconds())


def _score_candidate(
    science_mid: datetime,
    science_sperange: str | None,
    science_slitpos: float | None,
    cand: Candidate,
    *,
    consider_sperange: bool,
    consider_slitpos: bool,
) -> tuple:
    dt = _abs_dt_s(science_mid, cand.date_time_utc)

    sper_m = 0
    if consider_sperange and science_sperange and cand.sperange:
        sper_m = 0 if str(science_sperange).strip() == str(cand.sperange).strip() else 1

    slit_d = 0.0
    if consider_slitpos and (science_slitpos is not None) and (cand.slit_pos is not None):
        slit_d = abs(float(science_slitpos) - float(cand.slit_pos))
    else:
        # Put "unknown" at the end of tie-breaking.
        slit_d = 1.0e12

    # Deterministic final tiebreaker: calib_id
    return (dt, sper_m, slit_d, cand.calib_id)


@dataclass(frozen=True)
class SelectionResult:
    selected_id: str | None
    n_pool: int
    n_hard_compatible: int
    abs_dt_s: float | None
    sperange_mismatch: bool | None
    slitpos_diff: float | None
    tie_n: int | None
    tie_break: str | None


def select_best(
    *,
    kind: Kind,
    science_key: ConfigKey,
    science_mid: datetime,
    science_sperange: str | None,
    science_slitpos: float | None,
    pool: Sequence[Candidate],
    allow_readout_diff: bool = False,
) -> tuple[SelectionResult, list[dict]]:
    """Select best calibration candidate.

    Returns (selection_result, warnings).
    """

    warns: list[dict] = []
    n_pool = len(pool)
    hard = [
        c
        for c in pool
        if hard_compatible(science_key, c, kind, allow_readout_diff=allow_readout_diff)
    ]
    # NOTE: n_hard_compatible reflects the must-match key *before* any
    # preference heuristics (like preferring same readout for flats).
    hard_all = list(hard)
    n_hard = len(hard_all)

    # P0-F/P0-E/P0-G: flats/arcs may allow readout gain/rate mismatch, but when such
    # candidates exist we still prefer a calibration with the *same* readout.
    if allow_readout_diff and kind in ("flat", "arc") and hard_all:
        same = [c for c in hard_all if c.readout == science_key.readout]
        if same:
            hard = same
        else:
            hard = hard_all

    if n_pool == 0:
        return (
            SelectionResult(
                None,
                n_pool=n_pool,
                n_hard_compatible=n_hard,
                abs_dt_s=None,
                sperange_mismatch=None,
                slitpos_diff=None,
                tie_n=None,
                tie_break=None,
            ),
            [
                {
                    "code": f"{kind.upper()}_POOL_EMPTY",
                    "severity": "ERROR",
                    "message": f"No {kind} frames found in calibration pool.",
                }
            ],
        )

    if n_hard == 0:
        hard_desc = "instrument/geometry/readout" + ("/spectro" if kind != "bias" else "")
        if allow_readout_diff and kind in ("flat", "arc"):
            hard_desc = "instrument/geometry/node/spectro"  # readout rate/gain allowed
        return (
            SelectionResult(
                None,
                n_pool=n_pool,
                n_hard_compatible=n_hard,
                abs_dt_s=None,
                sperange_mismatch=None,
                slitpos_diff=None,
                tie_n=None,
                tie_break=None,
            ),
            [
                {
                    "code": f"{kind.upper()}_NO_HARD_MATCH",
                    "severity": "ERROR",
                    "message": (
                        f"No {kind} candidates satisfy must-match keys ({hard_desc})."
                    ),
                }
            ],
        )

    consider_sperange = True
    consider_slitpos = kind in ("flat", "arc")

    scored = [
        (
            _score_candidate(
                science_mid,
                science_sperange,
                science_slitpos,
                c,
                consider_sperange=consider_sperange,
                consider_slitpos=consider_slitpos,
            ),
            c,
        )
        for c in hard
    ]
    scored.sort(key=lambda x: x[0])

    best_score = scored[0][0]
    tied = [c for s, c in scored if s[:3] == best_score[:3]]
    selected = scored[0][1]

    # Diagnostics for selection meta
    abs_dt = _abs_dt_s(science_mid, selected.date_time_utc)
    sper_mismatch: bool | None = None
    if science_sperange and selected.sperange:
        sper_mismatch = str(science_sperange).strip() != str(selected.sperange).strip()
    slit_d: float | None = None
    if (science_slitpos is not None) and (selected.slit_pos is not None):
        slit_d = abs(float(science_slitpos) - float(selected.slit_pos))

    tie_n: int | None = None
    tie_break: str | None = None
    if len(tied) > 1:
        tie_n = len(tied)
        tie_break = "calib_id"
        warns.append(
            {
                "code": f"{kind.upper()}_MULTIPLE_BEST",
                "severity": "WARN",
                "message": f"Multiple equally-good {kind} candidates; selected deterministically by id.",
                "context": {"tied_ids": [c.calib_id for c in tied]},
            }
        )

    return (
        SelectionResult(
            selected_id=selected.calib_id,
            n_pool=n_pool,
            n_hard_compatible=n_hard,
            abs_dt_s=abs_dt,
            sperange_mismatch=sper_mismatch,
            slitpos_diff=slit_d,
            tie_n=tie_n,
            tie_break=tie_break,
        ),
        (
            warns
            + (
                [
                    {
                        "code": "FLAT_READOUT_MISMATCH_ALLOWED" if kind == "flat" else "ARC_READOUT_MISMATCH_ALLOWED",
                        "severity": "WARN",
                        "message": (
                            "Selected flat differs in readout gain/rate from science (allowed; preferred same readout if available)."
                            if kind == "flat"
                            else "Selected arc differs in readout gain/rate from science (allowed; preferred same readout if available)."
                        ),
                        "context": {
                            "science_readout": {
                                "node": science_key.readout.node,
                                "rate": science_key.readout.rate,
                                "gain": science_key.readout.gain,
                            },
                            "selected_readout": {
                                "node": selected.readout.node,
                                "rate": selected.readout.rate,
                                "gain": selected.readout.gain,
                            },
                            "policy": "prefer_same_readout_but_allow",
                            "selected_by": "min(abs_dt_s) then sperange/slitpos then id",
                        },
                    }
                ]
                if (
                    allow_readout_diff
                    and kind in ("flat", "arc")
                    and hard_all
                    and (selected.readout != science_key.readout)
                    and not any(c.readout == science_key.readout for c in hard_all)
                )
                else []
            )
        ),
    )


def select_flat_set(
    *,
    science_key: ConfigKey,
    science_mid: datetime,
    science_sperange: str | None,
    science_slitpos: float | None,
    pool: Sequence[Candidate],
    max_abs_dt_s: float | None,
    allow_readout_diff: bool = True,
    fallback_best_id: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Select a *set* of flat IDs to combine for a science_set.

    Rules
    -----
    - Start from hard-compatible flats (instrument/geometry + node + spectro).
    - If ``max_abs_dt_s`` is provided (>0), keep only frames inside that time
      window around ``science_mid``.
    - If the window is empty but there exist hard-compatible flats, fall back to
      the single best flat (``fallback_best_id`` when available) and emit a WARN.

    Returns (flat_ids, warnings).
    """

    warns: list[dict] = []

    hard = [
        c
        for c in pool
        if hard_compatible(science_key, c, "flat", allow_readout_diff=allow_readout_diff)
    ]
    if not hard:
        return [], warns

    # P0-F/P0-E: if gain/rate mismatches are allowed for flats, still prefer
    # flats with the same readout as science when such frames exist.
    hard_all = list(hard)
    if allow_readout_diff and hard_all:
        same = [c for c in hard_all if c.readout == science_key.readout]
        if same:
            hard = same
        else:
            hard = hard_all

    selected: list[Candidate] = hard
    win = max_abs_dt_s
    if win is not None and float(win) > 0:
        inwin = [c for c in hard if _abs_dt_s(science_mid, c.date_time_utc) <= float(win)]
        if inwin:
            selected = inwin
        else:
            # fall back to the best candidate
            fb = None
            if fallback_best_id:
                for c in hard:
                    if c.calib_id == fallback_best_id:
                        fb = c
                        break
            if fb is None:
                # deterministic: nearest-in-time then id
                hard_sorted = sorted(hard, key=lambda c: (_abs_dt_s(science_mid, c.date_time_utc), c.calib_id))
                fb = hard_sorted[0]
            selected = [fb]
            warns.append(
                {
                    "code": "FLAT_NO_IN_TIME_WINDOW",
                    "severity": "WARN",
                    "message": "No hard-compatible flats inside the requested time window; using the best available flat only.",
                    "context": {
                        "time_window_s": float(win),
                        "fallback_flat_id": str(fb.calib_id),
                    },
                }
            )

    # deterministic ordering
    selected = sorted(selected, key=lambda c: (_abs_dt_s(science_mid, c.date_time_utc), c.calib_id))
    return [c.calib_id for c in selected], warns
