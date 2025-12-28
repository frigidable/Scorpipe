"""Product naming policy (single source of truth for filenames/suffixes).

We intentionally keep filenames **predictable** and **unique**.
Historically, a few products had multiple spellings (e.g. ``*_sky_sub`` vs
``*_skysub``). This module centralizes naming so stages cannot diverge.

Design
------
* Canonical suffixes are *lowercase* and *underscore-free* where reasonable.
* Per-exposure filenames are ``<raw_stem>_<suffix>.<ext>``.
* Readers should accept legacy spellings, but writers must use canonical.
"""

from __future__ import annotations

from dataclasses import dataclass


# Canonical suffixes (extend carefully).
SUF_SKYSUB = "skysub"
SUF_SKYMODEL = "skymodel"
SUF_CC = "cc"  # cross-correlation / flexure / wavelength shift
SUF_LIN = "lin"  # generic linearized alias (legacy)
SUF_RECT = "rectified"
SUF_STACK2D = "stack2d"
SUF_LAMBDA_MAP = "lambda_map"


_CANON_SUFFIXES: set[str] = {
    SUF_SKYSUB,
    SUF_SKYMODEL,
    SUF_CC,
    SUF_LIN,
    SUF_RECT,
    SUF_STACK2D,
    SUF_LAMBDA_MAP,
}


def per_exp_name(raw_stem: str, suffix: str, *, ext: str = "fits") -> str:
    """Return canonical per-exposure filename.

    Parameters
    ----------
    raw_stem
        Stem of the raw FITS file (without extension).
    suffix
        Canonical suffix from this module.
    ext
        Extension without dot (default: ``fits``).
    """
    s = (suffix or "").strip().lower()
    if s not in _CANON_SUFFIXES:
        raise ValueError(f"Unknown product suffix: {suffix!r}")
    stem = (raw_stem or "").strip()
    if not stem:
        raise ValueError("raw_stem must be non-empty")
    e = (ext or "").strip().lstrip(".")
    if not e:
        raise ValueError("ext must be non-empty")
    return f"{stem}_{s}.{e}"


def sky_sub_fits_name(raw_stem: str) -> str:
    return per_exp_name(raw_stem, SUF_SKYSUB, ext="fits")


def sky_model_fits_name(raw_stem: str) -> str:
    return per_exp_name(raw_stem, SUF_SKYMODEL, ext="fits")


def legacy_sky_sub_fits_names(raw_stem: str) -> tuple[str, ...]:
    """Known legacy spellings for sky-subtracted frames."""
    stem = (raw_stem or "").strip()
    if not stem:
        return tuple()
    return (
        f"{stem}_sky_sub.fits",  # very common legacy
        f"{stem}_skysub.fits",  # canonical (keep for convenience)
    )


@dataclass(frozen=True)
class NamingReport:
    canonical: str
    legacy: tuple[str, ...]


def naming_report_for_skysub(raw_stem: str) -> NamingReport:
    return NamingReport(
        canonical=sky_sub_fits_name(raw_stem),
        legacy=legacy_sky_sub_fits_names(raw_stem),
    )
