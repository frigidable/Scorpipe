"""Independent sky-line atlas contract and conservative blend-aware shift probe.

Hanuschik (2003) UVES sky-line CENTER is an *observed air centroid*, not a
laboratory rest wavelength. Intensities from Paranal are NEVER imposed on
BTA exposures, and the measured shift is NEVER an automatically applicable
wavelength correction.

This module provides a diagnostic for one SCIENCE frame/window at a time,
independent of the self-fitted Kelson sky model. It cannot replace a
full λ-dependent, y-dependent, exposure-dependent flexure calibration.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class SkyReference:
    wavelength_air_A: float
    reference_kind: Literal["uves_observed_centroid", "laboratory_air"]
    source_id: str
    source_table: str | None = None
    peak_intensity_source: float | None = None
    uv_fwhm_A: float | None = None


def read_uves_airglow_csv(path: Path | str) -> list[SkyReference]:
    """Read a separately supplied, measured-air ESO UVES sky-line table.

    Never translate a measured UVES centroid into a laboratory line by
    changing a metadata label. The original ESO flux/intensity values are
    provenance, not fixed fitted component ratios.
    """
    required = {"source_table", "lambda_air_A", "peak_INT", "FWHM_A"}
    records: list[SkyReference] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("UVES sky catalogue is missing required measured-air fields")
        for number, row in enumerate(reader, start=2):
            try:
                wave = float(row["lambda_air_A"])
                intensity = float(row["peak_INT"])
                fwhm = float(row["FWHM_A"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed UVES line at CSV row {number}") from exc
            if not (
                np.isfinite(wave) and wave > 0
                and np.isfinite(intensity) and intensity >= 0
                and np.isfinite(fwhm) and fwhm > 0
            ):
                raise ValueError(f"Nonphysical UVES line at CSV row {number}")
            source_table = str(row["source_table"]).strip()
            if not source_table:
                raise ValueError(f"Missing UVES source table at row {number}")
            records.append(SkyReference(
                wavelength_air_A=wave,
                reference_kind="uves_observed_centroid",
                source_id="Hanuschik2003_UVES_airglow",
                source_table=source_table,
                peak_intensity_source=intensity,
                uv_fwhm_A=fwhm,
            ))
    if not records:
        raise ValueError("UVES sky catalogue contains no lines")
    return sorted(records, key=lambda rec: rec.wavelength_air_A)


def group_blended_lines(
    refs: Sequence[SkyReference],
    *, instrument_fwhm_A: float,
    fwhm_separation_factor: float = 1.0,
) -> list[list[SkyReference]]:
    """Treat connected lines closer than LSF FWHM as one unresolved feature."""
    if not np.isfinite(instrument_fwhm_A) or instrument_fwhm_A <= 0:
        raise ValueError("Positive finite instrumental FWHM is mandatory")
    if not np.isfinite(fwhm_separation_factor) or fwhm_separation_factor <= 0:
        raise ValueError("Invalid blend proximity factor")
    ordered = sorted(refs, key=lambda r: r.wavelength_air_A)
    groups: list[list[SkyReference]] = []
    for item in ordered:
        if not groups or (
            item.wavelength_air_A - groups[-1][-1].wavelength_air_A
            > instrument_fwhm_A * fwhm_separation_factor
        ):
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _valid_probe(
    wavelength_air_A: np.ndarray,
    signal: np.ndarray,
    noise_sigma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = np.asarray(wavelength_air_A, dtype=float)
    flux = np.asarray(signal, dtype=float)
    noise = np.asarray(noise_sigma, dtype=float)
    if noise.ndim == 0:
        noise = np.full(wave.shape, float(noise))
    if not (wave.ndim == flux.ndim == noise.ndim == 1):
        raise ValueError("Sky probe requires 1D arrays")
    if wave.shape != flux.shape or wave.shape != noise.shape:
        raise ValueError("Sky probe wavelength, SCI and sigma shapes must agree")
    good = (
        np.isfinite(wave) & np.isfinite(flux)
        & np.isfinite(noise) & (noise > 0)
    )
    wave, flux, noise = wave[good], flux[good], noise[good]
    if wave.size < 12 or np.unique(wave).size < 12:
        raise ValueError("Too few usable independent wavelength samples")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("Sky probe wavelengths must be strictly increasing")
    return wave, flux, noise


def fit_blended_sky_shift(
    wavelength_air_A: np.ndarray,
    signal: np.ndarray,
    noise_sigma: np.ndarray | float,
    refs: Sequence[SkyReference],
    *,
    wavelength_frame: Literal["air", "vacuum"],
    instrument_fwhm_A: float,
    max_abs_shift_A: float = 1.5,
    shift_step_A: float = 0.04,
    min_detection_delta_chi2: float = 25.0,
) -> dict:
    """Scan a common wavelength shift, freely fitting every blend amplitude.

    Weighted continuum nuisance terms are marginalized by least squares;
    line amplitudes are non-negative NNLS coefficients. The result is
    diagnostic only even for known laboratory air wavelengths. Neither
    a formal curvature uncertainty nor UVES airglow centroids are a
    certificate that a correction is safe to apply.
    """
    if wavelength_frame != "air":
        raise ValueError(
            "Sky reference wavelengths are air Angstrom; explicitly "
            "convert any vacuum wavelength array before fitting"
        )
    if not refs:
        raise ValueError("At least one external sky line is required")
    if any(r.reference_kind not in {"uves_observed_centroid","laboratory_air"} for r in refs):
        raise ValueError("Unsupported sky reference type")
    # Observed UVES centroids and atomic laboratory wavelengths need not
    # share an identical absolute origin. Reject mixed fits rather than
    # disguising their different provenance behind a single delta.
    if len({r.reference_kind for r in refs}) != 1:
        raise ValueError(
            "Cannot mix laboratory rest wavelengths with observed UVES sky centroids"
        )
    if len(refs) > 8:
        raise ValueError(
            "Sky-line fit exceeds eight components; split/review the unresolved "
            "blend instead of fitting an under-constrained catalogue segment"
        )
    if not (np.isfinite(instrument_fwhm_A) and instrument_fwhm_A > 0):
        raise ValueError("An independently measured finite instrumental FWHM is required")
    if not (np.isfinite(max_abs_shift_A) and max_abs_shift_A > 0
            and np.isfinite(shift_step_A) and 0 < shift_step_A < max_abs_shift_A):
        raise ValueError("Invalid wavelength shift search grid")
    if not (np.isfinite(min_detection_delta_chi2) and min_detection_delta_chi2 >= 0):
        raise ValueError("Invalid minimum detection significance")
    wave, flux, noise = _valid_probe(wavelength_air_A,signal,noise_sigma)
    x = (wave - float(np.median(wave))) / max(np.ptp(wave), 1e-9)
    inv = 1.0/noise
    baseline = np.column_stack((np.ones_like(x), x))*inv[:,None]
    weighted = flux*inv
    baseline_coef = np.linalg.lstsq(baseline, weighted, rcond=None)[0]
    baseline_only = weighted - baseline @ baseline_coef
    chi2_no_lines = float(np.dot(baseline_only,baseline_only))
    # Orthogonal projection removes the two unconstrained continuum
    # components before NNLS for nonnegative (freely varying) sky lines.
    bt = np.linalg.pinv(baseline)
    rest = np.asarray([r.wavelength_air_A for r in refs],float)
    sigma_lsf = float(instrument_fwhm_A)/2.354820045
    ngrid = int(round(2*max_abs_shift_A/shift_step_A)) + 1
    grid = np.linspace(-max_abs_shift_A,max_abs_shift_A,ngrid)
    chi2 = np.empty(grid.size, dtype=float)
    amps = np.empty((grid.size,rest.size), dtype=float)
    for i, shift in enumerate(grid):
        profile = np.exp(-0.5*((wave[:,None] - rest[None,:] - shift)/sigma_lsf)**2)
        profile *= inv[:,None]
        projected = profile - baseline @ (bt @ profile)
        amp, _ = nnls(projected,baseline_only,maxiter=50*rest.size+500)
        amps[i] = amp
        residual = baseline_only - projected @ amp
        chi2[i] = float(np.dot(residual,residual))
    k = int(np.argmin(chi2))
    detected = float(chi2_no_lines-chi2[k])
    near_boundary = k in (0,1,len(grid)-2,len(grid)-1)
    competing = np.where(chi2 <= chi2[k] + 4.0)[0]
    ambiguous = bool(competing.size and (
        grid[competing].max()-grid[competing].min()>0.5
    ))
    status = "PROVISIONAL" if (
        detected >= min_detection_delta_chi2 and not near_boundary and not ambiguous
    ) else "UNRELIABLE"
    return {
        "status": status,
        "wavelength_frame": "air",
        "reference_types": sorted(set(r.reference_kind for r in refs)),
        "sources": sorted(set(r.source_id for r in refs)),
        "shift_A": float(grid[k]),
        "shift_search_grid_A": [float(grid[0]),float(grid[-1]),float(grid[1]-grid[0])],
        "instrument_fwhm_A": float(instrument_fwhm_A),
        "n_sky_lines": int(len(refs)),
        "n_samples": int(wave.size),
        "detection_delta_chi2": detected,
        "chi2_min": float(chi2[k]),
        "chi2_continuum_only": chi2_no_lines,
        "peak_amplitudes_free": [float(t) for t in amps[k]],
        "grid_boundary": bool(near_boundary),
        "shift_degenerate": bool(ambiguous),
        "reference_is_lab_standard": bool(all(
            r.reference_kind == "laboratory_air" for r in refs
        )),
        "auto_apply_allowed": False,
        "science_wavelength_zero_point_approved": False,
        "reason": (
            "Single-band sky shift is provisional and cannot establish "
            "wavelength/spatial/time-dependent flexure; source UVES intensities "
            "were not fixed and observed UVES centroids are not laboratory rest lines."
        ),
    }
