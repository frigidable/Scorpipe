"""Lambda-map (λ(x,y)) validation.

Implements the strict contract from P1-B:
- 2D array (ny, nx)
- explicit wavelength unit + reference in header
- values are finite in (almost) all pixels
- monotonic (or near-monotonic) along dispersion axis x

The validator returns a diagnostics dict and raises
:class:`LambdaMapValidationError` on hard failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits

from scorpio_pipe.frame_signature import FrameSignature, format_signature_mismatch


class LambdaMapValidationError(RuntimeError):
    """Raised when lambda_map.fits violates the strict contract."""


def _norm_wave_unit(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    ss = s.lower().replace("å", "angstrom").replace(" ", "")
    if ss in {"a", "aa", "ang", "angs", "angstrom", "ångström", "angstroms"}:
        return "Angstrom"
    if ss in {"nm", "nanometer", "nanometers"}:
        return "nm"
    if ss in {"pix", "pixel", "pixels"}:
        return "pix"
    return s


def _read_unit_and_ref(hdr: fits.Header) -> tuple[str, str, str]:
    """Return (unit, waveref, source_tag)."""
    for k in ("WAVEUNIT", "LAMUNIT", "CUNIT1", "BUNIT"):
        v = hdr.get(k)
        if v not in (None, ""):
            return (
                _norm_wave_unit(str(v)),
                str(hdr.get("WAVEREF", "") or "").strip().lower(),
                k,
            )
    return "", str(hdr.get("WAVEREF", "") or "").strip().lower(), "heuristic"


@dataclass(frozen=True)
class LambdaMapDiagnostics:
    shape: tuple[int, int]
    unit: str
    waveref: str
    unit_source: str
    valid_frac: float
    lam_min: float
    lam_max: float
    monotonic_sign: int
    monotonic_bad_frac: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "unit": self.unit,
            "waveref": self.waveref,
            "unit_source": self.unit_source,
            "valid_frac": float(self.valid_frac),
            "range": [float(self.lam_min), float(self.lam_max)],
            "monotonic_sign": int(self.monotonic_sign),
            "monotonic_bad_frac": float(self.monotonic_bad_frac),
        }


def validate_lambda_map(
    path: str | Path,
    *,
    expected_signature: FrameSignature | None = None,
    expected_shape: tuple[int, int] | None = None,
    expected_unit: str | None = None,
    expected_waveref: Literal["air", "vacuum"] | None = None,
    max_invalid_frac: float = 1e-3,
    monotonic_bad_frac_max: float = 0.01,
    sample_rows: int = 7,
) -> LambdaMapDiagnostics:
    """Validate lambda_map.fits and return diagnostics.

    Parameters are tuned to be strict enough to prevent silent science errors,
    while remaining robust to small masked regions.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"lambda_map not found: {p}")

    with fits.open(p, memmap=False) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data)
        hdr = hdu.header

    if data.ndim != 2:
        raise LambdaMapValidationError("lambda_map must be a 2D image (ny, nx)")

    ny, nx = int(data.shape[0]), int(data.shape[1])

    if expected_shape is not None and (ny, nx) != tuple(expected_shape):
        raise LambdaMapValidationError(
            f"lambda_map shape mismatch: got {ny}x{nx}, expected {expected_shape[0]}x{expected_shape[1]}"
        )

    if expected_signature is not None:
        got_sig = FrameSignature.from_header(hdr, fallback_shape=(ny, nx))
        if not got_sig.is_compatible_with(expected_signature):
            # Header may not carry full signature (binning/ROI). Treat shape mismatch as fatal,
            # but provide a readable diff when available.
            if got_sig.shape != expected_signature.shape:
                raise LambdaMapValidationError(
                    "lambda_map signature mismatch: "
                    + format_signature_mismatch(expected=expected_signature, got=got_sig, path=p)
                )

    unit, waveref, src = _read_unit_and_ref(hdr)
    # Prefer explicit wavelength metadata. For legacy or synthetic inputs we
    # allow heuristic/absent metadata *unless* the caller requires a specific
    # unit/waveref.
    if (src == "heuristic" or not unit) and expected_unit is not None:
        raise LambdaMapValidationError(
            "lambda_map.fits is missing explicit wavelength unit metadata (WAVEUNIT/LAMUNIT/CUNIT1/BUNIT)"
        )
    if not unit:
        # Last-resort assumption: Angstrom in air. We preserve the fact that
        # this was assumed in the returned diagnostics via unit_source.
        unit = "angstrom"
        if not waveref:
            waveref = "air"
        src = "assumed"

    if expected_unit is not None:
        exp_u = _norm_wave_unit(expected_unit)
        if exp_u and unit != exp_u:
            raise LambdaMapValidationError(f"lambda_map unit mismatch: got {unit}, expected {exp_u}")

    if waveref not in {"air", "vacuum"}:
        if expected_waveref is not None:
            raise LambdaMapValidationError(
                f"lambda_map WAVEREF must be 'air' or 'vacuum' (got {waveref!r})"
            )
    if expected_waveref is not None and waveref != expected_waveref:
        raise LambdaMapValidationError(f"lambda_map waveref mismatch: got {waveref}, expected {expected_waveref}")

    d = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(d)
    valid_frac = float(np.mean(finite)) if finite.size else 0.0
    invalid_frac = 1.0 - valid_frac
    if invalid_frac > float(max_invalid_frac):
        raise LambdaMapValidationError(
            f"lambda_map has too many invalid pixels: invalid_frac={invalid_frac:.4g} > {max_invalid_frac:.4g}"
        )

    if not np.any(finite):
        raise LambdaMapValidationError("lambda_map has no finite values")

    lam_min = float(np.nanmin(np.where(finite, d, np.nan)))
    lam_max = float(np.nanmax(np.where(finite, d, np.nan)))
    if not (np.isfinite(lam_min) and np.isfinite(lam_max) and lam_max > lam_min):
        raise LambdaMapValidationError("lambda_map has an invalid wavelength range")

    # --- monotonicity along x ---
    ys = np.linspace(0, ny - 1, num=min(sample_rows, ny), dtype=int)
    signs: list[int] = []
    bad_fracs: list[float] = []
    for y in ys:
        row = d[y, :]
        ok = np.isfinite(row)
        if ok.sum() < max(16, nx // 8):
            continue
        dr = np.diff(row)
        okd = np.isfinite(dr)
        if okd.sum() < max(16, nx // 8):
            continue
        med = float(np.nanmedian(dr[okd]))
        sgn = 1 if med > 0 else (-1 if med < 0 else 0)
        if sgn == 0:
            continue
        bad = float(np.mean((dr[okd] * sgn) <= 0.0))
        signs.append(sgn)
        bad_fracs.append(bad)

    if not signs:
        raise LambdaMapValidationError("lambda_map monotonicity check failed (not enough valid rows)")

    monotonic_sign = int(1 if np.median(signs) >= 0 else -1)
    monotonic_bad_frac = float(np.median(bad_fracs)) if bad_fracs else 1.0

    if monotonic_bad_frac > float(monotonic_bad_frac_max):
        raise LambdaMapValidationError(
            f"lambda_map is not monotonic along dispersion axis x: bad_frac={monotonic_bad_frac:.3f} > {monotonic_bad_frac_max:.3f}"
        )

    return LambdaMapDiagnostics(
        shape=(ny, nx),
        unit=unit,
        waveref=waveref or "",
        unit_source=src,
        valid_frac=valid_frac,
        lam_min=lam_min,
        lam_max=lam_max,
        monotonic_sign=monotonic_sign,
        monotonic_bad_frac=monotonic_bad_frac,
    )
