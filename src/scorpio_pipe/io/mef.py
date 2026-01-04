from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER

DEFAULT_FATAL_BITS = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


from scorpio_pipe.version import as_header_cards
from scorpio_pipe import maskbits


@dataclass(frozen=True)
class WaveGrid:
    """Linear wavelength grid description for rectified products.

    Parameters
    ----------
    lambda0 : float
        Wavelength at pixel 1 (CRVAL1) in `unit`.
    dlambda : float
        Step per pixel (CDELT1) in `unit`.
    nlam : int
        Number of wavelength pixels along axis 1.
    unit : str
        FITS WCS unit string (e.g. 'Angstrom', 'nm').
    """

    lambda0: float
    dlambda: float
    nlam: int
    unit: str = "Angstrom"

    wave_ref: str = "UNKNOWN"
    def to_wcs_cards(self) -> dict[str, Any]:
        ref = str(self.wave_ref or "UNKNOWN").strip().upper()
        ctype = "AWAV" if ref in ("AIR", "A", "AWAV") else "WAVE"
        return {
            "CRVAL1": float(self.lambda0),
            "CDELT1": float(self.dlambda),
            "CRPIX1": 1.0,
            "CTYPE1": ctype,
            "CUNIT1": str(self.unit),
            "WAVEREF": ref.lower() if ref != "UNKNOWN" else "unknown",
            "WAVEUNIT": str(self.unit),
        }


def _apply_cards(hdr: fits.Header, cards: dict[str, Any]) -> None:
    for k, v in cards.items():
        try:
            hdr[k] = v
        except Exception:
            # Keep going even if some cards are non-standard
            pass


def write_sci_var_mask(
    path: str | Path,
    sci: np.ndarray,
    *,
    var: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    header: fits.Header | None = None,
    grid: WaveGrid | None = None,
    cov: np.ndarray | None = None,
    overwrite: bool = True,
    validate: bool = True,
    extra_hdus: list[fits.ImageHDU] | None = None,
    primary_data: np.ndarray | None = None,
) -> Path:
    """Write a multi-extension FITS (MEF): SCI [+VAR] [+MASK] [+extras].

    The primary HDU keeps provenance; science arrays live in EXTNAME=SCI etc.

    Notes
    -----
    - `mask` is expected to be a uint16 bitmask using :mod:`scorpio_pipe.maskbits`.
    - `var` is expected to be variance in the same units as `sci` squared.
    - If provided, `cov` is a per-pixel integer coverage map (number of contributing samples).
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    sci = np.asarray(sci, dtype=float)
    if var is not None:
        var = np.asarray(var, dtype=float)
        if var.shape != sci.shape:
            raise ValueError(f"VAR shape {var.shape} != SCI shape {sci.shape}")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != sci.shape:
            raise ValueError(f"MASK shape {mask.shape} != SCI shape {sci.shape}")
        if mask.dtype != np.uint16:
            # NumPy 2.0 strictness: allow a copy when needed.
            mask = np.asarray(mask, dtype=np.uint16)

    if cov is not None:
        cov = np.asarray(cov)
        if cov.shape != sci.shape:
            raise ValueError(f"COV shape {cov.shape} != SCI shape {sci.shape}")

    # Sanitize non-finite pixels: enforce fully-finite SCI/VAR at boundaries.
    # We mark such pixels as NO_COVERAGE and zero them.
    scorpnan = 0
    try:
        bad_sci = ~np.isfinite(sci)
        bad_var = ~np.isfinite(var) if var is not None else None
        if bad_var is not None:
            scorpnan = int(np.count_nonzero(bad_sci) + np.count_nonzero(bad_var))
            bad = bad_sci | bad_var
        else:
            scorpnan = int(np.count_nonzero(bad_sci))
            bad = bad_sci

        if scorpnan > 0:
            if mask is None:
                mask = np.zeros(sci.shape, dtype=np.uint16)
            mask = np.asarray(mask, dtype=np.uint16)
            mask[bad] = np.bitwise_or(mask[bad], np.uint16(NO_COVERAGE))
            sci[bad] = 0.0
            if var is not None:
                var[bad] = 0.0
    except Exception:
        scorpnan = 0

    phdr = fits.Header() if header is None else fits.Header(header)
    _apply_cards(phdr, as_header_cards(prefix="SCORP"))
    # Record the strict mask schema (even if MASK is absent; this makes downstream checks simpler).
    _apply_cards(phdr, maskbits.header_cards(prefix="SCORP"))

    if scorpnan > 0:
        try:
            phdr["SCORPNAN"] = (int(scorpnan), "Non-finite SCI/VAR sanitized to 0 (NO_COVERAGE)")
        except Exception:
            pass

    # Store grid metadata both as WCS cards and explicit SCORP_* keywords
    if grid is not None:
        _apply_cards(phdr, grid.to_wcs_cards())
        _apply_cards(
            phdr,
            {
                "SCORP_L0": float(grid.lambda0),
                "SCORP_DL": float(grid.dlambda),
                "SCORP_NL": int(grid.nlam),
                "SCORP_LU": str(grid.unit),
            },
        )
        # Ensure explicit wavelength unit keyword exists for downstream tooling.
        if "WAVEUNIT" not in phdr:
            try:
                phdr["WAVEUNIT"] = (str(grid.unit), "Wavelength unit (explicit)")
            except Exception:
                pass

    hdus: list[fits.HDUBase] = []
    if primary_data is not None:
        hdus.append(
            fits.PrimaryHDU(
                data=np.asarray(primary_data, dtype=np.float32), header=phdr
            )
        )
    else:
        hdus.append(fits.PrimaryHDU(header=phdr))

    shdr = fits.Header()
    if grid is not None:
        _apply_cards(shdr, grid.to_wcs_cards())
    # Propagate key unit metadata to SCI for self-describing products.
    # (VAR/MASK have different physical units, so we do not copy BUNIT there.)
    for k in ("BUNIT", "WAVEUNIT", "WAVEREF"):
        if k in phdr and k not in shdr:
            try:
                shdr[k] = phdr[k]
            except Exception:
                pass
    sci_hdu = fits.ImageHDU(
        data=np.asarray(sci, dtype=np.float32), header=shdr, name="SCI"
    )
    hdus.append(sci_hdu)

    if var is not None:
        vhdr = fits.Header()
        if grid is not None:
            _apply_cards(vhdr, grid.to_wcs_cards())
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(var, dtype=np.float32), header=vhdr, name="VAR"
            )
        )

    if mask is not None:
        mhdr = fits.Header()
        if grid is not None:
            _apply_cards(mhdr, grid.to_wcs_cards())
        _apply_cards(mhdr, maskbits.header_cards(prefix="SCORP"))
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(mask, dtype=np.uint16), header=mhdr, name="MASK"
            )
        )

    if cov is not None:
        chdr = fits.Header()
        if grid is not None:
            _apply_cards(chdr, grid.to_wcs_cards())
        # Coverage is unitless (counts)
        try:
            chdr["BUNIT"] = ("count", "Coverage samples per pixel")
        except Exception:
            pass
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(cov, dtype=np.int16), header=chdr, name="COV"
            )
        )

    if extra_hdus:
        hdus.extend(extra_hdus)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def read_sci_var_mask(
    path: str | Path,
    *,
    validate: bool = True,
    fatal_bits: int = DEFAULT_FATAL_BITS,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
    """Read a MEF product with SCI/VAR/MASK extensions.

    If ``validate=True`` (default), run a lightweight SCI/VAR/MASK contract check
    and raise a :class:`ValueError` on violations.
    """
    path = Path(path).expanduser().resolve()
    with fits.open(path) as hdul:
        hdr = fits.Header(hdul[0].header)
        sci = np.asarray(hdul["SCI"].data, dtype=float)
        var = None
        mask = None
        if "VAR" in hdul:
            var = np.asarray(hdul["VAR"].data, dtype=float)
        if "MASK" in hdul:
            mask = np.asarray(hdul["MASK"].data, dtype=np.uint16)

    if validate:
        issues = validate_sci_var_mask(sci, var, mask, fatal_bits=int(fatal_bits))
        if issues:
            raise ValueError(f"MEF contract violation in {path}: {issues}")

    return sci, var, mask, hdr


def try_read_cov(path: str | Path) -> np.ndarray | None:
    """Return COV extension as int16 if present, else None."""
    path = Path(path).expanduser().resolve()
    try:
        with fits.open(path) as hdul:
            if "COV" not in hdul:
                return None
            return np.asarray(hdul["COV"].data, dtype=np.int16)
    except Exception:
        return None


def validate_sci_var_mask(
    sci: np.ndarray,
    var: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    cov: np.ndarray | None = None,
    fatal_bits: int | None = None,
) -> list[dict[str, Any]]:
    """Validate core 2D product arrays.

    Returns a list of issue dicts (empty means OK). This is intentionally
    lightweight (no I/O) so stages can call it cheaply.

    Rules
    -----
    - Shapes must match.
    - Where pixels are *not* fatally masked, SCI must be finite and VAR must be
      finite and >= 0.
    - MASK must be uint16.
    - If COV is present, it must be integer and non-negative.
    """
    issues: list[dict[str, Any]] = []

    sci = np.asarray(sci)
    if var is not None:
        var = np.asarray(var)
        if var.shape != sci.shape:
            issues.append({"code": "SHAPE", "message": f"VAR shape {var.shape} != SCI shape {sci.shape}"})
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != sci.shape:
            issues.append({"code": "SHAPE", "message": f"MASK shape {mask.shape} != SCI shape {sci.shape}"})
        if mask.dtype != np.uint16:
            issues.append({"code": "DTYPE", "message": f"MASK dtype {mask.dtype} != uint16"})

    if cov is not None:
        cov = np.asarray(cov)
        if cov.shape != sci.shape:
            issues.append({"code": "SHAPE", "message": f"COV shape {cov.shape} != SCI shape {sci.shape}"})
        if not np.issubdtype(cov.dtype, np.integer):
            issues.append({"code": "DTYPE", "message": f"COV dtype {cov.dtype} is not integer"})
        else:
            try:
                if np.nanmin(cov) < 0:
                    issues.append({"code": "RANGE", "message": "COV has negative values"})
            except Exception:
                pass

    if var is not None:
        good: np.ndarray
        if mask is not None and fatal_bits is not None:
            good = (mask & np.uint16(int(fatal_bits))) == 0
        else:
            good = np.ones(sci.shape, dtype=bool)

        try:
            if not np.isfinite(sci[good]).all():
                issues.append({"code": "NAN", "message": "SCI has non-finite values in unmasked pixels"})
        except Exception:
            pass

        try:
            v = var[good]
            if not np.isfinite(v).all():
                issues.append({"code": "NAN", "message": "VAR has non-finite values in unmasked pixels"})
            if np.nanmin(v) < 0:
                issues.append({"code": "RANGE", "message": "VAR has negative values in unmasked pixels"})
        except Exception:
            pass

    return issues


def try_read_grid(hdr: fits.Header) -> WaveGrid | None:
    """Best-effort parse linear wavelength grid from FITS header."""
    try:
        l0 = hdr.get("SCORP_L0", hdr.get("CRVAL1", None))
        dl = hdr.get("SCORP_DL", hdr.get("CDELT1", None))
        nl = hdr.get("SCORP_NL", hdr.get("NAXIS1", None))
        lu = hdr.get("SCORP_LU", hdr.get("CUNIT1", "Angstrom"))
        wr = hdr.get("SCORP_WR", hdr.get("WAVEREF", "UNKNOWN"))
        if l0 is None or dl is None or nl is None:
            return None
        return WaveGrid(
            lambda0=float(l0), dlambda=float(dl), nlam=int(nl), unit=str(lu), wave_ref=str(wr)
        )
    except Exception:
        return None
