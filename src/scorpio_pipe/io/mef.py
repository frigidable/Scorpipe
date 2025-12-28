from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.version import as_header_cards


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

    def to_wcs_cards(self) -> dict[str, Any]:
        return {
            "CRVAL1": float(self.lambda0),
            "CDELT1": float(self.dlambda),
            "CRPIX1": 1.0,
            "CTYPE1": "WAVE",
            "CUNIT1": str(self.unit),
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
    overwrite: bool = True,
    extra_hdus: list[fits.ImageHDU] | None = None,
    primary_data: np.ndarray | None = None,
) -> Path:
    """Write a multi-extension FITS (MEF): SCI [+VAR] [+MASK] [+extras].

    The primary HDU keeps provenance; science arrays live in EXTNAME=SCI etc.

    Notes
    -----
    - `mask` is expected to be a uint16 bitmask (but we do not enforce a schema here).
    - `var` is expected to be variance in the same units as `sci` squared.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    sci = np.asarray(sci)
    if var is not None:
        var = np.asarray(var)
        if var.shape != sci.shape:
            raise ValueError(f"VAR shape {var.shape} != SCI shape {sci.shape}")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != sci.shape:
            raise ValueError(f"MASK shape {mask.shape} != SCI shape {sci.shape}")
        if mask.dtype != np.uint16:
            # NumPy 2.0 strictness: allow a copy when needed.
            mask = np.asarray(mask, dtype=np.uint16)

    phdr = fits.Header() if header is None else fits.Header(header)
    _apply_cards(phdr, as_header_cards(prefix="SCORP"))

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
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(mask, dtype=np.uint16), header=mhdr, name="MASK"
            )
        )

    if extra_hdus:
        hdus.extend(extra_hdus)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path


def read_sci_var_mask(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, fits.Header]:
    """Read a MEF product with SCI/VAR/MASK extensions."""
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
    return sci, var, mask, hdr


def try_read_grid(hdr: fits.Header) -> WaveGrid | None:
    """Best-effort parse linear wavelength grid from FITS header."""
    try:
        l0 = hdr.get("SCORP_L0", hdr.get("CRVAL1", None))
        dl = hdr.get("SCORP_DL", hdr.get("CDELT1", None))
        nl = hdr.get("SCORP_NL", hdr.get("NAXIS1", None))
        lu = hdr.get("SCORP_LU", hdr.get("CUNIT1", "Angstrom"))
        if l0 is None or dl is None or nl is None:
            return None
        return WaveGrid(
            lambda0=float(l0), dlambda=float(dl), nlam=int(nl), unit=str(lu)
        )
    except Exception:
        return None
