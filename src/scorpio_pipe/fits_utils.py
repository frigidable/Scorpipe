from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from astropy.io import fits


_SCALE_KEYS = ("BZERO", "BSCALE", "BLANK")


def _pick_first_image_hdu(
    hdul: fits.HDUList,
) -> tuple[int, fits.ImageHDU | fits.PrimaryHDU | Any]:
    """Return (index, hdu) for the first image-like HDU.

    Some SCORPIO products may be MEF-like, with the image stored in extension 1.
    """

    last_err: Exception | None = None
    for i, h in enumerate(hdul):
        try:
            data = getattr(h, "data", None)
            if data is None:
                continue
            a = np.asarray(data)
            if a.ndim >= 2:
                return i, h
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    raise ValueError("No image-like HDU found")


def _needs_unscaled_memmap_off(hdul: fits.HDUList) -> bool:
    """Return True if any image HDU declares scaling/blanking keywords.

    Astropy cannot apply BZERO/BSCALE/BLANK scaling when the image is memory-mapped.
    In that case we must open with memmap=False to get correct scaled values.
    """
    for h in hdul:
        # PrimaryHDU / ImageHDU / CompImageHDU all have .header
        try:
            hdr = h.header
        except Exception:
            continue
        if any(k in hdr for k in _SCALE_KEYS):
            return True
    return False


def open_fits_smart(
    path: Union[str, Path],
    *,
    memmap: bool | str = "auto",
    prefer_memmap: bool | None = None,
    **kwargs,
) -> fits.HDUList:
    """Open a FITS file with "no surprises" defaults.

    Parameters
    ----------
    memmap
        ``True``/``False`` behave like :func:`astropy.io.fits.open`. ``"auto"``
        (default) will attempt memmap=True first but will automatically fall
        back to memmap=False when scaling/blanking keywords (BZERO/BSCALE/BLANK)
        are present. This avoids the well-known Astropy error:
        "Cannot load a memory-mapped image ... Set memmap=False".
    prefer_memmap
        Backward compatible alias for older code. If provided, it overrides
        ``memmap``.
    """

    # Backward compatibility: older call sites pass prefer_memmap.
    if prefer_memmap is not None:
        memmap = bool(prefer_memmap)

    path = str(path)
    mm = memmap

    if mm is False:
        return fits.open(path, memmap=False, **kwargs)

    # memmap=True or 'auto'
    hdul = fits.open(path, memmap=True, **kwargs)
    try:
        if (mm == "auto") and _needs_unscaled_memmap_off(hdul):
            hdul.close()
            hdul = fits.open(path, memmap=False, **kwargs)
    except Exception:
        # If anything goes wrong during header inspection, fall back to memmap=False.
        try:
            hdul.close()
        except Exception:
            pass
        hdul = fits.open(path, memmap=False, **kwargs)

    return hdul


def getdata_smart(
    path: Union[str, Path],
    *,
    prefer_memmap: bool = True,
    ext: Optional[Union[int, str]] = None,
    **kwargs,
):
    """fits.getdata equivalent with safe handling of BSCALE/BZERO/BLANK."""
    # fits.getdata does not let us inspect headers first without reading the data.
    # So use open_fits_smart and then access .data.
    with open_fits_smart(path, prefer_memmap=prefer_memmap, **kwargs) as hdul:
        h = hdul[0] if ext is None else hdul[ext]
        return np.asarray(h.data)


def read_image_smart(
    path: Union[str, Path],
    *,
    ext: Optional[Union[int, str]] = None,
    memmap: bool | str = "auto",
    dtype: Any = np.float32,
    squeeze: bool = True,
    **kwargs,
) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    """Read an image-like HDU into a normalized ndarray.

    Returns
    -------
    data
        NumPy array, materialized (not tied to an open file).
    header
        Header of the chosen HDU.
    info
        Small dictionary with loader details (memmap_used, scaling keys, dtype...).
    """

    memmap_used: bool | None = None

    hdu_index: int | None = None

    with open_fits_smart(
        path,
        memmap=memmap,
        ignore_missing_end=True,
        ignore_missing_simple=True,
        do_not_scale_image_data=False,
        **kwargs,
    ) as hdul:
        try:
            memmap_used = bool(getattr(getattr(hdul, "_file", None), "memmap", False))
        except Exception:
            memmap_used = None
        if ext is None:
            hdu_index, h = _pick_first_image_hdu(hdul)
        else:
            hdu_index = int(ext) if isinstance(ext, int) else None
            h = hdul[ext]
        hdr = h.header
        data = np.asarray(h.data)

    if squeeze and data.ndim > 2:
        data = np.squeeze(data)

    # Normalize dtype in a NumPy 2.0-safe way.
    data = np.asarray(data, dtype=dtype)

    info = {
        "memmap_requested": memmap,
        "memmap_used": memmap_used,
        "hdu_index": hdu_index,
        "has_scaling": any(k in hdr for k in _SCALE_KEYS),
        "bzero": hdr.get("BZERO"),
        "bscale": hdr.get("BSCALE"),
        "blank": hdr.get("BLANK"),
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    }
    return data, hdr, info
