"""Strict typed sidecar reader for cosmics-cleaned primary-HDU FITS.

The legacy *clean.fits primary image remains unchanged.  New products stamp
SATMSK=DQ_V1 on both that image and its mandatory uint16 DQ companion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from scorpio_pipe.maskbits import COSMIC


def read_clean_dq_sidecar(
    clean_path: Path, hdr: fits.Header, shape: tuple[int, int],
) -> np.ndarray | None:
    """Return the adjacent science DQ; refuse loss/corruption of qualified DQ.

    Old 0/1 cosmic masks are normalized to the COSMIC bit.  Unlike legacy
    masks, DQ_V1 must have matching provenance on BOTH files and never
    converts zero/one content into detector evidence.
    """
    path = Path(clean_path)
    if not path.stem.endswith("_clean"):
        return None
    qualified = str(hdr.get("SATMSK") or "").strip().upper() == "DQ_V1"
    dq_path = path.with_name(path.stem + "_mask.fits")
    if not dq_path.exists():
        if qualified:
            raise ValueError(f"Missing qualified cosmics DQ sidecar: {dq_path}")
        return None

    with fits.open(dq_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"Empty cosmics DQ sidecar: {dq_path}")
        m = np.asarray(hdul[0].data)
        dq_hdr = hdul[0].header
        if qualified:
            if str(dq_hdr.get("SATMSK") or "").strip().upper() != "DQ_V1":
                raise ValueError(f"Invalid cosmics DQ_V1 sidecar marker: {dq_path}")
            for key in ("SATQ", "SATADU", "SATMARG", "SATSRC"):
                if key in hdr and (
                    key not in dq_hdr or str(hdr[key]) != str(dq_hdr[key])
                ):
                    raise ValueError(
                        f"Cosmics DQ provenance {key} differs from cleaned SCI: {dq_path}"
                    )
        elif str(hdr.get("SATQ") or "").strip().upper() == "RAW_EVAL":
            # A copied RAW_EVAL header without qualified sidecar is not proof.
            raise ValueError(f"Unqualified cosmics DQ with RAW_EVAL header: {dq_path}")

    if m.shape != tuple(shape) or not np.issubdtype(m.dtype, np.integer):
        raise ValueError(f"Cosmics DQ has wrong shape or noninteger dtype: {dq_path}")
    if np.any(m < 0) or np.any(m > np.iinfo(np.uint16).max):
        raise ValueError(f"Cosmics DQ outside uint16 range: {dq_path}")
    out = np.asarray(m, dtype=np.uint16)
    if not qualified and (not out.size or int(out.max()) <= 1):
        out = np.where(out > 0, COSMIC, 0).astype(np.uint16)
    return out
