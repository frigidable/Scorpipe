from __future__ import annotations

"""Atomic FITS writes.

Why
---
Pipeline stages write many FITS products. A crash/kill during write may leave a
truncated / partially-written FITS that then breaks resume logic.

We write to a temp file in the *same directory* and then swap it into place via
``os.replace`` (atomic on POSIX and Windows for same-filesystem paths).

Notes
-----
- This helper intentionally lives in :mod:`scorpio_pipe.io` as a single point of
  truth. Stages should not call ``fits.writeto(..., overwrite=True)`` directly.
- A leftover ``*.tmp`` is safe: it never replaces the target unless the write
  completed successfully.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from astropy.io import fits


def _fsync_path(path: Path) -> None:
    """Best-effort durability (flush file + directory metadata)."""
    try:
        with path.open("rb") as f:
            os.fsync(f.fileno())
    except Exception:
        # Durability is best-effort; atomicity is provided by os.replace.
        return

    # Fsync directory entry where supported (POSIX). No-op / may fail on Windows.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def atomic_write_fits(
    path: str | Path,
    obj: Any,
    *,
    header: fits.Header | None = None,
    overwrite: bool = True,
    tmp_suffix: str = ".tmp",
    **writeto_kwargs: Any,
) -> Path:
    """Atomically write a FITS product.

    Parameters
    ----------
    path
        Destination FITS path.
    obj
        One of:
        - ``astropy.io.fits.HDUList`` or any HDU instance supporting ``.writeto``;
        - a numpy-like data array to be written as a primary HDU.
    header
        FITS header to be used when ``obj`` is a data array.
    overwrite
        If False and destination exists, raise :class:`FileExistsError`.
    tmp_suffix
        Suffix for the temporary file name (default: ".tmp").
    writeto_kwargs
        Passed through to Astropy ``writeto`` (except ``overwrite`` which is
        managed internally).

    Returns
    -------
    Path
        The final destination path.
    """
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing FITS: {dst}")

    fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=tmp_suffix, dir=str(dst.parent))
    tmp_path = Path(tmp)
    # Important for Windows: close the handle before Astropy opens the file.
    os.close(fd)

    try:
        # Always allow writing to the temp path (it already exists).
        if hasattr(obj, "writeto") and not isinstance(obj, (bytes, str, Path)):
            # HDUList / HDU
            obj.writeto(tmp_path, overwrite=True, **{k: v for k, v in writeto_kwargs.items() if k != "overwrite"})
        else:
            # data array
            fits.writeto(
                tmp_path,
                obj,
                header=header,
                overwrite=True,
                **{k: v for k, v in writeto_kwargs.items() if k != "overwrite"},
            )

        _fsync_path(tmp_path)
        os.replace(str(tmp_path), str(dst))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return dst
