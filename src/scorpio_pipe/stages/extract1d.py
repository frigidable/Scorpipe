from __future__ import annotations

"""1D spectrum extraction.

Input: sky-subtracted, linearized 2D frame I(y, λ)
Output: 1D spectrum F(λ) obtained by summing/averaging rows in the
user-defined object region.
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from astropy.io import fits

from scorpio_pipe.wavesol_paths import resolve_work_dir
from scorpio_pipe.plot_style import mpl_style


def _wave_from_header(hdr: fits.Header, n: int) -> np.ndarray:
    crpix = float(hdr.get("CRPIX1", 1.0))
    crval = float(hdr.get("CRVAL1", 0.0))
    cdelt = float(hdr.get("CDELT1", 1.0))
    i = np.arange(n, dtype=float)
    return crval + (i + 1 - crpix) * cdelt


def _get_roi(cfg: dict[str, Any]) -> Tuple[int, int]:
    sky = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
    # v5 UI stores ROI as flat keys under [sky], e.g. roi_obj_y0.
    # Older/experimental configs may store a nested dict under [sky][roi].
    if "roi_obj_y0" in sky or "roi_obj_y1" in sky:
        y0 = sky.get("roi_obj_y0")
        y1 = sky.get("roi_obj_y1")
    else:
        roi = (sky.get("roi") or {}) if isinstance(sky.get("roi"), dict) else {}
        y0 = roi.get("obj_y0")
        y1 = roi.get("obj_y1")
    if y0 is None or y1 is None:
        raise ValueError(
            "Sky ROI is missing: please select OBJECT region first (Sky stage → Select regions…)."
        )
    y0 = int(y0)
    y1 = int(y1)
    if y1 < y0:
        y0, y1 = y1, y0
    return y0, y1


def extract_1d(cfg: dict[str, Any], *, in_fits: Path | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    """Extract 1D spectrum from the sky-subtracted linearized frame."""

    e1d = (cfg.get("extract1d") or {}) if isinstance(cfg.get("extract1d"), dict) else {}
    if not bool(e1d.get("enabled", True)):
        return {}

    work_dir = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = work_dir / "spec"
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_fits is None:
        in_fits = work_dir / "sky" / "obj_sky_sub.fits"

    if not Path(in_fits).exists():
        raise FileNotFoundError(f"Missing input for extract1d: {in_fits}")

    y0, y1 = _get_roi(cfg)

    var2d = None
    mask2d = None
    with fits.open(in_fits) as hdul:
        data = hdul[0].data.astype(float, copy=False)
        hdr = hdul[0].header.copy()
        if "VAR" in hdul:
            var2d = np.array(hdul["VAR"].data, dtype=float)
        if "MASK" in hdul:
            mask2d = np.array(hdul["MASK"].data, dtype=np.uint16)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data in {in_fits}, got shape={getattr(data, 'shape', None)}")

    ny, nx = data.shape
    y0c = max(0, min(ny - 1, y0))
    y1c = max(0, min(ny - 1, y1))
    if y1c < y0c:
        y0c, y1c = y1c, y0c

    method = str(e1d.get("method", "sum")).lower().strip()
    slab = data[y0c : y1c + 1, :]

    if method == "mean":
        flux = np.nanmean(slab, axis=0)
    else:
        flux = np.nansum(slab, axis=0)

    wave = _wave_from_header(hdr, nx)

    # Save as 1D image with WCS-like keywords
    out_fits = out_dir / "spectrum_1d.fits"
    ohdr = fits.Header()
    for k in ("OBJECT", "DATE-OBS", "EXPTIME", "INSTRUME", "TELESCOP"):
        if k in hdr:
            ohdr[k] = hdr[k]
    ohdr["CRPIX1"] = float(hdr.get("CRPIX1", 1.0))
    ohdr["CRVAL1"] = float(hdr.get("CRVAL1", wave[0]))
    ohdr["CDELT1"] = float(hdr.get("CDELT1", wave[1] - wave[0] if len(wave) > 1 else 1.0))
    ohdr["CTYPE1"] = str(hdr.get("CTYPE1", "WAVE"))
    ohdr["CUNIT1"] = str(hdr.get("CUNIT1", "Angstrom"))
    ohdr["EXTNAME"] = "SPEC1D"
    ohdr["YOBJ0"] = int(y0c)
    ohdr["YOBJ1"] = int(y1c)
    # Also store explicit wavelength vector as a table (helps external tools)
    # Variance / mask propagation (optional)
    nrows = int(y1c - y0c + 1)
    var1d = None
    if var2d is not None:
        slabv = np.asarray(var2d[y0c : y1c + 1, :], dtype=float)
        if method.lower() == "mean" and nrows > 0:
            var1d = np.nansum(slabv, axis=0) / float(nrows * nrows)
        else:
            var1d = np.nansum(slabv, axis=0)

    mask1d = None
    if mask2d is not None:
        slabm = np.asarray(mask2d[y0c : y1c + 1, :])
        mask1d = np.bitwise_or.reduce(slabm, axis=0).astype(np.uint16)

    cols = [
        fits.Column(name="WAVE", format="D", array=wave.astype(float)),
        fits.Column(name="FLUX", format="D", array=flux.astype(float)),
    ]
    if var1d is not None:
        cols.append(fits.Column(name="VAR", format="D", array=var1d.astype(float)))
    if mask1d is not None:
        cols.append(fits.Column(name="MASK", format="K", array=mask1d.astype(np.uint16)))

    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.header["EXTNAME"] = "SPECTRUM"
    fits.HDUList([fits.PrimaryHDU(flux.astype(np.float32), header=ohdr), tbhdu]).writeto(out_fits, overwrite=True)

    # CSV (easy diff/export)
    out_csv = out_dir / "spectrum_1d.csv"
    try:
        import csv

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["wave_A", "flux"]
            if var1d is not None:
                header.append("var")
            if mask1d is not None:
                header.append("mask")
            w.writerow(header)
            for i in range(nx):
                row = [float(wave[i]), float(flux[i])]
                if var1d is not None:
                    row.append(float(var1d[i]))
                if mask1d is not None:
                    row.append(int(mask1d[i]))
                w.writerow(row)
    except Exception:
        out_csv = None

    # PNG quicklook
    out_png = out_dir / "spectrum_1d.png"
    if bool(e1d.get("save_png", True)):
        try:
            import matplotlib.pyplot as plt

            with mpl_style():
                fig = plt.figure(figsize=(7.0, 3.4))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(wave, flux)
                ax.set_xlabel("Wavelength (Å)")
                ax.set_ylabel("Flux (ADU)")
                ax.set_title("Extracted 1D spectrum")
                fig.savefig(out_png)
                plt.close(fig)
        except Exception:
            pass

    done = out_dir / "extract1d_done.json"
    import json

    payload = {
        "stage": "extract1d",
        "input": str(in_fits),
        "output_fits": str(out_fits),
        "output_png": str(out_png),
        "method": method,
        "y0": int(y0c),
        "y1": int(y1c),
        "n_lambda": int(nx),
    }
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"spectrum_1d": out_fits, "spectrum_1d_png": out_png}
