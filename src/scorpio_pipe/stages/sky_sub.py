from __future__ import annotations

"""Night-sky subtraction (Kelson-like baseline implementation).

v5.0 implements a pragmatic, fast variant suitable for interactive use:

1) build a robust sky spectrum S(λ) from user-selected sky rows
   (top + bottom regions, excluding the object rows)
2) smooth S(λ) with a cubic B-spline fit with iterative sigma clipping
3) model spatial variation with a low-order polynomial in y:
      sky(y,λ) ≈ a(y) * S(λ) + b(y)
   where a(y), b(y) are fitted using sky rows only
4) subtract the model and write:
   - sky_model.fits
   - obj_sky_sub.fits

This is *not* a full re-implementation of Kelson (2003) on unrectified data,
but provides the same user-facing semantics and can be swapped out by a more
advanced method later.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import csv
import shutil
import numpy as np
from astropy.io import fits

from scorpio_pipe.wavesol_paths import resolve_work_dir
from scorpio_pipe.plot_style import mpl_style


def _write_mef(
    path: Path,
    sci: np.ndarray,
    hdr: fits.Header,
    *,
    var: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """Write a simple MEF: PRIMARY=SCI, optional VAR/MASK extensions."""

    hdus: list[fits.HDUBase] = [fits.PrimaryHDU(np.asarray(sci), header=hdr)]
    if var is not None:
        hdus.append(fits.ImageHDU(np.asarray(var), name="VAR"))
    if mask is not None:
        hdus.append(fits.ImageHDU(np.asarray(mask), name="MASK"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


@dataclass(frozen=True)
class ROI:
    obj_y0: int
    obj_y1: int
    sky_top_y0: int
    sky_top_y1: int
    sky_bot_y0: int
    sky_bot_y1: int


def _roi_from_cfg(cfg: dict[str, Any]) -> ROI:
    sky = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
    roi = (sky.get("roi") or {}) if isinstance(sky.get("roi"), dict) else {}

    def _g(*keys: str, default: int | None = None) -> int:
        for k in keys:
            if k in roi and roi[k] is not None:
                return int(roi[k])
        if default is None:
            raise KeyError(f"Missing ROI key(s): {keys}")
        return int(default)

    return ROI(
        obj_y0=_g("obj_y0", "obj_ymin"),
        obj_y1=_g("obj_y1", "obj_ymax"),
        sky_top_y0=_g("sky_top_y0", "sky_up_y0", "sky1_y0"),
        sky_top_y1=_g("sky_top_y1", "sky_up_y1", "sky1_y1"),
        sky_bot_y0=_g("sky_bot_y0", "sky_down_y0", "sky2_y0"),
        sky_bot_y1=_g("sky_bot_y1", "sky_down_y1", "sky2_y1"),
    )


def _wave_from_header(hdr: fits.Header, n: int) -> np.ndarray:
    crval = float(hdr.get("CRVAL1", 0.0))
    cdelt = float(hdr.get("CDELT1", 1.0))
    crpix = float(hdr.get("CRPIX1", 1.0))
    # FITS WCS: world = CRVAL + (pix-CRPIX)*CDELT, pix is 1-based
    pix = np.arange(n, dtype=float) + 1.0
    return crval + (pix - crpix) * cdelt


def _bspline_basis(x: np.ndarray, t: np.ndarray, deg: int) -> np.ndarray:
    """Evaluate B-spline basis matrix B(x) for knot vector t and degree deg.

    Returns B of shape (len(x), n_basis).
    Pure NumPy Cox–de Boor recursion. Sufficient for moderate sizes (N~2000).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if deg < 0:
        raise ValueError("deg must be >=0")
    n_basis = len(t) - deg - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector length")

    # k=0
    B = np.zeros((x.size, n_basis), dtype=float)
    for i in range(n_basis):
        left = t[i]
        right = t[i + 1]
        sel = (x >= left) & (x < right)
        B[sel, i] = 1.0
    # include last point exactly at end
    B[x == t[-1], -1] = 1.0

    # recursion
    for k in range(1, deg + 1):
        Bk = np.zeros_like(B)
        for i in range(n_basis):
            d1 = t[i + k] - t[i]
            d2 = t[i + k + 1] - t[i + 1]
            term1 = 0.0
            term2 = 0.0
            if d1 > 0:
                term1 = (x - t[i]) / d1 * B[:, i]
            if d2 > 0 and i + 1 < n_basis:
                term2 = (t[i + k + 1] - x) / d2 * B[:, i + 1]
            Bk[:, i] = term1 + term2
        B = Bk
    return B


def _fit_bspline_1d(x: np.ndarray, y: np.ndarray, *, step: float, deg: int = 3, sigma_clip: float = 3.0, maxiter: int = 6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < max(20, deg + 5):
        return y

    x0 = float(np.nanmin(x[m]))
    x1 = float(np.nanmax(x[m]))
    if step <= 0:
        step = (x1 - x0) / 200.0

    # internal knots (excluding endpoints)
    internal = np.arange(x0, x1 + step, step, dtype=float)
    if internal.size < 4:
        return y

    # open knot vector
    t = np.concatenate([
        np.full(deg + 1, x0, dtype=float),
        internal[1:-1],
        np.full(deg + 1, x1, dtype=float),
    ])

    B = _bspline_basis(x, t, deg)

    mask = m.copy()
    for _ in range(int(maxiter)):
        xm = x[mask]
        ym = y[mask]
        Bm = B[mask, :]
        if xm.size < max(20, deg + 5):
            break
        # least squares
        try:
            c, *_ = np.linalg.lstsq(Bm, ym, rcond=None)
        except Exception:
            break
        yfit = B @ c
        resid = y - yfit
        r = resid[mask]
        if r.size < 20:
            break
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med))
        sigma = 1.4826 * mad if mad > 0 else np.nanstd(r)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_mask = m & (np.abs(resid - med) <= sigma_clip * sigma)
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask

    # final
    try:
        c, *_ = np.linalg.lstsq(B[mask, :], y[mask], rcond=None)
        return B @ c
    except Exception:
        return y


def run_sky_sub(cfg: dict[str, Any], *, lin_fits: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    """Run sky subtraction.

    Parameters
    ----------
    cfg
        Resolved config dict.
    lin_fits
        Optional path to the linearized stacked frame. If None,
        uses work_dir/lin/obj_sum_lin.fits.
    out_dir
        Output directory. Defaults to work_dir/sky.
    """

    sky_cfg = (cfg.get("sky") or {}) if isinstance(cfg.get("sky"), dict) else {}
    # v5.12 defaults (best-practice): products/ as canonical outputs.
    wd = resolve_work_dir(cfg)
    products_root = wd / "products"
    legacy_root = wd / "sky"
    if not bool(sky_cfg.get("enabled", True)):
        # Write a marker anyway to keep resume/QC stable.
        wd = resolve_work_dir(cfg)
        out_dir = Path(out_dir) if out_dir is not None else (wd / "sky")
        out_dir.mkdir(parents=True, exist_ok=True)
        done = out_dir / "sky_sub_done.json"
        done.write_text(json.dumps({"skipped": True, "reason": "sky.enabled=false"}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"skipped": True, "reason": "sky.enabled=false", "out_dir": str(out_dir)}

    roi = _roi_from_cfg(cfg)

    out_dir = Path(out_dir) if out_dir is not None else (products_root / "sky")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_exposure = bool(sky_cfg.get("per_exposure", False))
    stack_after = bool(sky_cfg.get("stack_after", False))
    save_per_exp_model = bool(sky_cfg.get("save_per_exp_model", False))
    save_spectrum_1d = bool(sky_cfg.get("save_spectrum_1d", False))

    # Helper: mirror a product into legacy folder for backward compatibility.
    def _mirror_legacy(src: Path, rel: str) -> None:
        try:
            dst = legacy_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception:
            pass

    def _process_one(lin_path: Path, *, tag: str, base_dir: Path, write_model: bool) -> dict[str, Any]:
        var = None
        mask = None
        with fits.open(lin_path, memmap=False) as hdul:
            data = np.array(hdul[0].data, dtype=float)
            hdr = hdul[0].header.copy()
            if "VAR" in hdul:
                try:
                    var = np.array(hdul["VAR"].data, dtype=float)
                except Exception:
                    var = None
            if "MASK" in hdul:
                try:
                    mask = np.array(hdul["MASK"].data, dtype=np.uint16)
                except Exception:
                    mask = None

        ny, nx = data.shape
        wave = _wave_from_header(hdr, nx)

        def _clip(a: int) -> int:
            return int(np.clip(a, 0, ny - 1))

        obj_y0, obj_y1 = sorted((_clip(roi.obj_y0), _clip(roi.obj_y1)))
        st0, st1 = sorted((_clip(roi.sky_top_y0), _clip(roi.sky_top_y1)))
        sb0, sb1 = sorted((_clip(roi.sky_bot_y0), _clip(roi.sky_bot_y1)))

        sky_rows = np.zeros(ny, dtype=bool)
        sky_rows[st0 : st1 + 1] = True
        sky_rows[sb0 : sb1 + 1] = True
        sky_rows[obj_y0 : obj_y1 + 1] = False
        if sky_rows.sum() < 3:
            raise ValueError("Sky ROI is too small (need at least a few rows)")

        sky_pix = data[sky_rows, :]
        sky_spec_raw = np.nanmedian(sky_pix, axis=0)

        deg = int(sky_cfg.get("bsp_degree", 3))
        step = float(sky_cfg.get("bsp_step_A", 3.0))
        sigma_clip = float(sky_cfg.get("sigma_clip", 3.0))
        maxiter = int(sky_cfg.get("maxiter", 6))
        sky_spec = _fit_bspline_1d(wave, sky_spec_raw, step=step, deg=deg, sigma_clip=sigma_clip, maxiter=maxiter)

        use_spatial = bool(sky_cfg.get("use_spatial_scale", True))
        poly_deg = int(sky_cfg.get("spatial_poly_deg", 1))
        if poly_deg < 0:
            poly_deg = 0
        a_y = np.ones(ny, dtype=float)
        b_y = np.zeros(ny, dtype=float)
        if use_spatial:
            ys = np.where(sky_rows)[0].astype(float)
            a_s = []
            b_s = []
            for y in ys.astype(int):
                row = data[y, :]
                m = np.isfinite(row) & np.isfinite(sky_spec)
                if m.sum() < 30:
                    a_s.append(np.nan)
                    b_s.append(np.nan)
                    continue
                X = np.vstack([sky_spec[m], np.ones(m.sum(), dtype=float)]).T
                try:
                    (a, b), *_ = np.linalg.lstsq(X, row[m], rcond=None)
                except Exception:
                    a, b = np.nan, np.nan
                a_s.append(a)
                b_s.append(b)
            a_s = np.asarray(a_s, dtype=float)
            b_s = np.asarray(b_s, dtype=float)
            good = np.isfinite(a_s) & np.isfinite(b_s)
            if good.sum() >= max(poly_deg + 2, 5):
                pa = np.polyfit(ys[good], a_s[good], deg=poly_deg)
                pb = np.polyfit(ys[good], b_s[good], deg=poly_deg)
                y_all = np.arange(ny, dtype=float)
                a_y = np.polyval(pa, y_all)
                b_y = np.polyval(pb, y_all)

        sky_model = a_y[:, None] * sky_spec[None, :] + b_y[:, None]
        sky_sub = data - sky_model

        hdr_out = hdr.copy()
        hdr_out["HISTORY"] = "Scorpio Pipe v5.12: sky subtraction"
        hdr_out["SKYMETH"] = str(sky_cfg.get("method", "kelson"))
        hdr_out["OBJY0"] = int(obj_y0)
        hdr_out["OBJY1"] = int(obj_y1)

        # per-exposure naming if needed
        sky_model_path = base_dir / f"{tag}_sky_model.fits"
        sky_sub_path = base_dir / f"{tag}_sky_sub.fits"
        sky_spec_csv = base_dir / f"{tag}_sky_spectrum.csv"
        sky_spec_json = base_dir / f"{tag}_sky_spectrum.json"

        if write_model:
            _write_mef(sky_model_path, np.asarray(sky_model, dtype=np.float32), hdr_out)

        _write_mef(
            sky_sub_path,
            np.asarray(sky_sub, dtype=np.float32),
            hdr_out,
            var=None if var is None else np.asarray(var, dtype=np.float32),
            mask=None if mask is None else np.asarray(mask, dtype=np.uint16),
        )

        # 1D sky spectrum export (optional)
        if save_spectrum_1d:
            try:
                with sky_spec_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["wave_A", "sky_raw", "sky_fit"])
                    for i in range(nx):
                        w.writerow([float(wave[i]), float(sky_spec_raw[i]), float(sky_spec[i])])
                sky_spec_json.write_text(
                    json.dumps(
                        {
                            "tag": tag,
                            "wave_A": wave.tolist(),
                            "sky_raw": sky_spec_raw.tolist(),
                            "sky_fit": sky_spec.tolist(),
                            "bsp": {"degree": deg, "step_A": step, "sigma_clip": sigma_clip, "maxiter": maxiter},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # QC metrics in sky rows
        resid = sky_sub[sky_rows, :]
        q = {
            "tag": tag,
            "rms_sky": float(np.nanstd(resid)),
            "mae_sky": float(np.nanmedian(np.abs(resid))),
            "n_sky_rows": int(sky_rows.sum()),
        }
        return {
            "ok": True,
            "lin_fits": str(lin_path),
            "sky_model": str(sky_model_path) if write_model else None,
            "sky_sub": str(sky_sub_path),
            "sky_spec_csv": str(sky_spec_csv) if (save_spectrum_1d and sky_spec_csv.exists()) else None,
            "sky_spec_json": str(sky_spec_json) if (save_spectrum_1d and sky_spec_json.exists()) else None,
            "qc": q,
        }

    # Determine inputs
    if not per_exposure:
        if lin_fits is None:
            lin_fits = wd / "lin" / "obj_sum_lin.fits"
        lin_fits = Path(lin_fits)
        if not lin_fits.exists():
            raise FileNotFoundError(f"Missing linearized sum: {lin_fits} (run linearize first)")
        one = _process_one(
            lin_fits,
            tag="obj",
            base_dir=out_dir,
            write_model=bool(sky_cfg.get("save_sky_model", True)),
        )
        # Mirror to legacy names
        if one.get("sky_sub"):
            _mirror_legacy(Path(one["sky_sub"]), "obj_sky_sub.fits")
        if one.get("sky_model"):
            _mirror_legacy(Path(one["sky_model"]), "sky_model.fits")
        payload = {"mode": "stack", "out_dir": str(out_dir), "result": one}
    else:
        per_dir = wd / "lin" / "per_exp"
        if not per_dir.exists():
            raise FileNotFoundError(f"Missing per-exposure linearized frames: {per_dir}")
        out_per = out_dir / "per_exp"
        out_per.mkdir(parents=True, exist_ok=True)
        files = sorted(per_dir.glob("*.fits"))
        if not files:
            raise FileNotFoundError(f"No per-exposure linearized FITS found in {per_dir}")
        results: list[dict[str, Any]] = []
        for f in files:
            tag = f.stem
            res = _process_one(
                f,
                tag=tag,
                base_dir=out_per,
                write_model=bool(sky_cfg.get("save_sky_model", True)) and save_per_exp_model,
            )
            results.append(res)
        payload = {"mode": "per_exposure", "out_dir": str(out_dir), "per_exp": results, "stack_after": stack_after}

        # Run stacking as part of Sky, if requested.
        if stack_after:
            try:
                from .stack2d import run_stack2d

                sky_sub_files = [Path(r["sky_sub"]) for r in results if r.get("sky_sub")]
                stk = run_stack2d(cfg, inputs=sky_sub_files, out_dir=products_root / "stack")
                payload["stack2d"] = stk

                # For downstream stages (extract1d) and backward compatibility:
                # publish a combined product under the legacy and products/sky names.
                try:
                    src = Path(stk.get("output_fits", ""))
                    if src.exists():
                        comb = out_dir / "obj_sky_sub.fits"
                        shutil.copy2(src, comb)
                        _mirror_legacy(comb, "obj_sky_sub.fits")
                        payload["combined_sky_sub"] = str(comb)
                except Exception:
                    pass
            except Exception as e:
                payload["stack2d_error"] = str(e)

    done = out_dir / "sky_sub_done.json"
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _mirror_legacy(done, "sky_sub_done.json")
    return payload
