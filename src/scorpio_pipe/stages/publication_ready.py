from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.mef import read_sci_var_mask, write_sci_var_mask, WaveGrid
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.wavesol_paths import wavesol_dir
from scorpio_pipe.workspace_paths import stage_dir


def _load_primary_or_sci(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None and "SCI" in hdul:
            data = hdul["SCI"].data
        if data is None:
            raise ValueError(f"No image data in {path}")
        hdr = hdul[0].header.copy()
    return np.asarray(data, dtype=float), hdr


def _pick_stack_path(work_dir: Path) -> Path:
    sdir = stage_dir(work_dir, "stack2d")
    for name in ("stack2d.fits", "stacked2d.fits"):
        p = sdir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing stack2d output in {sdir}")


def _approx_grid_from_row(arr: np.ndarray, hdr: fits.Header) -> WaveGrid | None:
    ny, nx = arr.shape
    y0 = int(np.clip(int(round(float(hdr.get("SLITPOS", ny / 2.0) or ny / 2.0))), 0, ny - 1))
    row = np.asarray(arr[y0], dtype=float)
    m = np.isfinite(row)
    if int(np.sum(m)) < 8:
        row = np.nanmedian(arr, axis=0)
        m = np.isfinite(row)
    if int(np.sum(m)) < 8:
        return None
    x = np.flatnonzero(m).astype(float)
    y = row[m]
    coeffs = np.polyfit(x, y, deg=1)
    return WaveGrid(lambda0=float(np.polyval(coeffs, 0.0)), dlambda=float(coeffs[0]), nlam=int(nx), unit=str(hdr.get("WAVEUNIT", "Angstrom") or "Angstrom"), wave_ref=str(hdr.get("WAVEREF", "air") or "air"))


def run_publication_ready(cfg: dict[str, Any], *, out_dir: Path | None = None) -> dict[str, Any]:
    work_dir = resolve_work_dir(cfg)
    out_dir = Path(out_dir) if out_dir is not None else (Path(work_dir) / "publication_ready")
    out_dir.mkdir(parents=True, exist_ok=True)

    stack_path = _pick_stack_path(Path(work_dir))
    sci, var, mask, shdr = read_sci_var_mask(stack_path, validate=False)

    wdir = wavesol_dir(cfg, run_root=work_dir)
    wl_map, wl_hdr = _load_primary_or_sci(wdir / "lambda_map.fits")
    lsf_map, lsf_hdr = _load_primary_or_sci(Path(work_dir) / "lsf_model" / "lsf_map.fits")
    if wl_map.shape != lsf_map.shape:
        raise ValueError(f"lambda_map shape {wl_map.shape} != lsf_map shape {lsf_map.shape}")

    # Never extrapolate dismap into areas where the authoritative wavelength map is undefined.
    dismap = np.asarray(lsf_map, dtype=float)
    dismap[~np.isfinite(wl_map)] = np.nan

    grid_stack = None
    try:
        grid_stack = WaveGrid(
            lambda0=float(shdr.get("CRVAL1")),
            dlambda=float(shdr.get("CDELT1", shdr.get("CD1_1"))),
            nlam=int(sci.shape[1]),
            unit=str(shdr.get("CUNIT1", shdr.get("WAVEUNIT", "Angstrom")) or "Angstrom"),
            wave_ref=str(shdr.get("WAVEREF", "air") or "air"),
        )
    except Exception:
        grid_stack = None

    grid_raw = _approx_grid_from_row(wl_map, wl_hdr)

    ph_spectrum = fits.Header(shdr)
    ph_spectrum["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    ph_spectrum["PUBRDY"] = (True, "Publication-ready export")
    ph_spectrum["BUNIT"] = str(ph_spectrum.get("BUNIT", "adu/s") or "adu/s")
    ph_spectrum = add_provenance(ph_spectrum, cfg, stage="publication_ready")

    spectrum_path = out_dir / "Spectrum.fts"
    write_sci_var_mask(spectrum_path, sci.astype(np.float32), var=None if var is None else var.astype(np.float32), mask=None if mask is None else mask.astype(np.uint16), header=ph_spectrum, grid=grid_stack, primary_data=sci.astype(np.float32))

    ph_wl = fits.Header(wl_hdr)
    ph_wl["BUNIT"] = (str(wl_hdr.get("WAVEUNIT", "Angstrom") or "Angstrom"), "Wavelength unit")
    ph_wl["MAPTYPE"] = ("lambda", "Per-pixel wavelength map")
    ph_wl["PUBRDY"] = (True, "Publication-ready export")
    ph_wl["SCORPWCS"] = ("approx_central_row", "Primary WCS is approximate; SCI pixels are authoritative")
    ph_wl = add_provenance(ph_wl, cfg, stage="publication_ready")
    wl_path = out_dir / "wl.fts"
    write_sci_var_mask(wl_path, wl_map.astype(np.float32), header=ph_wl, grid=grid_raw, primary_data=wl_map.astype(np.float32))

    ph_dis = fits.Header(lsf_hdr)
    ph_dis["BUNIT"] = ("Angstrom", "Instrumental sigma")
    ph_dis["LSFUNIT"] = ("sigma_A", "Instrumental resolution quantity")
    ph_dis["MAPTYPE"] = ("dismap", "Per-pixel instrumental sigma map")
    ph_dis["PUBRDY"] = (True, "Publication-ready export")
    ph_dis["SCORPWCS"] = ("approx_central_row", "Primary WCS is approximate; SCI pixels are authoritative")
    ph_dis = add_provenance(ph_dis, cfg, stage="publication_ready")
    dis_path = out_dir / "dismap.fts"
    write_sci_var_mask(dis_path, dismap.astype(np.float32), header=ph_dis, grid=grid_raw, primary_data=dismap.astype(np.float32))

    sidecar = {
        "stage": "publication_ready",
        "pipeline_version": str(PIPELINE_VERSION),
        "inputs": {
            "stack2d": str(stack_path),
            "lambda_map": str(wdir / "lambda_map.fits"),
            "lsf_map": str(Path(work_dir) / "lsf_model" / "lsf_map.fits"),
        },
        "outputs": {
            "spectrum": str(spectrum_path),
            "wl": str(wl_path),
            "dismap": str(dis_path),
        },
        "semantics": {
            "Spectrum.fts": "linearized stacked science frame on exact linear wavelength WCS",
            "wl.fts": "authoritative per-pixel wavelength map in detector geometry; data values are authoritative, WCS is approximate helper",
            "dismap.fts": "authoritative per-pixel instrumental sigma map in detector geometry; no values are extrapolated outside finite wavelength support",
        },
        "shapes": {
            "spectrum": [int(v) for v in sci.shape],
            "wl": [int(v) for v in wl_map.shape],
            "dismap": [int(v) for v in dismap.shape],
        },
    }
    atomic_write_json(out_dir / "publication_provenance.json", sidecar)
    atomic_write_json(out_dir / "done.json", {"ok": True, **sidecar["outputs"]})
    return sidecar
