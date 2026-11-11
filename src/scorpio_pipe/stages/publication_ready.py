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




def _write_nan_preserving_sci(path: Path, data: np.ndarray, *, header: fits.Header, grid: WaveGrid | None, primary_data: np.ndarray | None = None) -> None:
    phdr = fits.Header(header)
    if grid is not None:
        for k, v in grid.to_wcs_cards().items():
            try:
                phdr[k] = v
            except Exception:
                pass
    hdus = [fits.PrimaryHDU(data=None if primary_data is None else np.asarray(primary_data, dtype=np.float32), header=phdr)]
    shdr = fits.Header()
    if grid is not None:
        for k, v in grid.to_wcs_cards().items():
            try:
                shdr[k] = v
            except Exception:
                pass
    for k in ("BUNIT", "WAVEUNIT", "WAVEREF", "MAPTYPE", "LSFUNIT", "SCORPWCS", "PUBRDY", "EXPORTOK", "VALLEVEL", "SCIQPASS", "PIXGEOM", "MAPALGN"):
        if k in phdr and k not in shdr:
            try:
                shdr[k] = phdr[k]
            except Exception:
                pass
    hdus.append(fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=shdr, name="SCI"))
    fits.HDUList(hdus).writeto(path, overwrite=True, output_verify="silentfix")

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


def _validate_export_coordinate_provenance(
    stack_hdr: fits.Header, wl_hdr: fits.Header, lsf_hdr: fits.Header,
) -> dict[str, str]:
    """Check declared spectral coordinate compatibility, not pixel co-registration.

    The rectified stack and the detector maps are fundamentally different
    pixel spaces. Even coincident array shapes do not certify co-registration.
    An absent declaration is recorded as unknown, never inferred from shape.
    """

    def medium(hdr: fits.Header) -> str | None:
        raw = str(hdr.get("WAVEREF") or "").strip().lower()
        if not raw:
            return None
        if raw not in {"air", "vacuum"}:
            raise ValueError(f"Publication input has unsupported WAVEREF={raw!r}")
        return raw

    def unit(hdr: fits.Header) -> str | None:
        tokens = [
            str(hdr.get(key) or "").strip().lower()
            for key in ("WAVEUNIT", "CUNIT1")
        ]
        found: list[str] = []
        for token in tokens:
            if not token:
                continue
            if token in {"a", "aa", "ang", "angstrom", "angstroms", "ångström", "å"}:
                found.append("Angstrom")
            elif token in {"nm", "nanometer", "nanometers"}:
                found.append("nm")
            else:
                raise ValueError(
                    f"Publication input has unsupported wavelength unit={token!r}"
                )
        if len(set(found)) > 1:
            raise ValueError("Publication input has inconsistent WAVEUNIT and CUNIT1")
        return found[0] if found else None

    refs = [medium(h) for h in (stack_hdr, wl_hdr, lsf_hdr)]
    units = [unit(h) for h in (stack_hdr, wl_hdr, lsf_hdr)]
    if len({r for r in refs if r is not None}) > 1:
        raise ValueError(f"Publication inputs disagree on air/vacuum medium: {refs}")
    if len({u for u in units if u is not None}) > 1:
        raise ValueError(f"Publication inputs disagree on wavelength unit: {units}")
    return {
        "wave_ref": next((r for r in refs if r is not None), "unknown"),
        "wave_unit": next((u for u in units if u is not None), "unknown"),
        "coordinate_declarations": (
            "COMPLETE" if all(r is not None for r in refs)
            and all(u is not None for u in units) else "INCOMPLETE"
        ),
    }


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
    if sci.ndim != 2 or wl_map.ndim != 2:
        raise ValueError("Publication requires 2D stack and 2D detector maps")
    coordinate_provenance = _validate_export_coordinate_provenance(
        shdr, wl_hdr, lsf_hdr
    )

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

    # Export completion cannot certify astrophysical science accuracy.
    # Publication readiness requires independent sky, LSF, covariance,
    # flux-calibration and validation gates not enforced by this writer.
    def _stamp_export_status(h: fits.Header) -> fits.Header:
        h["EXPORTOK"] = (True, "Requested export product created")
        h["PUBRDY"] = (False, "Scientific publication validation incomplete")
        h["SCIQPASS"] = (False, "Science QC not certified by export")
        h["VALLEVEL"] = ("EXPORT_ONLY", "No independent science validation")
        return h

    ph_spectrum = fits.Header(shdr)
    ph_spectrum["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    ph_spectrum = _stamp_export_status(ph_spectrum)
    ph_spectrum["PIXGEOM"] = ("RECTIFIED", "Pixel coordinate system of SCI")
    ph_spectrum["MAPALGN"] = (False, "Not pixel-aligned with detector maps")
    ph_spectrum["BUNIT"] = str(ph_spectrum.get("BUNIT", "adu/s") or "adu/s")
    ph_spectrum = add_provenance(ph_spectrum, cfg, stage="publication_ready")

    spectrum_path = out_dir / "Spectrum.fts"
    write_sci_var_mask(spectrum_path, sci.astype(np.float32), var=None if var is None else var.astype(np.float32), mask=None if mask is None else mask.astype(np.uint16), header=ph_spectrum, grid=grid_stack, primary_data=sci.astype(np.float32))

    ph_wl = fits.Header(wl_hdr)
    ph_wl["BUNIT"] = (str(wl_hdr.get("WAVEUNIT", "Angstrom") or "Angstrom"), "Wavelength unit")
    ph_wl["MAPTYPE"] = ("lambda", "Per-pixel wavelength map")
    ph_wl["PIXGEOM"] = ("DETECTOR", "Original detector pixel coordinates")
    ph_wl["MAPALGN"] = (False, "Not pixel-aligned with rectified Spectrum")
    ph_wl = _stamp_export_status(ph_wl)
    ph_wl["SCORPWCS"] = ("approx_central_row", "Primary WCS is approximate; SCI pixels are authoritative")
    ph_wl = add_provenance(ph_wl, cfg, stage="publication_ready")
    wl_path = out_dir / "wl.fts"
    _write_nan_preserving_sci(wl_path, wl_map.astype(np.float32), header=ph_wl, grid=grid_raw, primary_data=wl_map.astype(np.float32))

    ph_dis = fits.Header(lsf_hdr)
    ph_dis["BUNIT"] = ("Angstrom", "Instrumental sigma")
    ph_dis["LSFUNIT"] = ("sigma_A", "Instrumental resolution quantity")
    ph_dis["MAPTYPE"] = ("dismap", "Per-pixel instrumental sigma map")
    ph_dis["PIXGEOM"] = ("DETECTOR", "Original detector pixel coordinates")
    ph_dis["MAPALGN"] = (False, "Not pixel-aligned with rectified Spectrum")
    ph_dis = _stamp_export_status(ph_dis)
    ph_dis["SCORPWCS"] = ("approx_central_row", "Primary WCS is approximate; SCI pixels are authoritative")
    ph_dis = add_provenance(ph_dis, cfg, stage="publication_ready")
    dis_path = out_dir / "dismap.fts"
    _write_nan_preserving_sci(dis_path, dismap.astype(np.float32), header=ph_dis, grid=grid_raw, primary_data=dismap.astype(np.float32))

    sidecar = {
        "stage": "publication_ready",
        "geometry_contract_version": 1,
        "pixel_alignment": {
            "spectrum": "RECTIFIED",
            "wl": "DETECTOR",
            "dismap": "DETECTOR",
            "spectrum_detector_maps_coregistered": False,
            # Equal image shapes and detector labels are not independent
            # proof of alignment between maps built by separate stages.
            "wl_dismap_coregistered": None,
            "detector_map_alignment_qualification": "NOT_INDEPENDENTLY_VERIFIED",
            "direct_pixelwise_comparison_allowed": False,
            "same_shape_does_not_imply_alignment": True,
        },
        "coordinate_provenance": coordinate_provenance,
        "pipeline_version": str(PIPELINE_VERSION),
        "export_ok": True,
        "science_qc_pass": False,
        "validation_level": "export_only",
        "publication_ready": False,
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
            "wl.fts": "authoritative per-pixel wavelength map in DETECTOR geometry; NOT aligned with rectified Spectrum.fts, WCS approximate",
            "dismap.fts": "detector-geometry instrumental sigma; NOT effective rectified/stacked 1D LSF; invalid where wavelength map undefined",
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
