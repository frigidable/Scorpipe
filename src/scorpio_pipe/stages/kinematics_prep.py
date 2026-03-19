from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.provenance import add_provenance
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.workspace_paths import stage_dir

_C_KMS = 299792.458


def _load_spec1d(path: Path) -> tuple[fits.Header, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        trace = {name: np.asarray(hdul["SPEC_TRACE"].data[name]) for name in hdul["SPEC_TRACE"].columns.names}
        fixed = {name: np.asarray(hdul["SPEC_FIXED"].data[name]) for name in hdul["SPEC_FIXED"].columns.names}
    return hdr, trace, fixed


def _load_trace_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_lsf_summary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        s1d = obj.get("summary_1d", {}) if isinstance(obj, dict) else {}
        lam = np.asarray(s1d.get("lambda_A", []), dtype=float)
        sig = np.asarray(s1d.get("sigma_A", []), dtype=float)
        m = np.isfinite(lam) & np.isfinite(sig)
        return lam[m], sig[m]
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)


def _write_readme(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Kinematics prep",
        "",
        "This folder is a semantically explicit export for external cross-correlation / pPXF workflows.",
        "",
        f"- Preferred spectrum: {payload['preferred_product']}",
        f"- Wavelength reference: {payload['wave_ref']}",
        f"- Velocity scale advisory: {payload['velscale_kms']:.4f} km/s per pixel" if payload.get('velscale_kms') is not None else "- Velocity scale advisory: unavailable",
        f"- Median instrumental sigma: {payload['sigma_inst_A_median']:.4f} A" if payload.get('sigma_inst_A_median') is not None else "- Median instrumental sigma: unavailable",
        "",
        "## Notes",
        "",
        "- `spec_kin.fits` keeps the selected 1D science spectrum plus explicit instrumental-resolution columns.",
        "- `masks.json` contains wavelength intervals to inspect / mask, including skyline-residual windows when available.",
        "- Ensure the template library uses the same air/vacuum convention as the exported science spectrum.",
        "- For logarithmic rebinning, use the advisory velocity scale from the FITS header or this README, not a hard-coded value.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_kinematics_prep(cfg: dict[str, Any], *, out_dir: Path | None = None) -> dict[str, Any]:
    work_dir = resolve_work_dir(cfg)
    out_dir = Path(out_dir) if out_dir is not None else (Path(work_dir) / "kinematics_prep")
    out_dir.mkdir(parents=True, exist_ok=True)

    edir = stage_dir(work_dir, "extract1d")
    spec_path = edir / "spec1d.fits"
    trace_json_path = edir / "trace.json"
    hdr, trace, fixed = _load_spec1d(spec_path)
    trace_meta = _load_trace_json(trace_json_path)

    kcfg = cfg.get("kinematics_prep") if isinstance(cfg.get("kinematics_prep"), dict) else {}
    preferred = str(kcfg.get("preferred_product", "trace") or "trace").strip().lower()
    if preferred not in {"trace", "fixed"}:
        preferred = "trace"
    src = trace if preferred == "trace" else fixed

    wave = np.asarray(src["LAMBDA"], dtype=float)
    flux = np.asarray(src[f"FLUX_{preferred.upper()}"], dtype=float)
    var = np.asarray(src[f"VAR_{preferred.upper()}"], dtype=float)
    mask = np.asarray(src[f"MASK_{preferred.upper()}"], dtype=np.int32)
    sky = np.asarray(src[f"SKY_{preferred.upper()}"], dtype=float)
    npix = np.asarray(src[f"NPIX_{preferred.upper()}"], dtype=np.int16)

    lsf_lam, lsf_sig = _load_lsf_summary(Path(work_dir) / "lsf_model" / "lsf_model.json")
    sigma_inst = np.full_like(wave, np.nan, dtype=float)
    if lsf_lam.size >= 2:
        m = np.isfinite(wave)
        sigma_inst[m] = np.interp(wave[m], lsf_lam, lsf_sig, left=lsf_sig[0], right=lsf_sig[-1])

    good_l = np.isfinite(wave) & (wave > 0)
    velscale = None
    if int(np.sum(good_l)) >= 4:
        loglam = np.log(wave[good_l])
        dlog = np.diff(loglam)
        if np.isfinite(dlog).any():
            velscale = float(_C_KMS * np.nanmedian(dlog))

    sigma_inst_kms = np.full_like(sigma_inst, np.nan, dtype=float)
    m_sig = np.isfinite(sigma_inst) & np.isfinite(wave) & (wave > 0)
    sigma_inst_kms[m_sig] = _C_KMS * sigma_inst[m_sig] / wave[m_sig]

    skyline_intervals: list[dict[str, float]] = []
    sky_windows_path = Path(work_dir) / "sky_refine" / "sky_residual_windows.json"
    if sky_windows_path.exists():
        try:
            wins = json.loads(sky_windows_path.read_text(encoding="utf-8"))
            for w in wins:
                lam0 = w.get("lambda_min_A") if isinstance(w, dict) else None
                lam1 = w.get("lambda_max_A") if isinstance(w, dict) else None
                if lam0 is None or lam1 is None:
                    lam0 = w.get("lam_min_A") if isinstance(w, dict) else None
                    lam1 = w.get("lam_max_A") if isinstance(w, dict) else None
                if lam0 is None or lam1 is None:
                    continue
                skyline_intervals.append(
                    {
                        "lambda_min_A": float(min(lam0, lam1)),
                        "lambda_max_A": float(max(lam0, lam1)),
                        "classification": str(w.get("classification", "sky_residual")),
                        "action": str(w.get("action", "inspect")),
                    }
                )
        except Exception:
            skyline_intervals = []

    bad = (mask != 0) | ~np.isfinite(flux) | ~np.isfinite(var) | (npix <= 0)
    bad_intervals: list[dict[str, float]] = []
    if wave.size:
        start = None
        for i, b in enumerate(bad):
            if bool(b) and start is None:
                start = i
            elif (not bool(b)) and start is not None:
                bad_intervals.append({"lambda_min_A": float(wave[start]), "lambda_max_A": float(wave[i - 1]), "reason": "mask_or_no_coverage"})
                start = None
        if start is not None:
            bad_intervals.append({"lambda_min_A": float(wave[start]), "lambda_max_A": float(wave[-1]), "reason": "mask_or_no_coverage"})

    ph = fits.Header(hdr)
    ph["PIPEVER"] = (str(PIPELINE_VERSION), "Pipeline version")
    ph["KINPREP"] = (True, "Kinematics-prep export")
    ph["KINPROD"] = (preferred.upper(), "Selected extraction product")
    ph["WAVEREF"] = str(ph.get("WAVEREF", "unknown") or "unknown")
    if velscale is not None:
        ph["VELSCALE"] = (float(velscale), "Suggested log-rebin velocity scale [km/s]")
    sig_med = float(np.nanmedian(sigma_inst)) if np.isfinite(sigma_inst).any() else None
    if sig_med is not None:
        ph["SIGINST"] = (float(sig_med), "Median instrumental sigma [A]")
        ph["FWHMINST"] = (float(2.355 * sig_med), "Median instrumental FWHM [A]")
    ph = add_provenance(ph, cfg, stage="kinematics_prep")

    cols = [
        fits.Column(name="LAMBDA", format="D", array=np.asarray(wave, dtype=np.float64)),
        fits.Column(name="FLUX", format="D", array=np.asarray(flux, dtype=np.float64)),
        fits.Column(name="VAR", format="D", array=np.asarray(var, dtype=np.float64)),
        fits.Column(name="MASK", format="J", array=np.asarray(mask, dtype=np.int32)),
        fits.Column(name="SKY", format="D", array=np.asarray(sky, dtype=np.float64)),
        fits.Column(name="NPIX", format="I", array=np.asarray(npix, dtype=np.int16)),
        fits.Column(name="SIGMA_INST_A", format="D", array=np.asarray(sigma_inst, dtype=np.float64)),
        fits.Column(name="SIGMA_INST_KMS", format="D", array=np.asarray(sigma_inst_kms, dtype=np.float64)),
    ]
    hdul = fits.HDUList([fits.PrimaryHDU(header=ph), fits.BinTableHDU.from_columns(cols, name="SPEC_KIN")])
    spec_out = out_dir / "spec_kin.fits"
    hdul.writeto(spec_out, overwrite=True)

    masks_payload = {
        "preferred_product": preferred,
        "bad_intervals": bad_intervals,
        "skyline_residual_intervals": skyline_intervals,
        "trace_meta": {
            "trace_method": trace_meta.get("trace_model", {}).get("method") if isinstance(trace_meta, dict) else None,
            "aperture": trace_meta.get("aperture") if isinstance(trace_meta, dict) else None,
        },
    }
    atomic_write_json(out_dir / "masks.json", masks_payload)

    readme_payload = {
        "preferred_product": preferred,
        "wave_ref": str(ph.get("WAVEREF", "unknown") or "unknown"),
        "velscale_kms": velscale,
        "sigma_inst_A_median": sig_med,
    }
    _write_readme(out_dir / "README.md", readme_payload)
    atomic_write_json(out_dir / "done.json", {"ok": True, "spec_kin_fits": str(spec_out), "masks_json": str(out_dir / 'masks.json'), "readme_md": str(out_dir / 'README.md')})
    return {"ok": True, **readme_payload, "spec_kin_fits": str(spec_out)}
