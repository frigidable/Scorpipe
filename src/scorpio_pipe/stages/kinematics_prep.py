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


def _load_arc_lsf_proxy(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read detector-coordinate arc LSF only as an *unqualified proxy*.

    The arc map precedes rectification, stacking and extraction. Even a
    measured arc LSF cannot be directly used as the final 1D science LSF.
    """
    meta: dict[str, Any] = {
        "source": "unavailable", "wave_ref": None, "wave_unit": None,
        "status": "NO_ARC_PROXY",
    }
    if not path.is_file():
        return np.array([], dtype=float), np.array([], dtype=float), meta
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        source = str(obj.get("source") or "unknown")
        inst = obj.get("instrument") if isinstance(obj.get("instrument"), dict) else {}
        unit = str(inst.get("wave_unit") or "").strip().lower()
        medium = str(inst.get("wave_ref") or "").strip().lower()
        meta.update({"source": source, "wave_unit": unit, "wave_ref": medium})
        if unit not in {"angstrom", "a", "aa", "ang", "ångström"}:
            meta["status"] = "ARC_UNIT_UNVERIFIED"
            return np.array([], dtype=float), np.array([], dtype=float), meta
        if medium not in {"air", "vacuum"}:
            meta["status"] = "ARC_MEDIUM_UNKNOWN"
            return np.array([], dtype=float), np.array([], dtype=float), meta
        s1d = obj.get("summary_1d", {})
        lam = np.asarray(s1d.get("lambda_A", []), dtype=float)
        sig = np.asarray(s1d.get("sigma_A", []), dtype=float)
        if (lam.ndim != 1 or lam.size < 2 or sig.shape != lam.shape
                or not np.isfinite(lam).all() or not np.isfinite(sig).all()
                or np.any(lam <= 0) or np.any(sig <= 0)
                or not np.all(np.diff(lam) > 0)):
            meta["status"] = "ARC_PROXY_INVALID"
            return np.array([], dtype=float), np.array([], dtype=float), meta
        meta["status"] = "ARC_PROXY_ONLY"
        return lam, sig, meta
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        meta["status"] = "ARC_PROXY_INVALID"
        return np.array([], dtype=float), np.array([], dtype=float), meta


def _science_wavelength_in_angstrom(wave: np.ndarray, hdr: fits.Header) -> np.ndarray | None:
    unit = str(hdr.get("WAVEUNIT") or hdr.get("CUNIT1") or "").strip().lower()
    if unit in {"angstrom", "a", "aa", "ang", "ångström"}:
        return np.asarray(wave, dtype=float)
    if unit in {"nm", "nanometer", "nanometers"}:
        return np.asarray(wave, dtype=float) * 10.0
    return None


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
        "- `SIGMA_INST_A` and `SIGMA_INST_KMS` are NaN until a validated *post-reduction* LSF exists.",
        "- `SIGMA_ARC_PROXY_A` and `SIGMA_ARC_PROXY_KMS` are detector-arc diagnostics ONLY.",
        "- `masks.json` contains wavelength intervals to inspect / mask, including skyline-residual windows when available.",
        "- LSF status: " + str(payload.get("lsf_status", "UNQUALIFIED")) + "; NOT ELIGIBLE for template-resolution matching.",
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

    arc_path = Path(work_dir) / "lsf_model" / "lsf_model.json"
    lsf_lam, lsf_sig, arc_meta = _load_arc_lsf_proxy(arc_path)
    # A detector-geometry arc sigma is NOT the effective 1D LSF after
    # rectification/stacking/extraction. Never populate SIGMA_INST_* with it.
    sigma_inst = np.full_like(wave, np.nan, dtype=float)
    sigma_arc_proxy = np.full_like(wave, np.nan, dtype=float)
    science_wavelength_A = _science_wavelength_in_angstrom(wave, hdr)
    science_medium = str(hdr.get("WAVEREF") or "").strip().lower()
    if science_wavelength_A is None:
        arc_meta["status"] = "SCIENCE_UNIT_UNKNOWN"
    elif science_medium not in {"air", "vacuum"}:
        arc_meta["status"] = "SCIENCE_MEDIUM_UNKNOWN"
    elif arc_meta["status"] == "ARC_PROXY_ONLY" and science_medium != arc_meta["wave_ref"]:
        arc_meta["status"] = "ARC_MEDIUM_MISMATCH"
    elif arc_meta["status"] == "ARC_PROXY_ONLY":
        m = (np.isfinite(science_wavelength_A) &
             (science_wavelength_A >= lsf_lam[0]) &
             (science_wavelength_A <= lsf_lam[-1]))
        sigma_arc_proxy[m] = np.interp(science_wavelength_A[m], lsf_lam, lsf_sig)
        arc_meta["n_covered"] = int(np.count_nonzero(m))
        arc_meta["n_total"] = int(wave.size)
        if not np.any(m):
            arc_meta["status"] = "ARC_PROXY_NO_OVERLAP"

    good_l = np.isfinite(wave) & (wave > 0)
    velscale = None
    if int(np.sum(good_l)) >= 4:
        loglam = np.log(wave[good_l])
        dlog = np.diff(loglam)
        if np.isfinite(dlog).any():
            velscale = float(_C_KMS * np.nanmedian(dlog))

    sigma_inst_kms = np.full_like(sigma_inst, np.nan, dtype=float)
    sigma_arc_proxy_kms = np.full_like(sigma_inst, np.nan, dtype=float)
    if science_wavelength_A is not None:
        m_arc = np.isfinite(sigma_arc_proxy) & np.isfinite(science_wavelength_A) & (science_wavelength_A > 0)
        sigma_arc_proxy_kms[m_arc] = _C_KMS * sigma_arc_proxy[m_arc] / science_wavelength_A[m_arc]

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
    sig_med = None  # The effective extracted LSF has not been qualified.
    arc_sig_med = float(np.nanmedian(sigma_arc_proxy)) if np.isfinite(sigma_arc_proxy).any() else None
    ph["LSFQUAL"] = (False, "Effective 1D LSF is NOT qualified")
    ph["LSFSTAT"] = (str(arc_meta["status"]), "Arc-only LSF provenance; not effective")
    if arc_sig_med is not None:
        ph["ARCSIG"] = (float(arc_sig_med), "Unqualified detector-arc sigma [A]")
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
        fits.Column(name="SIGMA_ARC_PROXY_A", format="D", array=np.asarray(sigma_arc_proxy, dtype=np.float64)),
        fits.Column(name="SIGMA_ARC_PROXY_KMS", format="D", array=np.asarray(sigma_arc_proxy_kms, dtype=np.float64)),
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
    atomic_write_json(out_dir / "lsf_provenance.json", {
        "effective_1d_lsf_qualified": False,
        "arc_proxy_status": str(arc_meta["status"]),
        "arc_proxy_source": str(arc_meta["source"]),
        "arc_source_path": str(arc_path),
        "arc_medium": arc_meta["wave_ref"],
        "arc_wave_unit": arc_meta["wave_unit"],
        "science_wave_ref": science_medium,
        "n_proxy_covered": int(np.count_nonzero(np.isfinite(sigma_arc_proxy))),
        "n_science_bins": int(wave.size),
        "no_edge_extrapolation": True,
        "limitations": [
            "Detector-arc LSF has not been propagated through spectral rectification",
            "Spatial shift/robust stacking and extraction operators are unmodeled",
            "Slit illumination, flexure and source morphology can alter object LSF",
            "Do not supply SIGMA_ARC_PROXY_A as pPXF galaxy resolution",
        ],
    })

    readme_payload = {
        "preferred_product": preferred,
        "wave_ref": str(ph.get("WAVEREF", "unknown") or "unknown"),
        "velscale_kms": velscale,
        "sigma_inst_A_median": sig_med,
        "sigma_arc_proxy_A_median": arc_sig_med,
        "lsf_status": str(arc_meta["status"]),
        "lsf_qualified": False,
    }
    _write_readme(out_dir / "README.md", readme_payload)
    atomic_write_json(out_dir / "done.json", {"ok": True, "lsf_qualified": False, "spec_kin_fits": str(spec_out), "masks_json": str(out_dir / 'masks.json'), "lsf_provenance_json": str(out_dir / 'lsf_provenance.json'), "readme_md": str(out_dir / 'README.md')})
    return {"ok": True, **readme_payload, "spec_kin_fits": str(spec_out)}
