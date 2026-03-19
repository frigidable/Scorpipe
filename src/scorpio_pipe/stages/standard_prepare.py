from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.calib.association import (
    StandardAssociationReport,
    StandardCompatibilityStatus,
    assess_standard_association,
)
from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
from scorpio_pipe.extract import (
    build_trace_model,
    choose_standard_half_width,
    enclosed_flux_curve,
    estimate_contamination_proxy,
    linear_wave_axis,
    read_mef,
    rectify_with_lambda_map,
    safe_json,
)
from scorpio_pipe.io.atomic import atomic_write_json
from scorpio_pipe.io.done_json import write_done_json
from scorpio_pipe.io.mef import WaveGrid, write_sci_var_mask
from scorpio_pipe.instruments import parse_frame_meta
from scorpio_pipe.maskbits import BADPIX, COSMIC, NO_COVERAGE, REJECTED, SATURATED, USER
from scorpio_pipe.pipeline.context import StageResult
from scorpio_pipe.qc.flags import make_flag, max_severity
from scorpio_pipe.stages.extract1d import run_extract1d
from scorpio_pipe.stages.flatfield import apply_flat
from scorpio_pipe.stages.calib import _resolve_superflat_path
from scorpio_pipe.workspace_paths import stage_dir

_FATAL_BITS = int(NO_COVERAGE | BADPIX | COSMIC | SATURATED | USER | REJECTED)


@dataclass(frozen=True)
class _Candidate:
    role: str
    path: Path
    meta: Any
    report: StandardAssociationReport | None


_STATUS_SCORE = {
    StandardCompatibilityStatus.ABSOLUTE_TRUSTED: 4,
    StandardCompatibilityStatus.ABSOLUTE_APPROX: 3,
    StandardCompatibilityStatus.RELATIVE_RESPONSE_ONLY: 2,
    StandardCompatibilityStatus.RELATIVE_TELLURIC_ONLY: 1,
    StandardCompatibilityStatus.INCOMPATIBLE: 0,
}


def _load_meta(path: Path):
    hdr = fits.getheader(path, 0)
    return parse_frame_meta(hdr, strict=False).with_frame_id(path.stem)


def _association_sort_key(c: _Candidate) -> tuple[int, int, float, float, str]:
    role_pref = 1 if c.role == "standard" else 0
    if c.report is None:
        return (-1, role_pref, float("inf"), float("inf"), c.path.name)
    score = int(_STATUS_SCORE.get(c.report.status, 0))
    checks = c.report.checks or {}
    dt = checks.get("time_delta_hours")
    am = checks.get("airmass_delta")
    dtf = float(dt) if isinstance(dt, (int, float)) and math.isfinite(float(dt)) else float("inf")
    amf = float(am) if isinstance(am, (int, float)) and math.isfinite(float(am)) else float("inf")
    return (score, role_pref, -dtf, -amf, c.path.name)


def _select_standard_candidate(cfg: dict[str, Any]) -> tuple[_Candidate | None, list[_Candidate], Any | None]:
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    science_meta = None
    obj_frames = [Path(str(x)) for x in (frames.get("obj") or [])]
    if obj_frames:
        try:
            science_meta = _load_meta(obj_frames[0])
        except Exception:
            science_meta = None

    candidates: list[_Candidate] = []
    for role_key, role_name in (("standard", "standard"), ("telluric_standard", "telluric_standard")):
        for raw in (frames.get(role_key) or []):
            p = Path(str(raw))
            try:
                meta = _load_meta(p)
            except Exception:
                continue
            rep = assess_standard_association(science_meta, meta, standard_role=role_name) if science_meta is not None else None
            candidates.append(_Candidate(role=role_name, path=p, meta=meta, report=rep))

    if not candidates:
        return None, [], science_meta

    selected = sorted(candidates, key=_association_sort_key, reverse=True)[0]
    return selected, candidates, science_meta


def _stem(path: Path) -> str:
    return path.stem


def _try_existing_rectified(run_root: Path, raw_path: Path, cfg: dict[str, Any]) -> Path | None:
    fcfg = cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}
    explicit = str((fcfg or {}).get("standard_rectified_path") or (fcfg or {}).get("standard_input_path") or "").strip()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    lin_dir = stage_dir(run_root, "linearize")
    s = _stem(raw_path)
    for name in (f"{s}_skysub.fits", f"{s}_rectified.fits", f"{s}_lin.fits"):
        p = lin_dir / name
        if p.exists():
            return p
    return None


def _resolve_wave_grid(cfg: dict[str, Any], run_root: Path, lam_map: np.ndarray, lam_hdr: fits.Header) -> tuple[float, float, int, str, str]:
    lin_dir = stage_dir(run_root, "linearize")
    for cand in (lin_dir / "done.json", lin_dir / "linearize_done.json"):
        if cand.exists():
            try:
                payload = json.loads(cand.read_text(encoding="utf-8"))
                grid = payload.get("grid") if isinstance(payload, dict) else {}
                wave0 = float(grid.get("lam_start"))
                dw = float(grid.get("lam_step"))
                nlam = int(grid.get("nlam"))
                unit = str(grid.get("unit") or lam_hdr.get("WAVEUNIT") or lam_hdr.get("CUNIT1") or "Angstrom")
                waveref = str(grid.get("ref") or lam_hdr.get("WAVEREF") or "air")
                if math.isfinite(wave0) and math.isfinite(dw) and dw > 0 and nlam > 0:
                    return wave0, dw, nlam, unit, waveref
            except Exception:
                pass
    prev = lin_dir / "lin_preview.fits"
    if prev.exists():
        try:
            hdr = fits.getheader(prev, 0)
            wave0 = float(hdr.get("CRVAL1"))
            dw = float(hdr.get("CDELT1", hdr.get("CD1_1")))
            nlam = int(hdr.get("SCORP_NL") or hdr.get("NAXIS1") or 0)
            unit = str(hdr.get("WAVEUNIT") or hdr.get("CUNIT1") or "Angstrom")
            waveref = str(hdr.get("WAVEREF") or "air")
            if math.isfinite(wave0) and math.isfinite(dw) and dw > 0 and nlam > 0:
                return wave0, dw, nlam, unit, waveref
        except Exception:
            pass
    from scorpio_pipe.stages.linearize import _compute_output_grid

    lcfg = cfg.get("linearize") if isinstance(cfg.get("linearize"), dict) else {}
    dw_hint = lcfg.get("lambda_step_A", lcfg.get("dw", lcfg.get("wave_step")))
    try:
        dw_hint_f = float(dw_hint)
    except Exception:
        cen = np.asarray(lam_map[lam_map.shape[0] // 2], dtype=float)
        d = np.diff(cen[np.isfinite(cen)])
        d = d[np.isfinite(d)]
        dw_hint_f = float(np.nanmedian(np.abs(d))) if d.size else 1.0
    unit = str(lam_hdr.get("WAVEUNIT") or lam_hdr.get("CUNIT1") or "Angstrom")
    waveref = str(lam_hdr.get("WAVEREF") or "air")
    wave0, wmax, dw, nlam = _compute_output_grid(
        lam_map,
        dw_hint=float(dw_hint_f),
        mode=str(lcfg.get("grid_mode", "intersection")),
        lo_pct=float(lcfg.get("grid_lo_pct", 1.0)),
        hi_pct=float(lcfg.get("grid_hi_pct", 99.0)),
        imin_pct=float(lcfg.get("grid_intersection_min_pct", 95.0)),
        imax_pct=float(lcfg.get("grid_intersection_max_pct", 5.0)),
    )
    _ = wmax
    return float(wave0), float(dw), int(nlam), unit, waveref


def _prepare_rectified_standard(
    cfg: dict[str, Any],
    *,
    run_root: Path,
    stage_dir_path: Path,
    raw_path: Path,
) -> tuple[Path, dict[str, Any]]:
    existing = _try_existing_rectified(run_root, raw_path, cfg)
    if existing is not None:
        return existing, {"mode": "reuse_existing", "input": str(existing)}

    prep_meta: dict[str, Any] = {"mode": "inline_rectify", "raw_input": str(raw_path)}
    work_input = raw_path
    fcfg = cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}
    if bool((fcfg or {}).get("standard_apply_flat", True)):
        try:
            superflat_path = _resolve_superflat_path(cfg, run_root)
            if Path(superflat_path).exists():
                flat_path = stage_dir_path / f"{raw_path.stem}_flatcorr.fits"
                apply_flat(raw_path, Path(superflat_path), flat_path, cfg=cfg, do_bias_subtract=True)
                work_input = flat_path
                prep_meta["flatfielded_input"] = str(flat_path)
                prep_meta["superflat_path"] = str(superflat_path)
        except Exception as e:
            prep_meta["flatfield_warning"] = str(e)

    lambda_map_path = PRODUCT_CATALOG.best_path(cfg, "lambda_map", run_root=run_root)
    lam_map = np.asarray(fits.getdata(lambda_map_path), dtype=float)
    lam_hdr = fits.getheader(lambda_map_path, 0)
    wave0, dw, nlam, unit, waveref = _resolve_wave_grid(cfg, run_root, lam_map, lam_hdr)
    rect = rectify_with_lambda_map(
        cfg=cfg,
        in_path=work_input,
        lambda_map=lam_map,
        wave0=float(wave0),
        dw=float(dw),
        nlam=int(nlam),
        unit=str(unit),
        frame_signature_key=raw_path.stem,
        lambda_map_key=str(lambda_map_path),
    )
    out_path = stage_dir_path / "standard_rectified.fits"
    hdr = fits.Header(rect.get("hdr") or {})
    hdr["INPUT2D"] = (str(raw_path), "Standard raw/calibrated input")
    hdr["SRCROLE"] = ("standard", "Prepared for standard-star extraction")
    write_sci_var_mask(
        out_path,
        np.asarray(rect["rect"], dtype=float),
        var=np.asarray(rect.get("rect_var"), dtype=float),
        mask=np.asarray(rect.get("rect_mask"), dtype=np.uint16),
        header=hdr,
        grid=WaveGrid(lambda0=float(wave0), dlambda=float(dw), nlam=int(nlam), unit=str(unit), wave_ref=str(waveref)),
        cov=np.asarray(rect.get("rect_cov"), dtype=float),
        validate=True,
    )
    prep_meta.update(
        {
            "rectified_path": str(out_path),
            "lambda_map_path": str(lambda_map_path),
            "wave_grid": {
                "wave0": float(wave0),
                "dw": float(dw),
                "nlam": int(nlam),
                "unit": str(unit),
                "waveref": str(waveref),
            },
        }
    )
    return out_path, prep_meta


def _extract_cfg(base_cfg: dict[str, Any], *, method: str, aperture_half_width: int | None = None) -> dict[str, Any]:
    cfg2 = copy.deepcopy(base_cfg)
    ecfg = dict((cfg2.get("extract1d") if isinstance(cfg2.get("extract1d"), dict) else {}) or {})
    ecfg["enabled"] = True
    ecfg["method"] = str(method)
    ecfg["local_sky_correction"] = True
    ecfg["min_good_frac"] = float(ecfg.get("min_good_frac", 0.6) or 0.6)
    if aperture_half_width is not None:
        ecfg["aperture_half_width"] = int(aperture_half_width)
    cfg2["extract1d"] = ecfg
    return cfg2


def _probe_base_aperture(cfg: dict[str, Any], *, rectified_input: Path, out_dir: Path) -> dict[str, Any]:
    res = run_extract1d(_extract_cfg(cfg, method="boxcar"), in_fits=rectified_input, out_dir=out_dir)
    trace_json = Path(str(res.get("trace_json") or out_dir / "trace.json"))
    payload = json.loads(trace_json.read_text(encoding="utf-8"))
    ap = payload.get("aperture") if isinstance(payload, dict) else {}
    meta = ap if isinstance(ap, dict) else {}
    base_hw = int(meta.get("half_width_pix") or 0) if isinstance(meta.get("half_width_pix"), (int, float)) else 0
    if base_hw <= 0:
        done = json.loads((out_dir / "done.json").read_text(encoding="utf-8"))
        base_hw = int((((done.get("extra") or {}).get("trace") or {}).get("aperture_half_width") or 6))
    return {"base_half_width": int(max(1, base_hw)), "trace_payload": payload, "result": res}


def _patch_standard_fits(path: Path, *, candidate: _Candidate, report: StandardAssociationReport | None, safe_for: str) -> None:
    try:
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdr = hdul[0].header
            hdr["STAGE"] = ("standard_prepare", "Pipeline stage")
            hdr["STDROLE"] = (str(candidate.role), "Selected calibration role")
            hdr["STDNAME"] = (str(candidate.meta.object_name or candidate.path.stem)[:68], "Selected standard object")
            if report is not None:
                hdr["STDSTAT"] = (str(report.status.value)[:68], "Association compatibility")
            hdr["STDQUAL"] = (str(safe_for)[:68], "Calibration safety class")
            hdul.flush()
    except Exception:
        pass


def run_standard_prepare(cfg: dict[str, Any], *, run_root: Path, stage_dir_path: Path) -> StageResult:
    stage_dir_path.mkdir(parents=True, exist_ok=True)
    plan_json = stage_dir_path / "standard_prepare_plan.json"
    qc_json = stage_dir_path / "standard_qc.json"
    trace_json = stage_dir_path / "standard_trace.json"
    standard_fits = stage_dir_path / "standard_1d.fits"
    preview_png = stage_dir_path / "standard_preview.png"

    selected, candidates, science_meta = _select_standard_candidate(cfg)
    if selected is None:
        atomic_write_json(plan_json, {"stage": "standard_prepare", "status": "blocked", "reason": "no_standard_frames"}, indent=2, ensure_ascii=False)
        write_done_json(stage="standard_prepare", stage_dir=stage_dir_path, status="blocked", outputs={"plan_json": str(plan_json)})
        return StageResult.blocked(products={"standard_prepare_plan_json": str(plan_json)}, artifacts=[str(plan_json)], messages=["No standard or telluric-standard frames configured"])

    if selected.report is not None and selected.report.status == StandardCompatibilityStatus.INCOMPATIBLE:
        payload = {
            "stage": "standard_prepare",
            "status": "blocked",
            "reason": "all_candidates_incompatible",
            "selected_candidate": str(selected.path),
            "association": selected.report.as_dict(),
            "candidates": [
                {"path": str(c.path), "role": c.role, "association": (c.report.as_dict() if c.report is not None else None)}
                for c in candidates
            ],
        }
        atomic_write_json(plan_json, payload, indent=2, ensure_ascii=False)
        write_done_json(stage="standard_prepare", stage_dir=stage_dir_path, status="blocked", outputs={"plan_json": str(plan_json)}, extra=payload)
        return StageResult.blocked(products={"standard_prepare_plan_json": str(plan_json)}, artifacts=[str(plan_json)], messages=["Standard frames exist but are incompatible with the selected science setup"])

    rectified_input, prep_meta = _prepare_rectified_standard(cfg, run_root=run_root, stage_dir_path=stage_dir_path, raw_path=selected.path)

    probe = _probe_base_aperture(cfg, rectified_input=rectified_input, out_dir=stage_dir_path / "_probe_extract")
    sci, var, mask, hdr = read_mef(rectified_input)
    wave = linear_wave_axis(hdr, sci.shape[1])
    tcfg = dict((cfg.get("extract1d") if isinstance(cfg.get("extract1d"), dict) else {}) or {})
    trace_probe = build_trace_model(
        sci,
        var,
        mask,
        wave,
        roi={},
        sky_windows=None,
        cfg=tcfg,
        fatal_bits=int(_FATAL_BITS),
    )
    policy_cfg = dict((((cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}) or {}).get("standard_aperture") or {}) if isinstance((((cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}) or {}).get("standard_aperture")), dict) else {})
    ap_policy = choose_standard_half_width(int(probe["base_half_width"]), ny=int(sci.shape[0]), cfg=policy_cfg)
    selected_hw = int(ap_policy["selected_half_width"])

    wide_dir = stage_dir_path / "_wide_extract"
    wide_res = run_extract1d(_extract_cfg(cfg, method="boxcar", aperture_half_width=selected_hw), in_fits=rectified_input, out_dir=wide_dir)
    wide_trace = json.loads((wide_dir / "trace.json").read_text(encoding="utf-8"))
    wide_done = json.loads((wide_dir / "done.json").read_text(encoding="utf-8"))

    diag_opt_payload = None
    if bool((((cfg.get("fluxcal") if isinstance(cfg.get("fluxcal"), dict) else {}) or {}).get("standard_optimal_diag", True))):
        opt_dir = stage_dir_path / "_diag_optimal"
        opt_res = run_extract1d(_extract_cfg(cfg, method="optimal", aperture_half_width=selected_hw), in_fits=rectified_input, out_dir=opt_dir)
        try:
            opt_done = json.loads((opt_dir / "done.json").read_text(encoding="utf-8"))
            diag_opt_payload = {
                "outputs": {k: v for k, v in opt_res.items() if k.endswith("fits") or k.endswith("png") or k.endswith("json")},
                "metrics": ((opt_done.get("extra") or {}).get("extract") or {}),
            }
        except Exception:
            diag_opt_payload = None

    shutil.copy2(wide_dir / "spec1d.fits", standard_fits)
    if (wide_dir / "spec1d.png").exists():
        shutil.copy2(wide_dir / "spec1d.png", preview_png)

    # Detailed QC / physics-oriented diagnostics.
    y_trace = np.asarray(wide_trace.get("y_trace") or trace_probe.y_trace, dtype=float)
    y_fixed = np.asarray(wide_trace.get("y_fixed") or trace_probe.y_fixed, dtype=float)
    sky_windows = ((wide_trace.get("geometry") or {}).get("sky_windows") or (wide_trace.get("sky_windows_used") or []))
    if not isinstance(sky_windows, list):
        sky_windows = []
    flux_curve = enclosed_flux_curve(
        sci,
        var,
        mask,
        y_trace,
        sky_windows=sky_windows,
        half_widths=[max(2, selected_hw // 2), selected_hw, int(round(selected_hw * 1.5)), max(selected_hw + 2, int(round(selected_hw * 2.0)))],
        min_good_frac=float((((cfg.get("extract1d") if isinstance(cfg.get("extract1d"), dict) else {}) or {}).get("min_good_frac", 0.6) or 0.6)),
    )
    try:
        idx_selected = flux_curve["half_widths"].index(int(selected_hw))
        capture_proxy = float(flux_curve["capture_fraction"][idx_selected])
    except Exception:
        capture_proxy = 0.0
    contam = estimate_contamination_proxy(
        sci,
        y_center=float(np.nanmedian(y_trace)) if np.isfinite(y_trace).any() else 0.5 * max(0, sci.shape[0] - 1),
        sky_windows=sky_windows,
        selected_half_width=selected_hw,
    )
    edge_frac = float(np.mean((y_trace - selected_hw <= 0.5) | (y_trace + selected_hw >= float(sci.shape[0] - 1.5)))) if y_trace.size else 1.0
    total_ap = max(1.0, float(2 * selected_hw + 1))
    npix_median = float((((wide_done.get("metrics") or {}).get("trace_npix_median") or 0.0)))
    masked_fraction = float(max(0.0, 1.0 - npix_median / total_ap))
    drift = float(np.nanpercentile(y_trace, 95) - np.nanpercentile(y_trace, 5)) if np.isfinite(y_trace).any() else float("nan")
    smooth = ((wide_trace.get("trace_model") or {}).get("meta") or {}).get("rms_pix")
    smooth_f = float(smooth) if isinstance(smooth, (int, float)) and math.isfinite(float(smooth)) else None

    report = selected.report
    if report is None:
        safe_for = "response_only"
    elif selected.role == "telluric_standard":
        safe_for = "telluric_or_relative_only"
    elif report.is_absolute_compatible and capture_proxy >= 0.95 and edge_frac <= 0.02 and float(contam.get("secondary_to_primary") or 0.0) < 0.12:
        safe_for = "absolute_candidate"
    elif report.is_absolute_compatible and capture_proxy >= 0.90:
        safe_for = "approx_absolute"
    else:
        safe_for = "response_only"

    flags = []
    if report is not None and report.status != StandardCompatibilityStatus.ABSOLUTE_TRUSTED:
        flags.append(make_flag("STANDARD_ASSOCIATION_DEGRADED", "WARN", "Selected standard is degraded relative to the science setup", value=str(report.status.value)))
    if capture_proxy < 0.90:
        flags.append(make_flag("STANDARD_APERTURE_TOO_NARROW", "WARN", "Wide-box extraction may still under-capture the standard-star flux", value=float(capture_proxy), bad_le=0.90))
    if edge_frac > 0.02:
        flags.append(make_flag("STANDARD_EDGE_TRUNCATION", "WARN", "A non-negligible fraction of the standard aperture touches detector edges", value=float(edge_frac), bad_ge=0.02))
    if masked_fraction > 0.10:
        flags.append(make_flag("STANDARD_MASKED_FRACTION_HIGH", "WARN", "The standard aperture contains a high masked-pixel fraction", value=float(masked_fraction), bad_ge=0.10))
    if float(contam.get("secondary_to_primary") or 0.0) >= 0.15:
        flags.append(make_flag("STANDARD_CONTAMINATION_RISK", "WARN", "A secondary spatial peak may contaminate the standard extraction", value=float(contam.get("secondary_to_primary") or 0.0), bad_ge=0.15))
    if smooth_f is not None and smooth_f > 1.5:
        flags.append(make_flag("STANDARD_TRACE_NOT_SMOOTH", "WARN", "The standard-star trace fit is relatively rough", value=float(smooth_f), bad_ge=1.5))

    qc_payload = {
        "stage": "standard_prepare",
        "selected_standard": {
            "path": str(selected.path),
            "role": str(selected.role),
            "object_name": str(selected.meta.object_name or selected.path.stem),
            "frame_id": str(getattr(selected.meta, "frame_id", selected.path.stem) or selected.path.stem),
        },
        "science_reference": {
            "path": str((Path(str((cfg.get("frames") or {}).get("obj", [""])[0])) if ((cfg.get("frames") or {}).get("obj")) else "")),
            "object_name": (str(science_meta.object_name) if science_meta is not None else None),
        },
        "association": (report.as_dict() if report is not None else None),
        "safe_for": str(safe_for),
        "status_hint": (str(report.status.value) if report is not None else "NO_SCIENCE_REFERENCE"),
        "diagnostics": {
            "centroid_drift_pix_p95_p5": float(drift) if math.isfinite(drift) else None,
            "aperture_half_width_pix": int(selected_hw),
            "masked_fraction": float(masked_fraction),
            "edge_truncation_fraction": float(edge_frac),
            "trace_smoothness_rms_pix": smooth_f,
            "flux_capture_proxy": float(capture_proxy),
            "contamination": safe_json(contam),
            "enclosed_flux_curve": safe_json(flux_curve),
            "trace_valid_fraction": (((wide_trace.get("trace_model") or {}).get("blocks_valid") or None)),
        },
        "optimal_diag": safe_json(diag_opt_payload),
        "flags": flags,
        "max_severity": max_severity(flags),
    }
    atomic_write_json(qc_json, qc_payload, indent=2, ensure_ascii=False)

    trace_payload = dict(wide_trace)
    trace_payload.update(
        {
            "stage": "standard_prepare",
            "selected_standard": qc_payload["selected_standard"],
            "association": qc_payload["association"],
            "aperture_policy": safe_json(ap_policy),
            "rectification": safe_json(prep_meta),
            "safe_for": str(safe_for),
        }
    )
    atomic_write_json(trace_json, trace_payload, indent=2, ensure_ascii=False)

    _patch_standard_fits(standard_fits, candidate=selected, report=report, safe_for=safe_for)

    plan_payload = {
        "stage": "standard_prepare",
        "status": "warn" if flags else "ok",
        "selected_standard": qc_payload["selected_standard"],
        "association": qc_payload["association"],
        "rectification": safe_json(prep_meta),
        "aperture_policy": safe_json(ap_policy),
        "safe_for": str(safe_for),
        "candidate_reports": [
            {
                "path": str(c.path),
                "role": str(c.role),
                "association": (c.report.as_dict() if c.report is not None else None),
            }
            for c in candidates
        ],
    }
    atomic_write_json(plan_json, plan_payload, indent=2, ensure_ascii=False)

    status = "warn" if flags else "ok"
    write_done_json(
        stage="standard_prepare",
        stage_dir=stage_dir_path,
        status=status,
        outputs={
            "standard_prepare_plan_json": str(plan_json),
            "standard_1d_fits": str(standard_fits),
            "standard_trace_json": str(trace_json),
            "standard_qc_json": str(qc_json),
            "standard_preview_png": str(preview_png) if preview_png.exists() else None,
            "standard_rectified_fits": str(rectified_input),
        },
        metrics={
            "flux_capture_proxy": float(capture_proxy),
            "edge_truncation_fraction": float(edge_frac),
            "masked_fraction": float(masked_fraction),
        },
        flags=flags,
        qc={"flags": flags, "max_severity": max_severity(flags)},
        extra={"selected_standard": qc_payload["selected_standard"], "association": qc_payload["association"], "safe_for": str(safe_for)},
    )

    products = {
        "standard_prepare_plan_json": str(plan_json),
        "standard_1d_fits": str(standard_fits),
        "standard_trace_json": str(trace_json),
        "standard_qc_json": str(qc_json),
    }
    artifacts = [str(plan_json), str(standard_fits), str(trace_json), str(qc_json)]
    if preview_png.exists():
        artifacts.append(str(preview_png))
    if Path(rectified_input) != selected.path:
        artifacts.append(str(rectified_input))
    return (StageResult.warn if flags else StageResult.ok)(products=products, artifacts=artifacts, qc=qc_payload, messages=[f"Selected {selected.meta.object_name or selected.path.stem} ({selected.role})"])
