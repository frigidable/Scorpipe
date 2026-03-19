from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scorpio_pipe.io.atomic import atomic_write_json, atomic_write_text


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _rel(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _score_from_metric(value: float | None, *, good: float, bad: float, higher_is_better: bool) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if higher_is_better:
        if v >= good:
            return 1.0
        if v <= bad:
            return 0.0
        return max(0.0, min(1.0, (v - bad) / max(1e-12, good - bad)))
    if v <= good:
        return 1.0
    if v >= bad:
        return 0.0
    return max(0.0, min(1.0, (bad - v) / max(1e-12, bad - good)))


def compute_observer_trust_score(payload: Mapping[str, Any]) -> dict[str, Any]:
    comp: dict[str, float] = {}
    wave = payload.get("wavesolution") if isinstance(payload.get("wavesolution"), dict) else {}
    sky = payload.get("sky") if isinstance(payload.get("sky"), dict) else {}
    ext = payload.get("extraction") if isinstance(payload.get("extraction"), dict) else {}
    geom = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}

    wave_rms = _score_from_metric(wave.get("rms_A"), good=0.15, bad=0.60, higher_is_better=False)
    if wave_rms is not None:
        comp["wavesolution_rms"] = wave_rms
    wave_trust = _score_from_metric(wave.get("trust_score"), good=0.85, bad=0.35, higher_is_better=True)
    if wave_trust is not None:
        comp["wavesolution_trust"] = wave_trust
    geom_trust = _score_from_metric(geom.get("trust_score"), good=0.85, bad=0.35, higher_is_better=True)
    if geom_trust is not None:
        comp["geometry"] = geom_trust
    sky_gain = _score_from_metric(sky.get("sigma_gain"), good=0.10, bad=-0.20, higher_is_better=True)
    if sky_gain is not None:
        comp["sky_refine_gain"] = sky_gain
    ext_trace = _score_from_metric(ext.get("trace_stability_px"), good=0.05, bad=0.40, higher_is_better=False)
    if ext_trace is not None:
        comp["trace_stability"] = ext_trace
    ext_cont = _score_from_metric(ext.get("contamination_risk"), good=0.05, bad=0.65, higher_is_better=False)
    if ext_cont is not None:
        comp["contamination_risk"] = ext_cont

    overall = None if not comp else round(float(sum(comp.values()) / max(1, len(comp))), 3)
    status = "unknown"
    if overall is not None:
        if overall >= 0.80:
            status = "high"
        elif overall >= 0.55:
            status = "moderate"
        else:
            status = "low"
    return {"overall": overall, "status": status, "components": comp}


def build_observer_dashboard(cfg_or_path: Mapping[str, Any] | str | Path, *, run_root: Path | None = None) -> dict[str, Any]:
    from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG
    from scorpio_pipe.config import load_config_any
    from scorpio_pipe.operational_presets import infer_setup_preset
    from scorpio_pipe.recovery import build_recovery_plan

    if isinstance(cfg_or_path, Mapping):
        cfg = dict(cfg_or_path)
    else:
        cfg = load_config_any(cfg_or_path)
    rr = Path(run_root or cfg.get("work_dir") or ".").expanduser().resolve()

    def best(key: str) -> Path | None:
        try:
            return PRODUCT_CATALOG.best_path(cfg, key, run_root=rr)
        except Exception:
            return None

    preset = infer_setup_preset(cfg)
    geom_model = _read_json(best("geometry_model_json"))
    geom_qc = _read_json(best("geometry_qc_json"))
    wave2d = _read_json(best("wavesolution_2d_json"))
    wave_done = _read_json(best("wave_done_json"))
    geom_used = _read_json(best("geometry_prior_used_json"))
    slit = _read_json(best("slit_sanity_json"))
    sky_qc = _read_json(best("qc_sky_refine_json"))
    ext_qc = _read_json(best("extraction_qc_json"))
    qc_report = _read_json(best("qc_json"))
    recovery = build_recovery_plan(cfg, qc_payload=qc_report, run_root=rr)

    geometry_metrics = geom_model.get("metrics") if isinstance(geom_model.get("metrics"), dict) else {}
    wave_metrics = wave_done.get("metrics") if isinstance(wave_done.get("metrics"), dict) else {}
    sky_metrics = sky_qc.get("aggregate_metrics") if isinstance(sky_qc.get("aggregate_metrics"), dict) else {}
    ext_metrics = ext_qc.get("diagnostics") if isinstance(ext_qc.get("diagnostics"), dict) else {}

    payload = {
        "setup_preset": preset.as_dict(),
        "geometry": {
            "source": geom_model.get("source") or geom_qc.get("source"),
            "trust_score": geometry_metrics.get("trust_score"),
            "fit_rms_px": geometry_metrics.get("fit_rms_px"),
            "max_severity": geom_qc.get("max_severity") or geom_qc.get("status"),
            "qc_path": _rel(rr, best("geometry_qc_json")),
            "model_path": _rel(rr, best("geometry_model_json")),
        },
        "slit_sanity": {
            "status": slit.get("status") or slit.get("max_severity"),
            "effective_slit_width_px": slit.get("effective_slit_width_px"),
            "declared_slit": slit.get("declared_slit") or ((slit.get("setup") or {}).get("slit") if isinstance(slit.get("setup"), dict) else None),
            "qc_path": _rel(rr, best("slit_sanity_json")),
        },
        "wavesolution": {
            "model_kind": wave2d.get("kind"),
            "rms_A": wave2d.get("rms_A") or wave_metrics.get("rms2d_A"),
            "n_used": wave2d.get("n_used") or wave_metrics.get("n_lines_used"),
            "trust_score": wave_metrics.get("wave_trust_score"),
            "line_density": wave_metrics.get("wave_line_density"),
            "geometry_prior_median_abs_resid_px": geom_used.get("median_abs_resid_px"),
            "preview_path": _rel(rr, best("line_id_overview_png")),
            "geometry_overlay_path": _rel(rr, best("geometry_overlay_png")),
            "qc_path": _rel(rr, best("wave_done_json")),
        },
        "sky": {
            "status": sky_qc.get("status") or sky_qc.get("max_severity"),
            "n_windows_total": sky_qc.get("n_windows_total"),
            "sigma_before": sky_metrics.get("sky_sigma_before_median"),
            "sigma_after": sky_metrics.get("sky_sigma_after_median"),
            "sigma_gain": sky_metrics.get("sky_sigma_gain_median"),
            "observer_pack_path": _rel(rr, best("sky_observer_pack_md")),
            "qc_path": _rel(rr, best("qc_sky_refine_json")),
        },
        "extraction": {
            "trace_stability_px": ext_metrics.get("trace_stability_px"),
            "contamination_risk": ext_metrics.get("contamination_risk"),
            "fixed_vs_trace_disagreement_sigma": ext_metrics.get("fixed_vs_trace_disagreement_sigma"),
            "aperture_sensitivity_frac": ext_metrics.get("aperture_sensitivity_frac"),
            "background_asymmetry_sigma": ext_metrics.get("background_asymmetry_sigma"),
            "diagnostics_path": _rel(rr, best("extraction_diagnostics_png")),
            "qc_path": _rel(rr, best("extraction_qc_json")),
        },
        "publication_ready": {
            "spectrum_path": _rel(rr, best("publication_spectrum_fts")),
            "lambda_map_path": _rel(rr, best("publication_wl_fts")),
            "dismap_path": _rel(rr, best("publication_dismap_fts")),
            "provenance_path": _rel(rr, best("publication_provenance_json")),
        },
        "kinematics_prep": {
            "spec_kin_path": _rel(rr, best("spec_kin_fits")),
            "masks_path": _rel(rr, best("kinematics_masks_json")),
            "readme_path": _rel(rr, best("kinematics_readme_md")),
        },
        "qc_report_path": _rel(rr, best("qc_json")),
        "recovery_plan": recovery,
    }
    payload["trust_score"] = compute_observer_trust_score(payload)
    return payload


def render_observer_dashboard_md(payload: Mapping[str, Any]) -> str:
    tp = payload.get("trust_score") if isinstance(payload.get("trust_score"), dict) else {}
    lines = ["# Observer dashboard", ""]
    lines.append(f"- Overall trust score: **{tp.get('overall', 'n/a')}** ({tp.get('status', 'unknown')})")
    setup = payload.get("setup_preset") if isinstance(payload.get("setup_preset"), dict) else {}
    if setup:
        lines.append(f"- Setup preset: **{setup.get('label') or setup.get('key') or 'unknown'}**")
        for r in setup.get("reasons") or []:
            lines.append(f"  - {r}")

    geom = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    wave = payload.get("wavesolution") if isinstance(payload.get("wavesolution"), dict) else {}
    lines.extend(["", "## Geometry / wavelength"])
    lines.append(f"- Geometry trust: {geom.get('trust_score', 'n/a')} (RMS px: {geom.get('fit_rms_px', 'n/a')})")
    lines.append(f"- Wave RMS: {wave.get('rms_A', 'n/a')} Å; trust: {wave.get('trust_score', 'n/a')}; line density: {wave.get('line_density', 'n/a')}")

    sky = payload.get("sky") if isinstance(payload.get("sky"), dict) else {}
    ext = payload.get("extraction") if isinstance(payload.get("extraction"), dict) else {}
    lines.extend(["", "## Sky / extraction"])
    lines.append(f"- Sky refine status: {sky.get('status', 'n/a')} (sigma gain: {sky.get('sigma_gain', 'n/a')})")
    lines.append(f"- Extraction trace stability: {ext.get('trace_stability_px', 'n/a')} px; contamination risk: {ext.get('contamination_risk', 'n/a')}")

    rec = payload.get("recovery_plan") if isinstance(payload.get("recovery_plan"), dict) else {}
    actions = rec.get("actions") if isinstance(rec.get("actions"), list) else []
    lines.extend(["", "## Suggested recovery actions"])
    if not actions:
        lines.append("- No urgent recovery actions detected.")
    else:
        for a in actions:
            lines.append(f"- **{a.get('code')}: {a.get('title')}**")
            why = str(a.get("why") or "").strip()
            if why:
                lines.append(f"  - Why: {why}")
            for step in a.get("steps") or []:
                lines.append(f"  - {step}")
            retry = a.get("retry_tasks") or []
            if retry:
                lines.append(f"  - Retry: `{', '.join(str(x) for x in retry)}`")
    lines.append("")
    return "\n".join(lines)


def render_observer_dashboard_html(payload: Mapping[str, Any]) -> str:
    import html

    md = render_observer_dashboard_md(payload)
    blocks: list[str] = []
    for line in md.splitlines():
        if line.startswith("# "):
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            blocks.append(f"<p>{html.escape(line[2:])}</p>")
        elif line.startswith("  - "):
            blocks.append(f"<p style='margin-left:24px;color:#cdd6e3'>{html.escape(line[4:])}</p>")
        elif line.strip():
            blocks.append(f"<p>{html.escape(line)}</p>")
    body = "\n".join(blocks)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f16;color:#e7eefc;padding:24px;max-width:980px;margin:auto} h1,h2{margin:0.3em 0} p{margin:0.35em 0} code{background:#111827;border:1px solid #263445;border-radius:8px;padding:1px 6px}</style>"
        f"</head><body>{body}</body></html>"
    )


def write_observer_dashboard(cfg_or_path: Mapping[str, Any] | str | Path, *, out_dir: Path, run_root: Path | None = None) -> dict[str, Path]:
    payload = build_observer_dashboard(cfg_or_path, run_root=run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_p = out_dir / "observer_dashboard.json"
    md_p = out_dir / "observer_dashboard.md"
    html_p = out_dir / "observer_dashboard.html"
    rec_p = out_dir / "recovery_plan.json"
    atomic_write_json(json_p, payload, indent=2, ensure_ascii=False)
    atomic_write_json(rec_p, payload.get("recovery_plan") or {}, indent=2, ensure_ascii=False)
    atomic_write_text(md_p, render_observer_dashboard_md(payload), encoding="utf-8")
    atomic_write_text(html_p, render_observer_dashboard_html(payload), encoding="utf-8")
    return {
        "observer_dashboard_json": json_p,
        "observer_dashboard_md": md_p,
        "observer_dashboard_html": html_p,
        "recovery_plan_json": rec_p,
    }


__all__ = [
    "build_observer_dashboard",
    "compute_observer_trust_score",
    "render_observer_dashboard_md",
    "render_observer_dashboard_html",
    "write_observer_dashboard",
]
