from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.products import Product, group_by_stage, list_products
from scorpio_pipe.qc_thresholds import build_alerts, compute_thresholds
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.wavesol_paths import resolve_work_dir


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_cols(p: Path, ncols: int) -> np.ndarray | None:
    try:
        arr = np.genfromtxt(p, delimiter=",", names=False, dtype=float, skip_header=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < ncols:
            return None
        return arr
    except Exception:
        return None


def _rel(work_dir: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(work_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def _fmt_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "—"
    if s < 60:
        return f"{s:.1f} s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m {r:.0f}s"


def _metrics_wavesol(work_dir: Path, products: list[Product]) -> dict[str, Any]:
    m: dict[str, Any] = {}

    p1 = next((p.path for p in products if p.key == "wavesol_1d_json" and p.path.exists()), None)
    if p1:
        js = _read_json(p1) or {}
        m["wavesol_1d"] = {
            "deg": js.get("deg"),
            "rms_A": js.get("rms_A"),
            "n_pairs": js.get("n_pairs"),
            "n_used": js.get("n_used"),
        }

    p2 = next((p.path for p in products if p.key == "wavesol_2d_json" and p.path.exists()), None)
    if p2:
        js = _read_json(p2) or {}
        m2 = {
            "kind": js.get("kind"),
            "rms_A": js.get("rms_A"),
            "n_points": js.get("n_points"),
            "n_used": js.get("n_used"),
            "rejected_lines_A": js.get("rejected_lines_A", []),
        }
        # include both model RMS if present
        for k in ("power", "chebyshev"):
            if isinstance(js.get(k), dict):
                m2[f"{k}_rms_A"] = js[k].get("rms_A")
        m["wavesol_2d"] = m2

    # quick residual stats
    pres = next((p.path for p in products if p.key == "residuals_2d" and p.path.exists()), None)
    if pres:
        arr = _read_csv_cols(pres, 4)
        if arr is not None and arr.size:
            resid = np.asarray(arr[:, 3], float)
            resid = resid[np.isfinite(resid)]
            if resid.size:
                m["residuals_2d"] = {
                    "median_A": float(np.median(resid)),
                    "p95_abs_A": float(np.percentile(np.abs(resid), 95)),
                    "p99_abs_A": float(np.percentile(np.abs(resid), 99)),
                }

    return m


def _metrics_cosmics(products: list[Product]) -> dict[str, Any]:
    p = next((p.path for p in products if p.key == "cosmics_summary" and p.path.exists()), None)
    if not p:
        return {}
    js = _read_json(p) or {}
    # support disabled output
    if js.get("disabled"):
        return {"disabled": True}
    # old/new schema tolerance
    if "kind" in js and "replaced_fraction" in js:
        return {
            "kind": js.get("kind"),
            "n_frames": js.get("n_frames"),
            "k": js.get("k"),
            "replaced_pixels": js.get("replaced_pixels"),
            "replaced_fraction": js.get("replaced_fraction"),
            "per_frame_fraction": js.get("per_frame_fraction", []),
        }
    return js


def _metrics_timings(work_dir: Path) -> dict[str, Any]:
    p = work_dir / "report" / "timings.json"
    if not p.exists():
        return {}
    rows = _read_json(p)
    if not isinstance(rows, list):
        return {}

    by_stage: dict[str, list[float]] = {}
    for r in rows:
        try:
            stage = str(r.get("stage"))
            sec = float(r.get("seconds"))
            by_stage.setdefault(stage, []).append(sec)
        except Exception:
            continue
    agg = {
        st: {
            "n": len(v),
            "last_s": float(v[-1]),
            "avg_s": float(np.mean(v)) if v else None,
            "sum_s": float(np.sum(v)) if v else None,
        }
        for st, v in by_stage.items()
    }
    total_last = float(sum(v["last_s"] for v in agg.values() if isinstance(v.get("last_s"), (int, float)))) if agg else 0.0
    return {"stages": agg, "total_last_s": total_last}


def _metrics_sky(products: list[Product]) -> dict[str, Any]:
    """Sky-subtraction QC, aggregated from sky_sub_done.json."""

    p = next((p.path for p in products if p.key == "sky_done" and p.path.exists()), None)
    if not p:
        return {}
    js = _read_json(p) or {}
    per = js.get("per_exposure") or js.get("per_exp") or []
    rms = []
    sh_pix = []
    sh_A = []
    for r in per:
        try:
            m = r.get("metrics") or r.get("qc") or {}
            v = float(m.get("rms_sky"))
            if np.isfinite(v):
                rms.append(v)
                try:
                    sp = m.get("flexure_shift_pix")
                    if sp is not None:
                        spv = float(sp)
                        if np.isfinite(spv):
                            sh_pix.append(spv)
                    sa = m.get("flexure_shift_A")
                    if sa is not None:
                        sav = float(sa)
                        if np.isfinite(sav):
                            sh_A.append(sav)
                except Exception:
                    pass
        except Exception:
            continue
    out: dict[str, Any] = {"n_frames": int(len(per))}
    if rms:
        out.update({
            "rms_sky_median": float(np.median(rms)),
            "rms_sky_p90": float(np.percentile(rms, 90)),
        })
        if sh_pix:
            out.update({
                "flexure_shift_pix_median": float(np.median(sh_pix)),
                "flexure_shift_pix_p90_abs": float(np.percentile(np.abs(sh_pix), 90)),
            })
        if sh_A:
            out.update({
                "flexure_shift_A_median": float(np.median(sh_A)),
                "flexure_shift_A_p90_abs": float(np.percentile(np.abs(sh_A), 90)),
            })
    return out


def _metrics_stack(products: list[Product]) -> dict[str, Any]:
    p = next((p.path for p in products if p.key == "stack2d_done" and p.path.exists()), None)
    if not p:
        return {}
    js = _read_json(p) or {}
    out: dict[str, Any] = {
        "n_inputs": js.get("n_inputs"),
        "shape": js.get("shape"),
        "method": js.get("method"),
        "y_align_enabled": js.get("y_align_enabled"),
    }
    offs = js.get("y_offsets") or []
    try:
        vals = [float(o.get("y_shift_pix")) for o in offs if o and (o.get("y_shift_pix") is not None)]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out.update({
                "y_shift_pix_median": float(np.median(vals)),
                "y_shift_pix_p90_abs": float(np.percentile(np.abs(vals), 90)),
            })
    except Exception:
        pass
    return out


def _metrics_spec(products: list[Product]) -> dict[str, Any]:
    """Compute rough S/N from spec1d (median of |flux|/sqrt(var) in good pixels)."""

    p = next((p.path for p in products if p.key == "spec1d_fits" and p.path.exists()), None)
    if not p:
        return {}
    try:
        from astropy.io import fits

        with fits.open(p) as hdul:
            flux = np.asarray(hdul[0].data, float)
            var = np.asarray(hdul["VAR"].data, float) if "VAR" in hdul else None
            msk = np.asarray(hdul["MASK"].data, int) if "MASK" in hdul else None
        if var is None:
            return {}
        good = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        if msk is not None:
            good &= (msk == 0)
        if not np.any(good):
            return {}
        snr = np.abs(flux[good]) / np.sqrt(var[good])
        snr = snr[np.isfinite(snr)]
        if snr.size == 0:
            return {}
        return {
            "snr_median": float(np.median(snr)),
            "snr_p90": float(np.percentile(snr, 90)),
        }
    except Exception:
        return {}


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_products_table(work_dir: Path, products: list[Product]) -> str:
    rows = []
    for p in products:
        rel = _rel(work_dir, p.path)
        ex = p.exists()
        cls = "ok" if ex else ("req" if not p.optional else "miss")
        size = p.size()
        size_s = "" if size is None else f"{size/1024:.1f} KB" if size < 1024 * 1024 else f"{size/1024/1024:.2f} MB"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{_html_escape(p.stage)}</td>"
            f"<td>{_html_escape(p.key)}</td>"
            f"<td><a href='../{_html_escape(rel)}'>{_html_escape(rel)}</a></td>"
            f"<td>{'yes' if ex else 'no'}</td>"
            f"<td>{_html_escape(p.kind)}</td>"
            f"<td>{_html_escape(size_s)}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Stage</th><th>Key</th><th>Path</th><th>Exists</th><th>Kind</th><th>Size</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_stage_summary(products: list[Product]) -> str:
    g = group_by_stage(products)
    rows = []
    for st in sorted(g.keys()):
        ps = g[st]
        n = len(ps)
        ok = sum(1 for p in ps if p.exists())
        req = [p for p in ps if not p.optional]
        req_ok = sum(1 for p in req if p.exists())
        rows.append(
            f"<tr><td>{_html_escape(st)}</td>"
            f"<td>{ok}/{n}</td><td>{req_ok}/{len(req)}</td></tr>"
        )
    return (
        "<table class='mini'>"
        "<thead><tr><th>Stage</th><th>Products</th><th>Required</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_gallery(work_dir: Path, products: list[Product]) -> str:
    wanted = [
        "lin_preview_png",
        "cosmics_coverage_png",
        "cosmics_sum_png",
        "superneon_png",
        "wavesol_1d_png",
        "wavelength_matrix",
        "residuals_2d_png",
        "coverage_png",
        "spec1d_png",
    ]
    items = []
    for key in wanted:
        p = next((x for x in products if x.key == key and x.path.exists()), None)
        if not p:
            continue
        rel = _rel(work_dir, p.path)
        items.append(
            "<div class='card'>"
            f"<div class='cap'>{_html_escape(key)}</div>"
            f"<a href='../{_html_escape(rel)}'><img src='../{_html_escape(rel)}' alt='{_html_escape(key)}'></a>"
            "</div>"
        )
    if not items:
        return "<p class='muted'>(No quicklook images yet)</p>"
    return "<div class='gallery'>" + "".join(items) + "</div>"


def _render_alerts(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return "<p class='muted'>(No alerts)</p>"

    def _sev(a: dict[str, Any]) -> str:
        s = str(a.get('severity', 'info')).lower().strip()
        return s if s in {'ok','info','warn','bad'} else 'info'

    items = []
    for a in alerts:
        sev = _sev(a)
        msg = _html_escape(str(a.get('message', a.get('code', 'alert'))))
        code = _html_escape(str(a.get('code', '')))
        items.append(
            f"<div class='alert {sev}'>"
            f"<div class='acode'>{code}</div>"
            f"<div class='amsg'>{msg}</div>"
            f"</div>"
        )
    return "<div class='alerts'>" + "".join(items) + "</div>"


def build_qc_report(cfg: dict[str, Any], *, out_dir: str | Path | None = None, config_dir: Path | None = None) -> Path:
    """Build a lightweight QC report (JSON + HTML).

    Writes:
      - work_dir/report/qc_report.json
      - work_dir/report/index.html
    """

    if config_dir is not None and not cfg.get("config_dir"):
        cfg = dict(cfg)
        cfg["config_dir"] = str(config_dir)
    work_dir = resolve_work_dir(cfg)
    if out_dir is None:
        out_dir = work_dir / "report"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    products = list_products(cfg)
    # materialize relative paths for portability
    prod_rows = []
    for p in products:
        prod_rows.append(
            {
                "key": p.key,
                "stage": p.stage,
                "path": _rel(work_dir, p.path),
                "exists": p.exists(),
                "kind": p.kind,
                "optional": p.optional,
                "size": p.size(),
                "description": p.description,
            }
        )

    metrics: dict[str, Any] = {}
    metrics.update(_metrics_wavesol(work_dir, products))
    c = _metrics_cosmics(products)
    if c:
        metrics["cosmics"] = c
    t = _metrics_timings(work_dir)
    if t:
        metrics["timings"] = t

    s = _metrics_sky(products)
    if s:
        metrics["sky"] = s

    st = _metrics_stack(products)
    if st:
        metrics["stack"] = st

    sp = _metrics_spec(products)
    if sp:
        metrics["spec"] = sp

    thresholds, thresholds_meta = compute_thresholds(cfg)
    alerts = build_alerts(metrics, products=products, thresholds=thresholds)
    qc = {
        "thresholds": thresholds.to_dict(),
        "thresholds_meta": thresholds_meta,
        "alerts": alerts,
    }

    alert_counts = {
        "bad": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "bad"),
        "warn": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "warn"),
        "info": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "info"),
        "ok": sum(1 for a in alerts if str(a.get("severity", "")).lower() == "ok"),
        "total": len(alerts),
    }
    qc["alert_counts"] = alert_counts

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "work_dir": str(work_dir),
        "products": prod_rows,
        "metrics": metrics,
        "qc": qc,
    }

    out_json = out_dir / "qc_report.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- HTML ---
    timings_total = None
    try:
        timings_total = payload.get("metrics", {}).get("timings", {}).get("total_last_s")
    except Exception:
        timings_total = None

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Scorpio Pipe QC — v{_html_escape(PIPELINE_VERSION)}</title>
  <style>
    :root {{ --bg:#0b0f16; --card:#121a26; --text:#e7eefc; --muted:#9fb2d7; --ok:#1f9d55; --warn:#f59e0b; --bad:#ef4444; --line:#23314a; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background:var(--bg); color:var(--text); }}
    a {{ color:#8ab4ff; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:28px 18px 40px; }}
    .top {{ display:flex; gap:16px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; }}
    h1 {{ margin:0; font-size:22px; letter-spacing:0.2px; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .row {{ display:grid; grid-template-columns: 1fr; gap:14px; margin-top:16px; }}
    @media (min-width: 980px) {{ .row {{ grid-template-columns: 0.9fr 1.1fr; }} }}
    .box {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px 14px; }}
    .box h2 {{ margin:0 0 10px; font-size:16px; }}
    .muted {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ text-align:left; color:var(--muted); font-weight:600; }}
    table.mini td, table.mini th {{ padding:6px 8px; }}
    tr.ok td {{ color:#d7ffe6; }}
    tr.miss td {{ color:var(--warn); }}
    tr.req td {{ color:var(--bad); }}
    .kv {{ display:grid; grid-template-columns: 220px 1fr; gap:8px 12px; font-size:13px; }}
    .kv .k {{ color:var(--muted); }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid var(--line); background:#0f1520; font-size:12px; color:var(--muted); }}
    .gallery {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap:12px; }}
    .card {{ background:#0f1520; border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    .card img {{ display:block; width:100%; height:160px; object-fit:cover; }}
    .cap {{ padding:8px 10px; font-size:12px; color:var(--muted); border-bottom:1px solid var(--line); }}
    details {{ margin-top:10px; }}
    summary {{ cursor:pointer; color:var(--muted); }}
    code {{ background:#0f1520; padding:2px 6px; border-radius:8px; border:1px solid var(--line); }}
    .alerts {{ display:grid; gap:10px; }}
    .alert {{ border:1px solid var(--line); border-radius:14px; padding:10px 12px; background:#0f1520; }}
    .alert.ok {{ border-color: var(--ok); }}
    .alert.warn {{ border-color: var(--warn); }}
    .alert.bad {{ border-color: var(--bad); }}
    .alert .acode {{ font-size:11px; letter-spacing:0.2px; color:var(--muted); margin-bottom:4px; }}
    .alert .amsg {{ font-size:13px; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='top'>
      <div>
        <h1>Scorpio Pipe — QC report <span class='pill'>v{_html_escape(PIPELINE_VERSION)}</span></h1>
        <div class='meta'>Generated: {_html_escape(payload['generated_utc'])} · Work dir: <code>{_html_escape(str(work_dir))}</code></div>
      </div>
      <div class='meta'>{'' if timings_total is None else 'Last run total: ' + _html_escape(_fmt_seconds(float(timings_total)))}</div>
    </div>

    <div class='box' style='margin-top:14px'>
      <h2>Alerts</h2>
      {_render_alerts(alerts)}
      <div class='meta' style='margin-top:10px'>Bad: <code>{alert_counts['bad']}</code> · Warn: <code>{alert_counts['warn']}</code> · Total: <code>{alert_counts['total']}</code></div>
      <details>
        <summary>Show thresholds</summary>
        <pre style='white-space:pre-wrap; color:#cfe2ff; margin:10px 0 0'>{_html_escape(json.dumps(qc.get('thresholds', {}), indent=2, ensure_ascii=False))}</pre>
      </details>
    </div>

    <div class='row'>
      <div class='box'>
        <h2>Stage summary</h2>
        {_render_stage_summary(products)}
        <details>
          <summary>Show all products</summary>
          <div style='margin-top:10px'>{_render_products_table(work_dir, products)}</div>
        </details>
      </div>

      <div class='box'>
        <h2>Key metrics</h2>
        <div class='kv'>
          <div class='k'>Wavesol 1D RMS</div><div>{_html_escape(str(metrics.get('wavesol_1d', {}).get('rms_A', '—')))} Å</div>
          <div class='k'>Wavesol 2D RMS</div><div>{_html_escape(str(metrics.get('wavesol_2d', {}).get('rms_A', '—')))} Å</div>
          <div class='k'>2D model</div><div>{_html_escape(str(metrics.get('wavesol_2d', {}).get('kind', '—')))}</div>
          <div class='k'>Cosmics replaced</div><div>{_html_escape(str(metrics.get('cosmics', {}).get('replaced_fraction', '—')))}</div>
          <div class='k'>Residuals p95 |Δλ|</div><div>{_html_escape(str(metrics.get('residuals_2d', {}).get('p95_abs_A', '—')))} Å</div>
          <div class='k'>Sky RMS (median)</div><div>{_html_escape(str(metrics.get('sky', {}).get('rms_sky_median', '—')))}</div>
          <div class='k'>Stack N inputs</div><div>{_html_escape(str(metrics.get('stack', {}).get('n_inputs', '—')))}</div>
          <div class='k'>Spec S/N (median)</div><div>{_html_escape(str(metrics.get('spec', {}).get('snr_median', '—')))}</div>
        </div>
        <details>
          <summary>Show full metrics (JSON)</summary>
          <pre style='white-space:pre-wrap; color:#cfe2ff; margin:10px 0 0'>{_html_escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>
        </details>
      </div>
    </div>

    <div class='box' style='margin-top:14px'>
      <h2>Quicklooks</h2>
      {_render_gallery(work_dir, products)}
    </div>

    <div class='meta' style='margin-top:16px'>Tip: open <code>report/qc_report.json</code> to feed QC into notebooks, or <code>report/timings.json</code> to profile the pipeline.</div>
  </div>
</body>
</html>"""

    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html
