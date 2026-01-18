from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scorpio_pipe.products import Product, group_by_stage, list_products
from scorpio_pipe.qc_thresholds import build_alerts, compute_thresholds
from scorpio_pipe.version import PIPELINE_VERSION
from scorpio_pipe.paths import resolve_work_dir
from scorpio_pipe.workspace_paths import resolve_input_path, stage_dir

from scorpio_pipe.io.atomic import atomic_write_json, atomic_write_text
from scorpio_pipe.qc.flags import max_severity


class QCReportOutput(dict[str, Path]):
    """QC report output paths.

    Newer code/tests expect a mapping with at least ``{"json": Path, "html": Path}``.
    Older code/tests historically treated :func:`build_qc_report` return value as the
    HTML path directly.

    This tiny helper keeps both behaviours:

    - Mapping access: ``out["json"]`` / ``out["html"]``
    - Path-like attribute access delegated to the HTML path: ``out.exists()``,
      ``out.parent``, ``str(out)``, etc.
    """

    def __init__(
        self,
        *,
        json: Path,
        html: Path,
        legacy_json: Path | None = None,
        legacy_html: Path | None = None,
    ) -> None:
        super().__init__()
        self["json"] = Path(json)
        self["html"] = Path(html)
        if legacy_json is not None:
            self["legacy_json"] = Path(legacy_json)
        if legacy_html is not None:
            self["legacy_html"] = Path(legacy_html)

    # ------------------------------------------------------------------
    # Backward-compatible attribute aliases
    # ------------------------------------------------------------------
    @property
    def json_path(self) -> Path:
        """Alias for the JSON report path.

        Some legacy callers/tests expect ``out.json_path`` instead of
        mapping access ``out['json']``.
        """

        return self["json"]

    @property
    def html_path(self) -> Path:
        """Alias for the HTML report path (same as ``out['html']``)."""

        return self["html"]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        # Delegate unknown attributes to the HTML path to preserve old behaviour.
        return getattr(self["html"], name)

    def __fspath__(self) -> str:  # pragma: no cover
        return str(self["html"])

    def __str__(self) -> str:  # pragma: no cover
        return str(self["html"])

    def __repr__(self) -> str:  # pragma: no cover
        return f"QCReportOutput(html={self.get('html')!r}, json={self.get('json')!r})"


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


def _read_json_safely(p: Path) -> dict[str, Any] | None:
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _collect_stage_flags(work_dir: Path) -> list[dict[str, Any]]:
    """Aggregate QC flags from stage ``done.json`` files.

    Stages are the source of truth for the minimal mandatory flag set (P1-G).
    Here we only gather them into the QC report payload for convenience.
    """

    work_dir = Path(work_dir)
    out: list[dict[str, Any]] = []

    # Map legacy/short codes to the names used in the spec.
    code_map = {
        # Sky subtraction
        "OBJECT_EATING": "OBJECT_EATING_RISK",
        "SKY_COVERAGE_TOO_LOW": "COVERAGE_LOW",
        # Linearize
        "QC_LINEARIZE_COVERAGE": "COVERAGE_LOW",
        "QC_LINEARIZE_REJECTED": "REJECTED_FRAC_HIGH",
        # Stack
        "QC_STACK2D_COVERAGE": "COVERAGE_LOW",
        "QC_STACK2D_REJECTED": "REJECTED_FRAC_HIGH",
    }

    # Include flatfield (P0-F): calibration compatibility warnings (e.g.
    # ROTANGLE/SLITPOS mismatches) must be visible in the QC report.
    # Include extract so that NO_SKY_WINDOWS and other object-level QC appears.
    for stg in ("flatfield", "wavesol", "sky", "linearize", "stack", "extract"):
        try:
            d = _read_json_safely(stage_dir(work_dir, stg) / "done.json")
            if not d:
                continue
            flags = None
            qc = d.get("qc") if isinstance(d.get("qc"), dict) else None
            if qc and isinstance(qc.get("flags"), list):
                flags = qc.get("flags")
            if flags is None and isinstance(d.get("flags"), list):
                flags = d.get("flags")
            if not isinstance(flags, list):
                continue
            for fl in flags:
                if not isinstance(fl, dict):
                    continue
                row = dict(fl)
                row.setdefault("stage", stg)
                code = str(row.get("code") or "").strip()
                if code in code_map:
                    row["code"] = code_map[code]
                out.append(row)
        except Exception:
            continue

    return out


def _fmt_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "—"
    if s < 60:
        return f"{s:.1f} s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m {r:.0f}s"


def _metrics_wavesol(work_dir: Path, products: list[Product]) -> dict[str, Any]:
    """Wavesolution & residual QC summary.

    This function is intentionally tolerant to legacy product keys.
    """

    def _first_path(keys: set[str]) -> Path | None:
        for p in products:
            if p.key in keys and p.path.exists():
                return p.path
        return None

    m: dict[str, Any] = {}

    # --- 1D / 2D wavesolution JSON ---
    p1 = _first_path({"wavesolution_1d_json", "wavesol_1d_json"})
    if p1:
        js = _read_json(p1) or {}
        m["wavesol_1d"] = {
            "deg": js.get("deg"),
            "rms_A": js.get("rms_A"),
            "n_pairs": js.get("n_pairs"),
            "n_used": js.get("n_used"),
            "path": _rel(work_dir, p1),
        }

    p2 = _first_path({"wavesolution_2d_json", "wavesol_2d_json"})
    if p2:
        js = _read_json(p2) or {}
        m2 = {
            "kind": js.get("kind"),
            "rms_A": js.get("rms_A"),
            "n_points": js.get("n_points"),
            "n_used": js.get("n_used"),
            "rejected_lines_A": js.get("rejected_lines_A", []),
            "path": _rel(work_dir, p2),
        }
        for k in ("power", "chebyshev"):
            if isinstance(js.get(k), dict):
                m2[f"{k}_rms_A"] = js[k].get("rms_A")
        m["wavesol_2d"] = m2

    # --- residual statistics from 2D residual CSV ---
    pres = _first_path({"residuals_2d_csv", "residuals_2d"})
    if pres:
        arr = _read_csv_cols(pres, 4)
        if arr is not None and arr.size:
            resid = arr[:, 3].astype(float)
            resid = resid[np.isfinite(resid)]
            if resid.size:
                m["residuals_2d"] = {
                    "median_A": float(np.median(resid)),
                    "p95_abs_A": float(np.percentile(np.abs(resid), 95)),
                    "p99_abs_A": float(np.percentile(np.abs(resid), 99)),
                    "path": _rel(work_dir, pres),
                }

    return m



def _product_path(products: list[Product], key: str) -> Path | None:
    for p in products:
        if p.key == key:
            return p.path
    return None


def _product_path_any(products: list[Product], keys: tuple[str, ...], *, require_exists: bool = True) -> Path | None:
    for k in keys:
        p = _product_path(products, k)
        if not p:
            continue
        if (not require_exists) or p.exists():
            return p
    return None


def _metrics_calibs(work_dir: Path) -> dict[str, Any]:
    """Calibration masters + signatures (Block 03).

    Note: older done markers used `frame_signature`; newer tooling may expect
    `used_signature`. We expose both here.
    """
    try:
        from scorpio_pipe.work_layout import ensure_work_layout

        _ = ensure_work_layout(work_dir)
        out: dict[str, Any] = {}
        for name in ("superbias", "superflat"):
            done = resolve_input_path(
                f"{name}_done",
                work_dir,
                name,
                relpath=f"{name}_done.json",
            )
            if not done.exists():
                continue
            payload = _read_json(done) or {}
            used = payload.get("used_signature") or payload.get("frame_signature")
            exp = payload.get("expected_signature") or payload.get("expected_frame_signature")
            out[name] = {
                "done": _rel(work_dir, done),
                "n_inputs": payload.get("n_inputs"),
                "used_signature": used,
                "expected_signature": exp,
                "method": payload.get("method"),
                "qc": payload.get("qc"),
            }
        return out
    except Exception:
        return {}



def _metrics_calibration_associations(cfg: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Calibration association audit block (P0-B4).

    Summarizes which calibrations were associated to each science_set, and any
    QC-only compromises (match_reason / qc_deltas).
    """

    try:
        from scorpio_pipe.dataset.manifest_io import load_dataset_manifest

        man, path = load_dataset_manifest(cfg, work_dir, require=False, require_v3=False)
        if man is None or path is None:
            return {}

        # Aggregate manifest warnings
        warn_counts: dict[str, int] = {}
        for w in (man.warnings or []):
            code = str(getattr(w, "code", "") or "")
            if not code:
                continue
            warn_counts[code] = int(warn_counts.get(code, 0)) + 1

        by_set: list[dict[str, Any]] = []
        ss_by_id = {str(s.science_set_id): s for s in (man.science_sets or [])}
        for m in (man.matches or []):
            sid = str(m.science_set_id)
            ss = ss_by_id.get(sid)
            by_set.append(
                {
                    "science_set_id": sid,
                    "object": (ss.object if ss is not None else ""),
                    "config": (ss.config.as_compact_str() if ss is not None else ""),
                    "bias_id": (m.bias_id or None),
                    "flat_ids": list(m.flat_ids or ([] if m.flat_id is None else [m.flat_id])),
                    "arc_id": (m.arc_id or None),
                    "bias_meta": (m.bias_meta.model_dump(exclude_none=True) if m.bias_meta else None),
                    "flat_meta": (m.flat_meta.model_dump(exclude_none=True) if m.flat_meta else None),
                    "arc_meta": (m.arc_meta.model_dump(exclude_none=True) if m.arc_meta else None),
                }
            )

        return {
            "path": _rel(work_dir, path),
            "schema": getattr(man, "schema_id", None),
            "schema_version": getattr(man, "schema_version", None),
            "generated_utc": getattr(man, "generated_utc", None),
            "n_science_sets": int(len(man.science_sets or [])),
            "n_matches": int(len(man.matches or [])),
            "warnings": warn_counts,
            "by_set": by_set,
            "excluded_summary": (man.excluded_summary or {}),
        }
    except Exception:
        return {}



def _metrics_superneon(work_dir: Path, products: list[Product]) -> dict[str, Any]:
    qc_p = _product_path_any(products, ("superneon_qc_json",), require_exists=True)
    shifts_p = _product_path_any(products, ("superneon_shifts_json",), require_exists=True)
    out: dict[str, Any] = {}
    if qc_p and qc_p.exists():
        out["qc"] = _read_json(qc_p) or {}
        out["qc"]["path"] = _rel(work_dir, qc_p)
    if shifts_p and shifts_p.exists():
        sj = _read_json(shifts_p) or {}
        out["shifts"] = {
            "summary": (sj.get("summary") if isinstance(sj, dict) else None),
            "path": _rel(work_dir, shifts_p),
        }
    return out



def _metrics_wavesol_contract(products: list[Product]) -> dict[str, Any]:
    """Read unit/ref from lambda_map.fits header (Block 01)."""
    p = _product_path_any(products, ("lambda_map", "lambda_map_fits"), require_exists=True)
    if not p or not p.exists():
        return {}
    try:
        from astropy.io import fits

        hdr = fits.getheader(p, 0)
    except Exception:
        return {"path": str(p), "error": "failed_to_read_header"}

    wave_unit = hdr.get("WAVEUNIT") or hdr.get("CUNIT1")
    wave_ref = hdr.get("WAVEREF")
    ctype1 = hdr.get("CTYPE1")
    missing = []
    if not wave_unit:
        missing.append("WAVEUNIT/CUNIT1")
    if not wave_ref:
        missing.append("WAVEREF")
    if str(ctype1 or "").upper() != "WAVE":
        missing.append("CTYPE1=WAVE")

    return {
        "path": str(p),
        "wave_unit": wave_unit,
        "wave_ref": wave_ref,
        "ctype1": ctype1,
        "unit_ok": (len(missing) == 0),
        "missing": missing,
    }



def _metrics_signatures(products: list[Product]) -> dict[str, Any]:
    """FrameSignature consistency across input lists (manifest)."""
    p = _product_path_any(
        products,
        (
            "manifest",
            "manifest_legacy",
            "manifest_json",
            "manifest_legacy_json",
        ),
        require_exists=True,
    )
    if not p or not p.exists():
        return {}
    mj = _read_json(p)
    if not isinstance(mj, dict):
        return {}

    frames = mj.get("frames")
    bad: list[dict[str, Any]] = []
    if isinstance(frames, dict):
        for k, v in frames.items():
            if not isinstance(v, dict):
                continue
            if not bool(v.get("signature_consistent", True)):
                bad.append(
                    {
                        "kind": k,
                        "n": v.get("n"),
                        "frame_signature": v.get("frame_signature"),
                        "mismatches": v.get("signature_mismatches"),
                    }
                )

    return {"manifest": str(p), "bad_groups": bad}



def _metrics_cosmics(products: list[Product]) -> dict[str, Any]:
    p = next(
        (p.path for p in products if p.key == "cosmics_summary" and p.path.exists()),
        None,
    )
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
    # canonical: manifest/timings.json
    p = work_dir / "manifest" / "timings.json"
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
    total_last = (
        float(
            sum(
                v["last_s"]
                for v in agg.values()
                if isinstance(v.get("last_s"), (int, float))
            )
        )
        if agg
        else 0.0
    )
    return {"stages": agg, "total_last_s": total_last}


def _metrics_sky(products: list[Product]) -> dict[str, Any]:
    """Sky-subtraction QC, aggregated from sky_sub_done.json."""

    p = next(
        (p.path for p in products if p.key == "sky_done" and p.path.exists()), None
    )
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
        out.update(
            {
                "rms_sky_median": float(np.median(rms)),
                "rms_sky_p90": float(np.percentile(rms, 90)),
            }
        )
        if sh_pix:
            out.update(
                {
                    "flexure_shift_pix_median": float(np.median(sh_pix)),
                    "flexure_shift_pix_p90_abs": float(
                        np.percentile(np.abs(sh_pix), 90)
                    ),
                }
            )
        if sh_A:
            out.update(
                {
                    "flexure_shift_A_median": float(np.median(sh_A)),
                    "flexure_shift_A_p90_abs": float(np.percentile(np.abs(sh_A), 90)),
                }
            )
    return out


def _metrics_stack(products: list[Product]) -> dict[str, Any]:
    p = next(
        (p.path for p in products if p.key == "stack2d_done" and p.path.exists()), None
    )
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
        vals = [
            float(o.get("y_shift_pix"))
            for o in offs
            if o and (o.get("y_shift_pix") is not None)
        ]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out.update(
                {
                    "y_shift_pix_median": float(np.median(vals)),
                    "y_shift_pix_p90_abs": float(np.percentile(np.abs(vals), 90)),
                }
            )
    except Exception:
        pass
    return out


def _metrics_spec(products: list[Product]) -> dict[str, Any]:
    """Compute rough S/N from spec1d (median of |flux|/sqrt(var) in good pixels)."""

    p = next(
        (p.path for p in products if p.key == "spec1d_fits" and p.path.exists()), None
    )
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
            good &= msk == 0
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


def _metrics_linearize(work_dir: Path, products: list[Product]) -> dict[str, Any]:
    """Load linearize QC metrics (Block 05).

    The linearize stage writes work_dir/qc/linearize_qc.json.
    """
    try:
        p = next((x for x in products if x.key == "linearize_qc" and x.exists()), None)
        if p is None:
            return {}
        data = json.loads(Path(p.path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}

        # Keep only the stable, report-worthy subset.
        out: dict[str, Any] = {
            "preview": _rel(work_dir, Path(str(data.get("preview") or p.path))),
            "wave0": data.get("wave0"),
            "dw": data.get("dw"),
            "nlam": data.get("nlam"),
            "wave_unit": data.get("wave_unit"),
            "wave_ref": data.get("wave_ref"),
            "bunit": data.get("bunit"),
            "exptime_policy": data.get("exptime_policy"),
            "stacking": data.get("stacking"),
            "coverage": data.get("coverage"),
            "mask_summary": data.get("mask_summary"),
            "noise": data.get("noise"),
            "snr_abs": data.get("snr_abs"),
        }
        return out
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
        size_s = (
            ""
            if size is None
            else f"{size / 1024:.1f} KB"
            if size < 1024 * 1024
            else f"{size / 1024 / 1024:.2f} MB"
        )
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
        s = str(a.get("severity", "info")).lower().strip()
        return s if s in {"ok", "info", "warn", "bad"} else "info"

    items = []
    for a in alerts:
        sev = _sev(a)
        msg = _html_escape(str(a.get("message", a.get("code", "alert"))))
        code = _html_escape(str(a.get("code", "")))
        items.append(
            f"<div class='alert {sev}'>"
            f"<div class='acode'>{code}</div>"
            f"<div class='amsg'>{msg}</div>"
            f"</div>"
        )
    return "<div class='alerts'>" + "".join(items) + "</div>"


def build_qc_report(
    cfg: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    config_dir: Path | None = None,
) -> QCReportOutput:
    """Build a lightweight QC report (JSON + HTML).

    Writes (canonical):
      - work_dir/qc/qc_report.json
      - work_dir/index.html

    Backward compatibility:
      - also mirrors JSON into work_dir/manifest/qc_report.json
    """

    if config_dir is not None and not cfg.get("config_dir"):
        cfg = dict(cfg)
        cfg["config_dir"] = str(config_dir)
    work_dir = resolve_work_dir(cfg)
    qc_dir = Path(out_dir or (work_dir / "qc")).expanduser().resolve()
    qc_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir = (work_dir / "manifest").expanduser().resolve()
    legacy_dir.mkdir(parents=True, exist_ok=True)

    # Ensure products manifest exists (machine-readable index incl. per-exposure trees)
    try:
        from scorpio_pipe.products_manifest import write_products_manifest
        from scorpio_pipe.work_layout import ensure_work_layout

        # Products manifest remains in "manifest" for historical consumers.
        write_products_manifest(cfg=cfg, out_path=legacy_dir / "products_manifest.json")
    except Exception:
        pass

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

    cal = _metrics_calibs(work_dir)
    if cal:
        metrics["calibs"] = cal

    assoc = _metrics_calibration_associations(cfg, work_dir)
    if assoc:
        metrics["calibration_associations"] = assoc

    sig = _metrics_signatures(products)
    if sig:
        metrics["signatures"] = sig

    sn = _metrics_superneon(work_dir, products)
    if sn:
        metrics["superneon"] = sn

    wc = _metrics_wavesol_contract(products)
    if wc:
        metrics["wavesol_contract"] = wc

    lin = _metrics_linearize(work_dir, products)
    if lin:
        metrics["linearize"] = lin
        metrics["linearize_summary"] = lin
    c = _metrics_cosmics(products)
    if c:
        metrics["cosmics"] = c
    t = _metrics_timings(work_dir)
    if t:
        metrics["timings"] = t

    s = _metrics_sky(products)
    if s:
        metrics["sky"] = s

    # Linearize QC (v5.18+)
    try:
        from scorpio_pipe.work_layout import ensure_work_layout

        layout = ensure_work_layout(work_dir)
        # Canonical: qc/linearize_qc.json (stage may also point to its own file)
        for q in (
            layout.qc / "linearize_qc.json",
            legacy_dir / "linearize_qc.json",
            stage_dir(work_dir, "linearize") / "linearize_qc.json",
        ):
            if q.exists():
                metrics["linearize"] = json.loads(q.read_text(encoding="utf-8"))
                break
    except Exception:
        pass

    st = _metrics_stack(products)
    if st:
        metrics["stack"] = st

    sp = _metrics_spec(products)
    if sp:
        metrics["spec"] = sp

    thresholds, thresholds_meta = compute_thresholds(cfg)
    alerts = build_alerts(metrics, products=products, thresholds=thresholds)
    # Collect stage QC flags (minimal mandatory set is enforced by stage code;
    # we just aggregate them here for the GUI).
    stage_flags = _collect_stage_flags(work_dir)

    # P0-L: high-visibility warning when sky subtraction was skipped (pass-through).
    try:
        sky_codes = {"SKY_SUB_PASSTHROUGH", "UPSTREAM_SKY_PASSTHROUGH"}
        hits = [f for f in stage_flags if str(f.get("code") or "").strip() in sky_codes]
        if hits and not any(str(a.get("code") or "") == "QC_SKY_SUB_SKIPPED" for a in alerts):
            stems = set()
            for h in hits:
                for k in ("stem", "frame", "id", "path"):
                    v = h.get(k)
                    if isinstance(v, str) and v.strip():
                        stems.add(v.strip())
                        break
            n = len(stems) if stems else len(hits)
            alerts.insert(
                0,
                {
                    "severity": "warn",
                    "code": "QC_SKY_SUB_SKIPPED",
                    "message": f"Sky subtraction was skipped (pass-through) for {n} exposure(s); downstream products are degraded.",
                    "hint": "Inspect SKYMODEL_FAIL/QADEGRD. Consider stack2d.reject_if_mask_bits: [SKYMODEL_FAIL] or exclude affected frames in project_manifest.yaml.",
                    "n": int(n),
                },
            )
    except Exception:
        pass

    qc = {
        "thresholds": thresholds.to_dict(),
        "thresholds_meta": thresholds_meta,
        "alerts": alerts,
        "flags": stage_flags,
        "max_severity": max_severity(stage_flags),
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

    # P0-B3: surface reference context (manifest/reference_context.json).
    try:
        from scorpio_pipe.refs.context import load_reference_context

        ref_ctx = load_reference_context(work_dir)
        if isinstance(ref_ctx, dict) and ref_ctx.get("context_id"):
            payload["reference_context"] = {
                "context_id": str(ref_ctx.get("context_id")),
                "references": list(ref_ctx.get("references") or []),
            }
    except Exception:
        pass

    # P0-B6: contract compliance snapshot (non-fatal). This is a lightweight
    # provenance block for QA consumers to quickly verify that core contracts
    # are in place for a run.
    try:
        from scorpio_pipe.contracts.data_model import DATA_MODEL_VERSION
        from scorpio_pipe.contracts.validators import ProductContractError, validate_product
        from scorpio_pipe.stage_contracts import validate_contracts

        compliance: dict = {
            "schema": "scorpio_pipe.compliance",
            "schema_version": 1,
            "data_model_version": int(DATA_MODEL_VERSION),
            "stage_contracts_registry_ok": True,
            "reference_context_present": bool(payload.get("reference_context")),
            "product_contracts": [],
        }

        try:
            validate_contracts(cfg)
        except Exception as e:
            compliance["stage_contracts_registry_ok"] = False
            compliance["stage_contracts_error"] = str(e)

        # Validate a small, representative subset of products if they exist.
        key_allow = {"lambda_map", "stack2d_fits", "stacked2d_fits", "spec1d_fits"}
        checks = []
        for r in prod_rows:
            try:
                if not r.get("exists"):
                    continue
                if str(r.get("key") or "") not in key_allow:
                    continue
                rel = str(r.get("path") or "")
                pth = (work_dir / rel).resolve() if rel else None
                if pth is None or not pth.exists():
                    continue

                ok = True
                err = None
                try:
                    validate_product(pth, stage=str(r.get("stage") or "qc"))
                except ProductContractError as pe:
                    ok = False
                    err = {"code": pe.code, "message": pe.message}
                except Exception as e:
                    ok = False
                    err = {"code": "ERROR", "message": str(e)}

                checks.append({
                    "key": str(r.get("key") or ""),
                    "stage": str(r.get("stage") or ""),
                    "path": rel,
                    "ok": bool(ok),
                    "error": err,
                })
            except Exception:
                continue

        compliance["product_contracts"] = checks
        payload["compliance"] = compliance
    except Exception:
        pass

    out_json = qc_dir / "qc_report.json"
    legacy_json = legacy_dir / "qc_report.json"
    atomic_write_json(out_json, payload, indent=2, ensure_ascii=False)
    # Mirror legacy for backwards compatibility.
    try:
        atomic_write_json(legacy_json, payload, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # --- HTML ---
    timings_total = None
    try:
        timings_total = (
            payload.get("metrics", {}).get("timings", {}).get("total_last_s")
        )
    except Exception:
        timings_total = None

    # P0-B3: reference context box.
    ref_ctx_html = ""
    try:
        rc = payload.get("reference_context") or {}
        cid = rc.get("context_id")
        refs = rc.get("references") or []
        if cid:
            rows = ""
            for r in refs:
                if not isinstance(r, dict):
                    continue
                name = _html_escape(str(r.get("name") or ""))
                req = _html_escape(str(r.get("requested") or ""))
                sha = str(r.get("sha256") or "")
                sha12 = _html_escape(sha[:12] + ("…" if len(sha) > 12 else ""))
                ver = _html_escape(str(r.get("version") or ""))
                rows += f"<tr><td>{name}</td><td>{req}</td><td><code>{sha12}</code></td><td>{ver}</td></tr>"
            ref_ctx_html = (
                "<div class='box' style='margin-top:14px'>"
                "<h2>Reference context</h2>"
                f"<div class='meta'>context_id: <code>{_html_escape(str(cid))}</code></div>"
                "<details><summary>Show references</summary>"
                "<table class='mini'><thead><tr>"
                "<th>Name</th><th>Requested</th><th>SHA256</th><th>Version</th>"
                "</tr></thead><tbody>"
                f"{rows}"
                "</tbody></table></details></div>"
            )
    except Exception:
        ref_ctx_html = ""

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Scorpio Pipe QC — {_html_escape(PIPELINE_VERSION)}</title>
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
        <h1>Scorpio Pipe — QC report <span class='pill'>{_html_escape(PIPELINE_VERSION)}</span></h1>
        <div class='meta'>Generated: {_html_escape(payload["generated_utc"])} · Work dir: <code>{_html_escape(str(work_dir))}</code></div>
      </div>
      <div class='meta'>{"" if timings_total is None else "Last run total: " + _html_escape(_fmt_seconds(float(timings_total)))}</div>
    </div>

    {ref_ctx_html}

    <div class='box' style='margin-top:14px'>
      <h2>Alerts</h2>
      {_render_alerts(alerts)}
      <div class='meta' style='margin-top:10px'>Bad: <code>{alert_counts["bad"]}</code> · Warn: <code>{alert_counts["warn"]}</code> · Total: <code>{alert_counts["total"]}</code></div>
      <details>
        <summary>Show thresholds</summary>
        <pre style='white-space:pre-wrap; color:#cfe2ff; margin:10px 0 0'>{_html_escape(json.dumps(qc.get("thresholds", {}), indent=2, ensure_ascii=False))}</pre>
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
          <div class='k'>Wavesol 1D RMS</div><div>{_html_escape(str(metrics.get("wavesol_1d", {}).get("rms_A", "—")))} Å</div>
          <div class='k'>Wavesol 2D RMS</div><div>{_html_escape(str(metrics.get("wavesol_2d", {}).get("rms_A", "—")))} Å</div>
          <div class='k'>2D model</div><div>{_html_escape(str(metrics.get("wavesol_2d", {}).get("kind", "—")))}</div>
          <div class='k'>λ unit / ref</div><div>{_html_escape(str(metrics.get("wavesol_contract", {}).get("wave_unit", "—")))} · {_html_escape(str(metrics.get("wavesol_contract", {}).get("wave_ref", "—")))}</div>
          <div class='k'>Linearize normalize EXPTIME</div><div>{_html_escape(str(metrics.get("linearize", {}).get("exptime_policy", {}).get("normalize_exptime", "—")))}</div>
          <div class='k'>Linearize coverage (nonzero)</div><div>{_html_escape(str(metrics.get("linearize", {}).get("coverage", {}).get("nonzero_frac", "—")))}</div>
          <div class='k'>Linearize rejected</div><div>{_html_escape(str(metrics.get("linearize", {}).get("stacking", {}).get("rejected_fraction", "—")))}</div>
          <div class='k'>Superbias inputs</div><div>{_html_escape(str(metrics.get("calibs", {}).get("superbias", {}).get("n_inputs", "—")))}</div>
          <div class='k'>Superflat inputs</div><div>{_html_escape(str(metrics.get("calibs", {}).get("superflat", {}).get("n_inputs", "—")))}</div>
          <div class='k'>Signature mismatches</div><div>{_html_escape(str(len(metrics.get("signatures", {}).get("bad_groups", []))))}</div>
          <div class='k'>Cosmics replaced</div><div>{_html_escape(str(metrics.get("cosmics", {}).get("replaced_fraction", "—")))}</div>
          <div class='k'>Residuals p95 |Δλ|</div><div>{_html_escape(str(metrics.get("residuals_2d", {}).get("p95_abs_A", "—")))} Å</div>
          <div class='k'>Sky RMS (median)</div><div>{_html_escape(str(metrics.get("sky", {}).get("rms_sky_median", "—")))}</div>
          <div class='k'>Stack N inputs</div><div>{_html_escape(str(metrics.get("stack", {}).get("n_inputs", "—")))}</div>
          <div class='k'>Spec S/N (median)</div><div>{_html_escape(str(metrics.get("spec", {}).get("snr_median", "—")))}</div>
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

    <div class='meta' style='margin-top:16px'>Tip: open <code>qc/qc_report.json</code> to feed QC into notebooks. A legacy copy is also written into <code>manifest/qc_report.json</code>.</div>
  </div>
</body>
</html>"""

    # HTML is intentionally placed at the workspace root for easy access.
    out_html = work_dir / "index.html"
    atomic_write_text(out_html, html, encoding="utf-8")

    # Convenience mirrors
    qc_html = qc_dir / "qc_report.html"
    legacy_html = legacy_dir / "qc_report.html"
    try:
        atomic_write_text(qc_html, html, encoding="utf-8")
    except Exception:
        pass
    try:
        atomic_write_text(legacy_html, html, encoding="utf-8")
    except Exception:
        pass

    return QCReportOutput(json=out_json, html=out_html, legacy_json=legacy_json, legacy_html=legacy_html)
