"""QC thresholds + alerting.

The pipeline is used with different dispersers, binnings, and S/N regimes.

This module provides conservative defaults and a small alert engine used by
`scorpio_pipe.qc_report`. Defaults can be overridden in config.yaml via:

wavesol:
  qc:
    thresholds:
      wavesol_1d_rms_warn: 0.35
      wavesol_1d_rms_bad: 0.70

If `wavesol.qc.auto` is true (default), we apply a lightweight heuristic that
scales thresholds based on a number found in the disperser name (e.g. VPHG1200).
"""

from __future__ import annotations


import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Thresholds:
    wavesol_1d_rms_warn: float = 0.40
    wavesol_1d_rms_bad: float = 0.80

    wavesol_2d_rms_warn: float = 0.60
    wavesol_2d_rms_bad: float = 1.20

    resid_2d_p95_warn: float = 0.80
    resid_2d_p95_bad: float = 1.60

    cosmics_frac_warn: float = 0.03
    cosmics_frac_bad: float = 0.08

    linearize_cov_nonzero_warn: float = 0.95
    linearize_cov_nonzero_bad: float = 0.85

    linearize_rejected_frac_warn: float = 0.10
    linearize_rejected_frac_bad: float = 0.25

    # Sky-subtraction (detector-space) sanity checks
    # P1-C guardrails (sky subtraction): be stricter by default.
    sky_good_frac_warn: float = 0.80
    sky_good_frac_bad: float = 0.60

    sky_rows_frac_warn: float = 0.20
    sky_rows_frac_bad: float = 0.10

    sky_resid_mad_snr_warn: float = 3.0
    sky_resid_mad_snr_bad: float = 6.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "wavesol_1d_rms_warn": float(self.wavesol_1d_rms_warn),
            "wavesol_1d_rms_bad": float(self.wavesol_1d_rms_bad),
            "wavesol_2d_rms_warn": float(self.wavesol_2d_rms_warn),
            "wavesol_2d_rms_bad": float(self.wavesol_2d_rms_bad),
            "resid_2d_p95_warn": float(self.resid_2d_p95_warn),
            "resid_2d_p95_bad": float(self.resid_2d_p95_bad),
            "cosmics_frac_warn": float(self.cosmics_frac_warn),
            "cosmics_frac_bad": float(self.cosmics_frac_bad),
            "linearize_cov_nonzero_warn": float(self.linearize_cov_nonzero_warn),
            "linearize_cov_nonzero_bad": float(self.linearize_cov_nonzero_bad),
            "linearize_rejected_frac_warn": float(self.linearize_rejected_frac_warn),
            "linearize_rejected_frac_bad": float(self.linearize_rejected_frac_bad),
            "sky_good_frac_warn": float(self.sky_good_frac_warn),
            "sky_good_frac_bad": float(self.sky_good_frac_bad),
            "sky_rows_frac_warn": float(self.sky_rows_frac_warn),
            "sky_rows_frac_bad": float(self.sky_rows_frac_bad),
            "sky_resid_mad_snr_warn": float(self.sky_resid_mad_snr_warn),
            "sky_resid_mad_snr_bad": float(self.sky_resid_mad_snr_bad),
        }


def _pick_disperser(cfg: Dict[str, Any]) -> str | None:
    w = cfg.get("wavesol") if isinstance(cfg.get("wavesol"), dict) else {}
    if isinstance(w, dict):
        d = str(w.get("disperser") or "").strip()
        if d:
            return d
    s = cfg.get("setup") if isinstance(cfg.get("setup"), dict) else {}
    d2 = str((s or {}).get("disperser") or "").strip()
    return d2 or None


def _auto_scale_from_disperser(disperser: str | None) -> float:
    """Heuristic scaling factor for thresholds.

    Higher line density / higher resolution dispersers typically require tighter
    RMS values. We only use very broad scaling to avoid overconfidence.
    """

    if not disperser:
        return 1.0

    s = str(disperser)
    m = re.search(r"(\d{3,4})", s)
    if not m:
        return 1.0

    try:
        n = int(m.group(1))
    except Exception:
        return 1.0

    # Very conservative scaling (tunable later).
    if n >= 1500:
        return 0.60
    if n >= 1200:
        return 0.70
    if n >= 900:
        return 0.80
    if n <= 300:
        return 1.40
    if n <= 600:
        return 1.20
    return 1.0


def compute_thresholds(cfg: Dict[str, Any]) -> Tuple[Thresholds, Dict[str, Any]]:
    """Compute thresholds and return (thresholds, meta)."""

    # --- wavesolution auto-scaling (optional) ---
    w = cfg.get("wavesol") if isinstance(cfg.get("wavesol"), dict) else {}
    qc = (w or {}).get("qc") if isinstance((w or {}).get("qc"), dict) else {}

    auto = True
    try:
        auto = bool((qc or {}).get("auto", True))
    except Exception:
        auto = True

    thr = Thresholds()

    scale = 1.0
    disperser = _pick_disperser(cfg)
    if auto:
        scale = _auto_scale_from_disperser(disperser)
        if scale != 1.0:
            thr = Thresholds(
                wavesol_1d_rms_warn=thr.wavesol_1d_rms_warn * scale,
                wavesol_1d_rms_bad=thr.wavesol_1d_rms_bad * scale,
                wavesol_2d_rms_warn=thr.wavesol_2d_rms_warn * scale,
                wavesol_2d_rms_bad=thr.wavesol_2d_rms_bad * scale,
                resid_2d_p95_warn=thr.resid_2d_p95_warn * scale,
                resid_2d_p95_bad=thr.resid_2d_p95_bad * scale,
                cosmics_frac_warn=thr.cosmics_frac_warn,
                cosmics_frac_bad=thr.cosmics_frac_bad,
                linearize_cov_nonzero_warn=thr.linearize_cov_nonzero_warn,
                linearize_cov_nonzero_bad=thr.linearize_cov_nonzero_bad,
                linearize_rejected_frac_warn=thr.linearize_rejected_frac_warn,
                linearize_rejected_frac_bad=thr.linearize_rejected_frac_bad,
                sky_good_frac_warn=thr.sky_good_frac_warn,
                sky_good_frac_bad=thr.sky_good_frac_bad,
                sky_rows_frac_warn=thr.sky_rows_frac_warn,
                sky_rows_frac_bad=thr.sky_rows_frac_bad,
                sky_resid_mad_snr_warn=thr.sky_resid_mad_snr_warn,
                sky_resid_mad_snr_bad=thr.sky_resid_mad_snr_bad,
            )

    # --- user overrides (qc.thresholds + per-stage thresholds) ---
    w_overrides = (
        (qc or {}).get("thresholds") if isinstance((qc or {}).get("thresholds"), dict) else {}
    )

    l = cfg.get("linearize") if isinstance(cfg.get("linearize"), dict) else {}
    lqc = (l or {}).get("qc") if isinstance((l or {}).get("qc"), dict) else {}
    l_overrides = (
        (lqc or {}).get("thresholds") if isinstance((lqc or {}).get("thresholds"), dict) else {}
    )

    q = cfg.get("qc") if isinstance(cfg.get("qc"), dict) else {}
    q_overrides = (
        (q or {}).get("thresholds") if isinstance((q or {}).get("thresholds"), dict) else {}
    )

    s = cfg.get("sky") if isinstance(cfg.get("sky"), dict) else {}
    sqc = (s or {}).get("qc") if isinstance((s or {}).get("qc"), dict) else {}
    s_overrides = (
        (sqc or {}).get("thresholds") if isinstance((sqc or {}).get("thresholds"), dict) else {}
    )

    st2 = cfg.get("stack2d") if isinstance(cfg.get("stack2d"), dict) else {}
    st2qc = (st2 or {}).get("qc") if isinstance((st2 or {}).get("qc"), dict) else {}
    st2_overrides = (
        (st2qc or {}).get("thresholds") if isinstance((st2qc or {}).get("thresholds"), dict) else {}
    )

    d = thr.to_dict()
    applied: Dict[str, float] = {}
    for overrides in (q_overrides, w_overrides, l_overrides, s_overrides, st2_overrides):
        if not overrides:
            continue
        for k, v in overrides.items():
            kk = str(k).strip()
            try:
                vv = float(v)
            except Exception:
                continue
            if kk in d:
                d[kk] = vv
                applied[kk] = vv

    thr = Thresholds(**d)

    meta = {
        "auto": auto,
        "disperser": disperser,
        "scale": float(scale),
        "overrides_applied": applied,
    }
    return thr, meta


def _classify(value: float | None, warn: float, bad: float) -> str:
    if value is None:
        return "info"
    try:
        v = float(value)
    except Exception:
        return "info"
    if not (v == v):
        return "info"
    if v >= bad:
        return "bad"
    if v >= warn:
        return "warn"
    return "ok"


def build_alerts(
    metrics: Dict[str, Any], *, products: List[Any] | None, thresholds: Thresholds
) -> List[Dict[str, Any]]:
    """Create alert list for QC report.

    Parameters
    ----------
    metrics : dict
        QC metrics from qc_report.
    products : list[Product] | None
        If provided, missing required products are flagged.
    thresholds : Thresholds
        Thresholds used.
    """

    alerts: List[Dict[str, Any]] = []

    # Missing required products
    if products:
        for p in products:
            try:
                if (not p.optional) and (not p.exists()):
                    alerts.append(
                        {
                            "severity": "bad",
                            "code": "QC_MISSING_REQUIRED_PRODUCT",
                            "message": f"Missing required product: {p.key}",
                            "key": p.key,
                            "path": str(p.path),
                        }
                    )
            except Exception:
                continue

    # Wavesolution RMS values
    w1 = (
        (metrics.get("wavesol_1d") or {})
        if isinstance(metrics.get("wavesol_1d"), dict)
        else {}
    )
    w2 = (
        (metrics.get("wavesol_2d") or {})
        if isinstance(metrics.get("wavesol_2d"), dict)
        else {}
    )
    r2 = (
        (metrics.get("residuals_2d") or {})
        if isinstance(metrics.get("residuals_2d"), dict)
        else {}
    )

    v1 = w1.get("rms_A")
    sev1 = _classify(v1, thresholds.wavesol_1d_rms_warn, thresholds.wavesol_1d_rms_bad)
    if sev1 in {"warn", "bad"}:
        alerts.append(
            {
                "severity": sev1,
                "code": "QC_WAVESOL_1D_RMS",
                "message": f"1D wavesolution RMS is {v1} Å (warn≥{thresholds.wavesol_1d_rms_warn:.3g}, bad≥{thresholds.wavesol_1d_rms_bad:.3g})",
                "value": v1,
            }
        )

    v2 = w2.get("rms_A")
    sev2 = _classify(v2, thresholds.wavesol_2d_rms_warn, thresholds.wavesol_2d_rms_bad)
    if sev2 in {"warn", "bad"}:
        alerts.append(
            {
                "severity": sev2,
                "code": "QC_WAVESOL_2D_RMS",
                "message": f"2D wavesolution RMS is {v2} Å (warn≥{thresholds.wavesol_2d_rms_warn:.3g}, bad≥{thresholds.wavesol_2d_rms_bad:.3g})",
                "value": v2,
            }
        )

    v95 = r2.get("p95_abs_A")
    sev95 = _classify(v95, thresholds.resid_2d_p95_warn, thresholds.resid_2d_p95_bad)
    if sev95 in {"warn", "bad"}:
        alerts.append(
            {
                "severity": sev95,
                "code": "QC_RESIDUALS_2D_P95",
                "message": f"Residuals p95 |Δλ| is {v95} Å (warn≥{thresholds.resid_2d_p95_warn:.3g}, bad≥{thresholds.resid_2d_p95_bad:.3g})",
                "value": v95,
            }
        )

    c = (
        (metrics.get("cosmics") or {})
        if isinstance(metrics.get("cosmics"), dict)
        else {}
    )
    cf = c.get("replaced_fraction")
    try:
        cf_num = float(cf) if cf is not None else None
    except Exception:
        cf_num = None

    if cf_num is not None:
        sev = _classify(
            cf_num, thresholds.cosmics_frac_warn, thresholds.cosmics_frac_bad
        )
        if sev in {"warn", "bad"}:
            alerts.append(
                {
                    "severity": sev,
                    "code": "QC_COSMICS_FRACTION",
                    "message": f"Cosmics replaced fraction is {cf_num:.4g} (warn≥{thresholds.cosmics_frac_warn:.3g}, bad≥{thresholds.cosmics_frac_bad:.3g})",
                    "value": cf_num,
                }
            )



    # Wavesolution: 2D RMS should not get significantly worse than 1D
    try:
        v1f = float(v1) if v1 is not None else None
    except Exception:
        v1f = None
    try:
        v2f = float(v2) if v2 is not None else None
    except Exception:
        v2f = None

    if v1f is not None and v2f is not None:
        if v1f > 0 and v2f > max(v1f * 1.30, thresholds.wavesol_2d_rms_warn):
            alerts.append(
                {
                    "severity": "warn",
                    "code": "QC_WAVESOL_RMS_INCREASED",
                    "message": f"2D wavesolution RMS ({v2f:.4g} Å) is notably worse than 1D RMS ({v1f:.4g} Å). This may indicate a too-flexible/unstable 2D model or contaminated line list.",
                    "value": {"rms_1d_A": v1f, "rms_2d_A": v2f},
                }
            )

    # Wavelength-contract (units / reference) missing
    wc = (metrics.get("wavesol_contract") or {}) if isinstance(metrics.get("wavesol_contract"), dict) else {}
    if wc:
        if not bool(wc.get("unit_ok", True)):
            miss = wc.get("missing") or []
            alerts.append(
                {
                    "severity": "bad",
                    "code": "QC_WAVEUNIT_MISSING",
                    "message": f"lambda_map is missing required WAVE header keys ({', '.join(map(str, miss)) if miss else 'unknown'}). Units/ref become ambiguous.",
                    "missing": miss,
                }
            )

    # Signature mismatches (ROI/bin/readout) from manifest
    sig = (metrics.get("signatures") or {}) if isinstance(metrics.get("signatures"), dict) else {}
    bad_groups = sig.get("bad_groups") if isinstance(sig.get("bad_groups"), list) else []
    if bad_groups:
        alerts.append(
            {
                "severity": "bad",
                "code": "QC_SIGNATURE_MISMATCH",
                "message": f"Frame signatures are inconsistent in: {', '.join(map(str, bad_groups))}. Masters/products are not physically compatible.",
                "groups": bad_groups,
            }
        )

    # Linearize coverage / rejected fraction (if available)
    lin = (metrics.get("linearize") or {}) if isinstance(metrics.get("linearize"), dict) else {}
    cov = (lin.get("coverage") or {}) if isinstance(lin.get("coverage"), dict) else {}
    nz = cov.get("nonzero_frac")
    try:
        nzf = float(nz) if nz is not None else None
    except Exception:
        nzf = None
    if nzf is not None:
        sev = "ok"
        if nzf <= thresholds.linearize_cov_nonzero_bad:
            sev = "bad"
        elif nzf <= thresholds.linearize_cov_nonzero_warn:
            sev = "warn"
        if sev in {"warn", "bad"}:
            alerts.append(
                {
                    "severity": sev,
                    "code": "QC_LINEARIZE_COVERAGE",
                    "message": f"Linearize coverage nonzero fraction is {nzf:.4g} (warn≤{thresholds.linearize_cov_nonzero_warn:.3g}, bad≤{thresholds.linearize_cov_nonzero_bad:.3g}). Low coverage often means heavy masking, extrapolation cutoffs, or signature mismatch upstream.",
                    "value": nzf,
                }
            )

    st = (lin.get("stacking") or {}) if isinstance(lin.get("stacking"), dict) else {}
    rj = st.get("rejected_fraction")
    try:
        rjf = float(rj) if rj is not None else None
    except Exception:
        rjf = None

    if rjf is not None:
        sev = _classify(
            rjf, thresholds.linearize_rejected_frac_warn, thresholds.linearize_rejected_frac_bad
        )
        if sev in {"warn", "bad"}:
            alerts.append(
                {
                    "severity": sev,
                    "code": "QC_LINEARIZE_REJECTED",
                    "message": f"Linearize stacking rejected fraction is {rjf:.4g} (warn≥{thresholds.linearize_rejected_frac_warn:.3g}, bad≥{thresholds.linearize_rejected_frac_bad:.3g}). Consider tuning sigma-clip parameters.",
                    "value": rjf,
                }
            )

    exp = (lin.get("exptime_policy") or {}) if isinstance(lin.get("exptime_policy"), dict) else {}
    if exp.get("normalize_exptime") is False:
        alerts.append(
            {
                "severity": "info",
                "code": "QC_LINEARIZE_EXPTIME_POLICY",
                "message": "Linearize EXPTIME normalization is disabled (output is not ADU/s). This can be fine, but makes cross-night comparisons harder.",
            }
        )
    # If there are no alerts, still include a soft message when metrics are empty.
    if not alerts and (not metrics or metrics == {}):
        alerts.append(
            {
                "severity": "info",
                "code": "QC_EMPTY",
                "message": "No QC metrics yet (run some stages to populate report).",
            }
        )

    return alerts
