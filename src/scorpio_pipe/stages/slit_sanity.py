from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from astropy.io import fits

from scorpio_pipe.instrument_db import load_instrument_db
from scorpio_pipe.qc.flags import make_operational_flag, make_flag, max_severity


@dataclass
class SlitSanityResult:
    json_path: Path
    png_path: Path
    md_path: Path
    status: str
    expected_width_px: float | None
    median_width_px: float | None


def _iter_candidate_paths(cfg: dict[str, Any]) -> list[Path]:
    slit_cfg = cfg.get("slit_sanity") if isinstance(cfg.get("slit_sanity"), dict) else {}
    max_frames = int(slit_cfg.get("max_candidate_frames", 12) or 12)
    out: list[Path] = []
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    if not isinstance(frames, dict):
        return out
    for key in ("flat", "flatfield", "flat_list", "science", "obj", "object", "neon", "arc"):
        vals = frames.get(key)
        if isinstance(vals, (str, Path)):
            vals = [vals]
        if not isinstance(vals, (list, tuple)):
            continue
        for v in vals:
            p = Path(str(v))
            if not p.is_absolute():
                base = Path(str(cfg.get("data_dir") or cfg.get("project_root") or ".")).expanduser().resolve()
                p = (base / p).resolve()
            if p.exists() and p.suffix.lower() in {".fits", ".fit", ".fts"}:
                out.append(p)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq[: max(1, max_frames)]


def _fwhm_from_profile(profile: np.ndarray) -> float | None:
    prof = np.asarray(profile, float)
    if prof.size < 8 or not np.isfinite(prof).any():
        return None
    prof = np.nan_to_num(prof)
    prof = prof - np.median(prof)
    mx = float(np.max(prof))
    if not np.isfinite(mx) or mx <= 0:
        return None
    prof = prof / mx
    idx = int(np.argmax(prof))
    half = 0.5
    left = idx
    while left > 0 and prof[left] > half:
        left -= 1
    right = idx
    while right < (prof.size - 1) and prof[right] > half:
        right += 1
    if right <= left:
        return None
    return float(right - left)


def _measure_width(path: Path) -> tuple[float | None, dict[str, Any]]:
    with fits.open(path, memmap=False) as hdul:
        hdr = dict(hdul[0].header)
        data = np.asarray(hdul[0].data, dtype=float)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        return None, {"path": str(path), "reason": "not_2d"}
    H, W = data.shape
    x1 = int(max(0, round(W * 0.35)))
    x2 = int(min(W, round(W * 0.65)))
    y1 = int(max(0, round(H * 0.30)))
    y2 = int(min(H, round(H * 0.70)))
    prof_x = np.nanmean(data[y1:y2, x1:x2], axis=0)
    prof_y = np.nanmean(data[y1:y2, x1:x2], axis=1)
    wx = _fwhm_from_profile(prof_x)
    wy = _fwhm_from_profile(prof_y)
    cands = [v for v in (wx, wy) if v is not None and np.isfinite(v)]
    width = min(cands) if cands else None
    inst = str(hdr.get("INSTRUME") or hdr.get("INSTRUMENT") or "")
    slit = hdr.get("SLITWID", hdr.get("SLIT", None))
    try:
        slit = float(slit) if slit is not None and str(slit).strip() != "" else None
    except Exception:
        slit = None
    return width, {
        "path": str(path),
        "instrument": inst,
        "slit_arcsec": slit,
        "width_x_px": wx,
        "width_y_px": wy,
        "chosen_width_px": width,
    }


def run_slit_sanity(cfg: dict[str, Any], *, run_root: str | Path | None = None) -> SlitSanityResult:
    work_dir = Path(run_root).expanduser().resolve() if run_root is not None else Path(str(cfg.get("work_dir") or ".")).expanduser().resolve()
    qc_dir = work_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    json_path = qc_dir / "slit_sanity.json"
    png_path = qc_dir / "slit_profile.png"
    md_path = qc_dir / "slit_consistency.md"

    candidates = _iter_candidate_paths(cfg)
    rows: list[dict[str, Any]] = []
    widths: list[float] = []
    slit_vals: list[float] = []
    inst_name: str | None = None
    for p in candidates:
        width, meta = _measure_width(p)
        rows.append(meta)
        if width is not None:
            widths.append(float(width))
        slit = meta.get("slit_arcsec")
        if slit is not None:
            slit_vals.append(float(slit))
        if not inst_name and meta.get("instrument"):
            inst_name = str(meta.get("instrument"))

    expected_width_px = None
    expected_sources: dict[str, Any] = {}
    if slit_vals:
        slit_med = float(np.median(slit_vals))
        db = load_instrument_db()
        inst = db.get(inst_name or "") if inst_name else None
        if inst and inst.plate_scale_arcsec_per_pix:
            expected_width_px = float(slit_med / float(inst.plate_scale_arcsec_per_pix))
            expected_sources = {
                "instrument": inst.name,
                "plate_scale_arcsec_per_pix": float(inst.plate_scale_arcsec_per_pix),
                "slit_arcsec_median": slit_med,
            }

    median_width_px = float(np.median(widths)) if widths else None
    scatter_px = float(np.std(widths)) if len(widths) >= 2 else 0.0

    flags: list[dict[str, Any]] = []
    if len(set(round(v, 3) for v in slit_vals)) > 1:
        flags.append(
            make_operational_flag(
                "SLIT_METADATA_MISMATCH",
                severity="WARN",
                message="Different SLITWID values were found across candidate frames.",
                slit_values_arcsec=sorted({round(v, 3) for v in slit_vals}),
            )
        )
    if widths and scatter_px > max(1.0, 0.20 * max(1e-6, median_width_px or 1.0)):
        flags.append(
            make_operational_flag(
                "SLIT_METADATA_MISMATCH",
                severity="WARN",
                message="Measured slit width varies significantly across frames.",
                width_scatter_px=float(scatter_px),
            )
        )
    if expected_width_px is not None and median_width_px is not None:
        delta = abs(median_width_px - expected_width_px)
        if delta > max(2.0, 0.25 * expected_width_px):
            flags.append(
                make_operational_flag(
                    "SLIT_METADATA_MISMATCH",
                    severity="WARN",
                    message="Measured slit width is inconsistent with SLITWID and instrument plate scale.",
                    expected_width_px=float(expected_width_px),
                    measured_width_px=float(median_width_px),
                    delta_px=float(delta),
                )
            )
    if not widths:
        flags.append(
            make_flag(
                "SLIT_SANITY_NO_MEASUREMENT",
                "WARN",
                "Could not derive any slit-width proxy from candidate frames.",
            )
        )

    payload = {
        "stage": "slit_sanity",
        "schema_version": 1,
        "n_candidate_frames": int(len(candidates)),
        "measurements": rows,
        "expected": expected_sources or None,
        "median_width_px": median_width_px,
        "scatter_px": scatter_px,
        "flags": flags,
        "status": "warn" if max_severity(flags) == "WARN" else "ok",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# Slit consistency", "", f"Status: **{payload['status']}**", ""]
    if expected_sources:
        md_lines.append(
            f"Expected width: ~{expected_width_px:.2f} px from median SLITWID="
            f"{expected_sources['slit_arcsec_median']:.3f} arcsec and plate scale "
            f"{expected_sources['plate_scale_arcsec_per_pix']:.3f} arcsec/pix."
        )
        md_lines.append("")
    if median_width_px is not None:
        md_lines.append(f"Measured median width: **{median_width_px:.2f} px** (scatter {scatter_px:.2f} px).")
        md_lines.append("")
    if flags:
        md_lines.append("## Flags")
        for fl in flags:
            md_lines.append(f"- `{fl['code']}` {fl['message']}")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = [r.get("chosen_width_px") for r in rows if r.get("chosen_width_px") is not None]
        labels = [Path(str(r.get("path"))).name for r in rows if r.get("chosen_width_px") is not None]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * max(1, len(vals))), 3.2))
        if vals:
            ax.plot(range(len(vals)), vals, marker="o")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        if expected_width_px is not None:
            ax.axhline(expected_width_px, linestyle="--", linewidth=1.0)
        ax.set_ylabel("Width [pix]")
        ax.set_title("Slit sanity")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
    except Exception:
        png_path.write_bytes(b"")

    return SlitSanityResult(
        json_path=json_path,
        png_path=png_path,
        md_path=md_path,
        status=str(payload["status"]),
        expected_width_px=expected_width_px,
        median_width_px=median_width_px,
    )
