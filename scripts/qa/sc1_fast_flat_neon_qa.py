#!/usr/bin/env python3
"""Diagnostic-only fast-readout SCORPIO-1 FLAT/NEON comparison, no FITS writes.

For the audited N2146sat night only.  The geometric strips are candidates;
the mean of the strips is *not* a certified overscan calibration.
Usage:
    python scripts/qa/sc1_fast_flat_neon_qa.py --selftest
    python scripts/qa/sc1_fast_flat_neon_qa.py selected_frames.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SHAPE = (1032, 2080)
ROWS = slice(32, 1000)
COLS = slice(32, 2048)
LEFT = (ROWS, slice(2, 14))
RIGHT = (ROWS, slice(2066, 2078))
FLATS = ("s22800614.fts", "s22800615.fts", "s22800616.fts")
NEONS = ("s22800606.fts", "s22800613.fts")
MIN_BRIGHT = 2500.0  # bias-subtracted ADU, diagnostic threshold
SAT_LIMIT = 60000.0  # conservative raw ADU threshold, NOT a saturation calibration


def clipped_mean(data: np.ndarray) -> tuple[float, float, int]:
    a = np.asarray(data, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if len(a) < 100:
        raise ValueError("Too few usable pixels")
    for _ in range(4):
        center = np.median(a)
        scale = 1.4826 * np.median(np.abs(a - center))
        if scale <= 0:
            scale = float(np.std(a))
        if scale <= 0:
            break
        good = np.abs(a - center) < 4 * scale
        if good.sum() < 100:
            raise ValueError("Clipping removed the side-band sample")
        if good.all():
            break
        a = a[good]
    return float(a.mean()), float(a.std()), int(a.size)


def block_rms(image: np.ndarray, valid: np.ndarray, block: int = 64) -> float:
    h = (image.shape[0] // block) * block
    w = (image.shape[1] // block) * block
    d = image[:h, :w].astype(np.float64, copy=False)
    ok = valid[:h, :w]
    y, x = h // block, w // block
    sums = np.where(ok, d, 0.0).reshape(y, block, x, block).sum(axis=(1, 3))
    counts = ok.reshape(y, block, x, block).sum(axis=(1, 3))
    enough = counts >= block * block * 0.75
    if not np.any(enough):
        return float("nan")
    means = sums[enough] / counts[enough]
    means -= means.mean()
    return float(np.sqrt(np.mean(means * means)))


def pair_metrics(
    a: np.ndarray,
    b: np.ndarray,
    gain_e_per_adu: float,
    readnoise_adu: float,
) -> dict:
    """Compare independent exposures; not an absolute photon-transfer calibration."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Incompatible flat arrays")
    if gain_e_per_adu <= 0 or readnoise_adu < 0:
        raise ValueError("Invalid diagnostic gain/noise")

    usable = (np.isfinite(a) & np.isfinite(b)
              & (a > MIN_BRIGHT) & (b > MIN_BRIGHT)
              & (a < SAT_LIMIT) & (b < SAT_LIMIT))
    n = int(usable.sum())
    if n < 1000:
        raise RuntimeError(f"Too few bright valid flat pixels: {n}")

    scale = float(np.median(a[usable] / b[usable]))
    residual = (a - scale * b).astype(np.float64)
    values = residual[usable]
    med = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - med)))
    if mad <= 0:
        raise RuntimeError("Degenerate flat pair residual")
    clean = usable & (np.abs(residual - med) <= 5 * mad)
    # Approximate Poisson + read-noise prediction, only if header GAIN is e-/ADU.
    predicted = np.sqrt(
        (np.maximum(a[clean], 0.0)
         + scale**2 * np.maximum(b[clean], 0.0)) / gain_e_per_adu
        + (1 + scale**2) * readnoise_adu**2
    )
    centered = residual - med
    return {
        "relative_scale_a_over_b": scale,
        "n_bright_valid": n,
        "n_rejected_pair_outliers": int((usable & ~clean).sum()),
        "residual_median_adu": med,
        "residual_mad_sigma_adu": mad,
        "residual_mad_as_percent_of_flat_median": (
            100 * mad / float(np.median(a[usable]))
        ),
        "expected_sigma_adu_if_gain_e_per_adu": float(np.median(predicted)),
        "observed_over_expected_sigma": float(mad / np.median(predicted)),
        "64px_block_rms_adu": block_rms(centered, clean),
        "64px_block_rms_relative_percent": (
            100 * block_rms(centered, clean) / float(np.median(a[usable]))
        ),
    }


def load_one(path: Path, role: str):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
        arr = hdul[0].data
        if arr is None or arr.shape != SHAPE:
            raise RuntimeError(f"CCD geometry mismatch: {path.name}")
        if (abs(float(h["RATE"]) - 225) > 0.01
                or abs(float(h["GAIN"]) - 3.5) > 0.01):
            raise RuntimeError(f"Fast readout mismatch: {path.name}")
        data = np.asarray(arr, np.float32).copy()
        meta = {
            k: str(h.get(k, "")) for k in (
                "INSTRUME", "NODE", "RATE", "GAIN", "FOCUS", "DISPERSE",
                "GRISM", "SLIT", "EXPTIME", "EXPOSURE", "DATE-OBS", "TIME-OBS"
            )
        }
        if str(h.get("NODE", "")).strip().upper() != "A":
            raise RuntimeError(f"Unexpected CCD node: {path.name}")

    left, sl, nl = clipped_mean(data[LEFT])
    right, sr, nr = clipped_mean(data[RIGHT])
    level = 0.5 * (left + right)
    active = np.array(data[ROWS, COLS], dtype=np.float32, copy=True)
    active -= np.float32(level)
    info = {
        "role": role,
        "file": path.name,
        "left_mean_adu": left,
        "right_mean_adu": right,
        "candidate_side_mean_adu": level,
        "left_minus_right_adu": left - right,
        "left_std_adu": sl,
        "right_std_adu": sr,
        "left_n_after_clipping": nl,
        "right_n_after_clipping": nr,
        "active_median_after_side_subtraction_adu": float(np.median(active)),
        "active_mean_after_side_subtraction_adu": float(np.mean(active)),
        "fraction_active_raw_above_60000": float(np.mean(data[ROWS, COLS] > 60000)),
        "header": meta,
    }
    return active, info


def selftest():
    rng = np.random.default_rng(172)
    a = rng.normal(999.4, 2.0, 50000)
    a[0] = 65000
    assert abs(clipped_mean(a)[0] - 999.4) < 0.06

    yy, xx = np.mgrid[:128, :128]
    signal = 12000 + 7 * xx + 13 * yy
    n1 = rng.normal(0, np.sqrt(signal / 3.5 + 4), signal.shape)
    n2 = rng.normal(0, np.sqrt(signal / 3.5 + 4), signal.shape)
    flat1 = 1.003 * signal + n1
    flat2 = signal + n2
    flat1[10, 10] += 22000
    pair = pair_metrics(flat1, flat2, 3.5, 2)
    assert abs(pair["relative_scale_a_over_b"] - 1.003) < 0.0004
    assert pair["n_rejected_pair_outliers"] > 0
    assert 0.85 < pair["observed_over_expected_sigma"] < 1.15
    assert np.isfinite(pair["64px_block_rms_adu"])
    print("SELFTEST PASSED")


def run(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_key = {
        (r["role"].strip().lower(), Path(r["file"]).name.lower()): Path(r["file"])
        for r in rows
    }
    expected = [(role, name) for role, names in
                (("flat", FLATS), ("neon", NEONS)) for name in names]
    for key in expected:
        if key not in by_key:
            raise RuntimeError(f"Missing audited frame {key}")

    infos = []
    flat_images = []
    print("FAST-READOUT 225 / 3.5 SIDE-BAND QA")
    for role, name in expected:
        arr, info = load_one(by_key[(role, name)], role)
        infos.append(info)
        if role == "flat":
            flat_images.append(arr)
        print(
            f"{role:4} {name}: candidate side={info['candidate_side_mean_adu']:.4f}"
            f", L-R={info['left_minus_right_adu']:+.4f}"
            f", active-side={info['active_median_after_side_subtraction_adu']:.2f}"
        )

    # The side-band scatter is only a rough *diagnostic* read-noise proxy.
    noise = float(np.median([
        (r["left_std_adu"] + r["right_std_adu"]) / 2
        for r in infos if r["role"] == "flat"
    ]))
    pairs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        metrics = pair_metrics(flat_images[i], flat_images[j], 3.5, noise)
        metrics.update(flat_a=FLATS[i], flat_b=FLATS[j])
        pairs.append(metrics)
        print(
            f"PAIR {FLATS[i]}/{FLATS[j]}: "
            f"scale={metrics['relative_scale_a_over_b']:.7f} "
            f"robust sigma={metrics['residual_mad_sigma_adu']:.3f} ADU "
            f"predicted={metrics['expected_sigma_adu_if_gain_e_per_adu']:.3f} ADU "
            f"ratio={metrics['observed_over_expected_sigma']:.3f} "
            f"64px RMS={metrics['64px_block_rms_adu']:.4f} ADU"
        )

    outdir = csv_path.parent
    json_path = outdir / "sc1_fast_flat_neon_summary.json"
    pair_path = outdir / "sc1_fast_flat_pairs.csv"
    side_path = outdir / "sc1_fast_flat_neon_sidebands.csv"

    summary = {
        "status": "DIAGNOSTIC_ONLY_HOLD",
        "n_flats": len(FLATS), "n_neons": len(NEONS),
        "candidate_sidebands_not_certified": True,
        "no_fast_mode_masterbias_available": True,
        "readout_rate": 225, "readout_gain_header": 3.5,
        "assumption_for_expected_noise": "Header GAIN is electrons per ADU, unverified",
        "note": "No physical overscan assignment, trim, or bias subtraction approved",
        "files": infos, "pairs": pairs,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with pair_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)
    simple_infos = [{k: v for k, v in r.items() if k != "header"} for r in infos]
    with side_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(simple_infos[0].keys()))
        writer.writeheader()
        writer.writerows(simple_infos)
    print("OUTPUT:", json_path, pair_path, side_path, sep="\n  ")
    print("READ-ONLY QA COMPLETE; FITS, CONFIG, PIPELINE UNCHANGED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: script.py --selftest | selected_frames.csv")
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        run(Path(sys.argv[1]))
