#!/usr/bin/env python3
"""Read-only identifiability test of dispersion-axis shift between SCORPIO-1 FLATs.

Fits only high-pass 1D spectral details; a slowly changing lamp continuum does NOT
constitute evidence of a detector shift. The test reports a shift only if spatial
bands/windows contain sufficiently strong, correlated high-frequency features.
Provisional candidate side-band regions; no production correction.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

NAMES = ("s22800614.fts", "s22800615.fts", "s22800616.fts")
SHAPE = (1032, 2080)
Y0, Y1 = 32, 992
X0, X1 = 32, 2048
STRIPES = 12
WINDOWS = ((150, 550), (550, 950), (950, 1350), (1350, 1750))
SHIFTS = np.linspace(-2.0, 2.0, 161)


def clipped_mean(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 100:
        raise ValueError("Insufficient side-band pixels")
    for _ in range(4):
        med = np.median(v)
        sigma = 1.4826 * np.median(np.abs(v - med))
        if sigma <= 0:
            sigma = float(np.std(v))
        if sigma <= 0:
            break
        ok = np.abs(v - med) < 4 * sigma
        if ok.sum() < 100:
            raise ValueError("Insufficient pixels after clipping")
        if ok.all():
            break
        v = v[ok]
    return float(v.mean())


def gaussian_smooth(v, sigma=22.0):
    v = np.asarray(v, dtype=np.float64)
    radius = int(np.ceil(4.0 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, (radius, radius), mode="reflect"), k, mode="valid")


def profile_features(prof):
    smooth = gaussian_smooth(prof)
    if not np.all(np.isfinite(smooth)) or np.min(smooth) <= 0:
        raise ValueError("Invalid FLAT continuum")
    high = prof / smooth - 1.0
    return high, smooth


def shift_window(a, b, *, nrows=80, gain_e_per_adu=3.5, x1=150, x2=550):
    """A(x) compared to B(x-dx); positive dx means B shifts right to A."""
    if a.shape != b.shape or a.ndim != 1 or x1 < 100 or x2 > len(a) - 80:
        raise ValueError("Invalid profile/window")
    if nrows <= 0 or gain_e_per_adu <= 0:
        raise ValueError("Invalid gain/stripe height")
    ha, ca = profile_features(a)
    hb, cb = profile_features(b)
    xx = np.arange(x1, x2)
    use = (ca[xx] > 2500) & (cb[xx] > 2500) & (a[xx] < 60000) & (b[xx] < 60000)
    n = int(use.sum())
    if n < (x2-x1) * 0.75:
        return dict(eligible=False, reason="LOW_ILLUMINATION", n_use=n)

    x = xx[use].astype(float)
    observed = ha[xx][use]
    observed -= observed.mean()
    strength = float(np.std(hb[xx][use]))
    poisson_median_rel = float(np.median(
        1.2533*np.sqrt(np.maximum(cb[xx][use], 1.0) / gain_e_per_adu / nrows)
        /cb[xx][use]
    ))
    # Strong spectral details are needed to identify a shift from lamp-color change.
    if strength < 3 * poisson_median_rel:
        return dict(eligible=False, reason="INSUFFICIENT_SPECTRAL_FEATURES",
                    n_use=n, feature_rms=strength, noise_floor=poisson_median_rel)
    denom = float(observed @ observed)
    if denom <= 0:
        return dict(eligible=False, reason="NO_VARIANCE", n_use=n)
    losses, corrs = [], []
    source_x = np.arange(len(hb), dtype=float)
    for delta in SHIFTS:
        ref = np.interp(x - delta, source_x, hb)
        ref = ref - ref.mean()
        norm = float(ref @ ref)
        if norm < 1e-15:
            losses.append(denom)
            corrs.append(0.0)
            continue
        dot = float(observed @ ref)
        amp = max(dot / norm, 0.0)
        losses.append(float(np.sum((observed - amp * ref) ** 2)))
        corrs.append(dot / np.sqrt(norm * denom))
    losses = np.asarray(losses)
    corrs = np.asarray(corrs)
    i = int(np.argmin(losses))
    zero_i = int(np.argmin(np.abs(SHIFTS)))
    ratio = float((losses[zero_i] - losses[i]) / max(losses[zero_i], 1e-30))
    informative = bool((0 < i < len(SHIFTS)-1)
                       and corrs[i] > 0.35 and ratio > 0.005)
    return dict(
        eligible=informative,
        reason=("INFORMATIVE" if informative else "AMBIGUOUS_SHIFT"),
        n_use=n, feature_rms=strength,
        noise_floor=poisson_median_rel,
        best_delta_px=float(SHIFTS[i]),
        corr_at_best=float(corrs[i]),
        improvement_over_zero=float(ratio),
        shift_at_boundary=bool(i in (0, len(SHIFTS)-1)),
    )


def selftest():
    rng = np.random.default_rng(171206)
    x = np.arange(2016.)
    continuum = 11500 * (1 + 0.14 * (x - 1000) / 1000)
    pattern = np.zeros(x.shape)
    for center in (212, 304, 388, 713, 780, 841, 1114, 1208, 1337, 1512, 1618, 1681):
        pattern -= 0.03*np.exp(-0.5 * ((x-center)/5.0)**2)
    b = continuum*(1 + pattern)
    shifted = np.interp(x - 0.45, x, pattern, left=pattern[0], right=pattern[-1])
    a = continuum*(1 + shifted)
    a += rng.normal(0, 4.0, len(x))
    b += rng.normal(0, 4.0, len(x))
    for x1, x2 in WINDOWS:
        res = shift_window(a, b, x1=x1, x2=x2)
        assert res["eligible"], res
        assert abs(res["best_delta_px"] - 0.45) < 0.15, res
    a = continuum + rng.normal(0, 4.0, len(x))
    b = continuum + rng.normal(0, 4.0, len(x))
    r = shift_window(a, b, x1=150, x2=550)
    assert not r["eligible"] and r["reason"] == "INSUFFICIENT_SPECTRAL_FEATURES", r
    print("SELFTEST PASSED")


def read_profiles(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
        dat = hdul[0].data
        if dat is None or dat.shape != SHAPE:
            raise RuntimeError(f"Unexpected CCD geometry: {path.name}")
        if (abs(float(h["RATE"]) - 225) > 0.01
            or abs(float(h["GAIN"]) - 3.5) > 0.01
            or str(h.get("NODE", "")).strip().upper() != "A"):
            raise RuntimeError(f"Unexpected readout mode: {path.name}")
        left = clipped_mean(dat[Y0:Y1, 2:14])
        right = clipped_mean(dat[Y0:Y1, 2066:2078])
        side = 0.5*(left+right)
        profiles = []
        for y in range(Y0, Y1, (Y1-Y0)//STRIPES):
            block = np.asarray(dat[y:y+80, X0:X1], dtype=np.float64)
            profiles.append(np.median(block, axis=0)-side)
    if len(profiles) != STRIPES:
        raise RuntimeError("Incorrect stripe count")
    return profiles, side


def run(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    by_name = {Path(r["file"]).name.lower(): Path(r["file"])
               for r in records if r["role"].strip().lower() == "flat"}
    if not all(n in by_name for n in NAMES):
        raise RuntimeError("Missing audited FLAT frames")
    profs = {}
    sides = {}
    for name in NAMES:
        profs[name], sides[name] = read_profiles(by_name[name])
        print(f"{name}: 12 spatial spectra, side mean={sides[name]:.4f} ADU", flush=True)
    rows = []
    groups = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = NAMES[i], NAMES[j]
        estimates = []
        reason_counts = {}
        for band in range(STRIPES):
            for window_idx, (low, high) in enumerate(WINDOWS):
                result = shift_window(profs[a][band], profs[b][band],
                                      nrows=80, x1=low, x2=high)
                rows.append(dict(flat_a=a, flat_b=b, spatial_stripe=band,
                                 xwindow=window_idx, x0=low, x1=high, **result))
                reason_counts[result["reason"]] = reason_counts.get(result["reason"], 0)+1
                if result["eligible"]:
                    estimates.append(result["best_delta_px"])
        group = dict(flat_a=a, flat_b=b, n_total=STRIPES*len(WINDOWS),
                     n_informative=len(estimates), reason_counts=reason_counts,
                     median_shift_px=float(np.median(estimates)) if estimates else None,
                     mad_shift_px=(float(1.4826*np.median(np.abs(
                         np.asarray(estimates)-np.median(estimates)))) if len(estimates)>1 else None),
                     status=("CANDIDATE_SHIFT_NEEDS_PHYSICAL_REVIEW" if len(estimates)>=12
                             else "INSUFFICIENT_CONSISTENT_SPECTRAL_FEATURES"))
        groups.append(group)
        print(f"{a} / {b}: informative={group['n_informative']}/48 "
              f"shift_candidate={group['median_shift_px']} px "
              f"MAD={group['mad_shift_px']} | {reason_counts}", flush=True)
    out = csv_path.parent
    csv_output = out / "sc1_fast_flat_shift_windows.csv"
    json_output = out / "sc1_fast_flat_shift_summary.json"
    with csv_output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_output.write_text(json.dumps({
        "status":"DIAGNOSTIC_ONLY_HOLD", "not_approved_as_shift_calibration":True,
        "candidate_sidebands_not_certified":True,
        "assumed_gain_e_per_adu":3.5,
        "band_rows":80,"windows_in_cropped_x_coordinates":WINDOWS,
        "shift_convention":"positive if B is shifted right to match A",
        "caveat":"High-pass correlations do not prove mechanical flexure; spectral feature scarcity is reported",
        "groups":groups,"side_means_adu":sides,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT:", json_output, csv_output, sep="\n  ")
    print("READ-ONLY; NO SCIENCE FITS, CONFIG OR CALIBRATION PRODUCT MODIFIED")


if __name__ == "__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: script.py --selftest | selected_frames.csv")
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        run(Path(sys.argv[1]))
