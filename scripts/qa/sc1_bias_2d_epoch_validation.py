#!/usr/bin/env python3
"""Read-only 2D bias residual QA for the audited 2024-12-06 SCORPIO-1 night.

This is a diagnostic, NOT a calibration stage.  All slices are provisional.
Usage: python scripts/qa/sc1_bias_2d_epoch_validation.py --selftest
       python scripts/qa/sc1_bias_2d_epoch_validation.py selected_frames.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROWS = slice(32, 1000)
COLUMNS = slice(32, 2048)
LEFT = (ROWS, slice(2, 14))
RIGHT = (ROWS, slice(2066, 2078))
BLOCK = 64


def clipped_mean(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("Insufficient side-band pixels")
    for _ in range(4):
        med = np.median(values)
        sigma = 1.4826 * np.median(np.abs(values - med))
        if sigma <= 0:
            sigma = float(np.std(values))
        if sigma <= 0:
            break
        good = np.abs(values - med) <= 4 * sigma
        if good.sum() < 100:
            raise ValueError("Overscan clipping failed")
        if good.all():
            break
        values = values[good]
    return float(np.mean(values))


class OneEachTrimmedMean:
    """Online pixelwise mean after discarding one minimum and one maximum."""
    def __init__(self):
        self.n = 0
        self.total = None
        self.lo = None
        self.hi = None

    def add(self, image):
        a = np.asarray(image, dtype=np.float32)
        if a.ndim != 2 or not np.isfinite(a).all():
            raise ValueError("Non-finite or non-2D input")
        if self.total is None:
            self.total = np.zeros(a.shape, dtype=np.float64)
            self.lo = a.copy()
            self.hi = a.copy()
        if a.shape != self.total.shape:
            raise ValueError("Incompatible shapes")
        self.total += a
        np.minimum(self.lo, a, out=self.lo)
        np.maximum(self.hi, a, out=self.hi)
        self.n += 1

    def result(self):
        if self.n < 5:
            raise RuntimeError("Insufficient frames for trimmed mean")
        return ((self.total - self.lo - self.hi) / (self.n - 2)).astype(np.float32)


def centered(a):
    return a - float(np.mean(a, dtype=np.float64))


def blocks(a):
    height, width = a.shape
    height = (height // BLOCK) * BLOCK
    width = (width // BLOCK) * BLOCK
    return np.asarray(a[:height, :width], np.float64).reshape(
        height // BLOCK, BLOCK, width // BLOCK, BLOCK
    ).mean(axis=(1, 3))


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(
        std=float(np.std(a)),
        p05=float(np.percentile(a, 5)),
        median=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        max_abs=float(np.max(np.abs(a))),
    )


def expected_inventory(rows):
    inventory = {Path(r["file"]).name.lower(): r for r in rows
                 if r["role"].strip().lower() == "bias"}
    early = [f"s228001{i:02d}.fts" for i in range(1, 21)]
    late = [f"s228017{i:02d}.fts" for i in range(2, 21)]
    flagged = ["s22801620.fts", "s22801701.fts"]
    if set(inventory) != set(early + flagged + late) or len(inventory) != 41:
        raise RuntimeError("Selected BIAS inventory differs from audited 41 frames")
    return inventory, early, late, flagged


def read_residual(row):
    from astropy.io import fits
    path = Path(row["file"])
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        head = hdul[0].header
        if data is None or data.shape != (1032, 2080):
            raise RuntimeError(f"Unexpected CCD shape: {path.name}")
        if (abs(float(head["RATE"]) - 145) > 0.01
                or abs(float(head["GAIN"]) - 0.9) > 0.01
                or str(head.get("NODE", "")).strip().upper() != "A"):
            raise RuntimeError(f"Unexpected CCD readout: {path.name}")
        left = clipped_mean(data[LEFT])
        right = clipped_mean(data[RIGHT])
        active = np.array(data[ROWS, COLUMNS], dtype=np.float32, copy=True)
    side = 0.5 * (left + right)
    active -= np.float32(side)
    return active, side, left - right


def selftest():
    rng = np.random.default_rng(20261106)
    sample = rng.normal(999.4, 3, size=25000)
    sample[0] = 65535
    assert abs(clipped_mean(sample) - 999.4) < 0.1
    pattern = np.linspace(-0.15, 0.15, 128)[None, :] + np.zeros((96, 128))
    a, b = OneEachTrimmedMean(), OneEachTrimmedMean()
    for i in range(20):
        a.add(pattern + 0.4 + rng.normal(0, 0.3, pattern.shape))
    for i in range(19):
        b.add(pattern - 0.2 + rng.normal(0, 0.3, pattern.shape))
    assert blocks(a.result()).shape == (1, 2)
    assert np.std(blocks(centered(a.result()) - centered(b.result()))) < 0.05
    print("SELFTEST PASSED")


def run(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        inventory, early, late, flagged = expected_inventory(list(csv.DictReader(f)))
    groups = dict(
        early20=early, late19=late,
        early_a=early[:10], early_b=early[10:],
        late_a=late[:9], late_b=late[9:],
    )
    accum = {name: OneEachTrimmedMean() for name in groups}
    ordered = early + flagged + late
    print("FIRST PASS: independent residual MasterBias maps", flush=True)
    for index, name in enumerate(ordered, 1):
        image, side, asym = read_residual(inventory[name])
        for group, filenames in groups.items():
            if name in filenames:
                accum[group].add(image)
        print(f"{index:02d}/41 {name}: side {side:.4f}, L-R {asym:+.4f}", flush=True)

    maps = {name: centered(acc.result()) for name, acc in accum.items()}
    blockmaps = {name: centered(blocks(im)) for name, im in maps.items()}
    delta = centered(blockmaps["late19"] - blockmaps["early20"])
    ea = centered(blockmaps["early_b"] - blockmaps["early_a"])
    la = centered(blockmaps["late_b"] - blockmaps["late_a"])

    cv_rows = []
    print("SECOND PASS: out-of-epoch prediction", flush=True)
    for index, name in enumerate(ordered, 1):
        image, _, _ = read_residual(inventory[name])
        observed = centered(blocks(image))
        model_name = "late19" if name in early else "early20"
        corrected = centered(observed - blockmaps[model_name])
        before = float(np.sqrt(np.mean(observed ** 2)))
        after = float(np.sqrt(np.mean(corrected ** 2)))
        epoch = "early20" if name in early else ("late19" if name in late else "flagged")
        cv_rows.append(dict(
            file=name, group=epoch, independent_model=model_name,
            block_rms_before_adu=before, block_rms_after_adu=after,
            ratio=after / before if before > 0 else None,
        ))
        print(f"{index:02d}/41 {name}: block RMS {before:.5f} -> {after:.5f} ADU", flush=True)

    cross = {}
    for group in ("early20", "late19", "flagged"):
        subset = [r for r in cv_rows if r["group"] == group]
        cross[group] = dict(
            n=len(subset),
            mean_before_adu=float(np.mean([r["block_rms_before_adu"] for r in subset])),
            mean_after_adu=float(np.mean([r["block_rms_after_adu"] for r in subset])),
            improved=sum(r["block_rms_after_adu"] < r["block_rms_before_adu"]
                         for r in subset),
        )

    summary = dict(
        status="DIAGNOSTIC_ONLY_HOLD",
        geometry_unverified=True,
        detector_shape=[1032, 2080],
        diagnostic_region_y=[32, 1000],
        diagnostic_region_x=[32, 2048],
        candidate_sidebands_left=[2, 14],
        candidate_sidebands_right=[2066, 2078],
        combined_using="per-pixel discard one min and one max",
        flagged_removed_only_for_epoch_masters=flagged,
        n_early=20, n_late_stable=19,
        block_size=BLOCK,
        early_pattern=stats(blockmaps["early20"]),
        late_pattern=stats(blockmaps["late19"]),
        between_epochs=stats(delta),
        early_internal_half_difference=stats(ea),
        late_internal_half_difference=stats(la),
        epoch_pattern_correlation=float(np.corrcoef(
            blockmaps["early20"].ravel(), blockmaps["late19"].ravel())[0, 1]),
        independent_transfer=cross,
    )
    root = csv_path.parent
    json_path = root / "sc1_bias_2d_epoch_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    blocks_path = root / "sc1_bias_2d_64px_blocks.csv"
    with blocks_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["block_y", "block_x", "early20_adu", "late19_adu",
                         "late_minus_early_adu", "early_half_difference_adu",
                         "late_half_difference_adu"])
        for y, x in np.ndindex(delta.shape):
            writer.writerow([y, x, float(blockmaps["early20"][y, x]),
                             float(blockmaps["late19"][y, x]), float(delta[y, x]),
                             float(ea[y, x]), float(la[y, x])])

    transfer_path = root / "sc1_bias_2d_crossvalidation_41.csv"
    with transfer_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(cv_rows[0]))
        writer.writeheader()
        writer.writerows(cv_rows)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        lim = max(np.percentile(np.abs(blockmaps["early20"]), 99),
                  np.percentile(np.abs(blockmaps["late19"]), 99), 0.01)
        for ax, matrix, label, scale in zip(
            axes,
            (blockmaps["early20"], blockmaps["late19"], delta),
            ("Early 20", "Late 19", "Late minus early"),
            (lim, lim, max(np.percentile(np.abs(delta), 99), 0.01)),
        ):
            img = ax.imshow(matrix, origin="lower", cmap="coolwarm",
                            vmin=-scale, vmax=scale, aspect="auto")
            ax.set_title(label + " [ADU]")
            ax.set_xlabel("X, 64-px blocks")
            ax.set_ylabel("Y, 64-px blocks")
            fig.colorbar(img, ax=ax)
        fig.savefig(root / "sc1_bias_2d_epoch_maps.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass

    print("EARLY pattern STD:", summary["early_pattern"]["std"])
    print("LATE pattern STD:", summary["late_pattern"]["std"])
    print("INTER-EPOCH difference STD:", summary["between_epochs"]["std"])
    print("INTRA-EPOCH early/late half differences:",
          summary["early_internal_half_difference"]["std"],
          summary["late_internal_half_difference"]["std"])
    print("INDEPENDENT TRANSFER:", cross)
    print("OUTPUT:", json_path, blocks_path, transfer_path, sep="\n  ")
    print("DIAGNOSTIC COMPLETE; INPUT FITS AND CONFIG UNMODIFIED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: script.py --selftest | selected_frames.csv")
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        run(Path(sys.argv[1]))
