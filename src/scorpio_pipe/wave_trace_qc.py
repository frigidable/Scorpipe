"""Fail-closed diagnostics for 2D wavelength calibration track identities.

Each trace contains many strongly correlated rows. Scientific checks therefore
operate on distinct laboratory lines as well as on individual samples.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebvander2d


def trace_identity_qc(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray,
    seed_x: np.ndarray, seed_lam: np.ndarray,
    *, collision_px: float = 1.5, max_collision_fraction: float = 0.01,
    min_seed_separation_px: float = 4.0,
) -> dict:
    """Detect adjacent traces that collapse or reverse their seeded order.

    Compare only rows where *both* lines are traced; an absent observation
    cannot be called a collision. Seed positions are independent manual
    LineID anchors and determine the expected separation and its sign.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)
    seed_x = np.asarray(seed_x, float)
    seed_lam = np.asarray(seed_lam, float)
    if not (x.size == y.size == lam.size and seed_x.size == seed_lam.size):
        raise ValueError("Trace and seed coordinate shapes disagree")
    if not all(np.all(np.isfinite(a)) for a in (x, y, lam, seed_x, seed_lam)):
        raise ValueError("Trace identity coordinates are not finite")
    if collision_px <= 0 or not (0 <= max_collision_fraction < 1):
        raise ValueError("Invalid collision QC thresholds")

    groups = np.unique(lam)
    seed_positions: dict[float, float] = {}
    for w, xx in zip(seed_lam, seed_x):
        if float(w) in seed_positions:
            raise ValueError("Duplicate manual wavelength seed")
        seed_positions[float(w)] = float(xx)

    pairs = []
    issues = []
    for wave in sorted(set(seed_positions).difference(float(t) for t in groups)):
        issues.append(
            f"Manual anchor {wave:.3f} A has no usable 2D trace"
        )
    if groups.size < 2:
        issues.append("Fewer than two independently traced laboratory lines")
    for a, b in zip(groups[:-1], groups[1:]):
        la, lb = float(a), float(b)
        if la not in seed_positions or lb not in seed_positions:
            raise ValueError("A traced line has no manual anchor")
        dx_seed = seed_positions[lb] - seed_positions[la]
        if abs(dx_seed) < min_seed_separation_px:
            issues.append(
                f"Unresolved manual anchors {la:.3f}/{lb:.3f} A "
                f"({dx_seed:.2f} px separation)"
            )

        ia = np.where(lam == a)[0]
        ib = np.where(lam == b)[0]
        common_y, ya, yb = np.intersect1d(
            y[ia], y[ib], assume_unique=False, return_indices=True
        )
        if common_y.size < 10:
            issues.append(
                f"Insufficient overlapping trace rows for {la:.3f}/{lb:.3f} A"
            )
            continue
        sep = x[ib[yb]] - x[ia[ya]]
        collision_fraction = float(np.mean(np.abs(sep) < collision_px))
        inversion_fraction = float(np.mean(sep * dx_seed <= 0))
        info = {
            "line_a_A": la, "line_b_A": lb,
            "seed_separation_px": float(dx_seed),
            "rows_compared": int(common_y.size),
            "collision_fraction": collision_fraction,
            "inversion_fraction": inversion_fraction,
            "median_abs_separation_px": float(np.median(np.abs(sep))),
        }
        pairs.append(info)
        if collision_fraction > max_collision_fraction:
            issues.append(
                f"Trace collision {la:.3f}/{lb:.3f} A: "
                f"{collision_fraction:.1%} of overlapping rows "
                f"within {collision_px:g} px"
            )
        if inversion_fraction > max_collision_fraction:
            issues.append(
                f"Trace order inversion {la:.3f}/{lb:.3f} A: "
                f"{inversion_fraction:.1%} of overlapping rows"
            )
    return {
        "status": "REJECT" if issues else "PASS",
        "n_unique_lines": int(groups.size),
        "adjacent_pairs": pairs,
        "issues": issues,
    }


def leave_one_line_out_cv(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray,
    *, max_samples_per_line: int = 24,
    degx: int = 3, degy: int = 2,
) -> dict:
    """Hold out *whole lines*, with balanced row samples for fitting.

    Uses a deliberately simple Chebyshev model as an independent geometric
    sanity check, not as a substitute for the main 2D solution. Per-row random
    CV would leak nearly identical points between train and validation sets.
    """
    x, y, lam = map(lambda a: np.asarray(a, float), (x, y, lam))
    if not (x.size == y.size == lam.size):
        raise ValueError("Cross-validation input sizes disagree")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(lam).all()):
        raise ValueError("Cross-validation contains nonfinite data")
    unique = np.unique(lam)
    if unique.size < max(degx + 2, 6):
        raise ValueError("Not enough independent lines for line-level cross-validation")
    if max_samples_per_line < 5:
        raise ValueError("Too few representative spatial samples per line")

    samples = []
    for w in unique:
        idx = np.where(lam == w)[0]
        idx = idx[np.argsort(y[idx])]
        picks = np.linspace(
            0, idx.size - 1,
            min(idx.size, max_samples_per_line), dtype=int,
        )
        samples.extend(idx[np.unique(picks)].tolist())
    ind = np.asarray(samples, dtype=int)
    xx, yy, ll = x[ind], y[ind], lam[ind]
    xo, xs = float((x.max()+x.min())/2), float((x.max()-x.min())/2)
    yo, ys = float((y.max()+y.min())/2), float((y.max()-y.min())/2)
    if not (xs > 0 and ys > 0):
        raise ValueError("Cross-validation lacks spatial/spectral extent")
    A = chebvander2d((xx-xo)/xs, (yy-yo)/ys, [degx,degy])
    errors, lines = [], []
    for w in unique:
        test = ll == w
        train = ~test
        coeff, _, rank, _ = np.linalg.lstsq(A[train], ll[train], rcond=None)
        if rank != A.shape[1]:
            raise ValueError(
                f"Line-level CV design is rank deficient for held-out {w:g} A"
            )
        resid = A[test] @ coeff - ll[test]
        errors.extend(resid.tolist())
        lines.append({
            "lambda_A": float(w),
            "n_holdout": int(test.sum()),
            "median_residual_A": float(np.median(resid)),
            "rms_A": float(np.sqrt(np.mean(resid**2))),
        })
    errors = np.asarray(errors, float)
    return {
        "rms_A": float(np.sqrt(np.mean(errors**2))),
        "p95_abs_A": float(np.percentile(np.abs(errors),95)),
        "max_abs_line_median_A": float(
            max(abs(row["median_residual_A"]) for row in lines)
        ),
        "n_independent_lines": int(unique.size),
        "model": f"chebyshev_x{degx}_y{degy}_whole_line_holdout",
        "per_line": lines,
    }


def evaluate_solution_2d(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray,
    residuals_A: np.ndarray, used_mask: np.ndarray,
    *, dispersion_A_per_px: float,
    min_used_fraction_per_line: float = 0.65,
    rms_limit_px: float = 1.0,
    p95_limit_px: float = 2.0,
    line_bias_limit_px: float = 1.5,
    cv_rms_limit_px: float = 1.25,
    cv_line_bias_limit_px: float = 2.0,
) -> dict:
    """Quantitative 2D quality gate in detector pixels, not setup-specific Å.

    A low global RMS cannot conceal a completely discarded or biased line.
    """
    x, y, lam, residuals_A = map(
        lambda a: np.asarray(a, float), (x, y, lam, residuals_A)
    )
    used_mask = np.asarray(used_mask, bool)
    if not (x.size == y.size == lam.size == residuals_A.size == used_mask.size):
        raise ValueError("2D QC arrays have different sizes")
    if not (np.isfinite(dispersion_A_per_px) and dispersion_A_per_px > 0):
        raise ValueError("2D QC requires finite positive dispersion")
    if not np.isfinite(residuals_A).all():
        raise ValueError("2D QC contains nonfinite residuals")
    if not np.any(used_mask):
        raise ValueError("2D QC has no accepted control points")

    used = residuals_A[used_mask]
    scale = float(dispersion_A_per_px)
    rms_px = float(np.sqrt(np.mean(used**2)) / scale)
    p95_px = float(np.percentile(np.abs(used), 95) / scale)
    issues, rows = [], []
    if rms_px > rms_limit_px:
        issues.append(f"2D RMS {rms_px:.3f} px exceeds {rms_limit_px:g} px")
    if p95_px > p95_limit_px:
        issues.append(f"2D P95 residual {p95_px:.3f} px exceeds {p95_limit_px:g} px")
    for w in np.unique(lam):
        group = lam == w
        kept = group & used_mask
        fraction = float(np.sum(kept)/np.sum(group))
        med = (float(np.median(residuals_A[kept])) if np.any(kept) else float("nan"))
        rows.append({
            "lambda_A": float(w), "n_total": int(np.sum(group)),
            "n_used": int(np.sum(kept)),
            "used_fraction": fraction,
            "median_residual_A": med,
        })
        if fraction < min_used_fraction_per_line:
            issues.append(
                f"Line {w:g} A: retained {fraction:.1%} of trace samples"
            )
        if not np.isfinite(med) or abs(med)/scale > line_bias_limit_px:
            issues.append(
                f"Line {w:g} A: median 2D bias "
                f"{(med / scale):.2f} px exceeds {line_bias_limit_px:g} px"
            )

    # CV uses unmasked traces so that sigma clipping cannot disguise wrong
    # identities by discarding the most difficult parts of entire lines.
    cv = leave_one_line_out_cv(x, y, lam)
    cv_rms_px = cv["rms_A"] / scale
    cv_max_bias_px = cv["max_abs_line_median_A"] / scale
    if cv_rms_px > cv_rms_limit_px:
        issues.append(
            f"Whole-line CV RMS {cv_rms_px:.3f} px exceeds {cv_rms_limit_px:g} px"
        )
    if cv_max_bias_px > cv_line_bias_limit_px:
        issues.append(
            f"Whole-line CV max line bias {cv_max_bias_px:.3f} px exceeds "
            f"{cv_line_bias_limit_px:g} px"
        )
    return {
        "status": "REJECT" if issues else "PASS",
        "dispersion_A_per_px": scale,
        "rms_px": rms_px,
        "p95_abs_px": p95_px,
        "line_level_cv": cv,
        "line_level_cv_rms_px": float(cv_rms_px),
        "line_level_cv_max_bias_px": float(cv_max_bias_px),
        "per_line": rows,
        "limits_px": {
            "rms": rms_limit_px, "p95": p95_limit_px,
            "line_bias": line_bias_limit_px,
            "cv_rms": cv_rms_limit_px,
            "cv_line_bias": cv_line_bias_limit_px,
        },
        "issues": issues,
    }
