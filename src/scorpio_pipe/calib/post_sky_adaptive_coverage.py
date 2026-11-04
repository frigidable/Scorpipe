"""Synthetic-only ablation of adaptive Stack2D and data-derived Extract1D profiles.

Calls the ACTUAL production kernels on each noisy tiny 2D stack. Measures
empirical variance/coverage against the *reported conditional* VAR for four
combinations: inverse-variance/HUBER stacking × boxcar/optimal extraction.

No science FITS are emitted and no observational or full-variance
qualification is inferred. This diagnostic intentionally starts AFTER
SkySub; wavelength and spatial registration are fixed identity maps.
"""
from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from astropy.io import fits

from scorpio_pipe.stages.extract1d import _boxcar_extract, _optimal_extract
from scorpio_pipe.stages.stack2d import _stack_chunk_streaming


_MODES = (
    "invvar_boxcar",
    "invvar_optimal",
    "huber_boxcar",
    "huber_optimal",
)


def synthetic_adaptive_stack_extract_coverage(
    *,
    n_realizations: int = 360,
    seed: int = 20261104,
    object_flux: float = 18.,
    detector_sigma: float = 5.,
    huber_c: float = 1.2,
    optimal_smooth_lambda_sigma: float = 1.4,
    outlier_sigma: float = 0.,
) -> dict[str, Any]:
    """Measure nonlinearity without silently treating weights as fixed.

    Each realization draws three independent 5x9 science frames around
    exactly known synthetic source flux. A constant, optional outlier
    (one frame, one pixel) is an explicitly *deterministic contaminant*,
    not a Gaussian random variable whose variance can be added to VAR.

    Every draw uses the production `_stack_chunk_streaming` (including
    per-pixel data-derived Huber downweighting and its conditional VAR)
    and the production `_boxcar_extract`/`_optimal_extract` kernels.
    Optimal extraction learns its profile FROM THE SAME noisy data and
    smooths it across neighboring wavelengths, potentially coupling bins.

    `invvar_boxcar` is the analytically soluble reference, whose truth
    and variance are known from independent Gaussian pixel summation.
    The other three are observationally *diagnostic* — no universal
    expected coverage threshold is imposed on adaptive methods.
    """
    if (isinstance(n_realizations, (bool, np.bool_))
            or not isinstance(n_realizations, (int, np.integer))
            or not 200 <= n_realizations <= 1200):
        raise ValueError("n_realizations must be integer in [200, 1200]")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    for name, value, lower in (
        ("object_flux", object_flux, 0.),
        ("detector_sigma", detector_sigma, 0.),
        ("huber_c", huber_c, 0.),
        ("optimal_smooth_lambda_sigma", optimal_smooth_lambda_sigma, 0.),
        ("outlier_sigma", outlier_sigma, 0.),
    ):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (float, int, np.floating, np.integer))
                or not np.isfinite(value)
                or (value <= 0 if lower == 0. and name != "outlier_sigma"
                    else value < lower)):
            raise ValueError(f"{name}: invalid finite synthetic control")

    ny, nlam, n_exposures = 5, 9, 3
    spatial_profile = np.array([.08, .24, .36, .24, .08])
    true_line = float(object_flux) * (
        1. + .25 * np.exp(-.5*((np.arange(nlam) - 4.)/1.3)**2)
    )
    source = spatial_profile[:, None]*true_line[None, :]
    input_variance = float(detector_sigma)**2
    truth = np.sum(source, axis=0)
    analytic_boxcar_variance = ny*input_variance/n_exposures
    rng = np.random.default_rng(int(seed))

    flux = {
        mode: np.empty((int(n_realizations), nlam), dtype=np.float64)
        for mode in _MODES
    }
    reported_var = {
        mode: np.empty_like(flux[mode]) for mode in _MODES
    }
    huber_suppressed = []
    profile_fallback = {"invvar_optimal": [], "huber_optimal": []}
    z95 = 1.959963984540054

    for k in range(int(n_realizations)):
        frames = (
            source[None, :, :] +
            rng.normal(scale=float(detector_sigma),
                       size=(n_exposures, ny, nlam))
        ).astype(np.float32)
        if outlier_sigma > 0:
            frames[0, 2, 4] += (
                float(outlier_sigma)*float(detector_sigma)
            )
        hduls = [
            fits.HDUList([
                fits.PrimaryHDU(data=frames[e]),
                fits.ImageHDU(
                    data=np.full((ny, nlam), input_variance,
                                 dtype=np.float32),
                    name="VAR",
                ),
                fits.ImageHDU(
                    data=np.zeros((ny, nlam), dtype=np.uint16),
                    name="MASK",
                ),
            ])
            for e in range(n_exposures)
        ]
        try:
            for stack_name, stack_method in (
                ("invvar", "invvar"),
                ("huber", "invvar_huber"),
            ):
                combined, variance, mask, cov, suppressed, _ = _stack_chunk_streaming(
                    hduls=hduls, y0=0, y1=ny,
                    method=stack_method, robust_iter=1,
                    huber_c=float(huber_c), clip_sigma=4.,
                    y_shifts=[0.]*n_exposures,
                    y_align_enabled=False,
                    sci_scales=[1.]*n_exposures,
                    var_scales=[1.]*n_exposures,
                    fatal_bits=np.uint16(0), var_floor=None,
                )
                if (not np.isfinite(combined).all()
                        or not np.isfinite(variance).all()
                        or np.any(variance <= 0)
                        or not np.array_equal(cov, np.full_like(cov, n_exposures))):
                    raise RuntimeError("Synthetic Stack2D produced incomplete coverage")
                if stack_name == "huber":
                    huber_suppressed.append(suppressed > 0)

                # The Stack2D REJECTED bit is informational for an
                # already-verified coadd. No fatal DQ exists here.
                center = np.full(nlam, 2., dtype=np.float64)
                box = _boxcar_extract(
                    combined, variance, mask, center,
                    ap_hw=2, fatal_bits=0, min_good_frac=1.,
                    sky_resid=None, apply_local_sky=False,
                )
                opt = _optimal_extract(
                    combined, variance, mask, center,
                    ap_hw=2, fatal_bits=0, min_good_frac=1.,
                    sky_resid=None, apply_local_sky=False,
                    smooth_lambda_sigma=float(optimal_smooth_lambda_sigma),
                    clip_negative_profile=True, backend="numpy",
                )
                for suffix, outcome in (("boxcar", box), ("optimal", opt)):
                    key = f"{stack_name}_{suffix}"
                    f = np.asarray(outcome[0], dtype=np.float64)
                    v = np.asarray(outcome[1], dtype=np.float64)
                    if (f.shape != (nlam,) or v.shape != (nlam,)
                            or not np.isfinite(f).all()
                            or not np.isfinite(v).all() or np.any(v <= 0)):
                        raise RuntimeError(f"Invalid {key} synthetic extraction")
                    flux[key][k] = f
                    reported_var[key][k] = v
                    if suffix == "optimal":
                        profile_fallback[key].append(
                            float(outcome[-1]["profile_fallback_frac"])
                        )
        finally:
            for h in hduls:
                h.close()

    modes = {}
    for name in _MODES:
        err = flux[name] - truth[None, :]
        empirical = np.var(err, axis=0, ddof=1)
        conditional = np.mean(reported_var[name], axis=0)
        coverage = np.mean(
            np.abs(err) <= z95*np.sqrt(reported_var[name]), axis=0,
        )
        mean_error = np.mean(err, axis=0)
        cov01 = float(np.cov(err[:, 3], err[:, 4], ddof=1)[0, 1])
        modes[name] = {
            "empirical_variance": empirical.tolist(),
            "mean_reported_conditional_variance": conditional.tolist(),
            "variance_ratio_empirical_to_reported": (
                empirical/conditional
            ).tolist(),
            "coverage95": coverage.tolist(),
            "mean_error": mean_error.tolist(),
            "neighbor_wavelength_covariance_3_4": cov01,
            "mean_abs_difference_from_true_flux": float(
                np.mean(np.abs(err))
            ),
            "fraction_profile_fallback": (
                float(np.mean(profile_fallback[name]))
                if name in profile_fallback else None
            ),
            "adaptive_stack": name.startswith("huber"),
            "data_derived_extraction_profile": name.endswith("optimal"),
        }

    reference = modes["invvar_boxcar"]
    ref_var = np.asarray(reference["empirical_variance"])
    ref_cov = np.asarray(reference["coverage95"])
    # Tolerances are sampling-aware and stated as a SEPARATE control,
    # not used to retroactively qualify adaptive methods.
    var_tol = 5.*sqrt(2./(n_realizations-1))
    coverage_tol = 5.*sqrt(.95*.05/n_realizations)
    reference_ok = bool(
        np.max(np.abs(ref_var/analytic_boxcar_variance-1.)) <= var_tol
        and np.max(np.abs(ref_cov-.95)) <= coverage_tol
        and outlier_sigma == 0.
    )
    return {
        "status": "REFERENCE_CONSISTENT" if reference_ok else
                  ("CONTAMINATED_REFERENCE" if outlier_sigma > 0
                   else "REFERENCE_MC_MISMATCH"),
        "scope": "SYNTHETIC_POST_SKYSUB_ADAPTIVE_STACK_EXTRACT",
        "n_realizations": int(n_realizations),
        "n_exposures": n_exposures,
        "n_rows": ny,
        "n_wavelength_bins": nlam,
        "injected_true_flux": truth.tolist(),
        "analytic_invvar_boxcar_variance": float(analytic_boxcar_variance),
        "reference_boxcar_variance_tolerance": float(var_tol),
        "reference_boxcar_coverage_tolerance": float(coverage_tol),
        "outlier_sigma": float(outlier_sigma),
        "fraction_huber_reweighted_realizations": float(
            np.mean(huber_suppressed)
        ),
        "modes": modes,
        "adaptive_intervals_are_descriptive_only": True,
        "science_uncertainty_qualified": False,
        "production_variance_release_allowed": False,
    }
