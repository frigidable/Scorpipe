# P1.4d7 — synthetic full-refit sigma-clipping sensitivity audit

**Status:** independent diagnostic for the algorithm in
`stages.sky_sub._fit_kelson_bspline`, NOT scientific uncertainty
calibration. Baseline: Windows-validated PR #153.

## Why this gate

The P1.4d3–d6 frozen SkySub residual operator is conditional on fixed
sky-window selection, clipped samples, wavelength knots and weights.
Production `kelson_raw` actually re-fits B-splines with SciPy LSQR and
iteratively excludes sky samples whose standardized residual exceeds
`clip_sigma`. Re-selecting samples is nonlinear and generally not
described by the frozen WLS derivative.

With synthetic science, variance, wavelength and sky candidate mask
held explicitly, `calib.sky_clipping_sensitivity.audit_synthetic_kelson_clipping`:

1. Executes the **real production Kelson fit** on baseline synthetic data.
2. Records the baseline final retained/rejected sample membership and
   constructs the P1.4d3 analytic frozen WLS projector for that membership.
3. Applies seeded Gaussian native CCD perturbations, or optionally
   explicit synthetic probe vectors. Runs the **full production fit again**
   on `D +/- epsilon v` each time, with the same candidate sky pixels,
   error-map weights, y geometry and algorithm settings, allowing
   **the actual sigma clipping to select new samples**.
4. Compares both final memberships with baseline and the full CCD
   central-difference residual and fitted-sky model derivatives with
   the analytic frozen operator.
5. Returns exactly one of `FROZEN_STABLE`,
   `SELECTION_CHANGED`, `NUMERIC_UNSTABLE`, `REFIT_FAILED`;
   any sample-membership change takes precedence over numerical
   derivative agreement. Non-converged LSQR statuses cannot be accepted.
   `science_uncertainty_qualified=false` is unconditional.

Two synthetic test regimes are independently controlled: (a) smoothly
varying spectral sky with modest Gaussian readout noise; (b) the same
scene plus a deliberately high-amplitude outlying sky pixel and an
explicit perturbation that crosses its sigma-clipping threshold.
There is also a deliberate zero-tolerance numerical rejection and
invalid geometry/variance/parameter gates. Reported covariance is NEVER
silently corrected from this diagnostic.

## Restrictions

This gate is **synthetic-only**: complete finite native CCD maps are
required, and a strict 100,000-pixel ceiling prevents accidental use
as an unbounded large-detector analysis. The wavelength map, spatial
sky-region selection, variance weights and B-spline knot domain are
held fixed. It tests the additional nonlinearity from re-running
sigma clipping and finite-iteration LSQR, not all of SkySub's image
segmentation, cosmic masking, MasterBias/Flat calibration or nonlinear
future stages.

A finite set of Gaussian perturbations cannot prove zero clipping
changes at all amplitudes. Conversely, a changed sample means the
linear conditional Jacobian cannot be assumed valid across the tested
perturbation range; this is NOT automatically a claim that the
final astrophysical measurement is biased.

References:
- Kelson (2003), ApJ 588, 813, doi:10.1086/374114.
- SciPy `lsqr` solver and stopping conditions,
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
