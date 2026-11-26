# P1.4d6 — empirical local Jacobian audit for actual SkySub LSQR

**Scope:** read-only diagnosis, fixed final sky selection and fitting
weights, no reduction/data-product change. Parent: PR #152.

## Motivation

The existing P1.4d3 operator uses converged, unregularized weighted least
squares and its analytic Jacobian `R=I-X(Xs.T W Xs)^(-1)Xs.T W S`.
Production `kelson_raw` uses `scipy.sparse.linalg.lsqr` with
`atol=btol=1e-10`, `iter_lim=2000`, and data-dependent sigma clipping.
Even when the converged problem has a unique answer, finite-iteration
LSQR need not respond linearly to new data, because iterations and
stopping conditions depend on the right-hand side.

`calib.sky_lsqr_derivative.assess_kelson_lsqr_directional_derivatives`
takes **actual linked** frozen-fit NPZ plus `skymodel_raw.fits`
and `skysub_raw.fits` from the same science run.

The diagnostic first requires P1.4d5 archive-to-FITS fidelity.
It then re-solves the archived *final fixed* weighted training problem
with the saved LSQR tolerances and iteration cap, comparing its full-CCD
model to the archived coefficients.

For reproducible native pixel noise directions `v`, it solves
`LSQR(Aw, b + epsilon*W^(1/2)Sv)` and
`LSQR(Aw, b - epsilon*W^(1/2)Sv)`.
This yields a central finite-difference estimate of the SkySub residual
Jacobian action on `v`. The analytic reference is `R@v` from
the previously developed frozen WLS operator.

Two independent relative L2 metrics are required:

- residual derivative discrepancy `||J_LSQR(v)-R v||/||R v||`;
- **sky-model derivative discrepancy**
  `||[v-J_LSQR(v)]-[v-R v]||/||v-R v||`.

The second metric prevents the direct native-pixel noise term from
hiding errors in the much smaller sky-model derivative. Its denominator
is never artificially replaced with the noise norm.

A numerical `status=MATCH` requires both discrepancies below explicit
user-controlled tolerances, replay-model consistency, and recorded
original/replay/perturbed LSQR stopping codes 1 or 2. Stop codes 3
(condition limit) and 7 (iteration limit), among others, forbid MATCH.
The result ALWAYS sets `science_uncertainty_qualified=false`.
Unstable comparisons remain a diagnostic report, not a variance estimate.

## Methodological limitations

This checks only **local frozen-mask directional behavior** of the
actual numerical solver on the archived final sky pixel set.
It does NOT differentiate: sigma-clipping decisions, ROI/object masks,
knots chosen from data, wavelength solution, sky method switching, flat
and bias estimator calibration, or cross-exposure sky fitting. It does
not by itself establish coverage of published confidence intervals.

The diagnostic constructs the existing complete sparse sky basis
from the archival wavelength map. On very large CCDs this can be
computationally and memory expensive. Execute opt-in on representative
real BTA/SCORPIO frames after the full regression gate; assess results
over multiple dither/readout/night configurations before considering
a production variance model.

SciPy LSQR documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
