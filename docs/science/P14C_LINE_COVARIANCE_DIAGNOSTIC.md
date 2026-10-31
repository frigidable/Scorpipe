# P1.4c — conditional line-window uncertainty with continuum estimation

**Status:** diagnostic-only mathematical component; does not change the
production line catalogue or its existing median-continuum estimator.
Addresses a limited part of Issue #42; not approved for observational science.

## Scientific distinction

In production, science/lines.py::measure_line computes the line flux after
subtracting a continuum estimated using sideband **medians**. The reported
err_stat = sqrt(sum(VAR_k * gradient(lambda)_k**2)) misses both spectral
off-diagonal covariance and statistical uncertainty of the fitted continuum.
The P1.4b covariance contraction cannot be substituted blindly: the actual
median estimator is nonlinear/data-dependent, and the 1D data have already
undergone sky subtraction, spatial extraction, alignment and often robust
stacking, without a preserved full covariance operator.

This change deliberately introduces a **different linear diagnostic**:

1. Input flux is spectral **density per wavelength**; input VAR is its
   squared density. Integrated electron counts must be converted using
   COUNT_V1 scaling, not passed as spectral densities.
2. Each line window is integrated using exact overlaps with input bin
   edges inferred from wavelength centers (the COUNT_V1 edge convention).
   A masked/unqualified pixel in the line window fails closed. Missing or
   entirely masked left/right continuum regions also fail closed.
3. Continuum sidebands are fit by a constant or affine-in-wavelength model
   via diagonal weighted least squares, with **fixed** inverse-variance
   weights. Order-one is default, unlike production median-continuum.
4. For line integration weights a and continuum fit operator
   H = (X^T W X)^-1 X^T W, the full-spectrum line estimator is

   F = a^T f_line - (a^T X_line) H f_side = u^T f.

   Therefore Var_diag(F) = sum(u_i**2 * VAR_i) includes the sideband
   continuum uncertainty. If a scientifically qualified covariance operator
   is supplied for the *same* flux-density vector, Var_cond(F) = u^T C u.
   P1.4b can contract this directly only with the full fixed operator
   from input CCD pixels to the measured output and with correct units.
5. Report explicitly DIAG_ONLY or CONDITIONAL_USER_OPERATOR. Even the
   latter is **not** a full observational covariance claim.
   The full functional u is returned for independent tests/covariance.

## Provenance boundaries and missing physics

- Production measure_line/measure_lines, FITS and line CSV are untouched.
  Results are accessible only through this explicit diagnostic API.
- Sideband diagonal WLS uses covariance for uncertainty AFTER fitting:
  it is not generalized least squares and need not be optimal under
  correlated errors, although its conditional variance is calculable.
- Center-inferred pixel edges treat the intervals between supplied samples
  as actual CCD pixel intervals. Missing wavelength pixels omitted from an
  array cannot be diagnosed from centers alone; upstream MASK/coverage
  must be checked. Nonmonotonic wavelength grids fail closed.
- Input C does not yet include uncertainty from sky fitting, calibration,
  cosmic inpainting, wavelength solutions, trace/profile fitting, fluxcal,
  robust Huber/clip stacking or subpixel spatial shifts. Model these before
  upgrading any science uncertainty status.
- This diagnostic does not compute EW, nonlinear median-continuum uncertainty,
  or correlated errors in pPXF. These require separate validation.

## Validation gates

Independent normal-equation WLS reference including continuum terms;
analytic zero-line tests with sloped continua; exact partial-bin overlaps;
masked continuum/line windows; invalid wavelength arrays; nonuniform
dispersion; dense covariance oracle and seeded Monte Carlo on 2-to-1-A
COUNT_V1 rectification. Independent seeded simulation demonstrates that
the existing median-continuum err_stat can underestimate uncertainties even
on uncorrelated 1D noise. Both Linux CI and Windows tests on identical
commit must pass before accepting the new diagnostic as software-qualified.

Further work: reconstruct the complete linear operator through fixed
Stack2D/Extract1D and flux-calibration transformations; separately quantify
adaptive-step and median-continuum uncertainties.
