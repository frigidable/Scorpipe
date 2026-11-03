# P1.4d3 — frozen Kelson B-spline SkySub covariance, including MasterBias

**Status: numerical operator and independent oracle; NOT a production variance
correction.** Parent: PR #149, whose FlatField component maps are real but
whose SkySub intentionally marks them SKY_LOST.

## Identifiable frozen sky operator

Kelson's raw-space sky model uses a wavelength/y-position B-spline design
matrix, fitted only on selected sky pixels. For a final fixed set of sky
pixels, fixed spline knots and weights, the weighted-linear least-squares
model can be represented as

    c = (X_fit.T W X_fit + ridge I)^-1 X_fit.T W S d
    r = d - X_all c = R d.

S selects fitted sky pixels from the full unrectified science image,
X_all evaluates the same sky basis everywhere, and R is the residual
operator. For ridge=0 this is the ordinary frozen weighted LS solution;
ridge is a distinct, explicitly parameterized regularization policy.

For downstream scientific functional F = u.T r, the correct derivative
with respect to the pre-sky CCD image is R.T u. This is the central
source of cross-pixel covariance after sky fitting. It is **incorrect**
to multiply each MasterBias sensitivity pixel independently by a
single fixed 'sky subtraction factor.'

## Grouped covariance

For science exposure e, its pre-SkySub independent noise has diagonal
variance V_ind,e and its native-pixel response to a common MasterBias
field b_g is s_e. After freezing sky fit for each exposure, a left
measurement u_e has native-pixel coefficients

    h_L,e = R_e.T u_L,e

Then

    C_ind = sum_e sum_pixel V_ind,e h_L,e h_R,e
    C_bias = sum_group sum_pixel V_B,g
             [sum_e_in_group s_e h_L,e] *
             [sum_e_in_group s_e h_R,e].

The shared bias is ONE random CCD image reused by each exposure within
its group. The code avoids full Npixel x Npixel covariance materialization;
its sparse Cholesky/normal-equation factorization retains dimensional cost
in the spline coefficient space, not raw CCD pixel space.

## Relation to actual Scorpipe

- `sky_sub._fit_kelson_bspline` actually fits with SciPy LSQR and
  iteratively clips sky pixels. It does NOT currently export the final
  sky-pixel indices, exact solver operator or all design/weight metadata
  needed to instantiate this diagnostic automatically.
- `frozen_kelson_bspline_operator` reuses Scorpipe's actual
  `_bspline_design_matrix`, including `y_order=0/1`; callers must
  provide the FINAL sky selection, variance and knot vector.
- The frozen least-squares inverse is a converged-model idealization,
  not an exact derivative of a finite-iteration LSQR solver or of
  clip selection. Ridge=0 does not match the old linear-spline ridge
  unless explicitly supplied.
- Wavelength solution, sky pixel selection, variance estimates, knots
  and sky geometry are frozen. All changes to robust clipping,
  self-consistent wavelength fitting, sky_scale_raw, external sky frames,
  sky-refine, flatfield estimators and MasterBias covariance between
  nominally independent groups are excluded.
- Current FlatField `VREST` remains unattributed and must **NOT**
  be passed as certified independent detector variance. MasterFlat
  uncertainty may remain correlated with the MasterBias.
- There is no invocation from production SkySub, Linearize or Stack2D.
  `VCMPQ=SKY_LOST` and `SBCOVQ=UNMODEL` stay in force.

## Validation

Tests compare direct WLS and transposed sparse adjoints with an
independently formed dense residual projector, test exact cancellation
of spline basis modes and nonlocal influence of sky pixels, same-group
vs independent-group covariance, Gaussian Monte Carlo, and finite/
complete geometry and noise contracts.

## Scientific references

- Kelson (2003), The Astrophysical Journal 588, 813,
  https://arxiv.org/abs/astro-ph/0303507
- Weighted least squares residual projection and covariance:
  https://mude.citg.tudelft.nl/book/2025/observation_theory/03_WeightedLSQ.html
