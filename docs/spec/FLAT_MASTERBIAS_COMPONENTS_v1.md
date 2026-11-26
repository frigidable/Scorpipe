# FlatField v1 MasterBias variance decomposition — pre-sky-only

## Scope and equations

The production FlatField stage can now publish a *conditional diagonal*
noise-component decomposition when an **indexed** MasterBias estimator
variance in ADU^2 is available and is known to have been included once in
the input's output VAR.

Let B be the selected MasterBias image (ADU), its estimator variance
V_B (ADU^2), F the frozen master flat response and G the gain (e-/ADU).
The flat-corrected science pixels are

    S_e = G (D_adu - B_adu) / F

and the derivative w.r.t. a shared bias pixel is

    BSENS = d S_e / d B_adu = -G/F.

This implies the native-pixel *marginal* contribution to a FlatField
science pixel's variance

    VBIAS = BSENS^2 * V_B.

Without changing the original total variance, FlatField writes:

- `SCI`, `VAR`, `MASK`: unchanged science products.
- `BVARADU`: source MasterBias estimator-variance image, ADU^2.
- `BSENS`: per-exposure detector-local bias sensitivity, e-/ADU.
- `VBIAS`: marginal bias contribution to output variance, e-^2.
- `VREST`: remaining diagonal contribution, `VAR - VBIAS`, e-^2.

FITS header `VCMPVER=1`, `VCMPQ=LOCAL` and `VCMPGID` identify
the component contract and exact MasterBias group. All four component
images are required, with finite values, valid geometry and numerical
closure `VAR = VREST + VBIAS` and
`VBIAS = BSENS^2 BVARADU` under the frozen FlatField model.
Reader-side validation is fail-closed.

**VREST must not be renamed independent detector variance.** It may include
shared MasterFlat uncertainty, and both flat and bias estimators may be
statistically dependent when the same bias calibration was used to construct
the MasterFlat. MasterFlat, gain/flat calibration uncertainty, and nonlinear
sky-model fitting are not represented by this derivative. This is not a
complete astrophysical uncertainty covariance model.

## SkySub boundary

Current SkySub fits the sky model from observed science data. It does
not yet propagate the differential sensitivity of the fitted sky model to
MasterBias. Accordingly it validates any input `VCMPVER=1` FlatField
product and deliberately drops the four component image extensions:
the sky output does **not** declare `VCMPVER=1`. Instead, it records
`VCMPQ=SKY_LOST`, preserving `SB_GID` and the original total
`VAR` without claiming a nonexistent post-sky component model.

Do not pass downstream `VAR` as an independent detector variance to the
P1.4d2 frozen-chain shared bias oracle. This PR is first-stage data
provenance, not the full Issue #117 fix.

## Remaining gates

1. Carry full group-aware derivative operators through SkySub. For a
   frozen linear sky model M=H S, propagate sensitivity using (I-H) BSENS.
   For nonlinear sky modeling, require model-specific Jacobians or
   independent injection / finite-difference validation.
2. Apply identical COUNT_V1 row plans, validity/DQ masks and geometry to
   independent detector noise and bias-response operators through Linearize.
3. Persist final actual stacking and extraction weights, and ensure shared
   MasterFlat covariance is not silently misclassified as independent.
4. Run multi-night simulation ensembles for 68/95% coverage with real
   SCORPIO bias/flat configurations and correlated sky and interpolation
   noise. Maintain `SBCOVQ=UNMODEL` until then.
