# P1.4e1 — full frozen SkySub → 1D grouped covariance (synthetic only)

**Status:** opt-in mathematical diagnostic. Parent: PR #160, independently
Windows QA validated. No observational data and no production science FITS
modification.

## Why a separate cross-stage gate is necessary

P1.4d3 establishes a frozen sky-fit residual operator `R_e` and grouped
MasterBias covariance. P1.4d1 establishes the matrix-free propagation of
*independent detector-pixel noise* through fixed COUNT_V1 Linearize,
subpixel spatial registration, final Stack2D weights, Extract1D maps
and FluxCal scales. Each is separately tested, but neither by itself
proves that the shared detector-noise field and sky-fitting correlations
are correctly composed through the entire 2D-to-1D chain.

The new `calib.sky_chain_grouped_covariance.
covariance_of_frozen_sky_chain_grouped_pair` composes:

```text
pre-sky CCD noise x_e
       | subtract fitted sky R_e x_e
       | COUNT_V1 row-wise spectral-overlap operator
       | subpixel y-registration
       | frozen normalized Stack2D coefficients
       | frozen TRACE/FIXED extraction coefficients
       | deterministic FluxCal transfer
       v
final 1-D scalar spectral functional F
```

The chained post-sky backward projection of `F` is `P_e`.
The full pre-sky gradient is `g_e = R_e.T @ P_e`. For a second
measurement `G` with gradient `h_e`:

```math
Cov(F,G) = sum_e sum_i V_ind[e,i] g[e,i] h[e,i]
         + sum_group sum_i B[group,i]
              (sum_e-in-group s[e,i] g[e,i])
              (sum_e-in-group s[e,i] h[e,i]).
```

Here `s[e,i]` is the **signed exposure-dependent pre-sky response
to exactly one common MasterBias error at detector pixel i**.
`V_ind[e,i]` must describe the independent **pre-sky** noise even
at pixels discarded by later resampling: those pixels may still
affect fitted sky if they were included in the original sky mask.
Using a post-sky diagonal `VAR` or zero-filling invalid pixels
silently would undercount propagated noise.

The routine uses existing validated COUNT_V1 row plans and exact
frozen spatial registration coefficients, but never assembles a full
native CCD × CCD covariance matrix. The same MasterBias group must
have identical raw CCD geometry, otherwise the computation aborts.
Missing or extra group IDs are rejected.

## Independent validation

`tests/test_p14e1_frozen_sky_to_1d_group_covariance.py` builds
a bounded **algebraic 4×4 synthetic CCD proxy** with two exposures,
different COUNT_V1 row operators, independent noise, signed per-pixel
MasterBias sensitivity and optional shared group. Its dense
reference explicitly composes forward residual matrices by independent
weighted least-squares, then the fixed post-sky chain without using the
new routine or the sparse SkySub adjoint.

A separate 28,000-realization Monte Carlo draws fresh shared
MasterBias pixel fields per trial, fresh independent detector noise
per exposure, and fits the nonzero sky with **dense QR for each
realization**, subtracts the known injected object and compares
both variances and the cross-covariance of two final extracted
spectral measurements to the grouped analytic covariance.

The experiment also checks missing sky/lineage mappings, malformed
grouped variance, invalid pre-sky noise on a pixel rejected by a
later stage, unsupported output bins, inconsistent stack weights
and missing row plans. It is an algebraic stress test; the earlier
B-spline Kelson tests are its algorithmic prerequisites.

## Limits

All operators and source masks are **frozen**. The test does not
derive on-the-fly extraction weights, differentiate clipping decisions,
couple uncertain wavelength solutions to source flux, derive MasterFlat
correlation fields, propagate the resampled covariance into an actual
published FITS product or attest an observational dataset. Testing
correct mathematics does NOT release scientific `VAR`:
`science_uncertainty_qualified=false`, `SBCOVQ=UNMODEL` and
`VCMPQ=SKY_LOST` remain scientifically unchanged.

No real observation inputs or output files are used in these tests.
