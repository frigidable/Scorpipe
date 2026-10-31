# P1.4d1 - matrix-free covariance of the frozen linear reduction chain

**Scope:** opt-in diagnostic mathematical primitive, independent of the production
SCI/VAR/FITS writers. Does not establish scientific/observational qualification.
Related: #42 and P1.4b/c. The subpixel-y bug fix is separately handled by #48;
this primitive does not modify that PR or production Stack2D.

## Measurement operator

Each independent exposure `e` provides CCD variance `V[e,j,i]`, a Boolean
valid-pixel mask, a COUNT_V1 overlap plan `R[e,j,k,i]`, a deterministic output
scale `S[e,j,k]`, a fixed spatial alignment `Y[e,y,j]`, and **final normalized**
stack coefficients `a[e,y,k]`. Let `E_L[y,k]` and `E_R[y,k]` be frozen spatial
extraction weights for two possibly different modes (TRACE/FIXED), `d[k]>0`
the fixed combined fluxcal transfer, and `u[k]`, `v[k]` the final spectral
measurement weights.

For any functional `u`, its coefficient on raw CCD pixel `(e,j,i)` is

```
g[e,j,i](u,E) = sum_k R[e,j,k,i] S[e,j,k] sum_y (
  Y[e,y,j] a[e,y,k] E[y,k] d[k] u[k]
).
```

If **input CCD pixel errors are independent** and every operator is frozen,

```
Cov(F_u, F_v) = sum_e sum_j sum_i V[e,j,i] g[e,j,i](u,E_L) g[e,j,i](v,E_R).
```

The implementation constructs the transpose of the shift operator as at most
two nonzero spatial links per output row, then reuses P1.4b's row-plan adjoint
for COUNT_V1. No dense covariance or full-slit spatial matrix is assembled.
Extra temporary workspace grows approximately as `O(n_lambda + n_y)` per
exposure, excluding caller-owned operator/CCD arrays.

## Mandatory input/provenance restrictions

- `stack_weights` are *final* normalized coefficients `w_e / sum(w_e)` from
  the actual frozen stack. Passing raw inverse variances, recomputing weights
  from published marginal `VAR`, or using synthetic weights as if they came
  from a real robust stack is scientifically invalid. Nonnegative weights
  must sum to one wherever the output is supported. No weights are permitted
  on geometrically unsupported y rows.
- Actual `Stack2D` also applies fatal-bit selection and potential
  variance-floor, Huber or clipping weights; users must supply coefficients
  **after** those decisions, including zero for rejected samples. This
  diagnostic cannot certify that the supplied weights reproduce production.
- Each row-plan's `valid_mask` must match the original production spectral
  rectification. The imported P1.4b helper contracts masked raw inputs at
  zero coefficient. True uncertainties of masks and detection are excluded.
- `output_scale` includes whichever COUNT_V1 density and exposure-time/rate
  factors are required by the actual Stack2D input unit. Prevent double
  normalization; input FITS `BUNIT`, `REBNORM` and `NORMEXP` must be verified.
- Fractional spatial shifts follow the corrected #48 interpolation geometry,
  but registration offsets are treated as exact. Missing/uncovered y geometry
  must have zero stack weight; interpolation across **DQ-fatal** rows must be
  rejected in production via those zero weights, not invented by this core.
- `E` is the **effective linear** aperture or optimal-extraction coefficient;
  for a frozen Horne-style profile `P` and variance `V_2D`, its coefficient is
  `(P/V_2D) / sum(P**2/V_2D)` on valid rows. TRACE and FIXED have distinct E;
  their detector noise may be correlated. If local-sky correction is enabled,
  its own (often nonlinear) estimator must be separately represented or
  the current covariance is incomplete.
- `d` includes fixed wavelength-dependent sensitivity response and any
  telluric transfer with production's clipping convention. The response-fit,
  telluric-fit, absolute flux-scale and extinction uncertainties are omitted.
- `u` may include a P1.4c line+linear-continuum functional. The production
  median-continuum estimator is nonlinear and is **not** thereby qualified.
- Results cannot be reconstructed from the extant stacked `SCI/VAR/MASK/COV`
  image alone. The FITS HDU named `COV` remains **coverage**, not covariance.

## Explicitly excluded statistical effects

Data-derived trace/optimal profile, y-shift estimation, sky-fit subtraction,
cosmic/inpainting decision, Huber/clipping decision, flat/bias calibration,
wavelength solution and LSF uncertainties, response and telluric fitting,
input CCD inter-pixel noise correlation, and empirical `eta(lambda)`
rescaling are not propagated. Neither `1/sum(w)` nor `eta(lambda)` is a general
substitute for the above contraction.

## Implemented numerical gates

- Independent dense **forward** chain reference compared with matrix-free
  **adjoint** output for positive/negative fractional y shifts, spatially
  varying output scales, different spectral overlap plans, masked CCD
  samples, nonuniform final normalized stack weights, signed functionals,
  and nontrivial flux calibration.
- Independent dense TRACE/FIXED cross-covariance and seeded Monte Carlo for
  both marginal variances and their joint covariance.
- Integer-shift source-row coverage without an unused second contributor.
- Fail-closed invalid mask/variance/scale/shift/weight/row-plan contracts.
- Smoke test at 90,000 spectral bins without a dense `N_lambda**2` array.

## Remaining qualification before science integration

The same frozen coefficients must be captured faithfully from real
`Linearize -> Stack2D -> Extract1D -> Fluxcal` runs without modifying historical
products. Check mixed valid/fatal spectral masks, nonuniform/reversed input
wavelength maps, partial CCD coverage, shape and physical units, continuum
functionals, realistic long-slit memory/time, and independent observational
noise simulations. Implement storage/versioning and provenance only after
those tests. Keep the public status `CONDITIONAL`, not `SCIENCE_QUALIFIED`.
