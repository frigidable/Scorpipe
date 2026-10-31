# P1.4b — conditional covariance of COUNT_V1 linear functionals

**Status:** draft numerical primitive, not in the production extraction or
stacking path; not a calibration of total observational uncertainty.
Related: issue #42 and the P1.3c COUNT_V1 overlap model.

## Objective and exact restricted model

Each detector CCD pixel i contains an integrated electron count x_i with
independent marginal variance V_i. Rectification uses the fixed geometrical
overlap coefficients R_ki = overlap(input_i, output_k) / width(input_i). A
Boolean validity mask excludes unqualified detector pixels. To convert
integrated output counts into electron density or count rates, supply a
positive deterministic per-bin scale S_k (1/(D_k t) if applicable).

For any linear spectral measurement F_u = sum_k u_k S_k y_k, its coefficient
on original input pixel i is g_i = sum_k u_k S_k R_ki, so

- Var(F_u) = sum_i g_i**2 V_i
- Cov(F_u,F_v) = sum_i g_i h_i V_i.

These equations reproduce every off-diagonal spectral covariance term caused
**solely** by sharing input detector pixels. For adjacent output bins split
from the same CCD input pixel (V=16), their individual diagonal variances can
each be 4, but the covariance is also 4. Adding those bins has true VAR=16,
not 8. This problem is not corrected by multiplying independent output
variances by a universal empirical factor.

## Bounded-memory method

`covariance_of_linearized_functionals` reuses the exact row-plan storage
already built for production resampling (`starts, counts, src_idx, weights`).
Rather than constructing C_out of size Nλ × Nλ, it accumulates input
coefficients by input index using `np.bincount` and contracts against V.
Memory is linear in the number of input pixels, output bins and nonzero
overlaps. For Nλ≈90000, a single dense float64 spectral covariance would be
64.8 GB (≈60.3 GiB) and is not a feasible per-row 2D output.

The Boolean `valid_mask` must be supplied explicitly and be identical to
the one used in the production flux/VAR operator. A uint16 DQ plane may NOT
be silently interpreted as a validity mask. Caller-supplied output scaling
must encode both wavelength-density and exposure-time normalization.

## Explicit omissions

This quantity conditions on a fixed wavelength grid, fixed masks, known
marginal V_i and independent CCD input noise. It excludes upstream correlations
and uncertainty in sky-model fitting, flat/bias calibration, cosmic inpainting,
wavelength solution, spectral extraction profile/tracing, interpolation in
spatial alignment and data-dependent robust stacking weights.

It does **not** replace `VAR`, update `SPEC_TRACE`/`SPEC_FIXED`, or claim
that the `COV` FITS extension contains noise covariance: `COV` remains the
coverage plane. Phase B requires a separate integration/reduction design and
qualified provenance before the resulting uncertainties are used in science.

## Acceptance gates

- Independent matrix-overlap analytic oracle, including nonuniform and
  reversed wavelength centers.
- Seeded Monte Carlo and known closed-form split-pixel counterexamples.
- Masked pixels, invalid variance, optional per-bin density/exposure scaling,
  negative measurement weights and malformed operator index shapes.
- Nλ≈90k long-grid smoke test with bounded intermediate memory.
- GitHub Linux CI and targeted Windows test on identical commit before
  changing the scientific status.
