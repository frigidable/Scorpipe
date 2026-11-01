# P1.4d2 — opt-in capture of frozen production operators

This step extends P1.4d1 (#49) with **recording** the effective linear
coefficients actually used by Stack2D and Extract1D. This is a diagnostic
provenance capability, **not** a FITS variance model and **not** a declaration
of science qualification. Default stage behavior and science FITS products
remain unchanged unless the operator sidecars are explicitly enabled.

## Configuration

Enable separately for each stage:

```yaml
stack2d:
  capture_frozen_operators: true
  capture_max_bytes: 2147483648
extract1d:
  capture_frozen_operators: true
  capture_max_bytes: 536870912
  local_sky_correction: false
```

Both flags default to false. The byte limits are explicit safety ceilings,
not allocation targets. Stack recording uses disk-backed per-exposure float64
`.npy` files. Extract recording uses float64 TRACE/FIXED maps in a distinct
`.npz` file. The output directories are unique per capture (not overwritten).

## Stack2D replay contract

For each wavelength pixel and registered spatial row the captured array
contains `a[e,y,k] = w_final[e,y,k] / sum_e w_final[e,y,k]`.
`w_final` includes the actual fatal-mask decision, positive-variance check,
optional variance floor, and final Huber/sigma-clip factor. For robust
methods, capture deliberately replays the *last iteration reference*
`last_mu`, **not** the final stacked image; these generally differ.

The extra read pass uses streaming y chunks; it does not construct an
`(n_exposures, ny, nlambda)` in-memory cube. Disk needs
`8 * n_exposures * ny * nlambda` bytes, plus filesystem overhead. Replayed
weights are validated for nonnegativity, finite values, and normalization
on covered output pixels. Covered and uncovered pixels are distinguished;
a sum of zero signals unavailable support, not zero detector noise.

Stack `manifest.json` contains a schema version, input ordering and full
SHA256 hashes, per-exposure frozen weight file hashes, stacked FITS hash,
actual spatial shifts, input unit scales, variance-floor/robust parameters,
mask bits and the resampling-contract stamp. It records the *normalized
coefficient*, not the original inverse variance; downstream covariance
must use original input detector variances separately.

## Extract1D replay contract

The maps `E_trace[y,k]` and `E_fixed[y,k]` are effective linear spatial
coefficients. Boxcar uses a coefficient of 1 for each accepted pixel,
mean uses `1/n_good`, and optimal uses the frozen Horne
`(P/V) / sum(P**2/V)` operator with its exact profile and mask from that
extraction. The diagnostic verifies numerical reconstruction of extracted
flux against `sum_y E[y,k] SCI[y,k]` before saving. Both NumPy and Numba
optimal paths are permitted; small floating-point summation differences
are tolerated. Their full input and output FITS are SHA256-linked.

The local sky correction is currently rejected **when capture is enabled**.
Production's sky residual estimator is data-dependent and requires separate
modeling. A captured operator must not silently omit its additive term.

## Scope boundaries (scientific honesty)

The capture makes Stack2D and Extract1D *weights* available. It does not
yet serialize COUNT_V1 row-plan input mapping, raw CCD variance and validity,
or the final fluxcal transfer vector required to instantiate every
`FrozenCovarianceExposure` automatically. Stacked FITS `COV` still means
**coverage**. Nor does fixed-operator covariance propagate sky-model fit,
registration uncertainty, clipping decision uncertainty, data-derived
trace/profile uncertainty, detector inter-pixel correlations, or
calibration/LSF/wavelength uncertainty.

The SHA256 hashes are for provenance integrity, not proof that all FITS
metadata and production scientific conventions have been correctly mapped.

## Qualification gates

1. Reconstruct stacked SCI numerically from the saved final normalized
   operator and the exact shifted per-exposure SCI (including nonzero y
   shifts, variance floors, Huber and clipping).
2. Verify captured TRACE/FIXED coefficient maps independently reconstruct
   the extracted flux, with masked samples and wavelength-dependent centers.
3. Confirm enabled and disabled runs produce identical SCI/VAR/MASK/COV and
   identical extracted science spectra.
4. Test limits, incomplete files and unsupported local sky fail closed.
5. Run synthetic GitHub CI, targeted Windows regression, then controlled
   real-observation validation. The documented status remains
   `CONDITIONAL_FROZEN_OPERATOR_ONLY` until full source-to-product
   provenance and physically realistic noise simulations pass.

No historical FITS/raw data should be touched; all tests should use
isolated worktrees and disposable temporary working directories.
