# P1.4d5 — actual Kelson fit -> sky FITS numerical fidelity audit

**Status:** optional read-only diagnostic, scientific uncertainty remains
unqualified. Baseline: PR #151, which archives the actual final post-clip
Kelson LSQR fit and links the archive from sky products.

`calib.sky_fit_audit.audit_frozen_kelson_products` takes the captured
`*_kelson_frozen_fit_v1.npz`, the corresponding
`*_skymodel_raw.fits`, and `*_skysub_raw.fits`.

It verifies:

1. Exact v1 archive schema; native CCD wavelength/row geometry,
   coefficients, valid fit indices, positive fitted-pixel variance
   and numeric LSQR diagnostics.
2. Both science FITS primary headers identify the archived model:
   `SKYMD=kelson_raw`, `SKYFRZV=1`, `SKYFRZQ=FIT_ONLY`,
   and `SKYFRZP` equals the given archive basename. An old or
   accidentally mismatched sidecar cannot masquerade as the current fit.
3. Reconstructs the same spline basis as the science stage in
   **bounded row chunks** and compares against the actual
   `skymodel_raw.fits` SCI plane, at explicitly specified
   absolute/relative tolerances appropriate to float32 FITS outputs.
4. On the exact final fitted sky pixels, computes the standardized
   residual norm `||skysub[fit_indices]/sqrt(fit_variance)||_2`
   and compares with the recorded terminal LSQR `r1norm`.
5. Reports solver stop code, iterations, and condition estimate.
   Stop 1/2 are represented as nominal convergence; this does not
   establish a full-rank inverse or the nonlinear fit derivative.

The result is `status=MATCH` only after both FITS model consistency
and LSQR residual consistency pass. A failed check raises ValueError.
The audit does not change or generate FITS, NPZ, QC or science VAR.

## Limits to interpretation

This is an artifact fidelity check. It cannot establish that the sky
model is *astrophysically correct*, that its uncertainty is calibrated,
that sigma-clipping has a negligible selection effect, or that the
frozen normal-equation factorization matches finite-iteration LSQR
derivatives. `VREST` also remains potentially correlated between
exposures due to shared MasterFlat estimation.

The input FITS and NPZ are required from the same successfully captured
science execution, and native raw geometry must match. `SCI/VAR/MASK`
in production remain unchanged and `SBCOVQ=UNMODEL` is not removed.

After Windows QA, use real SCORPIO datasets to characterize the
distribution of model reconstruction errors, LSQR stopping/convergence,
condition estimates and optical-edge/wavelength-map pathologies.
This must precede using grouped covariance in production science S/N.
