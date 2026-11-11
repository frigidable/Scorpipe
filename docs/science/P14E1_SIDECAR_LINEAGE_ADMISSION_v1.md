# P1.4e1 / P1.4d2 — auditable frozen-operator sidecar pairing (v1)

**Scope:** read-only, explicitly invoked provenance diagnostic. **SCIENCE HOLD.**
No production image or 1D spectrum is modified. FITS `COV` still denotes
coverage, *not* the covariance matrix.

## Established versus unestablished contracts

The original [PR #161](https://github.com/frigidable/Scorpipe/pull/161)
remains a separate historical pull-request record; direct Git blob
comparison confirms that its P1.4e1 grouped covariance **module, tests
and scientific specification are already present in vNext**.

P1.4e1 provides a correct *conditional mathematical interface*
for prescribed frozen operators:
- Frozen sky residual `R_e = I - X_all (X_fit^T W X_fit)^-1 X_fit^T W S`;
- native-count COUNT_V1 spectral row maps, Y registration, final
  normalized stack weights and TRACE/FIXED extraction weights;
- deterministic wavelength-dependent flux transfer;
- full pre-sky independent detector variance and *signed* MasterBias
  response per exposure with explicit common calibration groups.

For a pair of final spectral functionals, the independent and shared
terms can be evaluated matrix-free. This code does **not** automatically
construct and certify the actual historical `R_e`, input covariance,
COUNT_V1 row plans, group assignments, or final wavelength transfer
from a real Scorpipe observing run. Those are scientific inputs that
must not be guessed.

P1.4d2 records normalized final Stack2D weights, y shifts and TRACE/FIXED
extraction maps, separately. Merely having both sidecars does not
establish that the two maps derive from the same data product; the
extraction could have consumed a copied, stale or unrelated 2D stack.

## Read-only verification

`calib.frozen_covariance_admission.audit_frozen_operator_sidecar_pair`
checks that:
1. The two manifests use the same versioned schema, qualified **only**
   as frozen operators, with identical spatial/spectral shape and
   `float64` coefficients;
2. Every original science input, the Stack2D output and the extracted
   1D output still match the **original full-file SHA-256**;
3. The exact *path and SHA-256* of Extract1D's input matches the captured
   Stack2D output, not merely its wavelength grid or a coincidentally
   equal shape. Alternate copies/legacy aliases are rejected until
   their provenance is independently established;
4. Every stack sidecar has a valid hash, deterministic exposure order,
   expected shape and nonnegative finite coefficients, with normalized
   per-pixel sum on covered (row, wavelength) locations;
5. Both TRACE and FIXED coefficient maps match that grid and have no
   nonzero coefficient where the recorded Stack2D weights provide
   **no support at the exact spatial/spectral pixel**;
6. The stack manifest explicitly identifies `COUNT_V1` spectral
   resampling and finite per-exposure spatial shifts and positive SCI/VAR
   scaling; the extraction manifest identifies `input_mode=stack2d`
   and `apply_local_sky=false` (an additive data-derived correction
   cannot be silently represented by the frozen spatial maps);
7. Explicit disk/memory byte ceilings are respected; manifests cannot
   redirect sidecar paths outside their own capture directory.

A successful audit returns
`HOLD_UNBOUND_PRE_SKY_OPERATOR_LINEAGE`, together with
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false` **unconditionally**.
The missing prerequisites are reported as
`required_but_not_bound_by_this_audit`. They are not silently inferred
from HDF5/NPY/FITS dimensions or from file integrity checks.

### Explicit non-claims

- SHA-256 validates file identity only, not correctness of the original
  sky fit, detector gain/read noise, shared MasterBias uncertainty,
  wavelength convention or real SCORPIO astrometry and slit geometry.
- The sidecars are a frozen linearization. Real Huber, clipping and
  Horne profiles are data-adaptive; uncertainty of the selected weights
  and trace is **not** contained in the conditional covariance.
- The diagnostic is not linked to the output of
  `covariance_of_frozen_sky_chain_grouped_pair`; supplying exact,
  physically checked pre-sky inputs remains a separate science gate.
- MasterFlat/sky model uncertainty, spatial and spectral resampling
  correlation, effective science LSF, and observing-condition uncertainty
  are not covered by this capture-pair check.
- Synthetic file identity tests may use inert bytes named `.fits`;
  those tests are **not** observational FITS or end-to-end science tests.

## Next qualification gate

Freeze the actual production SkySub design matrix, final fit mask and
weights, wavelength row maps, all detector calibration component
variances, MasterBias grouping, and the final spectral transfer **from
one observer-reviewed reduction**. Bind their exact source hashes to
these audited P1.4d2 sidecars. Independently replay the 1D data
and verify the grouped analytic covariance against physically realistic
injection/recovery ensembles, including adaptive decisions. Only then
consider removing the variance-related science hold; effective LSF,
sky flexure, PA and source attribution require additional review.
