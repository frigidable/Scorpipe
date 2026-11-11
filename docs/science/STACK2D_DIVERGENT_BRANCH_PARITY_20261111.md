# Stack2D and DQ branch-equivalence check — 2026-11-11

**Target:** [Scorpipe vNext PR #180](https://github.com/frigidable/Scorpipe/pull/180), still Draft / SCIENCE HOLD.

## Historic branches and currently integrated contracts

- Old [#115](https://github.com/frigidable/Scorpipe/pull/115), `fix/stack2d-fixed-weight-variance-20261103`: the current vNext accumulator computes `sum(w_i**2*VAR_i)/sum(w_i)**2` and marks `STKVRALG=W2VAR` and `STKVARQ=COND_ONLY`. The later test suite `tests/test_stack2d_weighted_variance_contract.py` is **byte-identical** to the suite on historical [#122](https://github.com/frigidable/Scorpipe/pull/122), and includes independent Monte Carlo checks. This is a **fixed-weight conditional** estimator variance, not total robust-estimator sampling variance.
- Old [#52](https://github.com/frigidable/Scorpipe/pull/52), `fix/stack2d-nonfinite-huber-weights`: current `_huber_weight_block` gates invalid Huber residuals and avoids `0 * NaN`; in contrast to #52's older accumulator behavior, current `_WeightedMeanAccumulator` intentionally raises on nonfinite or negative supplied weights. Silent acceptance of invalid weights must **not** be reintroduced. The historical end-to-end synthetic regression for a half-pixel shifted two-frame Huber coadd was absent from the current test file; it is brought in by this PR.
- Old [#111](https://github.com/frigidable/Scorpipe/pull/111), `fix/extract-stacked-rejected-provenance-20261103`: the current `_verified_stack_rejection_provenance` and versioned output mask semantics are stricter than the old branch. The full `tests/test_extract_stacked_rejection_semantics.py` suite is byte-identical to the later history on [#120](https://github.com/frigidable/Scorpipe/pull/120). An accepted coadded pixel's `REJECTED` bit records rejected contributors rather than necessarily invalidating its signal. Unsupported legacy inputs remain fail-closed.
- [#121](https://github.com/frigidable/Scorpipe/pull/121) is already represented in the current 1D schema, with an **identical** `docs/spec/DQ1D_PROVENANCE_USABILITY_v1.md`. Output `MASK_*` means fatal 1D conditions under `DQ1DVER=1`, distinct from `DQ_ANY_*` aperture-input provenance; `FRAC_*` is geometric contributor fraction and must not be interpreted as recovered flux fraction.

## Science limitations

Production `STKVARQ=COND_ONLY` is **not** evidence that uncertainty in robust weights, clipping decisions, spatial shifts, SkySub fit parameters or shared MasterBias was fully propagated. Resampled pixels can have non-zero off-diagonal covariance even if the marginal 2D VAR is correct. These uncertainties remain unqualified for science-level line-flux detections and kinematics. The new test exercises only a synthetic edge where one shifted exposure has no support, not real FITS calibration.

Only the missing historical **test** is ported; no original branch is merged or deleted, no science correction is applied to observations, and `main` is unchanged. Close #52/#111 as superseded only after exact integration-target CI demonstrates equivalent behavior; update original source issues with an explicit audit trail.
