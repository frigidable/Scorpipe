# LineMeasure: conditional S/N is not a qualified detection — 2026-11-11

**Target:** [vNext integration PR #180](https://github.com/frigidable/Scorpipe/pull/180). **SCIENCE HOLD.**

## Finding

Stack2D explicitly emits `STKVARQ=COND_ONLY` for fixed-weight variance, `STKXCVQ=UNMODEL` for unresolved cross-exposure covariance and `STKYCVQ=UNMODEL` when spatial interpolation covariance is not propagated. The MasterBias lineage may also mark `SBCOVQ=UNMODEL`. These are scientific *limitations on uncertainty*. The existing LineMeasure code set `uncertainty_qualified=False` when such a flag was present, but withheld `detected=True` only if `SBGSTAT=SHARED` and `SBCOVQ=UNMODEL` together. Thus other unqualified Stack2D variance products could still report formally detected lines despite their own output warning that the S/N was conditional.

## Narrow correction

The integration change retains measured flux, S/N and errors as *diagnostics* but withholds `detected=True` and formal `upper_limit_3sigma` if any of these explicit unqualified statuses is present:

- `STKVARQ=COND_ONLY`
- `STKXCVQ=UNMODEL`
- `STKYCVQ=UNMODEL`
- `SBCOVQ=UNMODEL`

An explicitly unqualified candidate is retained as `conditional_detection_candidate`; reasons appear in `unqualified_uncertainty_causes`, and both line and stage JSON/QC expose `uncertainty_gate_triggered` / `uncertainty_detection_gate_triggered`. The existing, more specific shared-MasterBias reason and historical `shared_bias_variance_unqualified` flag remain unchanged. Any preexisting partial-coverage reason is preserved. A missing negative flag leaves `uncertainty_qualified=None`, **never `True`**, so omission of flags cannot confer scientific qualification.

Withheld detections are not passed into a Balmer decrement as formally detected lines. The stage records the number of *conditional candidate lines* separately from formally detected lines.

## What this does not establish

No propagation of robust-selection or correlated calibration noise is implemented by this change; that requires controlled Monte Carlo validation and operator/covariance propagation. It does not validate instrument LSF, wavelength zero-points, blends, residual tellurics, extinction, source aperture losses or real-science signal recovery. In particular, the `err_stat` numeric field remains a **conditional** uncertainty estimate and must not be published as a calibrated detection significance or upper limit.

The change is deliberately conservative and potentially changes downstream diagnostic counts. It should only enter the vNext integration track after exact-HEAD CI and review of any existing downstream assumptions. It must not authorize merging vNext to main or lifting SCIENCE HOLD.
