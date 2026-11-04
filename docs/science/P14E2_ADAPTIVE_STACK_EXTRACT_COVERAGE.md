# P1.4e2 — synthetic data-adaptive Stack2D and Extract1D coverage

**Status:** separate, read-only scientific diagnostic. Based on the
independently Windows-verified PR #161 (frozen SkySub-to-1D grouped
covariance). No observational inputs or production science changes.

## Motivation

P1.4e1 composes the *fixed* operator SkySub → COUNT_V1 Linearize →
spatial registration → Stack2D → Extract1D → FluxCal. This is a
conditional linear variance model and an essential independent oracle.
However, actual `Stack2D` can estimate **Huber weights from noisy
science frames**, and actual Horne-like `Extract1D` can estimate its
spatial profile from the same noisy stacked image. These operations
are nonlinear and data-dependent: the error distribution of the
final extracted flux need not coincide with variance computed while
pretending the selected weights and spatial profile are fixed.

`calib.post_sky_adaptive_coverage.
synthetic_adaptive_stack_extract_coverage` runs an explicit four-mode
ablation on identical **synthetic noise realizations**:

| Stack2D | Extract1D | Role |
| --- | --- | --- |
| inverse variance | boxcar | analytically solvable Gaussian reference |
| inverse variance | optimal | adaptive spatial-profile effect |
| Huber | boxcar | adaptive robust-weight effect |
| Huber | optimal | combined adaptive effect |

For every realization, three independent 5×9 synthetic CCD frames
contain the same known source and fresh independent Gaussian noise.
The experiment calls the **production**
`stages.stack2d._stack_chunk_streaming` kernel in each stacking
mode, then **production**
`stages.extract1d._boxcar_extract` and
`_optimal_extract`. The optimal profile uses the same noisy
measurements and wavelength smoothing, so even independent CCD noise
can produce correlations between adjacent final wavelengths.
The Huber-combined Stack2D `REJECTED` bit is treated as
informational for a verified coadd, while all synthetic CCD pixels
are valid.

For each mode the report includes empirical per-wavelength variance,
average reported conditional `VAR`, variance ratios, known-truth
95% interval coverage, mean measured error, adjacent-wavelength
empirical covariance, and Huber reweighting frequency.
The known-truth linear inverse-variance boxcar reference has
analytically predicted variance `ny*input_var/n_exp`; a sampling-aware
gate checks only **that reference**. Other adaptive modes receive
**descriptive diagnostics**, not a universal `PASS` status.

An optional deliberately injected deterministic, fixed outlier in one
exposure is a separate contaminated synthetic experiment. Its bias
cannot be explained by adding that deterministic outlier to a
Gaussian noise variance. In this case the reference status is
`CONTAMINATED_REFERENCE`, not a spurious calibrated coverage claim.

## Limitations

This P1.4e2 ablation **starts after sky subtraction** and holds
wavelength and spatial registration at identity. It isolates adaptive
stack and extraction decisions but does **not** yet re-run full
Kelson SkySub and physical resampling in the same Monte Carlo draw.
The input per-exposure variance is fixed and assumed known;
there is no MasterBias/Flat cross-exposure correlation in this
particular experiment. Those effects were diagnosed separately in
P1.4e1, but their combination with adaptive stack/profile remains
an open integration task.

A successful linear reference in these tiny synthetic scenes DOES
NOT qualify production `VAR`, normalize published error bars or
override `SBCOVQ=UNMODEL` and `VCMPQ=SKY_LOST`. Results set
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false` unconditionally.

Readout- and instrument-specific validation is explicitly deferred.
