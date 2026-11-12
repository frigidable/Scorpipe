# P1.4d9 — synthetic Monte Carlo coverage with real Kelson re-clipping

**Status:** opt-in diagnostic; no observational data, no science VAR updates.
Prerequisites: independently Windows-validated PR #155 (P1.4d8 fixed-mask
Monte Carlo) and PR #154 (P1.4d7 perturbation-sensitive clipping).

## Critical distinction from P1.4d8

P1.4d8 samples CCD independent noise plus correlated shared-MasterBias
errors, then re-fits the sky by independent dense QR with a **fixed
final training mask**. That is an appropriate mathematical control,
but says nothing about repeated data-dependent sigma clipping.

The new `calib.sky_mc_reclipping_coverage.synthetic_reclipped_sky_coverage`
creates a fully synthetic 12x24 native CCD scene with (i) exact
B-spline-generated sky truth, (ii) known nonzero object flux in rows
outside the initial sky windows, (iii) spatially varying detector
noise and MasterBias variance, and (iv) optionally a fixed injected
outlier in the sky ROI. It constructs a **baseline** conditional frozen
WLS operator and computes the P1.4d3 grouped covariance prediction
including shared MasterBias across two science exposures.

For each fresh Monte Carlo realization and each exposure, the test
samples independent CCD noise and one shared MasterBias error map per
group, then reruns the **actual production** `_fit_kelson_bspline`
(including sigma-clipping and SciPy LSQR) on the perturbed sky pixels.
The fitted sky is subtracted from all native-CCD pixels. Two
object-aperture observables are evaluated against the **known injected
object**, not re-centered to the observed mean. Thus systematic sky-fit
bias can reduce true interval coverage, even when a covariance estimate
is numerically close to empirical sampling variance.

Results include predicted independent/shared/total covariance matrices,
empirical covariance, covariance discrepancies, centered normal 68.3%
and 95% interval coverage, standardized means, LSQR stop-code failures,
per-exposure fraction of clipped pixels, and the fraction of trials where
either final fitted-pixel mask differs from its deterministic baseline.

`MC_CONSISTENT` requires no selection changes, all final LSQR stop codes
1 or 2, and conservative finite-sample covariance/coverage agreement.
`SELECTION_VARIABLE` is reported **whenever** a fitted sample changes,
even if coverage happens to remain close to nominal. This is a failure
of the globally assumed frozen-mask *derivative regime*, not proof that
the spectrum's uncertainty is biased. `NUMERIC_UNSTABLE` takes
precedence when any LSQR solution is non-nominal; `MC_MISMATCH` denotes
statistical inconsistency without a mask change.

Controls include `clip_sigma`, deterministic `contamination_sigma`,
`group_ids`, spline `y_order`, integer seed, and 200–3000 Monte Carlo
realizations. No real files are read, and no output files are written.

## Interpretation limits

The experiment freezes initial sky-window geometry, wavelength maps,
noise variances and B-spline knot support. It does NOT capture
data-derived sky geometry, cosmic-ray detection, noisy estimation of
variance/weights, dispersion-map uncertainty, correlated flat-field
estimator errors, shared sky-subtraction across exposures, CCD
resampling, object aperture selection or final published 1D inference.
It also assumes independent Gaussian detector pixels conditional on
the shared simulated MasterBias.

The Monte Carlo distribution is conditional on one specified synthetic
truth and contamination pattern; 68.3%/95% coverage are frequentist
benchmarks for these generated trials, not general observational
calibration statements. `science_uncertainty_qualified=false` is
reported **unconditionally**.

No production science `SCI/VAR/MASK`, `SBCOVQ=UNMODEL`, or
`VCMPQ=SKY_LOST` is modified.
