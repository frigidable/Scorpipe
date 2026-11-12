# SCORPIO-1 NGC2146sat — observational science qualification gates

**Status:** HOLD (2026-11-12). This is an evidence register and non-mutating
reduction protocol, not an assertion that the December 2024 raw data have been
read, reduced, or scientifically validated on GitHub CI.

## Evidence and physical interpretation

The SAO RAS instrument pages establish that SCORPIO-1 and SCORPIO-2 are
distinct focal reducers and have different CCD configurations. The SCORPIO-1
instrument pages are https://www.sao.ru/hq/lon/SCORPIO/main.html and
https://www.sao.ru/hq/lsfvo/devices/scorpio/param.html ; the SCORPIO-2
detector summary is
https://www.sao.ru/hq/lsfvo/devices/scorpio-2/params_eng.html .
The official SCORPIO-1 overview lists an EEV 42-40 active CCD format of
2048x2048 pixels; the exact native raw FITS dimensions depend on binning,
readout and any accompanying bias regions. Its published nominal gain modes
are **not proof of the meaning or calibration of particular night-specific
header `GAIN` values**.
These public instrument descriptions **do not certify an active/overscan
pixel map for this exact 2024 night**. A raw array of (1032, 2080), even
when `BINNING=1x2`, is insufficient by itself to declare the outer columns
overscan, the remaining region active silicon, or the absence of trimming.

Prior exact-night archived QA records 7 selected OBJ, 20 early BIAS, 3 FLAT
and 2 NEON. This cohort and the expected header values (145/0.9 for
BIAS/OBJ, 225/3.5 for FLAT/NEON, NODE A) are **night-specific audit
constraints**, not general specifications for all SCORPIO frames. The
historical scripts in closed #174 are retrievable as evidence but are **not
production-certified reductions**.

## Gate A — configuration and header-only preflight

Run only the read-only entry point:
`python scripts/qa/sc1_first_masterbias_gate.py --check <config.yaml> <external-new-report.json>`

The output location must be outside both the work tree and *all* selected
RAW frame directories. Never overwrite reports. The code verifies selection
counts and exact early20 basenames; existence and FITS header NAXIS; explicit parseable CCD binning;
same-mode FrameSignature compatibility; exact finite `RATE` and `GAIN`; NODE;
`bias_policy=strict`; `flatfield.enabled=false`;
`superneon.bias_sub=false`; and `project.wave_reference=air`. It refuses
pre-existing stage outputs. **PASS certifies only these checks**, not
binary FITS integrity, unselected candidate frames, illumination/overscan
physics, CCD data quality, or output SCI/VAR/DQ.

GitHub CI may exercise this code only on small synthetic FITS. It does
**not** have access to the observer's real FITS paths. An operator must
supply the real saved config and selected RAW files and examine the report.
Do **not** use archived `--run-superbias` or run a general CLI command
that regenerates a saved config as a substitute for this gate.

## Gate B — CCD geometry and physical overscan (HOLD)

Obtain detector ID, readout mode, binning, and raw `BIASSEC`/`DATASEC`
(and related) keywords directly from the native headers **for this night**.
Independently verify physical serial/parallel overscan from instrument
documentation or qualified instrument support, *and* test edge signals
against bias, FLAT, NEON and illuminated OBJ frames. Unverified outer
bands are just diagnostic regions, **not subtractable physical overscan**.
The archived geometry and edge-candidate QA in #174 is non-production
evidence; no automatic overscan application is authorized.

## Gate C — MasterBias and uncertainty (HOLD)

Use the audited early20 same-mode BIAS only; do not substitute late
exposures without a temporal model and observational evidence. Prior to
scientific use, compare MasterBias SCI with the independently reconstructed
input estimator, test VAR and DQ provenance, map persistent/single-frame
anomalies and row/common-mode temporal behavior. The Gaussian order
statistic factor for the median applies to independent identically
distributed samples; ADC quantization, common-mode correlations, drift
and data-dependent outlier rejection can invalidate nominal coverage.

The `bias_sigma_clip>0` plus median combination is deliberately rejected,
and the default median is regression-tested. Those implementation checks
do **not** establish correct uncertainty coverage for real observations.

## Gate D — wavelength solution, flexure and local LSF (HOLD)

Confirm whether the actual line list and line centers are expressed in
air/vacuum wavelengths and whether calibration exposures share detector
geometry and disperser/slit setup. Independently inspect identified arc
lines, rejected blends, 2-D residual structure and spatial/spectral
trends. Measure exposure-dependent zero-point/flexure where possible and
derive a wavelength- and position-dependent LSF from suitable calibration
features. A nominal instrument resolution or a scalar FWHM is not a
substitute for a qualified local LSF.

## Gate E — covariance through science products (HOLD)

Propagate detector noise, MasterBias uncertainty, sky subtraction,
resampling/linearization, stacking and extraction to the final 1-D
spectrum using the declared operators/masks. Synthetic tests of linear
functional covariance and Monte Carlo coverage are engineering
regressions; empirical injection-recovery and selected real-data
cross-checks must validate support, flux conservation, masked pixels,
cross-bin correlations and stated confidence intervals. Never describe
a diagonal variance image as proof of full spectral covariance.

## Exit decision

A gate changes from HOLD to qualified only when dated, independently
inspectable evidence exists (input observation IDs/hashes, exact commit,
configuration, diagnostics, acceptance criteria, outcomes and caveats).
Choose quantitative tolerances **before** inspecting science outputs and
justify them from the measurement goal and instrument stability; do not
invent pass/fail thresholds in the absence of data. A 7/7 GitHub CI or
successful Windows packaging run is *necessary engineering evidence*,
not release permission or evidence of science E2E.

Central #180 remains **Draft / SCIENCE HOLD**, version 8.0.5. Do not
modify original FITS, existing stage products, saved configuration or
`main` during these read-only qualifications.
