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

Choose the explicitly read-only mode according to the *existing* workspace state:

- **Before a first build in an untouched work directory:**
  `python scripts/qa/sc1_first_masterbias_gate.py --check <config.yaml> <external-new-report.json>`
  — fails closed if any prior stage products exist.
- **For an already reduced or partially reduced observation, including a
  saved `03_bias` directory:**
  `python scripts/qa/sc1_first_masterbias_gate.py --audit-existing <config.yaml> <external-new-report.json>`
  — performs identical selection/header checks but *only lists* existing
  stage outputs. It does not authorize creating, replacing or overwriting
  any prior product and is explicitly **not** a first-build readiness PASS.

A `--check` failure solely due to prior `03_bias` is a clean-workspace
safety gate, **not** evidence of scientifically invalid raw observations.
Never delete, rename or move old calibration products to make it pass.

For Windows PowerShell 5.1, an end-to-end **fail-stop** wrapper runs both
read-only diagnostics while preserving the existing work tree:

`& scripts/qa/run_sc1_n2146sat_audit.ps1 -ConfigPath <saved-config.yaml> -ReportDir <external-qa-directory> -PythonExe <python.exe>`

The wrapper sends a Python dependency probe through standard input instead
of a quote-sensitive `python -c` argument; it checks native exit codes before
each subsequent step and only prints a final success summary if both JSON
reports exist and validate. There is no need to remove or rename `03_bias`.
The external QA directory will be created only after the Python-side
out-of-RAW/out-of-work destination checks succeed.

The output location must be outside the work tree, the **configured RAW
root** and *all* selected RAW frame directories, including nested series. Never overwrite reports. The code verifies selection
counts and exact early20 basenames; existence and FITS header NAXIS; explicit parseable CCD binning;
same-mode FrameSignature compatibility; exact finite `RATE` and `GAIN`; NODE;
`bias_policy=strict`; `flatfield.enabled=false`;
`superneon.bias_sub=false`; and `project.wave_reference=air`. Only `--check` refuses
pre-existing stage outputs. In `--audit-existing` the report records
`existing_stage_products`, `mode=audit_existing_readonly` and
`stage_execution_authorized=false`. **PASS certifies only these checks**, not
binary FITS integrity, unselected candidate frames, illumination/overscan
physics, CCD data quality, or output SCI/VAR/DQ.

GitHub CI may exercise this code only on small synthetic FITS. It does
**not** have access to the observer's real FITS paths. An operator must
supply the real saved config and selected RAW files and examine the report.
Do **not** use archived `--run-superbias` or run a general CLI command
that regenerates a saved config as a substitute for this gate.

## Gate A2 — native RAW FITS readability and content fingerprints (HOLD for real night)

After the header-only `--audit-existing` check succeeds on the observer's
saved config, run this separate **read-only** check (which likewise tolerates
prior stage products without touching them) to verify the selected primary arrays are physically
present and finite and to capture reproducible source byte fingerprints:

`python scripts/qa/sc1_raw_pixel_readability_qa.py --inspect <config.yaml> <new-external-report.json>`

The program checks all 32 exact-night selected sources independently. It
computes whole-file SHA-256 fingerprints, verifies that the declared primary
FITS data block fits inside the actual file size (using Astropy's `datLoc`
and `datSpan`), then reads unscaled native primary pixels, verifies array
shape and finite real numeric samples, rejects integer FITS BLANK sentinels
and invalid scaling metadata, and records storage-value min/max/mean.
The report records a root-independent content manifest digest, refuses
writing inside RAW/work and refuses replacing prior reports. It is designed
to keep memory bounded by processing one primary image at a time.

**Limitations:** a generated SHA-256 is not an independently trusted original
observing archive hash; it proves only reproducible bytes at the time of the
inventory. FITS CHECKSUM/DATASUM keyword validity, unselected files, physical
gain and unit scaling, electronic bias removal, detector pixel geometry,
flatfield, and all covariance remain unvalidated. Any `NaN`, `Inf`, missing
or truncated source payload is an input-readability failure, not an approved
method to replace or silently interpolate a bad science exposure.

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
