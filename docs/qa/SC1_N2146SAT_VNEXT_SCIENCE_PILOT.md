# Scorpipe vNext — NGC2146sat controlled science-pilot baseline

**Pilot phase P0: source-evidence reconciliation, not a science reduction.**
**Case:** SCORPIO-1, VPHG1200B, 2024-12-06. **Science status: HOLD.**

## Why this phase exists

Scorpipe already has separate tools for (i) the saved seven-OBJ/zero-SKY
configuration and its early20 science-readout bias cohort, (ii) readability
and SHA-256 of 32 native source FITS, and (iii) the observer-confirmed
six-ON/one-OFF role plan and isolated config. Those evidence records can
become stale without any RAW image changing. Green CI and correct frame
role assignments do not certify CCD, wave, LSF, sky or uncertainty physics.

This pilot reuses these existing artifacts, rather than rerunning LineID26.
It supplements, not replaces, [the original science HOLD gates](./SC1_N2146SAT_SCIENCE_HOLD_GATES.md),
[the OFF-SKY role contract](./SC1_N2146SAT_OFFSET_SKY_ROLE_CONTRACT.md)
and [observatory release requirements](../spec/OBSERVATORY_RELEASE_GATES.md).

## Read-only gate

Invoke the committed scripts/qa/sc1_pilot_evidence_gate.py with the parameters
--saved-config, --raw-inventory, --role-contract and --out, pointing to the
**existing** original config, original 32-source JSON and isolated
role_contract.json, and to a NEW external JSON output.

It checks, without reopening RAW FITS images:

- Saved original config SHA-256 equals the one recorded by the actual
  observer's role-plan; the original seven-OBJ sample includes OFF04 once.
- The original RAW report's schema/status, complete 32-source list, exact
  original role counts and ordering, source paths, file existence and sizes,
  internally reconstructed source manifest hash and non-authorizing status.
- The role plan refers to the exact same saved config, RAW inventory path
  and input manifest SHA-256.
- Six ON07–12 and one dedicated OFF04 are disjoint and exhaustive, while
  twenty BIAS, three FLAT and two NEON retain original source identities.
- The current isolated config and project manifest still preserve those
  exact explicit roles, with no changed globs or excludes.
- Config, manifest and reports do not change during this check; output is
  exclusive and outside RAW, old workspace and isolated role-plan roots.

Any inconsistency causes a nonzero exit and **no success report**.
The status EVIDENCE_CONSISTENT_SCIENCE_HOLD means *archived engineering
evidence is mutually consistent*, not physical science qualification.
This program does not rehash the 32 source FITS bytes; those hashes were
verified in earlier observer workflows, not during this reconciliation.

Outputs always set science_qualified=false,
observatory_release_approved=false, stage_execution_authorized=false,
sky_only_rows_certified=0 and science_hold=true. Synthetic CI tests
cannot turn these flags true.

## Pilot dependency gates

| Gate | Available evidence | Required independent exit evidence |
| --- | --- | --- |
| A — input roles and provenance | Original 32-file SHA inventory and real 6+1 loader verification | Freeze current source bytes, environment and independent provenance before stage execution |
| B — detector geometry | SC1 1032×2080 native shape, readout groups, edge QA | Real physical active/overscan map, ADC/gain/read-noise model and detector response |
| C — MasterBias and FLAT | Early20 science-readout BIAS, distinct fast FLAT/ARC, finite-N estimator tests | Real BIAS SCI/VAR/DQ, time/common-mode covariance, slit illumination; do not divide raw 0.9-gain science by raw 3.5-gain lamp FLAT |
| D — wavelength/LSF/flexure | Historic _95 RMS≈0.316 Å and native ARC LineID18–25 studies | Independently identified unblended calibration lines, air/vac status, held-out 2D residuals, individual flexure and local effective source LSF |
| E — object and sky | LineID26c–m, suspicious ON y896–911 and y400–511, ~109 arcsec OFF pointing | Source isolation including faint extended light, independent held-out SKY_ONLY evidence, atmospheric temporal evolution and shared OFF noise |
| F — covariance | Frozen Kelson, COUNT_V1, Stack/Extract operators and adaptive synthetic MC | Full SCI/VAR/MASK + shared calibration covariance, known-truth injections, coverage and cross-bin correlations |
| G — end-to-end benchmarks | Native Windows packaging and extensive synthetic regression | Independent original-night SC1/SC2 reductions and physically predeclared line/velocity/sky/uncertainty tolerances |

Engineering input reconciliation does NOT close physical gates B–G.
Quantitative acceptance thresholds, source masks and held-out bands must
be justified and frozen before examining their real science outcomes.

## Operational sequence

1. Reconcile existing original config, RAW report and isolated role plan.
   Freeze source-report hashes and detect altered current configs.
2. Reuse previous SC1 CCD/BIAS diagnostics; verify physical overscan and
   calibration before any science processing. Do not overwrite 03_bias.
3. Qualify source-free sky slit rows independently, or keep SkySub and
   object extraction on HOLD. Not finding slitpos among the nine previously
   inspected original spectroscopy-directory headers does NOT prove that
   no full-night slit-position image exists.
4. Obtain independent wave, flexure and source-effective LSF evidence.
5. Validate actual SCI/VAR/MASK and common covariance in isolated, known-
   truth E2E tests; never treat saved _95/_96/done.json as a release oracle.
6. Bring a second SCORPIO-2 data set, such as NGC 2784, under comparable
   reporting contracts with independently chosen instrument tolerances.

For NGC2146sat, zero SKY_ONLY CCD rows are currently science-qualified.
LineID26m y≈900 centroid variations are in historical detector-coordinate
windows, not an independently measured slit registration or DAR. Nominal
exposure ratio and local 5199 raw feature evolution are not physical sky
transfer coefficients. A shared OFF creates covariance between ONs.

## Scope

Only external evidence reconciliation, synthetic tests and documentation
change. No production MasterBias, SkySub, WaveSolution, LSF, Stack2D,
Extract1D, GUI or output qualification flags are changed. PR #180 is still
Draft SCIENCE HOLD; #223/#224 remain independent unmerged Draft PRs.
No source FITS, existing old work products, saved config or main are modified.
