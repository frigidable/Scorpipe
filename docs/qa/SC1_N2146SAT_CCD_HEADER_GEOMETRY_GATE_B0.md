# SCORPIO-1 NGC2146sat — Gate B0 detector geometry intake

**State: SCIENCE HOLD.** This is a source-locked, header-only diagnostic,
not a physical CCD crop/overscan calibration. Stacked on pilot P0 (#225).

## Existing evidence and limits

Original December 2024 SCORPIO-1 science frames have 1032 x 2080 native
pixels and 1x2 binning. The already-existing real-data diagnostic scripts
sc1_bias_2d_epoch_validation.py, sc1_fast_flat_spatial_qa.py and
sc1_fast_flat_neon_qa.py use the candidate Python slices x=[2:14],
x=[2066:2078], and interior x=[32:2048] to examine electronic baseline
and lamp illumination. These slices are NOT a physically certified BIASSEC,
DATASEC, illumination or overscan correction.

An instrument-specific section and readout mode must be derived from
actual night files, readout documentation and physical tests. Astropy
explains the one-based inclusive FITS section conventions and additional
physical overscan qualification:
https://www.astropy.org/ccd-reduction-and-photometry-guide/v/2.0.1/notebooks/02-00-Handling-overscan-trimming-and-bias-subtraction.html
SCORPIO-1 manual:
https://www.sao.ru/hq/lsfvo/devices/scorpio/BSAO_eng.pdf

## New read-only Gate B0

Call the Python script scripts/qa/sc1_ccd_section_header_audit.py with
--pilot-report pointing to the already completed real P0 JSON and --out
pointing to a new separate JSON outside RAW, old work and isolated plan.

It verifies the saved original config, RAW inventory, observer role report,
and current isolated config/manifest against SHA-256 values from P0. It then
**rehashes all 32 current original FITS bytes** against the earlier raw
manifest. Unlike P0, the current RAW bytes really are reverified.

Only each primary FITS header is inspected: native image shape, readout
labels, BIASSEC, DATASEC, TRIMSEC, CCDSEC, DETSEC and recognized amplifier
sections. Simple FITS one-based inclusive rectangles are parsed into
Python zero-based half-open bounds, with reversed axes and malformed,
out-of-bounds or overlapping declared sections recorded explicitly.
Missing sections are not invented.

Exact original roles are still 20 BIAS, seven original OBJ (including
OFF04), three FLAT and two NEON; the observer-corrected six ON/one OFF
roles remain fixed by the P0 plan. All files remain immutable. The report
records source hashes, declared sections and conflicts, readout group
labels, and explicit no-science flags.

**Success status is merely HEADER_SECTION_INVENTORY_DIAGNOSTIC_SCIENCE_HOLD.**
A BIASSEC card is a header declaration, not proof of charge-unexposed
pixels; absence of BIASSEC is not proof of absent overscan. There is no
code path authorizing a physical correction or scientific export.

## Independent Gate B exit criteria

1. Independently confirm detector, controller, readout geometry,
   amplifier arrangement, axis orientation and actual physical pixel
   sections using night/instrument-specific authoritative references.
2. Reuse original prior candidate-edge QA, then test actual light/charge
   response across held-out BIAS/FLAT/NEON/ON/OFF exposures and assess
   drift and inhomogeneous electronics. Do not rerun prior 41-BIAS maps
   just to repeat preliminary estimates.
3. Separate science-readout RATE 145/GAIN 0.9 and fast lamp RATE 225/
   GAIN 3.5; these header values are not independently certified e-/ADU.
   No same-mode fast BIAS has yet been approved.
4. Prespecify physical tolerances and systematic model before using
   detector data to validate or reject proposed overscan models.
5. A subsequent distinct physical-method review must explicitly approve
   overscan subtraction and MasterBias propagation before engine use.

No RAW FITS, old 03_bias, legacy _95/_96, vNext, main or science-stage
implementation changes in this PR. CI only tests code-level safety.
