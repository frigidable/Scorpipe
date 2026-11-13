# NGC2146sat — source-free slit evidence intake (before any SKY_ONLY claim)

**Status:** DATA INTAKE ONLY / SCIENCE HOLD. Dated 2026-11-13. This is
not a sky-subtraction recipe or a positive sky-mask certificate.

## Why an archive inventory is necessary

Previous actual original-data studies already exist. Do not rerun the
LineID26c/d/e/f fits merely to produce new filenames or new summary
statistics. Their complete preserved source exports, signed reports and
checksums must instead be retrieved as the immutable basis for an
independent sky-mask validation.

- [LineID26d original multi-band and flat audit](../science/SCORPIO1_N2146SAT_LINEID26D_CONTINUUM_FLAT_AND_POINTING_GATE.md):
  eight wavelength windows, 65 spatial blocks, **0 SKY_ONLY_APPROVED**;
  ON-target broad excess near y400–511; stronger narrow structure
  y896–911; wavelength-dependent slit/flat response at the outer edges.
  Low empirical spatial-excess is not proof of source-free sky.
- [LineID26f original OFF SKY and [N I] probe](../science/SCORPIO1_N2146SAT_LINEID26F_CONFIRMED_OFFSET_SKY_AND_BIAS_GATE.md):
  one confirmed 600s offset exposure s22800604 and six 1200s
  ON-target frames s22800607–12; OFF and ON pointing differ about
  109 arcsec; 0 compact-source excesses at strict chosen thresholds in
  provisional OFF y64–967, but **0 source-free rows certified**.
- [LineID26g temporal protocol](../qa/SC1_N2146SAT_LINEID26G_SKY_RATE_HOLDOUT_PROTOCOL.md):
  provisional ON y96–319 (LOWER_REVIEW) and y640–863
  (UPPER_REVIEW) are REVIEW regions, not SKY_ONLY; conditional 5199
  behavior has no calibrated sky-flux interpretation.
- Additional LineID26h–l actual Windows QA: embedded NPZ-to-original
  raw CCD sample fidelity, all 32 spatial blocks with relative decline,
  and spectrally localized relative contrast; **none of these
  measurements independently certifies a source-free slit row**.

The instrument's mechanical ROTANGLE is not automatically the celestial
slit PA. The OFF acquisition pointing must not be treated as the same
sky field as the science exposures. An image-level registration of
sources is not itself a calibrated mapping onto detector y.

## The new read-only intake command

Run the committed
\`scripts/qa/sc1_sky_only_evidence_archive_inventory.py\` with
\`--root <old archival parent directory>\` and
\`--out <new external QA JSON>\`. It:

- searches only paths named for LineID26c/d/e/f, and only the archived
  JSON, CSV, NPZ, TXT, PNG or ZIP containers;
- records SHA-256 and file sizes for eligible artifacts and restricted
  top-level JSON keys, CSV header or archive member names;
- never reads raw FITS, unfolds NPZ arrays, copies source files, runs a
  pipeline stage, uses a pre-existing output filename, or changes the
  original archive;
- reports missing phases or over-large un-hashed files explicitly;
- always reports \`source_free_sky_mask_approved=false\` and
  \`certified_sky_rows=0\`.

A complete metadata inventory is *not* independent validation of
the archives' contained measurements; its purpose is to identify the
exact historical evidence for subsequent reading and reconciliation.
Original reports with stale role assumptions must be preserved and
interpreted in light of the observer-confirmed OFF/ON role contract.

## Prespecified scientific review — no qualification until complete

1. **Provenance and detector coordinate basis.** Reconcile the exact
   archived LineID26d/e/f reports and original FITS hashes with the
   signed 32-RAW role manifest, coordinate origin, 1032-row untrimmed
   detector geometry and each ON exposure. Never transfer an OFF
   row coordinate as the same celestial area without slit astrometry.
2. **Conservative source exclusion.** Quarantine the known ON-target
   excess structures at y400–511 and y896–911, their physically
   justified PSF/trace wings, and unqualified detector/slit edges.
   The broader historic review envelopes y320–620 and y880–927
   are *candidate safety buffers*, not a measured PSF or final
   automatically applied mask. The review windows y96–319 and
   y640–863 remain only candidate regions.
3. **Independent evidence of source-free status.** Independently
   derive slit registration and object traces from acquisition
   images and independently selected spectral bands. Check
   wavelength-dependent extended light, compact stars/galaxies,
   continuum slope, masked detector columns and flat throughput
   without using the 5199 sky-line contrast to choose regions.
4. **Held-out validation.** Freeze the proposed y intervals and model
   choices before checking separate spectral bands and separate
   ON frames, and preserve all failed cases. Holding out detector
   rows or bands from the same correlated observations is a
   robustness experiment, not an independent night of observing.
5. **Quantitative acceptance.** Specify science-driven maximum
   residual source contribution, noise and systematics before
   examining the holdout data. Calibrated SCI/VAR/MASK, the
   appropriate science-readout MasterBias, flat and illumination
   uncertainty, uncertainty from common calibration products and
   local wave/LSF are needed. A positive 5199 feature or a flat
   spatial intensity profile alone is insufficient.
6. **Result.** For every individual source and candidate interval,
   record \`REJECTED\`, \`UNVERIFIED\`, or independently
   \`QUALIFIED\` only with source bytes, threshold definitions,
   holdout results, celestial slit mapping and dated human review.
   Until all are satisfied, return **SKY_ONLY_UNVERIFIED** and never
   authorize a production SkySub fit on the basis of the ROI.

**No current positive scientific SKY_ONLY certification exists.** A
configuration ROI, a successful code regression, or the #223 geometric
ROI consistency patch is necessary infrastructure, not evidence of sky
purity. Neither the 600s/1200s exposure ratio nor the observed 5199
ON12/ON07 contrast ratio can be used as a physical OFF→ON sky scale.
The single shared OFF template also introduces cross-exposure
covariance that must be propagated separately.

This intake PR makes no change to \`main\`, \`sky_sub.py\`,
\`sky_geometry.py\`, Wave2D, LSF or the observed science FITS.
