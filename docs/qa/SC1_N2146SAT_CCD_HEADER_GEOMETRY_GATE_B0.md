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

## Real P0 checkpoint (15 November 2026)

The observer's existing source-consistency report is
sc1_pilot_evidence_20261115_221058_086.json, SHA-256
70097150a4eb30f9718f45e569ffe0d87001c7f98edfbccfbe40198286737309.
It confirms old input-manifest SHA-256
be6966c73b7f04aebdbca6739d5d38a279379933b2cf7c80ae567dd65f458765.
Gate B0 must compare **current RAW bytes** to this historical inventory,
rather than treating a P0 report as current pixel fidelity. Actual B0
results remain unknown until the Windows source files are inspected.

## Exact Windows observer result, 2026-11-15 — B0 complete, B physical HOLD

Observer executed the approved header-only SHA-locked #226 code on the
original Windows 2024-12-06 source cohort. Exact uploaded
`sc1_ccd_header_sections_20261115_231247_814.json` content SHA-256:
`52b6ca93daf7c14e24a0a79984e240bab44deb3aec94bfc609acf10ef26eaba2`.
The B0 report records status
`HEADER_SECTION_INVENTORY_DIAGNOSTIC_SCIENCE_HOLD`.
**32/32 current original-FITS bytes rehashed and matched** the pinned P0
manifest `be6966c73b7f04aebdbca6739d5d38a279379933b2cf7c80ae567dd65f458765`.
All have `DETECTOR=E2V CCD4240`, `NODE=A`, `BINNING=1x2`,
`NAXIS1=2080`, `NAXIS2=1032`, `BITPIX=16`,
`BSCALE=1`, `BZERO=32768`. Old 20 BIAS and seven original OBJ
(including actual dedicated OFF04) have RATE145 / GAIN0.9. The three
FLAT and two NEON have RATE225 / GAIN3.5.
**None of 32 headers declares BIASSEC, DATASEC, TRIMSEC, CCDSEC, DETSEC
or any supported amp-section key.** Hence zero parse errors/overlaps
means **zero declared sections**, not a tested valid geometry. No RAW pixel
arrays opened by the B0 script, no correction and no science qualification.

### Non-redundancy with November 8–9 archived 53-frame physical QA

The read-only research archive [#174](https://github.com/frigidable/Scorpipe/pull/174)
already examined *53* real SC1 FITS (41 BIAS, seven OBJ, three FLAT,
two NEON) and found exactly the same absence of header-section keywords.
Do **not** repeat those archived pixel studies merely because B0 saw no
sections. The distinct contribution of B0 is a **fresh whole-file SHA
reverification of the 32 selected exact-night inputs**, through the
successful P0 config/role evidence chain.

Historical original-data diagnostics, recorded in #174:
- All 12 illuminated original OBJ/FLAT/NEON frames exhibit abrupt x optical
  transitions at x15/16 and x2063/2064; flat response left neighbor
  ~+950 ADU and right ~+12,700 ADU over candidate baseline. This supports
  `raw[:,16:2064]` (2048 columns) as an **optical image-X hypothesis**,
  not a certified raw image DATASEC.
- Candidate x[0:16] and x[2064:2080] are **not constant padding**;
  per-pixel readout-noise-like spread ~3.4–3.5 ADU in slow exposures
  and ~2.2 ADU in fast. This supports real noise-like samples but not
  necessarily charge-unexposed electronic overscan.
- Normal 39 early/late BIAS L−R offset range **+0.421419 to +0.548112 ADU**;
  all seven original OBJ **+0.641667 to +0.820117 ADU**, outside
  that empirical range. This remains true over independent Y subregions
  to differing degrees and denies scientifically using the mean of
  both candidate X edges as an unvalidated OBJ electronic zero.
  Source light, history, transfer differences and readout modes are
  not disentangled; no p-values follow from these correlated summaries.
- Vertical optical response persists through y=1026; it sharply
  transitions by y=1027, with y1027..1029 possibly containing residual
  light, and y0 may be anomalous. A provisional 1024-row image hypothesis
  `raw[1:1025,16:2064]` is **not validated**; bright y1025/1026
  complicates active-versus-service-row classification.

### Exact instrument documentation: unresolved geometry lineage

SAO Laboratory of Advanced Developments identifies a SCORPIO camera
using an e2v CCD42-40 with **2048 x 2052 physical format**, controller
**DINACON-I**, whereas the detector manufacturer gives **2048 x 2048
active image pixels** and historic SCORPIO 2012/2013 user manuals show
**2068 x 2072 including overscan** (some old 1x2 controller examples use
2068 x 1046). None establishes the **actual 2024 raw file packing**.
The observed 2080 = 16+2048+16 and 1032 = 1024+8 are only
arithmetic decompositions. The controller/firmware's sample order,
region-of-interest setting, binning and readout clock pattern must be
independently established before a physical DATASEC, prescan, serial
overscan or parallel overscan is assigned.

Sources:
- https://www.sao.ru/hq/adlab/cameras.html
- https://www.teledynespaceimaging.com/en-us/Products_/Pages/CCD42-40.aspx
- https://www.sao.ru/hq/lsfvo/devices/scorpio/manuals/scorpio_manual_2013.pdf
- Original 53-frame #174 comments / reports dated November 8–9, 2026.

**Decision:** Engineering Gate B0 evidence acquisition COMPLETE;
physical Gate B remains **HOLD**. Do not crop, subtract candidate
overscan, average L/R electronic offsets, change pre-existing
MasterBias/VAR/DQ, or release data. Next independent piece of evidence
is the exact controller readout geometry/firmware and night-specific
physical response mapping, not another identical 53-frame edge scan.

## Historic DINACON readout controls: documented capability, not 2024 settings

**New documentary evidence**: the observatory's primary Dina System for
Windows user guide explains that Settings > Preferences > Camera has
two independent configurable fields, *Frame overscanned columns number*
and *Frame overscanned lines number*. These are **additional** samples
when the full CCD is read via one output node. The guide says that when
both output nodes are used simultaneously, additional overscan columns
and rows are not transferred. The December 2024 original FITS report
\`NODE=A\`, compatible with the single-node branch but **not proof** of
the actual 2024 controller program or its settings.

A separate SCORPIO manual, Afanasiev & Moiseev (2012), Appendix H,
specifies the *historic* DINACMD control syntax:
- \`-v oC oL\`: extra overscanned columns and lines (historic default 20,20);
- \`-x X1 X2\` and \`-y Y1 Y2\`: user-defined readout pixel/row ranges
  in zero-based controller notation;
- \`-b Bx By\`: pixel binning; \`-n A|B|AB\`: amplifier choice.

These values are **not** documented in the selected 2024 FITS by any
\`BIASSEC/DATASEC\` fields. The archived B0 tool also did not collect
all other header cards such as \`CRVAL1/CRVAL2\` or \`ORIGIN\`. The
historic DINACMD guide describes CRVAL1/CRVAL2 as detector X/Y offsets,
not spectral wavelength calibration; never interpret such RAW keys as
an astrophysical wavelength WCS without checking full WCS semantics.

The arithmetic hypothesis \`2080 = 2048 + 32\`,
\`1032 = (2052/2) + 6\` is **consistent** with one possible
full-readout plus extra-sample configuration. It is NOT a measurement
of \`-v 32 6\`, a claim that the older controller software ran in 2024,
or evidence of where those samples were put in the FITS image.
Use this hypothesis solely to identify the correct technical questions.

**Original technical-provenance search targets from SAO Dina System guide:**
the acquisition PC's \`CFG/chip.cfg\` (detector parameters),
\`CFG/init*.cfg\` (controller initialization),
\`CFG/fits.hdr\` (FITS template),
\`CFG/tele*.cfg\`, \`CFG/*.ldr\` (controller uploads),
\`CFG/*.ngr\` (clock timing diagrams), and any
documented DINA batch or night settings that explicitly persist
overscan/ROI/binning/node settings. A similarly named file on another
PC is not evidence of the 2024 night. Do not execute archived controller
programs, firmware or command files during an audit.

### Gate B1 technical-provenance discovery tool (read-only)

Use \`scripts/qa/sc1_dinacon_provenance_probe.py\` with:
\`--p0 <real-p0.json> --b0 <real-b0.json>\`
\`--scan-root <bounded-folder> --out <new-external-json>\`.
Repeat \`--scan-root\` for explicitly chosen local archive folders,
**never** scan the entire system drive automatically.

It requires the actual successful P0/B0 source reports and their
matching SHA-256 link, the original input-manifest SHA, and verifies
five representative unchanged original FITS whole-file hashes before
reading only their primary headers, including \`ORIGIN\`, \`FILE\`,
\`CRVAL1/2\`, \`MODE\`, \`COMMENT\` and \`HISTORY\` when present.
It searches filenames of known Dina System technical artifacts, records
their size/mtime/SHA and does NOT execute or interpret their contents.
No arbitrary \`.cfg\`/firmware is accepted as the actual 2024 setup.

A success or empty search is recorded only as
\`TECHNICAL_PROVENANCE_DISCOVERY_ONLY_SCIENCE_HOLD\`.
This program has **no pathway** to approve raw CCD image sections,
electronic bias subtraction, a source-free SKY slit mask, or science
qualification. If no contemporary controller setup can be established
from the user's local folders, an authoritative observing-night
record or instrument engineer's controller configuration is necessary.

**Original documentation references:**
- SAO Dina System user manual:
  https://www.sao.ru/hq/adlab/doc/ruk_polz_DinaSys.pdf,
  especially sections 2 and 6.3 and Camera Settings.
- SCORPIO user manual, 2012, Appendix H pp. 79–83:
  https://www.sao.ru/hq/lsfvo/devices/scorpio/scorp_man_2012.pdf
- CCD42-40 NIMO datasheet (2048×2052 physical readout,
  two selectable amplifiers):
  https://www.sao.ru/hq/adlab/doc/ccd42-40_nimo_bi.pdf

## Actual Windows Gate B1 result — negative local technical-file discovery

The observer executed the source-locked B1 script on 2026-11-15 and
attached `sc1_dinacon_provenance_20261115_233150_313.json`, SHA-256
`0a2d2fbe9c8f8e8a2f6a9d120da28c211912736a863b5f76e3bdfd7c6aaa4f22`.
Status `TECHNICAL_PROVENANCE_DISCOVERY_ONLY_SCIENCE_HOLD`.
The B1 chain fingerprints exactly match the already authenticated P0
(`70097150...`) and B0 (`52b6ca93...`) reports. All **five**
source FITS primary-header anchors (BIAS01, OFF04, ON07, NEON06,
FLAT14) were reverified byte-for-byte; no pixel samples were read.

Only two local roots, `Documents/PhD` and `Downloads`, were searched.
The tool screened **125427 filenames** and found **zero** matches to
its **predefined DINACON technical-file naming patterns**. This is a
negative finding **within a bounded filename search**; it does not
establish absence of technical logs on the original BTA controller
computer, within packaged archives, or under another naming scheme.
No engineering or science status was promoted by this finding.

**Important newly observed original header field**:
`ORIGIN='dCon5'` in all five independently checked original FITS,
with `CRVAL1=CRVAL2=0`, `MODE='Spectra'`, `NODE='A'` and
`DETECTOR='E2V CCD4240'`. The historical 2012 SCORPIO format
explicitly defines `ORIGIN` as the acquisition system identifier
and `CRVAL1/2` as detector X/Y offsets; those keys are NOT a
wavelength WCS. The designation `dCon5` might refer to a
DINACON-5-generation acquisition program or controller, but there is
**no direct source** equating this exact FITS string with the actual
2024 hardware. The SAO CCD camera catalogue's older DINACON-I label
must not be silently retained as a verified 2024 hardware identity;
neither may it be silently replaced by DINACON-5. This distinction is
a new explicitly **unresolved acquisition-provenance gate**.

A specific source request package with exact original hashes, the
competing instrument hypotheses, and technical questions is now at
[SC1_N2146SAT_2024_CONTROLLER_PROVENANCE_REQUEST.md](./SC1_N2146SAT_2024_CONTROLLER_PROVENANCE_REQUEST.md).
The SAO library also records 2021 technical report No. 348, a study
of CCD42-40 camera 05-002-20. It is a **potential lead**, not
identified as this SCORPIO detector:
https://www.sao.ru/hq/adlab/biblio.html

**Disposition:** B0 DONE as immutable-source metadata intake; B1
DONE as bounded technical-source discovery; physical Gate B STILL HOLD.
No further identical local filename searches are justified without
new information about archive roots or authoritative configuration
names. Prioritize original BTA CCD acquisition-machine configuration,
night log, 2021 camera report eligibility and instrument engineers.
