# SCORPIO-1 NGC2146sat (2024-12-06) — readout configuration enquiry packet

**Status:** technical-source request only. Real observation: SCIENCE HOLD.
**Purpose:** obtain exact detector/controller/FITS-packing evidence from SAO,
not to auto-authorize an overscan correction or crop.

## Source identification (exact authenticated observations)

The observer has frozen original 32-file N2146sat VPHG1200B input source
manifest SHA-256 `be6966c73b7f04aebdbca6739d5d38a279379933b2cf7c80ae567dd65f458765`.
Independent operator reports:
- P0 original saved config/32-file role and SHA inventory concordance:
  `sc1_pilot_evidence_20261115_221058_086.json`,
  SHA-256 `70097150a4eb30f9718f45e569ffe0d87001c7f98edfbccfbe40198286737309`.
- B0 original 32/32 whole-file SHA verification and primary header section
  inventory: `sc1_ccd_header_sections_20261115_231247_814.json`,
  SHA-256 `52b6ca93daf7c14e24a0a79984e240bab44deb3aec94bfc609acf10ef26eaba2`.
- B1 exact five-anchor SHA verification and two-folder technical filename
  discovery: `sc1_dinacon_provenance_20261115_233150_313.json`,
  SHA-256 `0a2d2fbe9c8f8e8a2f6a9d120da28c211912736a863b5f76e3bdfd7c6aaa4f22`.
  Screened **125427 names** in `Documents/PhD` and `Downloads`,
  **zero candidate controller configuration filenames**. This says nothing
  about a BTA acquisition computer or an arbitrary backup archive.

All five verified B1 anchors (one early BIAS, dedicated OFF04, ON07,
NEON06, FLAT14) share:
`INSTRUME=SCORPIO-1`, `DETECTOR=E2V CCD4240`,
`ORIGIN=dCon5`, `MODE=Spectra`, `NODE=A`,
`BINNING=1x2`, `NAXIS1=2080`, `NAXIS2=1032`,
`CRVAL1=0`, `CRVAL2=0`. All original 32 FITS lack detector section
keys `BIASSEC`, `DATASEC`, `DETSEC`, `CCDSEC`, `TRIMSEC`
and recognized amplifier variants; there is no physical section map
in the selected FITS headers.

**Readout cohorts**: BIAS20 and original OBJ7 (six ON+one OFF)
`RATE=145.0`, `GAIN=0.9`; FLAT3 and NEON2
`RATE=225.0`, `GAIN=3.5`. These header GAIN values have not been
independently cross-checked in units e-/ADU for these exact modes.

Five source header `DATE` samples:
BIAS01 2024-12-06T12:24:12;
OFF04 2024-12-06T15:29:27;
NEON06 2024-12-06T15:34:12;
ON07 2024-12-06T15:55:05;
FLAT14 2024-12-06T17:38:32.
`DATE-OBS=2024/06/12` is the established instrument-specific
YYYY/DD/MM acquisition format (**6 December 2024**, not 12 June).
Source anchors were independently SHA checked; no pixels were opened
by B0 or B1.

## Scientific observations already established, not to be remeasured blindly

Archived 8–9 November 2026 original-data science audit #174 on
53 RAW FITS (41 BIAS including two flagged, 7 OBJ, 3 FLAT, 2 NEON)
shows strong optical transitions at raw zero-based X15→16 and
X2063→2064 in all illuminated samples. Thus raw X[16:2064] is a
well-supported candidate **optical** 2048-column area, not an
authenticated `DATASEC`. The X[0:16] and X[2064:2080] sides
contain noise-like samples rather than fixed software pad, but whether
they are actual overscan, prescan, synthesized electronic samples, or
mixed is unknown.

39 normal BIAS have L−R side offset +0.421419…+0.548112 ADU; all 7
OBJ yield +0.641667…+0.820117 ADU, outside the observed BIAS
range. Do not simply average left/right side regions or derive science
electronic zero from a mismatch in readout history.

Y=0 can show electronic glitches. Y=1…1026 remain illuminated;
a sharp fall starts Y=1027 and FLAT Y1027…1029 retain residual
illumination. No Y trim or parallel-overscan region is certified.

## Documentary ambiguity: legacy camera vs 2024 acquisition software

- SAO Advanced Design Lab camera catalogue currently describes the
  historical SCORPIO CCD42-40 camera as **2048×2052 with DINACON-I**:
  https://www.sao.ru/hq/adlab/cameras.html
- Afanasiev & Moiseev (2012), SCORPIO Appendix H, describes an earlier
  readout controller software with configurable `-v oC oL`, `-x`,
  `-y`, `-b`, `-n`; it says raw `ORIGIN` is the **acquisition
  system identifier** and `CRVAL1/2` are detector coordinate offsets
  (not a spectral WCS):
  https://www.sao.ru/hq/lsfvo/devices/scorpio/scorp_man_2012.pdf
- Ardilanov et al. (2020), *Development of Large-Format Camera Systems
  Based on the Latest Generation Sensors for the 6-m Telescope*, says
  DINACON-5 is used on newer SAO CCD camera systems; **it does not say
  the old SCORPIO-I CCD42-40 was upgraded**:
  https://www.sao.ru/jet/conferences/pdfs/115.pdf
- Raw `ORIGIN=dCon5` makes a DINACON-5 connection **plausible**,
  but this string is **not yet independently decoded**, and a
  software provenance string need not equal actual hardware revision.
  Do NOT replace the old DINACON-I assumption with a DINACON-5 fact.
- The SAO 2021 scientific/technical report No. 348,
  **Afanasieva, Dodonov, Murzin & Ardilanov, study of CCD42-40 camera
  No. 05-002-20** (23 pp.), appears in the SAO publications catalogue.
  Its relevance to the **actual SCORPIO-1 camera** must be verified
  rather than assumed:
  https://www.sao.ru/hq/adlab/biblio.html

## Focused questions for BTA instrumentation/archive team

1. What **controller hardware model, serial number, firmware and host
   acquisition software version** produced `ORIGIN=dCon5` on
   2024-12-06 for SCORPIO-1 CCD42-40?
   Does `dCon5` identify a controller, the host data system, or
   only its FITS-header template?
2. What were the *actual* 2024 configuration values for readout ROI
   X/Y, `1×2` binning, node A, prescan/overscan/underscan and
   the number/order of extra serial and parallel samples? Which
   zero-based NumPy sections (including orientation/flip) correspond
   to physical illuminated pixels and electronic samples in the
   RAW `(1032,2080)` array?
3. Is the physical CCD42-40 array read as **2048×2048 active plus
   four vertical extra pixels**, or does the 2024 system use another
   device/clocking geometry? Are raw y1025/1026 physically illuminated,
   charge-transfer tail, or service rows?
4. Are the left X[0:16] and right X[2064:2080] regions genuine
   independently clocked electronic samples? Why could their
   relative zero differ systematically in 1200/600s OBJ vs BIAS
   of the same header RATE145/GAIN0.9?
5. Are true same-mode BIAS available for the fast lamp
   RATE225/GAIN3.5 exposures? What are the independently calibrated
   read noise, gain and recommended zero-point/BIAS/overscan method
   for these two 2024 modes?
6. Can the observing-night acquisition log, screenshot/export of
   controller preferences, `CFG/chip.cfg`, `init*.cfg`,
   `fits.hdr`, firmware or timing diagram be recovered with a
   timestamp/version linking them to **6 December 2024**?
   Is technical report No. 348 (2021) relevant to this device?

## Release contract

No technical document, filename or emailed answer automatically
certifies physical overscan. First identify the actual 2024
controller/exposure-to-header mapping with independent provenance.
Then evaluate the candidate physical section map against already
saved 53-FITS illumination and time/bias controls, with explicit
holdout tests and predeclared tolerances. Only a separately reviewed
science gate may authorize any crop, electronic zero estimate or
propagated MasterBias/VAR treatment. All current stages and science
products remain unchanged. **Gate B: HOLD; Science: HOLD.**
