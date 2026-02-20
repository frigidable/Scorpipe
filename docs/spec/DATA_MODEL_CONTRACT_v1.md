# Scorpio Pipe — Data Model Contract v1

**Status:** stable (contract-law)  
**Data model version:** v1 (`SCORPDMV=1`)  

This document defines the *versioned* on-disk schema for Scorpio Pipe products.
It is the normative reference for what the pipeline produces and what downstream
stages/tools may assume.

Implementation of the validators lives in `src/scorpio_pipe/contracts/*` and is
enforced at stage boundaries by the engine (see `src/scorpio_pipe/pipeline/engine.py`).

## 1. Terms

- **Frame** — a raw detector exposure (typically a single-extension FITS).
- **Product** — a pipeline-generated artifact (FITS or JSON).
- **MEF2D** — a 2D Multi-Extension FITS with planes: **SCI**, **VAR**, **MASK**.
- **SPEC1D** — a FITS with a primary header + spectral tables.
- **LAMBDA_MAP** — a 2D FITS image mapping detector rectified coordinates to wavelength.

## 2. Global requirements for FITS products

Every pipeline FITS product that represents calibrated science data MUST include:

### 2.1 Required provenance/version headers

- `SCORPDMV` *(int)* — data model version. Must equal **1**.
- Noise provenance (either compact or standard keywords):
  - `GAIN` or `SCORPGN`
  - `RDNOISE` or `SCORPRN`
  - `NOISRC` or `SCORPNS` (free-form string describing how VAR was built)

If these are missing, the product is **contract-invalid**.

### 2.2 Wavelength reference

If a product declares a wavelength axis (e.g. `CTYPE1=WAVE*` or equivalent linear-WCS
keywords), it MUST declare the wavelength medium:

- `WAVEREF = air | vacuum`

`WAVEREF=unknown` is treated as missing for contract purposes.

### 2.3 Units

Science products must declare physical units consistently:

- `SCI` plane: `BUNIT` must be one of the pipeline-recognized science units
  (see `src/scorpio_pipe/contracts/units.py`).
- `VAR` plane: `BUNIT` must match the expected variance unit for the given
  science unit (e.g. `ADU^2` for `ADU`).

## 3. MEF2D products (SCI/VAR/MASK)

### 3.1 Structure

A MEF2D product MUST be a FITS with the following extensions:

- `SCI`  — floating point array (typically `float32`)
- `VAR`  — floating point array (`float32`), same shape as `SCI`
- `MASK` — integer bitmask array (`uint16` or `uint32`), same shape as `SCI`

Hard rules:

- `SCI`, `VAR`, `MASK` shapes must match exactly.
- `SCI` and `VAR` must be finite everywhere (no NaN/Inf).
- `VAR` must be **non-negative** everywhere.

### 3.2 Mask schema

MEF2D products MUST include the mask schema stamp in the primary header:

- `SCORPMKV` *(int)* — mask schema version (currently **2**)
- `SCORPMB0` … `SCORPMB9` — bit names for bits 0–9

Bit meanings are defined in `src/scorpio_pipe/contracts/maskbits.py` and are
contract-stable within a given `SCORPMKV`.

## 4. LAMBDA_MAP products

`lambda_map.fits` is a 2D FITS image that maps rectified detector coordinates to
wavelength.

Minimum requirements:

- Primary HDU contains a 2D floating-point image.
- Declares wavelength unit:
  - `WAVEUNIT` *(e.g. `Angstrom`)* OR a standard WCS unit keyword (`CUNIT1`).
- Declares wavelength medium:
  - `WAVEREF = air | vacuum`

The contract validator treats missing/invalid `WAVEREF` as a hard failure.

## 5. SPEC1D products

`spec1d.fits` is written by `12_extract`.

### 5.1 Structure

- Primary HDU: header-only (no required data payload).
- HDU `SPEC_TRACE`: binary table with at least:
  - `LAMBDA` (float64)
  - `FLUX_TRACE` (float32)
  - `VAR_TRACE` (float32)
  - `MASK_TRACE` (int32 bitmask)
- HDU `SPEC_FIXED`: binary table with at least:
  - `LAMBDA` (float64)
  - `FLUX_FIXED` (float32)
  - `VAR_FIXED` (float32)
  - `MASK_FIXED` (int32 bitmask)

Additional lightweight preview image HDUs may be present.

### 5.2 Header requirements

Primary header MUST include:

- `SCORPDMV=1`
- Wavelength axis declaration (linear WCS or equivalent) + `WAVEREF`
- Flux unit declaration:
  - `BUNIT` (and `FLUXUNIT` may duplicate it)

Non-finite values in spectral vectors are not allowed; writers must sanitize by
setting flux/var to 0 and marking `NO_COVERAGE` in the corresponding mask.

## 6. Variance rules

Variance propagation rules are part of the data model.

- VAR is interpreted as *per-pixel variance* in the same pixel grid as SCI.
- Operations that scale SCI must scale VAR appropriately.

Normative implementation:

- `src/scorpio_pipe/contracts/variance.py`

## 7. Versioning & migrations

- Any change to the meaning of existing required fields is a **breaking change**.
- Breaking changes require:
  1) bumping `SCORPDMV`,
  2) adding a migration note in `docs/VERSIONING.md` + `CHANGELOG.md`,
  3) updating validators and golden/regression tests.

