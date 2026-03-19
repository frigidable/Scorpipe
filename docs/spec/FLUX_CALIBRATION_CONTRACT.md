# Scorpio Pipe — Flux Calibration Contract

**Status:** draft-but-normative for architecture P0 groundwork  
**Scope:** semantically calibrated outputs produced by future calibration-domain stages  
**Primary implementation:** `src/scorpio_pipe/contracts/calibration.py` and `src/scorpio_pipe/contracts/validators.py`

This document defines how Scorpipe declares that a product is no longer a raw or
merely reduced file, but a spectrum/image that has passed through an explicit
calibration step.

The goal is simple: a calibrated FITS product must be **machine-readable**, not a
"trust me" artifact whose meaning lives only in logs or filenames.

## 1. Covered products

This contract currently applies to the calibration-domain products introduced at
P0:

- `fluxcal_spec1d.fits`
- `fluxcal_2d.fits`

The catalog may expand this set later, but any new calibrated output must obey
this same semantic model unless a new data-model version explicitly replaces it.

## 2. Required semantic headers

A calibrated product MUST carry the following primary-header keys:

- `CALSTAT` — calibration status enum
- `FLUXUNIT` — declared flux unit for the calibrated flux domain
- `REFCTID` — reference-context id used when the calibration was built/applied
- `SENSFUNC` — identifier/reference of the applied sensitivity function
- `WAVEREF` — wavelength convention (`air` or `vacuum`)

Additionally:

- `TELLURIC` is REQUIRED whenever the calibration status is telluric-aware
  (`RELATIVE_TELLURIC`, `ABSOLUTE_APPROX`, `ABSOLUTE_TRUSTED`).
- 2D calibrated products are a separate semantic class from 1D calibrated spectra:
  they MUST NOT claim `ABSOLUTE_TRUSTED`.

These headers are not optional bookkeeping. They are part of the product
contract and are validated at stage boundaries.

## 3. Calibration status vocabulary

Allowed `CALSTAT` values in the global vocabulary:

- `RAW`
- `RELATIVE`
- `RELATIVE_TELLURIC`
- `ABSOLUTE_APPROX`
- `ABSOLUTE_TRUSTED`

Per-product legality:

- `fluxcal_spec1d.fits`: `RELATIVE`, `RELATIVE_TELLURIC`, `ABSOLUTE_APPROX`, `ABSOLUTE_TRUSTED`
- `fluxcal_2d.fits`: `RELATIVE`, `RELATIVE_TELLURIC`, `ABSOLUTE_APPROX`

### Semantic meaning

- `RAW`
  - Reserved for uncalibrated data semantics.
  - A product validated as *calibrated* MUST NOT declare `CALSTAT=RAW`.
- `RELATIVE`
  - Relative spectral-shape calibration only; no telluric correction implied.
- `RELATIVE_TELLURIC`
  - Relative shape calibration plus telluric correction.
- `ABSOLUTE_APPROX`
  - Approximate absolute scale, with known association downgrade(s).
- `ABSOLUTE_TRUSTED`
  - Trusted absolute calibration within the association policy and provenance.

The status is intended to be the compact, machine-readable summary of the
standard/science association result plus what calibration steps were actually
applied.

## 4. Unit rules

Calibrated products MUST declare **physical flux-like units**.

Examples of accepted flux-like families include units containing tokens such as:

- `erg`
- `flam`
- `Jy`
- `W/m...`
- photon-counting physical-flux forms

Examples of units that are NOT valid for calibrated products:

- `electron`
- `e-`
- `ADU`

### 4.1 `FLUXUNIT` / `BUNIT` consistency

If both `FLUXUNIT` and `BUNIT` are present in the primary header, they must
describe the same unit family/rate semantics. Silent drift like
`FLUXUNIT=erg/s/cm2/A` with `BUNIT=electron` is contract-invalid.

### 4.2 Variance unit rule

For calibrated MEF products, `VAR` must carry the variance unit implied by the
calibrated SCI unit. In practice this means the VAR unit must correspond to the
squared form of the science flux unit.

## 5. Wavelength convention

Calibrated products MUST declare the wavelength convention with `WAVEREF`:

- `air`
- `vacuum`

This is required even if the product filename already implies a calibrated
context, because reference libraries, extinction curves, and telluric resources
are themselves wavelength-convention-sensitive.

## 6. Reference linkage

`REFCTID` links the product to a concrete reference-context snapshot built from
packaged and/or user-supplied reference resources.

This makes the following machine-traceable:

- which standard table/library version was used,
- which extinction curve was assumed,
- which telluric mask/template family participated,
- whether two calibrated products are comparable under the same reference basis.

`SENSFUNC` and `TELLURIC` complement `REFCTID` by recording the applied model
artifacts, not just the broader context bundle.

## 7. Validation policy

Validation is strict by design. A calibrated file fails contract validation if,
for example:

- `CALSTAT` is missing or invalid,
- `CALSTAT=RAW`,
- `FLUXUNIT`/`BUNIT` are missing for the calibrated domain,
- the declared units are not physical flux-like,
- `REFCTID` is missing,
- `SENSFUNC` is missing,
- `TELLURIC` is missing for telluric-aware statuses,
- `WAVEREF` is missing or invalid,
- base MEF/SPEC1D structural/unit contracts are violated.

## 8. Relationship to other contracts

## 8. Calibration-systematics companion product

`calibration_systematics.fits` is part of the calibrated-output contract family.
Its primary header MUST carry at least:

- `REFCTID`
- `CALSTAT`
- `WAVEREF`
- `SYSUNIT=fraction`

Its extension set MUST include:

- `WAVE`
- `CALSYS_REL`
- `CALSYS_TELL`
- `CALSYS_ABS`
- `CALSYS_TOTAL`

where `CALSYS_TOTAL` is the quadrature combination of the component systematic terms.

## 9. Relationship to other contracts

This document extends, but does not replace:

- `docs/spec/DATA_MODEL_CONTRACT_v1.md`

That document remains the normative source for the underlying FITS structure
(MEF/SPEC1D/LAMBDA_MAP, noise provenance, mask schema, generic wavelength/unit
rules). The present document adds the **semantic calibration layer** on top.
