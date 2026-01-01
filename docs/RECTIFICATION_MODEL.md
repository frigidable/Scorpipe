# Rectification model artifact (`rectification_model.json`)

This document describes the stable, machine-readable contract produced by **Wavelength Solution** and consumed by downstream stages (notably **Sky Subtraction** and **Linearization**).

## Why this exists

For long-slit spectroscopy, two downstream operations are extremely sensitive to the *truth* of the wavelength mapping:

- **Sky subtraction in detector geometry** (Kelson-style / sky-line aware methods) needs a consistent `λ(x,y)` map.
- **Linearization / rectification** must apply the *same* wavelength truth to SCI/VAR/MASK to avoid visually plausible but physically inconsistent products.

This artifact makes that truth explicit and versioned.

## Location

Written by stage `08_wavesol` next to `lambda_map.fits`:

- `08_wavesol/rectification_model.json`
- `08_wavesol/lambda_map.fits`

## Minimal schema (v1)

```json
{
  "model_version": "1",
  "created_utc": "2025-12-30T18:42:10+00:00",
  "pipeline_version": "v5.40.1",
  "stage": "wavesolution",

  "lambda_map": {
    "path": "lambda_map.fits",
    "sha256": "<hex>",
    "shape": [ny, nx],
    "dtype": "float32"
  },

  "wave_unit": "Angstrom",
  "wave_ref": "air",

  "source": {
    "wavesolution_1d_json": "wavesolution_1d.json",
    "wavesolution_2d_json": "wavesolution_2d.json",
    "done_json": "done.json"
  },

  "policies": {
    "var": {
      "name": "linear_combo_uncoorrelated",
      "formula": "VAR_out = Σ_k (a_k^2 * VAR_k)",
      "assumption": "input pixel errors are treated as uncorrelated after rectification"
    },
    "mask": {
      "name": "or_fatal_bits",
      "fatal_or_bits": [{"name":"NO_COVERAGE","value":1}, ...],
      "no_coverage_bit": {"name":"NO_COVERAGE","value":1},
      "schema_version": 1
    }
  }
}
```

## Consumer rules (Linearization)

1. Prefer `rectification_model.json` if present.
2. Resolve `lambda_map.path` relative to the wavesolution directory.
3. If `lambda_map.sha256` is present, verify it and fail fast on mismatch.
4. Record the hashes + policies into `10_linearize/done.json` for provenance.

## Scientific notes (why VAR/MASK policies are explicit)

- **VAR propagation** for a linear combination with weights `a_k` follows standard uncertainty propagation:
  `Var(Σ a_k x_k) = Σ a_k^2 Var(x_k)` when covariances are neglected.
- **MASK propagation** is deliberately strict: if any contributing input pixel is flagged as fatal, the output pixel inherits that bit; if no pixels contribute, set `NO_COVERAGE`.

These conventions keep rectification physically interpretable and reproducible.
