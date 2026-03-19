# LONG Comparison Harness v1

Scorpipe treats LONG as an operational benchmark, not as a truth oracle.

The comparison harness should compare bundles of machine-readable metrics across
at least five observer-relevant dimensions:

- wavelength solution;
- sky subtraction;
- extraction;
- response / calibration;
- science outputs.

The harness is intentionally separate from reduction stages so that benchmark
logic does not contaminate the core pipeline implementation.


## Implemented P0 deliverables

The public repository now ships:

- `src/scorpio_pipe/benchmark/` with the comparison engine and markdown/json report writers;
- `scorpio-pipe long-compare` CLI;
- `benchmarks/long_parity/` with a registry, case manifest schema, and a synthetic fixture;
- `.github/workflows/long_parity.yml` as the first release gate for the harness itself.

The synthetic fixture is intentionally **not** a scientific parity claim.
It exists to keep the machinery stable until the private real-night corpus is filled in.


## P5 additions

P5 extends the harness in two practical directions:

- `scorpio-pipe export-observer-bundle` writes a canonical observer bundle from an actual Scorpipe run so weekly CI can compare wave/sky/extract metrics without scraping ad-hoc JSON files.
- `scorpio-pipe certify-long-parity` builds a release-certification package (JSON/markdown/html) from the registry, available case manifests, golden time-budget evidence, known limitations, and observer sign-off.

The public `golden_mini_night.case.json` is an **operational baseline**, not a claim that the public synthetic night reaches scientific LONG parity.
