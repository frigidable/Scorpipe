# P1.2 local velocity patch integration — 2026-11-11

**SCIENCE HOLD.** Source is the observer's private Windows patch, SHA-256
`1ed70cf84b0dcc5cded0833ae3ce3c28b7c1317e956e0436a228e16b2d3a2a45`.
Local baseline `d145f171d057ce56516e0ae755a3afe9a9d80c95` and four original
Git blobs matched the uploaded patch header; all four unified diffs reconstructed
without context conflicts.

## Reviewed changes, separate from prior PR #180

- Extended `barycentric.py` opt-in utility with explicit RA units,
  UTC midpoint, proper coordinate frames, independent BTA/custom EarthLocation,
  offline builtin ephemeris/IERS-B, and computed (not applied) provenance.
- `project.barycentric_correction` only accepts OFF or None, preventing an
  unsupported automatic science-velocity correction from being silently ignored.
- Independent `SSYSOBS=TOPOCENT` header default even when a spectral axis already
  declares `SPECSYS=BARYCENT`. This does not change the coordinate values.
- Tests for direct Astropy comparison, leap/UTC/SCORPIO date parsing,
  no implicit RA units, no malformed location fallback, schema fail-closed,
  FITS header separation, and repeat/air-grid rejection.
- `CHANGELOG.md` and [local audit](../../science/P1_2_VELOCITY_CORRECTION_AUDIT_20261111.md).

The uploaded patch included **four modified tracked files**, not the author's
59 untracked local test files. The original Windows audit statistics belong
to the local dirty checkout; integration tests on this branch are an
independently authored subset. Do **not** infer new broad test coverage from
the original local test count.

The existing PR #180 code already multiplied the optical Doppler factor by
`1 + v_corr/c` (Astropy optical convention); this port preserves it and
expands metadata validation. No production code automatically applies the
correction, and none of `_01`, `_91`–`_96`, original FITS, Wave2D or LSF
are modified.

## CI gating and destination

Development child: `integration/p1-2-velocity-contract-20261111`, reviewed in
Draft PR #182. The CI workflow currently filters `pull_request` to
`branches: [main]`, so testing may temporarily target `main` for the
exact reviewed HEAD; **no merge into main is authorized**. Once CI passes,
restore the PR target to `integration/scorpipe-vnext-20261111`, then
fast-forward that integration branch only if its SHA still equals the verified
pre-port HEAD. This protects main and the observer's private local checkout.

Unresolved: complete original 59-test transfer, source FITS E2E and science
wavelength/flexure acceptance. Those remain separate gates.
