# Operational failure taxonomy

This document defines the observer-facing failure classes introduced in P0 for LONG parity work.

The point is not to replace detailed stage flags. The point is to give the pipeline a **stable operational vocabulary** so that:

- QC reports are intelligible at a glance;
- benchmark reports can group failures by mechanism;
- recovery runbooks can say what to try next.

## Canonical classes

| Code | Domain | Meaning | Typical next action |
|---|---|---|---|
| `WAVE_GEOMETRY_WEAK` | wavesolution | 2D wavelength geometry is weak, edge-constrained poorly, or spatial smoothness is suspicious | inspect geometry priors, rejected lines, λ residual atlas |
| `WAVE_LINESET_POOR` | wavesolution | arc line set is too sparse / ambiguous / mismatched to the setup | switch to per-grism preset or review line IDs |
| `SKY_WINDOWS_CONTAMINATED` | sky | sky stripes are contaminated by object flux, cosmics, or poor coverage | redefine sky windows and inspect upper/lower stripe plots |
| `SKY_MODEL_UNSTABLE` | sky | the sky model or flexure fit is numerically unstable | inspect skyline residuals, flexure diagnostics, and template compatibility |
| `SKY_RESIDUALS_STRONG` | sky | post-subtraction skyline structure remains scientifically important | enable/adjust residual-cleanup logic before trusting extraction |
| `SLIT_METADATA_MISMATCH` | calibration | slit metadata and calibrations disagree with the data | split datasets and verify slit position / rotation / readout consistency |
| `EXTRACT_TRACE_UNCERTAIN` | extraction | trace or aperture is too uncertain for a clean 1D product | compare fixed vs optimal extraction and inspect contamination |
| `PUBLICATION_PRODUCT_DEGRADED` | publication_ready | exportable science product exists but should not be treated as science-grade | fix upstream QC issue before publication/export |

## Design rules

1. Stage-local codes remain more specific and are still the source of truth.
2. A stage flag may optionally map to one of the canonical classes.
3. QC reports should aggregate both the raw flag and the canonical class.
4. Release gates care about the canonical class because it is stable across refactors.
