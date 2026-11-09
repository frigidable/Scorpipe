# Independent sky-reference and Stack2D pointing admission (2026-11-09)

**State:** engineering implementation, not scientific approval. The frozen
N2146sat _95 wavelength map and all preceding pilot workdirs are read-only.
Neither sky-reference diagnostic nor pointing check authorizes resuming
LSF, Sky, Linearize, Stack2D or publishing a kinematic result.

## Provenance and physical meaning

The ESO UVES atlas by Hanuschik (2003), doi:10.1051/0004-6361:20030885,
lists **measured atmospheric emission-line centroids in air**, widths and
intensities from Paranal. It is not a laboratory rest/ThAr/Ne catalogue,
and its measured line centroids must not be relabelled laboratory wavelengths.
Per-line intensities and blends vary with observatory/site/time.
ESO's original technical page describes the airglow extraction and
wavelength-scale verification purpose:
https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html

The user-provided 503-entry catalogue covering the _95 detector range may
be supplied as an external CSV to read_uves_airglow_csv(path). Scorpipe
must not assume that all 503 entries are independent reference lines at
the SCORPIO spectral resolution. SkyReference.reference_kind preserves
the distinction between uves_observed_centroid and laboratory_air.

group_blended_lines() forms proximity components using the **instrumental**
LSF FWHM, not the intrinsic UVES atlas line widths. It is a diagnostic
grouping, not a substitute for an accurately measured instrumental LSF.

fit_blended_sky_shift() evaluates a grid of common offsets for one
wavelength region in **one science exposure**. It fits a linear continuum
and non-negative, independently variable line amplitudes, without importing
UVES peak intensity ratios into the SAO fit. It requires explicit **air**
wavelengths and a separately measured instrumental FWHM. A vacuum/air
mismatch raises rather than silently converting an unknown convention.

Every fit has auto_apply_allowed=false and
science_wavelength_zero_point_approved=false. Even a good fit cannot
establish wavelength-, row- and time-dependent flexure or uniquely
deblend all weak unidentified lines. A single feature is never sufficient
to populate verified flexure in linearize. The pre-existing internal Kelson
sky self-fit correctly remains FLEXURE_UNVERIFIED.

## Source-grounded N2146sat risks

Raw SCORPIO-1 FITS s22800604.fts and s22800612.fts show different telescope
pointings: 109.03 arcsec angular separation; ROTANGLE 298.4 and
271.1 degrees. ROTANGLE is an instrument field/rotator value, not necessarily
the slit position angle on the sky. Matching OBJECT names do not establish
identical slit sky coverage.

stack_pointing_qc.check_stack_pointing compares candidate frames with the
first frame via spherical separation and circular rotator-angle difference.
run_stack2d writes stack_pointing_qc.json before any stacked image. Header
disagreement beyond default limits 5 arcsec or 1 degree causes
POINTING_QC REJECT. Config keys:
- stack2d.pointing_qc.max_pointing_sep_arcsec
- stack2d.pointing_qc.max_rotator_diff_deg
- stack2d.pointing_qc.require_complete_metadata

Missing or ambiguous coordinates are UNVERIFIED rather than a fabricated
PASS; require_complete_metadata=true makes missing values fatal.
A PASS is only *header-level* consistency, not astrometric slit registration,
verified on-sky PA, or proof of a common spatial footprint. Spatial
registration remains a separate science acceptance test.

## Acceptance before science use

1. Execute fourteen new targeted regressions plus existing Golden and
   wavesolution tests in a fresh detached worktree pinned to PR #178 HEAD.
   Verify tests for ~109-arcsec mismatch, RA wrapping, missing metadata,
   UVES observed-centroid provenance, mixed-reference/overcomplete-blend rejection, air-vacuum mismatch, and free-amplitude
   5200-A blend fitting.
2. For isolated future science data, preserve per-science ARC associations.
   Independently measure ARC shift across detector x and y; independently
   confirm at least two sky complexes with different wavelengths, including
   model/template and LSF uncertainties in the error budget.
3. Before correction, design a per-exposure SCIENCE flexure contract with
   lambda-map SHA256, raw frame SHA256, source wavelengths, spatial coverage,
   uncertainty and reviewed application status. Linearize must refuse stale
   or unverified measurements rather than fall back to a series-wide median.
4. Require verified slit WCS/spatial registration before stacking separate
   telescope pointings. A mechanical ROTANGLE and numerical y-align shift
   cannot by themselves guarantee this.

**Implementation boundary:** PR adds a provenance-safe, diagnostic-only sky
reference and a geometry admission screen. It does not automatically alter
sky subtraction, lambda_map, science wavelengths or downstream data.
