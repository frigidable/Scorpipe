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

1. Execute seventeen new targeted regressions plus existing Golden and
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


## Independent v11 original-science experiment: LSF is the limiting nuisance

A diagnostic re-fit of **frozen** N2146sat _95 raw OBJ04/OBJ12 and the
same untouched _95 lambda_map used 40 non-overlapping 19-row spectra
per input. The [N I] laboratory air components at 5197.902 and
5200.257 Å are only 2.355 Å apart. A common shift plus *freely fit*
line amplitudes at different **assumed fixed Gaussian FWHM values**
has the following **median** outcomes:

| Assumed FWHM Å | OBJ04 shift Å | OBJ12 shift Å | Difference Å |
|---:|---:|---:|---:|
| 3.1 | −0.188 | 0.000 | +0.188 |
| 4.5 | −0.125 | +0.063 | +0.188 |
| 5.6 | +0.250 | +0.388 | +0.138 |
| 6.5 | +0.900 | +0.975 | +0.075 |

These FWHM values are **not independently measured SCORPIO LSF** and
the extraction bins were not verified as uncontaminated sky. The exercise
tests numerical sensitivity, NOT a physical sequence of wavelength
solutions. The previous ~0.2 Å temporal drift was robust among the
previous blend models, but is **not** robust to all reasonable
unconstrained-FWHM/free-amplitude assumptions.

The old free-width [N I] fit gave effective Gaussian σ~2.4 Å
(FWHM~5.6 Å), but because two components separated by 2.355 Å are
partly unresolved, that best-fit width **cannot itself certify**
instrumental LSF. The new probe now reports
unresolved_reference_blend and
absolute_shift_deblend_identifiability instead of relying on a
high line-detection Δχ² to imply a trustworthy zero point.
The supplementary evaluate_lsf_sensitivity() runs >=3
independently justified candidate widths and reports the inferred
shift spread; any high spread or an unreliable individual fit
is UNSTABLE, but even PROVISIONAL always keeps auto_apply_allowed=false.

Follow-on science acceptance requires an LSF constrained by
isolated ARC features, sky extraction free from object emission,
at least two independent air-wavelength nightglow complexes and
per-exposure, per-row uncertainty evaluation. Reference source:
NIST N I air wavelengths in
https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds3sec5.pdf.
