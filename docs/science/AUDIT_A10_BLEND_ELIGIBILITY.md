# A10 — fixed-window emission fluxes are not independent line components

**Historical source:** [PR #62](https://github.com/frigidable/Scorpipe/pull/62), adapted against the current vNext `measure_lines` API and the A02/A03/A05/uncertainty gates. **Central integration:** [PR #180](https://github.com/frigidable/Scorpipe/pull/180), SCIENCE HOLD.

## Problem and quantitative definition

LineMeasure subtracts an estimated local continuum and integrates a fixed wavelength interval around each catalogue line. Those integrals may collect flux from neighbouring components even when every individual line is highly significant.

The current integration windows (air rest wavelengths) are:
- Hα 6562.80 ±18 Å and [N II] 6548.05 ±12 Å / 6583.45 ±12 Å. Both nominal [N II] intervals overlap the Hα interval.
- [S II] 6716.44 ±12 Å and [S II] 6730.82 ±12 Å. The intervals overlap one another.

An overlap diagnoses **potential contamination from another catalogue component**, not that the neighbour is actually emitting. Whether the true LSF-resolved line profiles overlap also depends on the instrumental resolution and intrinsic kinematics. Consequently, this change does not assign or subtract a contaminating line flux.

## Explicit output contract

`measure_lines` compares requested windows with the **full DEFAULT_LINES catalogue** (plus any custom requested lines), not only with simultaneously requested outputs. The air/vacuum rest wavelength conversion and redshift used in this comparison are identical to the integration centre convention. Each row gets:

- `blend_partners`: names of potentially overlapping catalogue line windows;
- `blend_flag`, `blend_warning` and `blend_window_overlap_free`: classifications of those integration windows;
- `fit_method=continuum_subtracted_window_integral` and `flux_interpretation=BLENDED_DIAGNOSTIC_ONLY` for known overlaps or `ISOLATED_WINDOW_DIAGNOSTIC` otherwise;
- `component_flux_qualified=false` **even when no known window overlaps**. A simple window integral is not a validated LSF-aware, covariance-aware component fit. This avoids interpreting an empty catalogue overlap list as a scientific certification.
- `flux_kind=window_integral_not_deblended` for any non-partial flagged blend, extending legacy Hα/[N II] protection to [S II]. Partial-window status remains a more severe independent limitation.

The existing `detected` flag retains its meaning as a numerical/conditional **window-signal** candidate under the pre-existing statistical uncertainty gates. The mere presence of `detected=true` **must never** imply an independently measured Hα, [N II] or [S II] component. Stage QC now reports blended and overlap-free conditional window candidates separately, including when the formal-covariance gate withholds detections.

The Balmer-decrement helper refuses rows labelled with any blend or explicitly unqualified component flux, while retaining compatibility with independently fitted external input whose physical line identity and eligibility are explicitly supported.

## Required follow-on science modelling

A scientifically publishable flux ratio requires an independent multi-component continuum+line fit with a validated effective instrumental LSF and source kinematics, controlled stellar Balmer absorption, physically justified common velocity constraints and [N II] doublet ratio, and propagation of diagonal and off-diagonal fit/calibration covariance. For examples, SDSS describes joint Gaussian-plus-background fitting and imposed physically known doublet ratios:
https://www.sdss4.org/dr12/algorithms/redshifts/
and https://www.sdss2.org/dr3/algorithms/speclinefits.php .

Even a line catalogued as isolated still needs line-profile and continuum tests. This correction is only a **fail-closed data semantics safeguard**; no original SCORPIO spectrum was reprocessed or certified, and SCIENCE HOLD remains.
