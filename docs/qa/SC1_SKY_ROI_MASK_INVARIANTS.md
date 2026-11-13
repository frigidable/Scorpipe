# Sky ROI mask consistency — narrow engineering safety correction

**Scope:** synthetic P0 invariant only, against Scorpipe vNext; **SCIENCE HOLD**.

`sky_sub._run_sky_sub_impl` uses `geom.mask_sky_y` (not the descriptive
`geom.sky_windows` list) to select all wavelength pixels for the sky model.
This makes mismatches between the returned mask and list scientifically
consequential.

Two observed source-level inconsistencies in `sky_geometry.compute_sky_geometry`:

1. A manually supplied ROI was checked for overlap with the **undilated**
   object band, but the final object-protection mask was expanded by
   `dilation_px`. Adjacent sky windows could overlap protected object rows.
   **Resolution:** reject the entire manual SKY ROI when it intersects the
   dilated object mask. Leave no sky rows, emit `ROI_PROTECTED_MARGIN_OVERLAP`
   at ERROR severity, and mark the ROI invalid. Do not silently shorten a
   user-selected window or substitute automatic pixels.
2. Automatic object detection removed too-narrow SKY islands from the
   `sky_windows` list but left those pixels set in `mask_sky_y`; SkySub thus
   used rows that the advertised minimum-width screen had rejected.
   **Resolution:** build the final `mask_sky_y` exclusively from accepted
   inclusive windows, after the minimum-width test.

These are necessary geometric invariants only. They **do not** validate the
physical source-free nature of any candidate rows: dispersed object light,
continuum wings, background galaxies, detector throughput, acquisition
slit registration, sky-line blends, temporal transfer and calibration noise
still require independent evidence. The existing LineID26d/e/f studies
approved **zero** `SKY_ONLY` rows, and 26g–l likewise do not certify any.

No production SkySub/LSF/WaveSolution or original FITS is executed or
modified by this PR. The new unit tests exercise: adjacent invalid manual
ROI, separated valid ROI, automatic narrow sky island, and automatic
no-valid-window failure.

The engineering fix must not be represented as an astrophysical SKY_ONLY
certificate or authorization to run a science reduction. The complete
source-free mask acceptance contract remains an independent future P0 gate.
