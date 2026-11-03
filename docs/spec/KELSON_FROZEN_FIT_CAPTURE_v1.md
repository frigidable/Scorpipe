# P1.4d4 — capturing actual post-clipping Kelson fit provenance

**Status: new optional frozen-fit input archive, NOT validated full variance
or production covariance.** Scientific baseline: PR #150, P1.4d3.

The new opt-in configuration is `sky.covariance_capture.enabled: true`
(or the backward-compatible `sky_sub` section if `sky` is absent).
The default is disabled, preserving existing production outputs and time.

For successful `kelson_raw` fits, the stage saves one atomic, versioned,
no-pickle NPZ file per science frame:
`<stem>_kelson_frozen_fit_v1.npz`. It contains:

- exact evaluated 2-D wavelength map used on that exposure;
- actual `y_norm_rows`, spline knot vector, degree, y-order and fitted
  coefficients used for sky-model evaluation;
- pixel indices of the FINAL sigma-clipping keep set in full raw-CCD geometry;
- corresponding positive variances used by the final WLS fit;
- explicit version (1) and status (`FROZEN_CONDITIONAL`).

The retained indices are mapped through the real `_fit_kelson_bspline`
internal finite-positive subset to prevent a silent error in pixel
provenance. Capture verifies selected lambda/y and solver cache values.
The primary header of current `skysub_raw.fits` links the file via
`SKYFRZP`, declares `SKYFRZV=1` and `SKYFRZQ=FIT_ONLY`.
Stage done.json records the same status/path, including fit size.

`load_frozen_kelson_fit` rejects unknown schemas and nonnumeric arrays;
it calls the P1.4d3 frozen WLS model constructor for later CONDITIONAL
covariance investigations. It does NOT establish equality to the iterative
LSQR solver's finite-tolerance Jacobian, differentiate the clip selection,
nor recover the sky fit estimator covariance unconditionally.

A failure of optional capture raises a `SKY_FROZEN_CAPTURE_FAILED` warning
without discarding otherwise valid sky-subtracted SCI/VAR/MASK. A failed
capture does not set `SKYFRZV`, and a stale legacy sidecar file cannot
be taken as current authoritative output without current-header linkage.

## Remaining scientific blockers
- Confirm frozen reconstructed coefficient map agrees with the production
  model on real SCORPIO data, including nonfinite wavelength map edges.
- Validate the exact LSQR residual, rank, convergence and conditioning.
- Carry correct native-pixel independent variance, MasterBias response
  through SkySub and forward to Linearize with group relationships.
- Propagate MasterFlat shared noise, sky estimator and clipping-selection
  effects before removing `SBCOVQ=UNMODEL` or qualifying measured S/N.
