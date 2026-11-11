# COUNT_V1 actual-validity capture and frozen replay (v1)

**Status: opt-in diagnostic; all science-release flags remain false.**

Linearize applies `COUNT_V1` after combining source FITS SCI/VAR/MASK,
cosmic-ray overlays, unit conversion, invalid-variance checks, saturation
DQ and wavelength geometry. The Boolean validity array used in
`_apply_linearize_row_plan` was previously transient. It cannot be
recovered exactly from the output MASK/COV alone.

When `linearize.capture_count_v1_validity=true`, each worker now records
the **actual pre-overlap Boolean pixel-validity mask** from the numerical
loop, not a post-hoc guessed mask. After the stage finishes writing
`<stem>_skysub.fits`, it writes a versioned diagnostic JSON+NPZ:
`<stem>_count_v1_actual_validity_v1.json/.npz`. It binds complete
SHA256 of the exact selected Linearize input and output science FITS
and of the compact coefficient archive; that archive contains the
effective (potentially flexure-shifted and Y-cropped) wavelength map
and the exact validity Boolean array. Normalization records COUNT_V1
output grid, wavelength density/bin convention and exposure-time
normalization.

The sidecar is not written when capture is off (default).
The writer and the auditor enforce explicit data-size ceilings.
Ancillary write failures are recorded per exposure and do not modify
SCI/VAR/MASK or replace a valid science frame.

The independent read-only
`calib.count_v1_capture.audit_count_v1_capture` checks source, output
and NPZ hashes, archive identity and native geometry. On the
**strictly qualified electron-unit source** with no subsequent
post-rectification sky correction, it reconstructs the production
COUNT_V1 row-overlap operator using the captured effective wavelength
grid and actual-validity masks, then compares per-exposure
SCI, diagonal VAR, and wavelength coverage COV to the saved FITS.
It checks normalizations and the output `REBNORM=COUNT_V1` and
`NORMEXP` contracts, not a fabricated inverse pixel mapping.

The optional `sky_lineage_manifest` argument independently executes
the full-file Kelson snapshot and two-SkySub-FITS replay, then requires
that the **exact** Linearize source path is the linked SkySub output.
It refuses a different SkyRefine product, an unrelated copied 2D FITS,
or an old/modified SkySub exposure; it does **not** infer ancestry from
matching CCD shapes or matching object names.

## Explicit boundaries

- This verifies a **frozen numerical operation**, including the
  *observed* validity and the exact spectral bin overlaps. It does not
  differentiate the formation of masks/weights or recover upstream
  detector noise correlations. The `VAR` replay remains the diagonal
  `Σ a_i² V_i` under the existing conditional independence assumption.
  Adjacent spectral bins sharing CCD pixels have nonzero off-diagonal
  covariance absent from output VAR.
- A SkyRefine output or any other source not bound to a real Kelson
  manifest remains unqualified. Likewise, any post-Linearize residual
  sky cleanup triggers
  `HOLD_UNCAPTURED_POST_LINEARIZE_SKY_CLEANUP` rather than pretending
  the simple COUNT_V1 map generated the final SCI.
- Non-electron or unresolved gain/unit transforms return
  `HOLD_UNQUALIFIED_SOURCE_UNIT_TRANSFORM` rather than guess at
  detector variance units.
- The effective wavelength grid is the one used by the worker, which
  may incorporate a flexure shift. Its relationship to an independent
  arc/sky-line validation still requires separate source verification.
- This does **not** yet bind the same native CCD science pixels to
  the P1.4g pre-sky independent-vs-shared MasterBias fields, nor
  integrate Stack2D's normalized weights, Y registration, extracted
  TRACE/FIXED maps or calibrated wavelength transfer into a
  publication-grade end-to-end covariance. It cannot establish the
  effective extracted 1D LSF.

Successful direct replay reports
`HOLD_REPLAYED_COUNT_V1_WITHOUT_SKY_TO_1D_ADMISSION` with
`science_uncertainty_qualified=false` and
`production_variance_release_allowed=false`.
The production FITS are never altered by this diagnostic.

The synthetic regressions run the **actual**
`_process_linearize_frame` numerical kernel with optional capture,
exercise density/bin and exposure-time normalizations, check all three
output arrays, and reject SHA256 changes, re-hashed false masks,
unsupported cleanup, inconsistent SkySub source identity and resource
limit violations. They are **not** tests on original observed FITS.
