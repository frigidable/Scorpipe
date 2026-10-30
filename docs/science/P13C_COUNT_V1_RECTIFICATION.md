# P1.3c — COUNT_V1 rectification implementation review

Status: draft, not yet approved for science data; supersedes the *algorithmic* findings in LINEARIZE_RESAMPLING_CONTRACT_ISSUE32.md only after independent review and integration.

## Scientific model

Detector input SCI is integrated electrons per input CCD pixel and input VAR is electron^2 per pixel. Each source pixel has wavelength width d_i, determined from the lambda_map wavelength centers. Each output bin has width D_k. Overlap length L_ki gives integrated-count coefficients R_ki = L_ki/d_i. SCI_out[k] = sum(R_ki SCI_in[i]), VAR_out[k] = sum(R_ki^2 VAR_in[i]). Distinct output bins share noise when overlapping the same input: Cov_out[k,l] = sum(R_ki R_li VAR_in[i]); diagonal VAR is not the full covariance.

Coverage uses a DIFFERENT coefficient, L_ki/D_k, for valid input pixels. This separation is necessary because fractional output coverage and fractional input count transfer are different quantities.

## Units and output modes

- Legacy configuration key bunit_mode='adu_bin': output physical units e- per output wavelength bin; default. The name is retained for compatibility but output is NOT ADU.
- Legacy configuration key bunit_mode='adu_per_unit': output spectral electron density e-/Angstrom (or e-/nm for that wavelength unit).
- normalize_exptime=True: divide SCI by exposure seconds and VAR by seconds squared. FITS BUNIT appends /s. A pre-normalized rate input is currently rejected to prevent double normalization and ambiguous units.
- SCI and VAR FITS extensions label e-/..., and the variance extension retains every wavelength-density dimension in its squared BUNIT.
- Production FITS primary headers include REBNORM='COUNT_V1'. This marker separates new output from pre-fix legacy rectifications.
- Stack2D prohibits a mixed set of marker-bearing and unmarked frames, or different explicit resampling contracts. Homogeneous historical unmarked stacks remain loadable as LEGACY_UNQUALIFIED and MUST NOT be used to claim publication-grade conservation.
- Standard-star inline rectification uses the same _process_linearize_frame kernel, which stamps its physical BUNIT and REBNORM before standard_prepare writes the FITS.

## Remaining limitations and follow-ups

- Spatial and spectral noise covariance after rebinning is NOT represented in the standard diagonal VAR image. The oracle tests estimate it; an uncertainty treatment is required for integrated line flux and precise kinematics.
- Input lambda_map has center-based inferred pixel edges. Distortions/nonmonotonicities, NaN gaps, slit boundaries and wavelength-solution uncertainty require a separate physical validation gate.
- Changing the science operator invalidates previously rectified FITS, later coadds and extracted spectra. They must be regenerated; the same file name and apparent WCS do NOT imply compatible data.
- The Huber robust stack variance estimator, subpixel y-shift covariance, sky-model fitting uncertainty and local sky subtraction are outside this PR.
- Issue #34 fail-closed MEF handling should be fixed separately before scientific production.
- Performance and memory should be checked on long BTA spectra (Nlambda about 90000) after correctness tests; no real exposures in this PR.

## Acceptance tests

The scientific resampling oracles added in PR #33 become hard PASS (no xfail). Additional synthetic tests cover electron versus ADU input, integrated versus density output, 0.5 and 1 Angstrom bins, exposure normalization, reversed/nonuniform dispersion, masked samples, FITS SCI/VAR header dimensions and Stack2D input unit interpretation. GitHub CI and local Windows tests must both pass. This is the necessary software gate, not sufficient real-observation qualification.

## Standard-star calibration and sensitivity transfer

- The shared standard-star rectification facade now follows science bin/density mode and the final science spectrum's exposure normalization (Linearize or Stack2D, depending on extraction input mode).
- Extract1D preserves REBNORM and NORMEXP in 1D primary headers. Sensfunc FITS stamps INBUNIT and INREBNM from the standard's extracted 1D spectrum, and fluxcal apply rejects explicitly incompatible science units or resampling schemes.
- Legacy sensfunc FITS without these stamps remain readable for backward compatibility; they cannot be treated as scientifically qualified COUNT_V1 responses without provenance review.
- Physical flux calibration still requires independently qualified extinction, standard identification, calibration uncertainty, and consistent wavelength medium; unit consistency is necessary but insufficient.
