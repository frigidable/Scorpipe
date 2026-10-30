# Issue #32 — Scientific audit of linearization units, flux and covariance

Date: 2026-10-31. Status: confirmed discrepancy in integrated-count convention; production code unchanged. No real observations processed.

## Current pipeline

flatfield, sky_sub and linearize._process_linearize_frame standardize detector SCI to electrons and VAR to electrons squared. Raw detector samples are integrated electrons (or ADU) per CCD pixel. Yet _build_linearize_row_plan constructs weights overlap/OUTPUT_width, and _apply_linearize_row_plan applies these directly to pixel-integrated SCI. The corresponding VAR helper squares the same weights.

In the default run_linearize bunit_mode=adu_bin, the output is multiplied by output wavelength-bin width and labeled BUNIT=ADU/bin, even though SCI/VAR have already been converted to electrons/electron-squared. An alternate density mode is labeled ADU/Angstrom (or another unit) with the same contradiction. Downstream Stack2D uses these unit strings for normalization.

## Independent physical model

Input pixel i has wavelength width d_i, integrated signal N_i electrons, independent diagonal variance V_i electron^2. Output pixel k has width D_k. Let L_ki be the intersection length of their wavelength intervals, or zero if disjoint.

R_ki = L_ki / d_i
N'_k = sum_i R_ki * N_i
Var(N'_k) = sum_i R_ki^2 * V_i
Cov(N'_k, N'_m) = sum_i R_ki * R_mi * V_i.

Assumption: piecewise constant density INSIDE each input CCD pixel. An output density is N'_k/D_k and its variance is Var(N'_k)/D_k^2. Exposure normalization, if requested, divides SCI by t and variance by t^2. For fully covered boundaries, sum_k N'_k = sum_i N_i. Valid-pixel coverage is sum_{i valid} L_ki/D_k. Missing input pixel counts contribute zero; recovery of unobserved flux would require a separate explicitly documented policy.

The Scorpipe COV extension is a coverage plane, not a noise-covariance matrix.

## Analytic counterexample

12 detector pixels, each 2 Angstrom wide, each with N=20 electrons and V=16 electron^2. Rebin onto 24 output bins, each 1 Angstrom wide, covering exactly the same range:

| Quantity | Physical count-conserving oracle | Present density operator applied to integrated counts |
| --- | ---: | ---: |
| Counts per output bin | 10 electrons | 20 electrons |
| Total counts | 240 electrons | 480 electrons |
| Diagonal variance per output bin | 4 electron^2 | 16 electron^2 |
| Covariance between neighboring output bins split from same input pixel | 4 electron^2 | not tracked |

The sum of two adjacent split-bin signals has variance 16 electron^2, not the diagonal-only sum 8 electron^2. This is a deterministic mathematical counterexample, not a measurement of SCORPIO data.

## Verification and future repair

tests/test_scientific_resampling_oracles.py provides an independent overlap-matrix oracle, nonuniform-width and mask tests, seeded Gaussian Monte Carlo covariance checks, and two STRICT xfail production-characterization tests. Expected failures signify current scientific defects, NOT passing physics. Previous white-noise statistical tests use the old density operator and therefore do not qualify the integrated-count convention.

A production fix must simultaneously establish: input integrated/density semantics; unit model and VAR units; correctly normalized flux and variance; reversed dispersion and masked-pixel policies; BUNIT/SCORPUM/NORMEXP compatibility; downstream Stack2D/Extract1D behavior; cache invalidation and backward compatibility. Only then remove strict xfails and require all production checks to pass. Independently review the scientific contract before scientific defaults change.

## References

- PypeIt sampling.Resample (integrated data versus density, conserve switch): https://pypeit.readthedocs.io/en/soar_triplespec/api/pypeit.sampling.html
- Carnall 2017, SpectRes: https://arxiv.org/abs/1705.05165
