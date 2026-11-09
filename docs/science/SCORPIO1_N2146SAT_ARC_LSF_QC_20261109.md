# N2146sat SCORPIO-1 ARC LSF: 2026-11-09 science HOLD

Frozen original _95 Wave2D is unchanged. The last Windows-tested PR head
(f8033c2) had **94/94 PASS**. Subsequent LSF patch requires fresh regression
tests and its own isolated new engineering pilot.

## Independently measured on original FITS

Raw SCORPIO-1/VPHG1200B neon exposures s22800606.fts and
s22800613.fts (nominal slit 1.2 arcsec) were analyzed separately with
21-row-median Gaussian+linear-background profiles, spatial step 27;
blended known lines and unstable fit windows were rejected.
SAO quotes nominal VPHG1200B FWHM ~5.5 Å, 0.87 Å/px:
https://www.sao.ru/hq/lsfvo/devices/scorpio/grisms/grisms.html

10 manual lines yielded 661 accepted single-line fits:
- Per-line-balanced median FWHM 5.776 Å (NEON06), 5.794 Å (NEON13).
- Global bin median 5.934 / 5.955 Å; not the per-line-balanced median.
- Central slit third FWHM 5.581 / 5.618 Å.
- Outer thirds approximately 6.0–6.3 Å.
- Matched-bin median NEON13 minus NEON06 FWHM +0.0208 Å.

A separate *joint two-Gaussian* fit to closely spaced ARC-line pairs
recovers FWHM 6.029 / 6.023 Å from the 5330.78+5341.60 Å pair.
The isolated 5080.38 Å reference gives 5.957 / 6.010 Å; their linear
interpolation to NI 5199 Å yields ~6.0 Å. Joint-pair fitting is
currently external diagnostics only. The production LSF algorithm
excludes close pairs and marks red-edge extrapolation as unverified.

## Proven production-method bias

Old _estimate_sigma_px evaluated clipped positive second moments within
a ±6 px window, using the window edges themselves as sky level.
A mathematically exact FWHM=5.8 Å Gaussian at 0.87 Å/px is estimated
as just 4.082 Å (−29.6%). Archived _95 SuperNeon at central
y=516 likewise returns 3.94 Å at 3888.65, 4.33 Å at 4471.48,
4.34 Å at 4921.93, 4.33 Å at 5080.38.
Single-row vs median-profile comparisons are indicative, not matched.
The synthetic truncation bias is deterministic.

New source design:
- Robust Gaussian+linear-background local ARC profile fit in
  src/scorpio_pipe/lsf_arc_profile.py.
- Reject truncation when fitting window fails to contain profile wings.
- Do not fit manual reference lines contaminated by known close neighbours.
- Median spatial strips, use local pixel-to-Angstrom derivative.
- Window choice may reference nominal expected instrument width, but
  the prior never substitutes for a measured FWHM.
- If fewer than 3 independently modeled lines survive: persist
  lsf_profile_qc_reject.json and raise before emitting an LSF FITS.
- Manual nominal fallback is opt-in only and NOT a science verification.
- Resolve missing disperser/slit metadata from explicit config because
  intermediate SuperNeon lacks important source headers.
- Record measured line-support range, extrapolated fraction and
  science_lsf_validated=false even on engineering-success FITS.
- Synthetic, low-SNR and empty-ARC regressions added.

## Original-science NI at ARC-constrained width

Forty spatial bins each in OBJ04 and OBJ12 were fitted around
laboratory-air NI 5197.902+5200.257 Å with widths interpolated
independently from both original ARC frames.
With fixed intensity ratio 5198/5200=1.759: medians +0.238/+0.438 Å,
paired median OBJ12−OBJ04 +0.175 Å, paired difference MAD ~0.111 Å.
With component amplitudes freely varying: +0.638/+0.700 Å,
paired median change +0.100 Å, paired MAD ~0.500 Å.
This degeneracy rules out claiming an independently verified absolute
sky zero point or applying any SCIENCE wavelength correction.
The ratio is an externally imposed model hypothesis, not independently
established from these observations. The spatial sky windows still need
physical review.

ARC line FWHM must never be equated automatically with point-source
stellar LSF; slit illumination and non-Gaussian wings matter:
https://doi.org/10.1051/0004-6361/200811467

## Next gate

Fresh pinned PR HEAD in isolated worktree; targeted Windows regressions
including P3 LSF publication/kinematics, Golden mini night and
the engineering selftest. If all pass, make a new isolated _96 pilot,
never modify _95 or use old multi-stage Resume. Scientific HOLD remains
until per-exposure slit/sky-line zero-point and LSF transfer tests pass.
