# N2146sat v21: raw NEON profile and red-line identity cross-check

**Result: REVIEW_REQUIRED / SCIENCE HOLD.** Source inputs remain frozen.
Do not write Wave2D/LSF FITS, move a science wavelength zero point or
enable downstream stacking/kinematics on this evidence.

## Laboratory identity is the first unresolved science problem

The user-supplied historical Moiseev source list (lines_long(1).tab)
contains the exact record **5341.6**; the reviewed 19-line
hand_pairs.txt contains **5341.60 Å**, with a prior position
x~2047.67, along with **5330.78 Å**. This is NOT a transcription error.

But the NIST *Strong Lines of Neon* Ne I AIR reference includes
**5330.7775, 5341.0938 and 5343.2834 Å**, with two separate
transitions only ~2.19 Å apart near the historic blended
5341.60 Å position. The difference between 5341.60 and the
closest of those laboratory lines is **+0.5062 Å**. An additional
weaker Ne I 5326.3968 Å is recorded in the 1950 interferometric
neon laboratory work. NIST relative handbook intensities are
not lamp-specific line ratio calibrations.

References:
- NIST Basic Atomic Spectroscopic Data, Strong Lines of Neon:
  https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm
- NIST Atomic Spectra Database:
  https://www.nist.gov/pml/atomic-spectra-database
- Saloman & Sansonetti, 2004, reviewed Ne I line compilation:
  https://www.nist.gov/publications/wavelengths-energy-level-classifications-and-energy-levels-spectrum-neutral-neon
- Interference Measurements in the Spectra of Neon and Natural Mercury,
  JOSA 40 (1950), region 5326-5343 Å:
  https://www.wip.osapublishing.org/josa/abstract.cfm?uri=josa-40-6-339
- ULySS LSF section explains that unresolved neighboring ARC lines
  can bias instrumental-profile estimation:
  https://doi.org/10.1051/0004-6361/200811467

The original local line list may deliberately tabulate an
**effective blended centroid** rather than individual laboratory
transitions. Its intent and possible empirical line-ratio/
instrument assumptions must be established BEFORE a Wave2D
recalibration. Do not simply replace a measured blended
centroid by the 5341.0938 line. The original _95 wavelength map
itself was built with the historic line list and is therefore
not an independent standard for this comparison. The other
18 manual references also require a cross-atlas review.

## Actual independent original-ARC measurements

Read the untouched raw **NEON06** s22800606.fts and **NEON13**
s22800613.fts. SHA256 respectively:
54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac
9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05

Read-only original _95/_96 wavelength map SHA256:
37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab
Original 19-line hand_pairs SHA256:
2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623

Diagnostics on **33 spatial rows per lamp**, 21-row raw
median profiles, common fit interval **5320–5352 Å**, shared
Gaussian width and pixel shift, separate nonnegative line
amplitudes and linear continuum (soft-L1 robust optimization).
Compare the following models on the EXACT SAME wavelength pixels:
- historic_2: 5330.78 + 5341.60 (historic effective reference)
- nist_3: 5330.7775 + 5341.0938 + 5343.2834
- nist_4: above plus 5326.3968

Independent local replay using a separate original-FITS reader
and exactly the v21 pure fitting functions: 66/66 successful
fits for each hypothesis, and 66/66 for wavelength-registered
historic_2. Median across both independent original lamps
and all supported slit rows:

| Component model | Median Gaussian-equivalent FWHM Å | Median relative profile RMS |
|---|---:|---:|
| historic_2 | **5.972** | 0.03165 |
| nist_3 | **5.636** | 0.03370 |
| nist_4 | **5.632** | 0.03323 |
| historic_2 after wavelength-trace resampling | **6.004** | 0.02980 |

Difference historic_2 - nist_3: **0.335 Å** in the same
window, enough to materially alter any derived LSF.
However, nist_3 has a slightly WORSE profile RMS median
over the full slit than historic_2; selection of the
physical line list CANNOT be made by minimizing this
correlated fit-residual RMS on one night.

The two original lamps show strongly correlated normalized
residual shapes: median paired-y Pearson r **0.9855**
(p10 0.9763, p90 0.9891) for the original two-component
hypothesis. This is NOT a p-value or 33 independent
confirmations of an unlisted line because detector,
wavelength map, atomic blends, and profile model are shared.
Wavelength-trace aligning each 21-row slit strip by the
_95 map changes median fitted red FWHM by only ~0.033 Å;
it also interpolates and can broaden line profiles.

On the four legacy joint pairs, more flexible pseudo-Voigt
fits did not show repeatable nonzero positive Lorentzian
wings in the narrow common window. Independently fitted
component widths in the red doublet differ by about 0.2 Å
(median ~5.83 versus ~6.03 Å), but these also are
conditional on possibly unresolved blends.

A separate four-line original-lamp scan found similar
residual motifs in both lamps. Strong coherent residual
peaks cannot be assigned to individual weak Ne lines
without full atlas coverage, correct line ratios and
residual diagnostic controls. Do NOT publish a claimed
new emission line from these results.

## Real data scientific decision

1. **Freeze and flag historic 5341.60** as a
   review-required effective/blended red LineID; do not
   automatically alter hand_pairs, _95 Wave2D or _96 LSF.
2. Do not treat the v17-v20 red pair as two isolated
   laboratory reference lines. Its strong raw-ARC
   repeatability does not establish an unbiased LSF.
3. Before considering _97, check every one of the 19
   references against a wavelength-convention-consistent
   Ne/Ar spectral atlas, and the Moiseev list documentation
   (especially multiplets 5341/5343, 5047 and 4182/4191).
4. Evaluate 3/4-line models with the **same fit windows**,
   spatially held-out line strengths, independent nights,
   residual cross-lamp covariance and instrumental line
   formation. Never confuse more fit parameters with
   independent predictive evidence.
5. Science LSF requires independent point-source/slit
   illumination transfer, stellar/sky calibration and
   per-exposure flexure checks.

## Repository boundaries

src/scorpio_pipe/lsf_neon_multiplet_diagnostic.py is a
pure diagnostic module. It reports REVIEW_REQUIRED and
DIAGNOSTIC_ONLY, explicitly marks all science approvals
FALSE, and does not import production Wave2D/LSF stages.
The source-locked real-FITS runner in scripts/qa generates
only new JSON + CSV outside the original input directories.
No production FITS has been touched by v21.

Windows v20b checks were already 149/149 PASS and real
curvature audit 4/4. This v21 change needs a NEW
Windows regression run before it can be considered
engineering-verified. PR #178 remains DRAFT.
