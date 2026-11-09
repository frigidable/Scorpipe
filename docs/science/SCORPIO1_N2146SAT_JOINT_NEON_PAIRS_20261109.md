# SCORPIO-1 N2146sat original Ne joint-pair experiment, 2026-11-09

**Engineering diagnostic only.** Windows PR #178 v16 passed 108/108 tests
plus selftest. All prior _95 Wave2D and _96 LSF outputs remain frozen.

## Independent original Ne arc measurements

Inputs: untouched BTA SCORPIO-1 raw lamps s22800606.fts and
s22800613.fts and the frozen _95 air-wavelength map copied to _96.
Manual pairs from the reviewed 19-line hand_pairs.txt.

For each separate lamp, 37 median-binned profiles (21 spatial rows
per bin, 27-row step) were examined for four close laboratory-line
pairs. Two Gaussian components have common sigma and detector shift,
independent positive strengths and linear background. A robust
soft-L1 loss is used; reject truncated detector edges, missing
components, poorly fitted profiles or bounded solutions.

| Pair laboratory air Å | NEON06 FWHM Å (accepted/37) | NEON13 FWHM Å (accepted/37) | Paired median lamp shift, px |
|---|---:|---:|---:|
| 4181.88/4190.71 | 5.101 (37) | 5.127 (37) | +0.147 |
| 4333.56/4345.17 | 6.152 (32) | 6.194 (32) | +0.158 |
| 5037.75/5047.74 | 5.766 (36) | 5.761 (37) | +0.104 |
| 5330.78/5341.60 | 6.023 (37) | 6.017 (37) | +0.112 |

These medians do NOT supply measurement errors. The red
5330.78/5341.60 pair is repeatably measurable with the two-line
model in both original lamps and is a potential red-side reference
for the [N I] night-sky complex at ~5200 Å. Old _96 only fitted
nine isolated lines up to 5080.38 Å, leaving 34.32% of finite
wavelength-map pixels outside the fitted span. Do not relabel
the old map empirically validated based on a new diagnostic.

Fitted width varies across pairs (4182/4191 is much narrower than
4334/4345); small lamp-to-lamp differences do NOT establish that
all pairs have uncontaminated, identical instrumental profiles.
Model uncertainty may involve weak neighbours, non-Gaussian wings,
incorrect line identity, residual sky/background or spatial sampling.
The independent LSF audit requires alternate continuum windows,
line held-out cross-validation, variable line ratio, and residual
asymmetry before adding joint-profile widths to production fits.

## Source and acceptance

New src/scorpio_pipe/lsf_arc_pair.py exports
fit_arc_pair_diagnostic(), returning REJECT with a reason or
DIAGNOSTIC_ONLY with common Gaussian FWHM, free amplitudes,
shift and profile residual. It does not apply to Wave2D,
LSF FITS, Sky, Linearize, Stack2D or publication products.

Six synthetic tests cover width and shift recovery, independent
lamp drift, absent second line, detector-edge truncation,
ambiguous close centers, and non-adoption by the LSF production
stage. No scientific correction is authorized by a good fit.
The SCORPIO sky-line [N I] zero-point is still sensitive to
component amplitudes and effective LSF assumptions.

The complete read-only independent audit with 296
pair-by-spatial-bin records, summary JSON and two plots is
available as N2146sat_NEON_joint_pair_audit.zip in conversation.
