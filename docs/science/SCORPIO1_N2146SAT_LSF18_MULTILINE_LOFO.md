# N2146sat LSF v18: independent two-ARC, feature-blocked validation

**State: DIAGNOSTIC_CANDIDATE / SCIENCE HOLD.** The original _95 Wave2D
and _96 isolated-line LSF products remain frozen. v18 does not replace
the production LSF map, alter sky/linearization, or approve stellar LSF.

## Provenance

Verified original Ne lamp SHA256:
- s22800606.fts: 54a6d638d2284b9bfd1d947df02399eaf0264f998667a830e144b1c437a61dac
- s22800613.fts: 9e2c98f2d2ba1125e18f84722854099f73e435a1fd7810f0b4df84ea07ab0c05
- Frozen _95 lambda map: 37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab
- Frozen 19-line manual LineID: 2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623
- Independent original-ARC isolated line-width CSV:
  bd59115bfa256621acf2a0e7ba95ffeb50eb91d1d844671f34117a2c345796fb
- Independent original-ARC joint-pair CSV:
  85c967f8d697498c30484f987db543d882f9378fc4a2dc5560bcb3d836cad7ac

The source-specific read-only QA entrypoint at
scripts/qa/lsf18_multiline_probe.py validates all six digests and
laboratory membership in the reviewed LineID list. It reuses
**precomputed** original-lamp profile fits and does not independently
re-fit raw FITS pixels. Source SHA proof is not line-fit proof.

## Independent group, not pixel, as statistical unit

One feature means a SINGLE isolated Ne laboratory line or a PAIR
jointly fit with common Gaussian width and independent amplitudes.
Duplicate physical laboratory members across features are forbidden.
Both lamp exposures at the same spatial sample are first paired
into ONE datum. Require at least 12 common bins, inter-lamp
median |FWHM difference| <=0.20 Å and p90 <=0.40 Å.

The diagnostic FWHM(lambda,y) is low-order: constant, lambda-linear,
y-linear, y-quadratic and a lambda-by-y interaction. Features get
equal total robust IRLS weight. Leave-one-feature-out (LOFO)
removes all bins and BOTH lamp repeats of the held-out laboratory
feature. Engineering gates: held-out median abs <=0.50 Å and
p90 abs <=0.95 Å. These exploratory thresholds are **not
confidence intervals or a certified LSF error budget**. They
were developed using the same N2146sat audit and require
independent-night external validation.

## Actual source audit

14 nonoverlapping laboratory groups: 13 with adequate paired
repeat measurements, 10 passing both lamp repeatability and LOFO.
Rejected features: 4181.88+4190.71 Å (LOFO median abs 0.783 Å);
4713.00 Å (1.180 Å); 4921.93 Å (0.536 Å);
5221.80 Å (only 3 common spatial bins). Do not delete any
such lines from the separately validated wavelength solution.

The red Ne 5330.78+5341.60 Å pair is accepted *for a candidate*:
ARC median abs FWHM difference ~0.006 Å; arc p90 ~0.047 Å;
LOFO median abs ~0.407 Å, p90 ~0.831 Å.
Equal-feature root-mean-square of accepted median-abs LOFO
errors is ~0.308 Å. This is an exploratory predictive
metric, not an absolute FWHM uncertainty.

Candidate line-center range: **3888.65–5336.19 Å**; maximum
internal accepted-line gap ~392.48 Å. Common supported y rows:
**74–911**. On frozen _95 wavelength map, this is 79.195%
wavelength-only bracketing (previous _96: 65.676%).
A candidate is rejected if accepted features share fewer than 100 CCD rows.
Applying BOTH spectral and shared spatial support leaves
**64.222%** of all pixels. Remaining 35.778% explicitly
unsupported. The preview function returns NaN there.

At ~5200 Å, the spatial-bin median across shared slit rows is
~5.996 Å for v18 versus ~6.046 Å for _96; this is **not** the
central-row prediction. At detector row y=516 and the closest
wavelength pixel (~5200.453 Å), the candidate is **5.762 Å**
and the frozen _96 value is **5.881 Å**. On mutually supported
pixels, median absolute model difference is 0.128 Å, p90 0.297 Å.

A second, more demanding model check compares original ARC
measurement to the *full model*, not just two-ARC repeatability.
The red pair near 5336 Å, y=506, yields **6.225/6.193 Å**
on original NEON06/13, while the fitted v18 smooth surface gives
~**5.793 Å** at the same wavelength and row, a **0.400–0.433 Å**
systematic model discrepancy. These two lamps agree with each
other, yet the fitted smooth model does not reproduce their
local profile at that y. Good LOFO criteria alone therefore
cannot be treated as red-pair scientific validation.

The post-selection model is sensitive to preliminary LOFO gates:
with median/p90 gates (0.40/0.85 Å), the red pair is excluded
and central y=516/5200 Å model predicts 5.52 Å **outside
the resulting line-bracketed range**; with (0.45/0.90)
the red pair is included and model predicts ~5.71 Å;
with default (0.50/0.95) ~5.76 Å, and with
(0.60/1.10) ~5.63 Å. These comparisons do not represent
a calibrated CI; they demonstrate selection sensitivity.

Keeping the default ten accepted features fixed, a whole-feature
leave-one-out *refit* predicts FWHM at 5200 Å and y=516 in
the range **5.627–5.854 Å** (spread 0.226 Å). This is an
internal influence measure, NOT an independent systematic
or formal measurement uncertainty.

The model is a diagnostic candidate only. Do not replace the
measured 5336 Å pair width with its smooth prediction or
use the v18 FWHM for science until the discrepancy is understood.

## Requirements for science-ready LSF

- Refit raw original Ne profiles inside the reproducible pipeline,
  quantify widths including alternate continuum windows, wing
  residuals/asymmetry and neighboring unidentified features.
- Validate selection/model complexity on *independent* nights and
  grisms; LOFO on same data is exploratory and selection-biased.
- Explicitly propagate model, wavelength-map and slit-position
  uncertainties with spatial correlation, not individual-pixel
  independent error bars.
- Distinguish ARC slit-illumination from sky-emission and point-source
  LSF; Gaussian FWHM is not the entire LSF.
- Verify absolute per-exposure science wavelength zero point from
  independent unblended atmospheric references and the actual
  telescope pointing/long-slit position.

SCORPIO-1/VPHG1200B nominal range/dispersion/FWHM:
https://www.sao.ru/hq/lsfvo/devices/scorpio/grisms/grisms.html

LSF can be wavelength/position dependent and non-Gaussian:
https://doi.org/10.1051/0004-6361/200811467

The production lsf_model.py is untouched by this v18 evidence probe.
SCIENCE HOLD remains.
