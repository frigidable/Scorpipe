# N2146sat _96 ARC LSF independent audit, 2026-11-09

Status: ENGINEERING LSF COMPLETE, SCIENCE HOLD. Frozen _96 is immutable.
The user-supplied archived run was direct run_lsf_model, not engine E2E.
Pinned commit: 6384522d5849deeba8ada9c99bd8b4f302e7413c.

## Archived input SHA256
- lambda_map.fits: 37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab
- superneon.fits: 001f36ab8cfbf1ed03c3af8a93768310f0e6407c9ced34c2fae43543e77243f9
- hand_pairs.txt: 2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623
- generated _96 lsf_map.fits: 45f219f044cc9a574778f2df8922928ab8c080a0379aec6c6472767bd118b11b

## Archived science measurements

Stage reports arc_measured, engineering ok, science_lsf_validated=false.
Source accepted 450 spatial-strip Gaussian-plus-linear-background
measurements from nine independent manual lines. It rejected
416 known close-line cases and 122 other profile fits.
Line 4713.00 and 5221.80 Å failed low_snr in all 52 sampled
spatial strips; edge-noise estimator may count structured wings
as noise and should be reviewed. These exclusions are LSF-only;
they do not warrant deleting lab wavelength anchors.
Four close-line pairs (4182/4191,4334/4345,5038/5048,
5331/5342 Å) are excluded from isolated Gaussian fitting.

LSF map shape 1032x2080, all 2,146,560 pixels numerically finite.
Reported LSF median FWHM ~5.96865 Å, p16–p84 ~5.45–6.66 Å,
minimum/maximum ~5.13–7.76 Å. The source wavelength map spans
3572.633–5385.218 Å. Nine measured ARC-line anchors span
only 3888.65–5080.38 Å. The ratio of wavelength pixels
inside this empirical span is **65.6764%**; the other
**34.3236%** have edge-constant numerical extrapolation.
The sky [N I] 5198/5200 Å doublet is NOT bracketed by the
measured LSF lines. An exact real-valued FITS value is not
proof of LSF measurement or science quality.

Even within the extreme anchor span, there is a 392.48 Å gap
from 4471.48 to 4863.96 Å, and adjacent fitted line widths
are not smoothly varying. The LSF there is interpolated,
not independently resolved at every wavelength.
The 5015.68 Å width may be line-dependent; it is not
confirmed by the earlier independent raw NEON isolated-line
catalogue. A separate joint fit near 5336 Å gave FWHM ~6 Å
but is NOT implemented in the production source LSF estimator.

Eight common lines are available in previous independent raw
NEON06 and NEON13 measurements. The RMS of the line-by-line
FWHM difference is ~0.110 Å. The largest discrepancy is
+0.260 Å at 4471.48 Å. The source methods use different
y-binning/weighting, so this is a consistency check, not
independent certification of absolute astrophysical LSF.

## Fix added after _96 to PR #178

The pure bracketed_arc_wavelength_support helper distinguishes
finite map pixels from those bracketed by fitted ARC lines.
Future runs write lsf_span_support.fits (uint8: 1 in the
measured outer line span, 0 outside). Headers carry LSFSPAN
and LSFSCI=false. QC records numerical coverage distinctly
from span-supported fraction, maximum between-line gap,
and a warning when wavelengths are not bracketed. Even
support=1 does not mean independent LSF shape, stellar/slit
illumination or wavelength flexure has been validated.

The user Windows regression suite for this follow-on
patch has NOT yet run. Do not modify or Resume frozen _96.
Science status remains HOLD.

## Scientific next gates
- Joint fitting of the 5330.78/5341.60 Å neon pair on each
  original lamp; independent LSF λ/y and time consistency.
- Robust rejection of anomalous line widths and close blends,
  with model-order and line-held-out residual checks.
- LSF transfer from sky-filling lamps to astrophysical sources
  using slit-illumination uncertainty.
- Independent multiple sky-complex absolute zero-point
  checks before per-exposure wavelength corrections.
- Verified celestial slit geometry and pointing before
  combining spectra at Stack2D.
