# LineID26g: conditional 5199-A feature time-transfer review

**Status: diagnostic only; SCIENCE HOLD. No reduction or sky subtraction.**

## Reuse already qualified archive — do not repeat it

The observer verified that s22800604.fts is one dedicated 600-second
OFF-SKY exposure and s22800607.fts through s22800612.fts are six
1200-second ON-TARGET exposures. Their isolated role contract was actually
checked on Windows using all 32 original source file hashes. Manifest:
be6966c73b7f04aebdbca6739d5d38a279379933b2cf7c80ae567dd65f458765.

Closed historical PR #178 contains the following *actual* evidence:
- LineID26c exported 7 x 1032 x 37 original native detector samples
  around the old-map conditional 5199-A feature. Its raw sample values
  were previously independently replayed.
- LineID26d exported eight continuum bands; the on-target data have a
  broad possible source excess around y=400..511 and a stronger narrow
  feature at y=896..911. The low throughput CCD edges are not safe sky.
- LineID26e compared original offset and target acquisition images,
  registered 21 compact common sources across 108.947 arcsec telescope
  pointing difference. This is not a certified source-to-slit mapping.
- LineID26f found no *strong* compact eight-band excess in provisional
  OFF-SKY rows 64..967; it certified ZERO SKY_ONLY rows. Its conditional
  5199 centroid offsets of ON07..12 vs OFF04 in combined candidate
  regions are +0.0581, +0.1378, +0.1170, +0.1590, +0.2055, +0.1585
  native pixels. Lower/upper differences and model ambiguity preclude
  treating this as a flexure measurement.
- LineID26f carried out 420 alternative fits, NOT 420 independent sky
  observations. Atomic [N I] line identity and doublet ratio remain
  unapproved. Source BIAS/FLAT physically valid correction and full sky
  estimator covariance remain HOLD.

Source record:
https://github.com/frigidable/Scorpipe/pull/178
Current qualification ledger:
https://github.com/frigidable/Scorpipe/issues/86

## Read-only calculation and its interpretation

New script: scripts/qa/sc1_lineid26g_temporal_review.py.

The historic LineID26c NPZ was previously reported on the user's Windows
machine at:
C:\Users\frigi\Documents\PhD\Scorpipe_LineID26c_FullSlit_v2_20261111_023442\lineid26c_fullslit_raw_native.npz

Use --inspect first to reveal keys, array shapes, dtypes and archive SHA256;
it does not write anything. Its literal NPZ key names and frame axis are
NOT independently available to the current GitHub CI. The analysis mode
fails closed unless one unique real numerical cube has shape 7x1032x37,
or --cube-key is explicitly given. The flag --assert-26c-frame-order is
required to record that the operator has checked the legacy NPZ mapping
OFF04, then ON07..12; this does not cryptographically authenticate that
mapping. The historical signed 26c report should be checked separately.

For each native detector row with 37 sampled x pixels, local background is
linearly interpolated between the *medians* of the first and last 7 pixels.
The summed residual for central pixel indices 13..23, including negative
values, is reported as a RAW LINE-CORE CONTRAST PROXY:

  q(frame, y) = sum over x=13..23 of
      [raw(frame,y,x) - linear_wing_continuum(frame,y,x)]

It is not a calibrated sky flux, Gaussian likelihood, formal S/N or a
source-independent wavelength-centroid measurement. Line wings, detector
structure and flexure can bias this proxy.

For the two previously used nonoverlapping provisional regions
LOWER_REVIEW y96..319 and UPPER_REVIEW y640..863, compute:
- Median raw line-core contrast, also divided by nominal exposure time
  (600 s OFF, 1200 s ON).
- Conditional ON/OFF proxy-rate ratio only for a positive OFF denominator.
- 16-row block medians and even/odd block diagnostic crosschecks.
- Exactly 14 region/frame summaries, without inventing uncertainties.

Those regions are RESEARCH CANDIDATES and not approved SKY_ONLY. Block
groups share detector, atmospheric and calibration systematics and cannot
serve as independent uncertainty samples. A small scatter of ON/OFF
ratios is not evidence that physical sky subtraction is allowed.

Run outputs must be new files outside all original RAW and work roots.
Neither input NPZ nor role_contract.json is modified. The output embeds
their SHA256 and ALL scientific approval flags are false.

## Required before a physical OFF-to-ON subtraction

1. Verify exact LineID26c source export and frame index/order against its
   previously signed report and the real 32-FITS role-contract fingerprint.
2. Qualify actual slow-readout MasterBias, calibrated SKY SCI/VAR/MASK,
   detector pixel geometry and source-independent bias/flat corrections.
   Do not divide raw OBJ by different-readout lamp FLAT (gain 3.5 vs 0.9).
3. Derive a physical sky-only slit mask using independent acquisition
   slit mapping, trace and bands, excluding source and low-throughput rows.
   Validate on held-out rows and wavelengths.
4. Use multiple independently verified night sky lines and physically
   valid wavelength/LSF profiles. Distinguish atmospheric temporal changes
   from shifts, profile variation, slit illumination and residual CCD bias.
5. Propagate sky-model estimator uncertainty and cross-covariance when
   one 600s OFF frame is reused for six science exposures. The same
   template contributes common uncertainty to every ON result.

External context: ESO X-Shooter slit offset pipeline uses individually
calibrated and paired ON/OFF exposures, with matching exposure lengths in
its standard recipe. This is methodological context, NOT a SCORPIO proof:
https://www.eso.org/observing/dfo/quality/XSHOOTER/pipeline/xsh_scired_slit_offset.html

**Not permitted**: automatic 2x SKY subtraction, changes to original
2024 native FITS, config, 03_bias, _95/_96, production Wave2D/LSF, _97,
or a release from SCIENCE HOLD.
