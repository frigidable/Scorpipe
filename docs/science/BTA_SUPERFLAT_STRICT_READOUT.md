# BTA: strict SuperFlat readout matching

First real-data N2146sat observations show two readout modes:

- Science and available 1×2 bias: GAIN 0.9, RATE 145, NODE A.
- Target-near Neon and Flat: GAIN 3.5, RATE 225, NODE A, same 1×2 geometry.

The 40 other bias frames use 2×2, GAIN 3.51, RATE 145.
No recorded candidate bias matches the high-gain 1×2, RATE 225 mode.

High-gain exposures themselves are viable calibration inputs:
Neon frames have P50 1034 ADU, P99.9 5569–5671 ADU;
three flats have P50 13357–13373 ADU and P99.9
16716–16755 ADU. No tested pixels exceed 65000 ADU.
This does not establish appropriate bias subtraction or full slit
illumination calibration.

Before this patch, build_superflat used the default master-bias
resolver (bias_policy=degraded) and forwarded its readout
mismatch authorization to _build_superflat_core. This could subtract
a 0.9/145 ADU master from a 3.5/225 flat when dimensions match.

This patch requires strict master-bias selection for SuperFlat,
checks the actual master-bias FITS FrameSignature against the flat
even in legacy runs lacking a bias index, and unconditionally
passes allow_readout_diff=False to core subtraction.
A matched master-bias file remains supported.

The change intentionally fails closed for the observed BTA night
until a readout-matched master bias or independently certified
calibration is supplied. It does not infer a bias for 225-rate mode.
This is a scientific guard, not an automatic calibration workaround.

All source FITS remain immutable. Added unit and source-policy tests.
