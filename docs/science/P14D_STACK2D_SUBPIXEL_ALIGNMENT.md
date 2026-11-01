# Stack2D fractional y-alignment: broadcasting and source-coverage fix

Issue #47. In the original SCI, VAR and MASK subpixel y-shift helpers, a
Boolean selector of shape (N_y, 1) was used directly on data of shape
(N_y, N_lambda). NumPy Boolean array indexing does not broadcast this
mask along wavelength. With a nonzero measured y-offset and more than one
spectral column, the old code raises IndexError inside Stack2D.

The corrected helpers select *rows* with flat integer indices. Wavelength
columns are retained, with full-vector interpolation coefficients. SCI
uses (1-f) SCI0 + f SCI1, and marginal VAR uses
(1-f)^2 VAR0 + f^2 VAR1, for a deterministic shift.

For exact integer-valued offsets passing through the subpixel alignment
path, the second neighbor is not needed at f=0. The new validity check
requires source row i0 to be inside the detector and only requires i1 when
f>0. Exact integer shifts do not OR a noncontributing neighbor's DQ bit
into the output. Out-of-bounds rows retain NaN (SCI, VAR) or the
NO_COVERAGE DQ bit.

The code still applies **conditional marginal** variance only. Spatial
interpolation introduces off-diagonal covariance of different output y
rows; if two output rows share a raw CCD row, their covariance is generally
positive. These corrections are prerequisites for, not a replacement for,
Issue #42 / P1.4d full operator transport. No changes to the noise
covariance FITS representation or to stacking weights are made here.

Tests include multi-column nonzero positive/negative subpixel shifts,
positive/negative integral nonzero shifts, edge coverage, bitwise OR DQ,
a seeded Monte Carlo demonstration of shared spatial-row covariance,
and a genuine Stack2D streaming smoke run with nonzero y alignment.
Existing old no-shift behavior remains unchanged.

Science qualification remains limited to synthetic data; SCORPIO-2
real observations with measured y alignment have not been reprocessed
in this PR.
