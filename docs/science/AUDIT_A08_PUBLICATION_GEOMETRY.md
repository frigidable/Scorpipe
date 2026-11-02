# Audit A08 — publication geometry must be explicit

The existing publication export writes `Spectrum.fts` using the
**rectified Stack2D spectral grid**, while `wl.fts` and `dismap.fts`
contain maps sampled on **detector coordinates** before rectification.
The raw detector maps must **not** be used as pixel-aligned companions
to the rectified spectrum, even when the dimensions happen to match.

## Contract

All three exported primary and SCI image headers now carry
`PIXGEOM` (`RECTIFIED` or `DETECTOR`) and `MAPALGN=F` (the
detector-map/rectified-spectrum pair is *not* co-registered).
`publication_provenance.json` contains a versioned, machine-readable
`pixel_alignment` object that declares the detector wavelength and
detector LSF maps mutually aligned but not aligned with the stack.
The exporter rejects contradictory declared wavelength medium and
units between the three input products instead of producing an
internally inconsistent bundle. The detector-map helper WCS remains
explicitly approximate; actual wavelength and sigma values remain
unaltered.

`dismap.fts` is a **detector-geometry instrumental sigma map**, not
the effective 2D/1D LSF of the registered/stacked scientific data.
Computing a matched rectified LSF requires a scientifically validated
LSF transfer model. A same-shaped numerical interpolation cannot
stand in for instrumental line-spread convolution.

## Validation

Independent tests build dissimilar and equal-shaped detector/rectified
arrays, verify FITS primary+SCI headers, exact data preservation, NaN
support, geometry JSON, and fail-closed incompatible medium/units.
No real SCORPIO observation has been verified by this change.

Standards context: FITS World Coordinate System Papers I–III
(https://science.data.nasa.gov/fits/documentation/wcs). WCS coordinates
are not a proof of a common detector/rectified sampling operator.

**Follow-up:** instrument-specific physically justified rectification
of the LSF, explicit 2D spatial WCS and model QA. This work does not
make the current publication bundle science qualified.
