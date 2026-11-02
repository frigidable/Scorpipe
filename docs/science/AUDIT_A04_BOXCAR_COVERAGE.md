# Audit A04 — correct NO_COVERAGE meaning for boxcar extraction

The previous `_boxcar_extract` OR-ed all input aperture mask bits into
the output mask. As a result, `NO_COVERAGE` on **one** geometrically
sampled but excluded input spatial row marked an otherwise supported
output wavelength bin `NO_COVERAGE`. Downstream science could reject
a spectrum despite finite accepted flux, variance and `NPIX>0`.

A 1D `NO_COVERAGE` flag must be derived from the accepted spatial
coverage and the configured `min_good_frac`, not blindly inherited
from any excluded input row. The patch suppresses only the forwarded
`NO_COVERAGE` bit, consistent with the existing optimal extraction path.
The actual acceptance threshold, flux, variance, and count of accepted
pixels are **unchanged**. All other input aperture DQ bits remain OR-ed
into the output as provenance (including `EDGE` and `BADPIX`).

The retained propagation of other scientifically fatal aperture bits
deserves its own follow-up: it can cause a line-measurement module to
exclude an entire wavelength bin even when some good aperture rows
contributed. A future provenance schema should distinguish
`aperture_contains_bad_sample` and `output_usable`. This PR deliberately
does not redefine existing mask semantics beyond A04.

Numerical tests cover one lost row, subthreshold accepted fraction,
zero support, mixed DQ bits, and comparison with optimal extraction.
The PypeIt boxcar extraction documentation also distinguishes
per-pixel input masks from the usability of extracted bins:
https://pypeit.readthedocs.io/en/stable/api/pypeit.core.extract.html .

No observations were modified. The PR must not be merged without
Linux CI, Windows exact-HEAD regressions and explicit review.
