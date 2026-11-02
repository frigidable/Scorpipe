# Audit A11a — detector-arc LSF is not the final extracted science LSF

## Confirmed scientific defect

Before A11, `stages/kinematics_prep.py` read `summary_1d.sigma_A`
from `lsf_model/lsf_model.json`, which is a **spatial median of
detector-coordinate arc-line widths**. It interpolated the result to
the 1D extracted spectrum and published this as `SIGMA_INST_A`,
`SIGMA_INST_KMS`, `SIGINST` and `FWHMINST` with no warning
that a different operator had been applied to the science data.

The actual science spectrum may have undergone:
1. CCD pixel-integration, detector flatfield and wavelength calibration;
2. COUNT_V1 spectral overlap integration into a different pixel grid;
3. per-exposure wavelength/flexure shifts and spatial y-alignment;
4. robust exposure stacking and wavelength-dependent weight selection;
5. TRACE/FIXED profile-weighted extraction with DQ exclusions.

No current production operator is demonstrated to propagate the arc
LSF through all these transforms. A Gaussian detector sigma therefore
cannot be treated as the effective post-reduction LSF required for
pPXF template matching and accurate stellar/gas velocity dispersion.

## Compatibility / fail-closed contract

This conservative, independent first slice changes only
`kinematics_prep` export semantics, not extraction or arc fitting.

- `SPEC_KIN.SIGMA_INST_A` and `SIGMA_INST_KMS`: all NaN until a
  separately qualified effective LSF is available. No misleading
  finite resolution is exported.
- `SIGMA_ARC_PROXY_A` and `SIGMA_ARC_PROXY_KMS`: retained detector
  arc-map median *diagnostics*, clearly not eligible for science
  template matching.
- The FITS primary header explicitly carries `LSFQUAL=F`,
  `LSFSTAT`, and optional `ARCSIG`; old `SIGINST` and
  `FWHMINST` keywords are withheld.
- `lsf_provenance.json` enumerates source, medium/unit checks, covered
  spectral bins, limitations, and `effective_1d_lsf_qualified=false`.
  The export remains available for inspection; `done.json` also
  records `lsf_qualified=false`.
- An arc proxy cannot be used when its documented detector spectral
  unit is not Angstrom (the current `lsf_model` may incorrectly label
  sigma measured in nm as sigma_A), or air/vacuum differs from the
  target 1D spectrum. Interpolation is restricted to measured
  wavelength support; **no extrapolation** to spectral edges.

This is an intentional downstream API semantics change. A consumer
must interpret the new columns and `LSFQUAL` rather than silently
substitute the proxy for `SIGMA_INST`.

## Science test roadmap: A11b onward

1. Validate detector arc LSF, sigma vs wavelength and spatial position,
   air/vac and native unit conventions using held-out arc lines.
2. Forward-model narrow monochromatic lines through the **same**
   COUNT_V1 mapping, spatial shifting, robust stacking and extraction,
   including fractional detector pixel integration.
3. Compare post-reduction line profiles against observed sky/arc
   features where geometry and illumination permit. For slit-filling
   arcs and point sources, do **not** assume identical slit illumination.
4. Preserve spectral kernel asymmetry or a sampled wavelength-dependent
   kernel if Gaussian second moments cannot describe the profile.
5. Treat model uncertainty, flexure drift and sample/pixel convolution
   explicitly. Compare pPXF velocity dispersion sensitivity to LSF
   errors before promoting any `SIGMA_INST_*` to science-qualified.

## Literature

- Law et al. (2016), AJ 152, 83, doi:10.3847/0004-6256/152/4/83.
  Documents MaNGA pixel-integration effects and systematic LSF biases.
- Law et al. (2021) and SDSS DR17 MaNGA LSF conventions:
  https://www.sdss4.org/dr17/manga/manga-data/working-with-manga-data/
  Distinguishes `LSFPRE` and `LSFPOST` and covariance of resampled data.
- Westfall et al. (2019), AJ 158, 231, section 7.4.3, spectral
  resolution matching for stellar kinematics:
  https://pure.port.ac.uk/ws/portalfiles/portal/20093750/Westfall_2019_AJ_158_231.pdf
- Weilbacher et al. (2020), A&A, MUSE pipeline and line profile
  propagation: doi:10.1051/0004-6361/202037855

None of these sources independently validates Scorpipe's current
detector-arc LSF. Qualification requires real SCORPIO-1/2 observations.
