# N2146sat SCORPIO-1: initial 19-line He/Ar/Ne atlas audit (2026-11-10)

**Status: preliminary LAB_ATLAS_REVIEW_REQUIRED / SCIENCE HOLD.**

Windows v21 completed 159/159 PASS and original raw-ARC diagnostic completed 264/264 fit attempts: 66/66 successful fits in each of four historical/alternative red-multiplet models. User's uploaded v21 CSV and JSON independently reconcile exactly. Frozen _95 and _96 unchanged.

## Input and scope

Reviewed 19 pairs in the archived _96 hand_pairs.txt, SHA-256 2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623. Original line source: user-provided Moiseev lines_long(1).tab; this appears on A. V. Moiseev's SCORPIO lecture dispersion plot, including 5341.60, 4863.96, 5080.38 and 5221.80 A. **Do not assume that the numeric columns 0/1/2/3 in the historical line list describe chemical species**; original key is not yet verified.

NIST Strong Lines tables for Helium, Argon and Neon have separate **AIR** wavelength sections. The comparison here is only an atomic **screen**: absent from those short strong-line tables does not imply absent from full NIST ASD or erroneous in the Moiseev working list. Labels describing neighbors are model-risk flags, NOT observed evidence that the neighbor necessarily contributes to this particular lamp.

## All 19 source anchors

| Historic air A | NIST identification / current provisional status | NIST strong-line air A | Review |
|---:|---|---:|---|
| 3888.65 | He I | 3888.6489 | MULTIPLET: also 3888.6046, 3888.6456 |
| 3948.98 | Ar I | 3948.979 | atlas wavelength agrees |
| 4044.42 | Ar I | 4044.418 | Ar II 4042.894 nearby |
| 4181.88 | Ar I | 4181.884 | Ar II 4179.297 nearby |
| 4190.71 | Ar I | 4190.713 | Ar I 4191.029 just +0.316 A |
| 4259.36 | Ar I | 4259.362 | atlas wavelength agrees |
| 4333.56 | Ar I | 4333.561 | Ar II 4332.030 and Ar I 4335.338 nearby |
| 4345.17 | Ar I | 4345.168 | Ar II 4348.064 nearby |
| 4471.48 | He I | 4471.479 | He I 4471.68 component |
| 4713.00 | He I candidate effective reference | 4713.146 | historical minus nearest = -0.146 A; 4713.38 component |
| 4863.96 | unresolved | — | full ASD/species/source check required |
| 4921.93 | He I | 4921.931 | atlas wavelength agrees |
| 5015.68 | He I | 5015.678 | atlas wavelength agrees |
| 5037.75 | Ne I | 5037.7512 | mixed Ne I / He I joint pair with 5047.74 |
| 5047.74 | He I | 5047.74 | NOT Ne I |
| 5080.38 | provisional Ne I, independently listed in calibration literature | — | absent NIST abbreviated strong-line table; full ASD check |
| 5221.80 | unresolved | — | full ASD/species/source check required |
| 5330.78 | Ne I | 5330.7775 | near 5326.3968 weaker reported Ne I |
| 5341.60 | candidate effective Ne I blend centroid | — | NIST Ne I 5341.0938 and 5343.2834: do not treat as isolated laboratory wavelength |

Classifiable directly from NIST Strong Lines: 6 He I, 7 Ar I, 2 unambiguous Ne I, 1 reviewed Ne blend region; 2 remain without a defensible atomic assignment (4863.96 and
5221.80); 5080.38 is independently established as Ne I but awaits
complete ASD air-wavelength/blend verification. One reference
5341.60 is a potential effective red blend rather than a known
single laboratory transition. A direct catalog match **does not** establish that the observed profile is free of blended neighbors.

## Follow-on independent evidence: laboratory identity and source-medium cautions

The original SCORPIO-1 instrument description of Afanasiev & Moiseev
states explicitly that the *SCORPIO-1* calibration illuminator uses a
**He-Ne-Ar gas-filled lamp**, not just a SCORPIO-2-only convention.
The BTA prime-focus adapter documentation likewise calls its
calibration mode NEON a He-Ne-Ar lamp. The original N2146sat
s22800606/s22800613 primary FITS headers state
INSTRUME=SCORPIO-1, IMAGETYP=neon, DISPERSE=VPHG1200B
and FILTERS=slit_1.2, but do not explicitly measure gas composition
or verify unchanged physical hardware on 2024-12-06.

Independent 2012 spectrograph commissioning dissertation classifies
5080.38 Angstrom as Ne I with reference to the NIST ASD line
database. A 2019 experimental gas-discharge spectroscopy paper
also identifies Ne I at **508.038 nm** and gives an atomic level
classification. Therefore **5080.38 is a supported Ne I historical
assignment**, but the exact NIST ASD observed-vs-Ritz AIR
wavelength and profile-blend purity have not yet been independently
exported in the current audit. The short NIST Handbook Strong Lines
table alone is NOT enough to refute the identification.

An experimental study reports **Ar I 522.127 nm**, numerically
5221.27 Angstrom. This could be near historic 5221.80 Angstrom,
but the source's reported air/vacuum convention has **not**
been established. Do not compute a physical residual between
them or assume an identification. In the code the candidate
is kept separate from all NIST-air laboratory transitions
with wavelength_medium=UNVERIFIED.

The 4863.96 Angstrom source anchor remains documentary only:
present in Moiseev's reference illustration and the published
SCORPIO-1 2013 user manual calibration plots, but the exact
species and laboratory transition cannot be verified from
the available full ASD output. We have **not** obtained a
complete NIST ASD 5.12 machine-readable line export during
this check, and should never label this search exhaustive.

For the red Ne I doublet the historic effective center at
5341.60 lies between NIST air 5341.0938 and 5343.2834.
If exactly two identical symmetric profile kernels and
nonnegative integrated line fluxes are assumed, producing
5341.60 as a blend centroid requires the red component to
contribute about **23.12% of total integrated flux**,
or a red-to-blue component ratio of about **0.3007**.
This is a **hypothetical necessary ratio**, not a measured
He-Ne-Ar lamp ratio.

At Gaussian FWHM=6 Angstrom the normalized line-shape
overlap for Ar I 4190.713/4191.029 is ~0.9962
(two-template Gram condition ~520), and for He I
4713.146/4713.380 ~0.9979 (condition ~949).
Independent free amplitudes are very poorly identified
by these low-resolution spectra even when the global
fit residual is small. The red 5341.0938/5343.2834
components have shape overlap ~0.831 and condition ~10.9,
but remain conditional on continuum, weak neighbors,
relative intensities and instrumental profile.

Source citations:
- Original SCORPIO-1 paper: https://www.sao.ru/hq/lsfvo/devices/scorpio/BSAO_eng.pdf
- BTA adapter: https://www.sao.ru/hq/lsfvo/adap/index-en.html
- Fabricius 2012, VIRUS-W line table 3.3:
  https://edoc.ub.uni-muenchen.de/14440/1/Fabricius_Maximilian.pdf
- Ne I / Ar I laboratory discharge study:
  https://www.mdpi.com/2218-2004/7/1/9
- Original SCORPIO-1 2013 manual:
  https://www.sao.ru/hq/lsfvo/devices/scorpio/manuals/scorpio_manual_2013.pdf

## Methodological consequences

1. The name NEON is a file/product type. SCORPIO-2 IFU instrumentation documentation explicitly calls its NEON exposure a **He-Ne-Ar lamp calibration**. This confirms that mixed-species NEON is a documented SCORPIO convention, **not** that the lamp in the December 2024 SCORPIO-1 exposure was necessarily identical; check observing log/hardware.
2. Existing LSF pair 5037.75 + 5047.74 combines **Ne I and He I**. Modeling both as neon-line amplitudes or lab strengths is physically incorrect.
3. 4181.88 + 4190.71 Ar I pair requires 4191.029 Ar I neighbor check; 4333.56 + 4345.17 Ar I pair requires Ar II 4348.064 and other neighbors. The v17–v20 two-component LSF fits cannot be assumed unbiased.
4. 4713.00 historical line center is 0.146 A shortward of NIST 4713.146 He I and He I 4713.38 is adjacent. The v18 LOFO outlier at 4713.00 may be a LineID/blend issue rather than purely model outlier.
5. Historical 5341.60 differs from Ne I 5341.0938 by 0.5062 A; separate Ne I at 5343.2834 can shift the effective centroid. In actual two-original-ARC v21 fits, 2-component vs 3-component Gaussian-equivalent FWHM differed in median by 0.335 A in the same detector interval. The historic two-line model had slightly lower normalized fit RMS; do not automatically change the wavelength solution.
6. The frozen _95 Wave2D was built using these historic line centers: residuals against that map are NOT independent evidence of atlas accuracy. A model-selection audit must jointly refit original ARC pixels with reference lab lines, branch competing identifications, block spatial samples, and evaluate held-out species/lines (avoiding extrapolation).
7. No replacement of line IDs, wavelength map, sky zero point or production LSF is permitted based on this screen.

## Sources

NIST Strong Lines He I, AIR: https://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable2.htm

NIST Strong Lines Ar I/II, AIR: https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable2.htm

NIST Strong Lines Ne I, AIR: https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm

NIST ASD (full transition data needed next): https://physics.nist.gov/asd

A. V. Moiseev, SCORPIO lecture III: https://www.sao.ru/hq/lon/SCORPIO/lectures/Scorpio_III.pdf

SCORPIO-2 IFU instrument/methods paper, He-Ne-Ar calibration: https://www.sao.ru/hq/lsfvo/devices/scorpio-2/SCORPIO-2_IFU_files/IFU_paper_rus.pdf

All data here is source-grounded but no external-night physical accuracy gate has been passed. **SCIENCE HOLD.**
