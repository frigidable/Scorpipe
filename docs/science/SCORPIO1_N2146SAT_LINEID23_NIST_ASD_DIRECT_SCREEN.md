# N2146sat SCORPIO-1 — LineID23 direct NIST ASD v5.12 laboratory screen (2026-11-10)

**Status: source-verified laboratory candidates only; SCIENCE HOLD.** Windows LineID22a has passed 172/172 tests and generated 19/19 header/source review. LineID23 synthetic tests currently 16/16 in an independent non-Windows environment; **no original ARC pixel fitting or Windows LineID23 acceptance has yet completed**. Frozen `_95`, `_96`, LSF, and original raw FITS remain unchanged; do not create `_97` or change production Wave2D.

## Primary source acquisition

Queried NIST **Atomic Spectra Database 5.12** (full line table, **not** abbreviated Strong Lines) for He I/II, Ne I/II, Ar I/II. **Fourteen** disjoint windows covering ±15 Å of all 19 historic anchors returned exactly **359 transition records**, with declared count matching parsed rows on each query: 17, 12, 23, 26, 27, 39, 37, 40, 15, 10, 36, 16, 24, 37. Wavelengths are **standard air (200–2000 nm)**; keep *observed* and *Ritz* separate, including Ritz-only fine structure. NIST intensities are source-dependent and **not** calibrated across elements or ionization stages. These transition records are not 359 independent measured features. Source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html ; documentation https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html ; version history https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml ; DOI 10.18434/T4W30F.

| Historical line Å | Direct NIST ASD observed air Å | Scientific status |
|---:|---|---|
| 3888.65 | He I 3888.648 | fine structure, not isolated |
| 3948.98 | Ar I 3948.979 | strong laboratory match, arc purity unknown |
| 4044.42 | Ar I 4044.418; Ar II 4042.8934; Ne I 4042.642 | neighbor review |
| 4181.88 | Ar I 4181.884; Ar II 4179.2972 | neighbor review |
| 4190.71 | Ar I 4190.713; Ar I 4191.029 | severely unresolved 0.316 Å components |
| 4259.36 | Ar I 4259.362; Ne I 4259.739 | neighbor review |
| 4333.56 | Ar I 4333.561; Ne I 4334.1267; Ar II 4332.0295 | asymmetric blend risk |
| 4345.17 | Ar I 4345.168; Ne I 4345.479; Ar II 4348.0635 | blend risk |
| 4471.48 | He I 4471.4802; Ne II 4471.586 | fine structure / ionic neighbor |
| **4713.00** | **Ne II 4712.968**, Ne I 4712.800, He I 4713.1457; He I Ritz-only 4713.375684 | multiple plausible components, historical value not a nonnegative He-only centroid |
| **4863.96** | Ne I 4863.081, 4864.3729, 4865.5065, 4866.476 | complex potential Ne blend, no isolated historical line confirmed |
| 4921.93 | He I 4921.9313 | laboratory match |
| 5015.68 | He I 5015.6783; Ne I 5015.187; Ar II 5017.1626 | neighboring species |
| 5037.75 | Ne I 5037.7512; Ar II 5037.770 (weak/blended tabulated) | close numeric match not proof of Ar II contribution |
| 5047.74 | He I 5047.738; Ne I 5046.608 | mixed He/Ne historical pair with 5037.75 |
| **5080.38** | **Ne I 5080.383** | direct ASD correspondence; profile isolation still unverified |
| **5221.80** | **Ar II 5221.864**, Ar I 5221.271, Ne I 5222.3517 | competing transitions, actual emitter unknown |
| 5330.78 | Ne I 5330.7775; Ne I 5330.672 | laboratory match with nearby weak component |
| **5341.60** | Ne I 5341.0938, 5343.0048, 5343.2834 | potential effective blend centroid, not a single lab line |

**Most material new finding:** Ne II at 4712.968 (0.032 Å from legacy 4713.00) and Ar II at 5221.864 (0.064 Å from 5221.80) require adding ion-stage alternatives, although their presence/brightness in this lamp is unconfirmed. The high relative intensity of a different transition is not evidence against these candidates, and line coordinate proximity alone does not establish an ID.

## Next validation gate

Use an **independent** 14-query NIST export with saved raw responses and SHA-256, then SHA-guard the original s22800606/s22800613 ARC pixel frames, historical `_96` wavelength map and hand-pairs. Fit competing physical and empirical mixture hypotheses on identical detector windows at 33 slit positions per original ARC. Both spatial samples and lamps are correlated; unweighted robust BIC/AICc are strictly **heuristic**, not independent line-ID probabilities or a valid stellar-kinematics LSF. The historical map localizes features but cannot independently validate itself. Check held-out laboratory transitions and sky/reference zero points before a new Wave2D.

**No automatic LineID rewrite, no science FITS writes, no release from SCIENCE HOLD.**