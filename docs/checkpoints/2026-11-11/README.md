# Scorpipe — integration and scientific checkpoint (2026-11-11)

**State:** reproducibility / engineering checkpoint, **NOT** a scientific release. **SCIENCE HOLD** for absolute Wave2D, scientific LSF, sky subtraction, per-exposure flexure, spatial stacking, final 1D and publication products. **Do not turn experimental diagnostics into science outputs.**

## Immutable lineage and branch roles

| Ref | Pinned commit / purpose |
|---|---|
| `archive/pr178-20261111-e1c1e3d2` | `e1c1e3d2072b7df953128e4e7f046d0efec91fc4`, exact pre-checkpoint PR #178 HEAD |
| `archive/local-base-20261111-d145f171` | `d145f171d057ce56516e0ae755a3afe9a9d80c95`, committed base of observer's local dirty working tree |
| `checkpoint/scorpipe-20261111-science-hold` | Branch based on the pinned PR #178 HEAD; **documentation-only changes** |
| PR #178 `fix/sc1-masterbias-gross-var-20261109` | Open Draft, base `qa/sc1-flat-profile-shift-and-route-20261106`. **Do not retarget, force-push or merge without a separate review.** |
| `main` | Unchanged by this checkpoint |

At inspection, PR #178 contained 142 commits and 59 changed files relative to its **non-main** base. Its HEAD was 470 commits ahead / 0 behind `main`. Only a skipped Dependabot workflow was surfaced at that exact HEAD; do **not** claim fresh full CI validation.

The observer's Windows checkout at this checkpoint was on `fix/barycentric-velocity-conventions`, HEAD `d145f171...`, with **staged, unstaged and untracked changes** (barycentric.py, schema.py, linearize.py, CHANGELOG.md, local QA archives, notes and CSV). **These uncommitted bytes are NOT stored in any GitHub branch**. They must first be exported separately using the non-destructive local checkpoint backup procedure in [RECOVERY.md](RECOVERY.md). Untracked inventory in the source report lists **only the first 20** changes; never infer completeness from it.

## Observing role truth: N2146sat (SCORPIO-1, 2024-12-06)

The observer explicitly confirmed:

| Frame IDs | Role | Meaning |
|---|---|---|
| `s22800604.fts` | **OFFSET_SKY** | dedicated 600s offset sky, approximately 108.947 arcsec from main target pointing |
| `s22800607.fts` through `s22800612.fts` | **OBJECT_FRAMES** | six 1200s science spectra of N2146sat |
| `s22800603.fts` | acquisition image of offset-sky field | R-band ~5s |
| `s22800605.fts` | acquisition image of main field | R-band ~30s |
| `s22800606.fts`, `s22800613.fts` | ARC / NEON | two original independent calibration frames |
| `s22800614.fts` through `s22800616.fts` | FLATS | fast-readout flats, `GAIN=3.5`, **not** the same detector readout as science |
| `s22800101.fts` through `s22800120.fts` | BIAS (previously selected) | twenty earlier-night bias files in original `config.yaml`; source selection existed and MasterBias has prior engineering audit |

Historical configs `N2146sat_VPHG1200B_01`, `_91`–`_96` **all contain OBJ04 within their seven `frames.obj` and no `frames.sky`**. Preserve as historical evidence, **do not edit retroactively**. For any **new isolated** science run create explicit `OBJECT_FRAMES=07..12` and `SKY_FRAMES=04`, with no overlap. **Do not globally exclude OBJ04**, as that would remove it from SKY as well. The loader's resolved configuration, not just its raw YAML, must be validated.

No intentional observing role implies every CCD slit row is free of contaminants; OFF SKY source-cleanliness is a separate scientific test.

## Actual engineering provenance (not a claim of scientific validity)

| Area | Achieved evidence | Remaining limitation |
|---|---|---|
| MasterBias | 20 selected real biases; earlier MasterBias SCI median reproduced exactly, readout matching and SCI/VAR/DQ/index contract; optional `GROSSMED` gross outlier mode, tests and engineering pilot | Provisional finite-N median estimator uncertainty, ADC quantization/time-dependent level, coverage; **do not redo from scratch** |
| `_91` / `_92` | preserved failed engineering runs; `_91` stops at Cosmics, `_92` at LineID | Historical failure evidence; never replace or overwrite |
| `_93` | Wave stage operationally `ok` but 2D science fit failed: line collisions and RMS `4.345 Å` | Product `ok` is **not** a scientific pass |
| `_94` | improved fail-closed Wave2D rejects insufficient supported line trajectories | Retain diagnostic failure |
| `_95` | Engineering Wave2D generated; independent arc analysis and source hash stored | Absolute wavelength zero/independent holdouts not validated |
| `_96` | Engineering arc-profile LSF product generated; source reports `science_lsf_validated=false` | Approx 34.3236% outside extreme measured ARC-line λ support; lamp LSF ≠ guaranteed stellar LSF |
| LineID22–25 | source audits, NIST ASD screening, native-pixel and blend-model diagnostics | No reliable independent nonhistoric holdout to certify absolute λ mapping |
| LineID26–26f | authenticated original OFF/ON CCD data comparisons; 420/420 conditional fit variants and zero strong compact 8-band candidate in provisional y64–967 OFF region | 420 fits share data; [N I] components/model ratios, y cleanliness, temporal sky evolution, zero point unproven |
| stack2d | `check_stack_pointing` fail-closed guard rejects >109″ OFF-vs-ON mismatch | Six ON exposures themselves have mechanical ROTANGLE variation; sky slit PA/trace registration not established |

LineID26f model-dependent candidate 5199 native-pixel offsets OBJ07–12 relative to OFF04: **+0.058, +0.138, +0.117, +0.159, +0.205, +0.158 px** (descriptive model sensitivity, **NOT a permitted flexure correction**).

## Actual source and processing graph

Existing source modules are retained, not reimplemented:
- MasterBias: `src/scorpio_pipe/stages/calib.py`, `src/scorpio_pipe/bias_uncertainty.py`, SCI/VAR/DQ/index; readout matching consumed by `cosmics.py`.
- Project roles: `src/scorpio_pipe/project_manifest.py` and `autocfg.py`: explicit `OBJECT_FRAMES`, `SKY_FRAMES`, manifest source.
- Science: `cosmics` → `superneon` → `lineid_prepare` → `wavesolution` → `lsf_model` → `sky` → `sky_refine` → `linearize` → `stack2d` → `extract1d`.
- Existing sky option: `sky.primary_method=sky_scale_raw` requires explicit SKY role and source compatibility. **Do not turn it on merely because OBJ04 is sky.** Assess bias-corrected/flat-compatible noise, contamination, time evolution and line calibration first.
- `stack_pointing_qc.py` checks telescope-pointing and **mechanical** `ROTANGLE`; it is **not** a celestial slit PA solution. Never weaken or bypass the guard to force a stack.
- `pipeline/recipes.py` contains legacy aliases `biascorr`/ `arclineid` adjacent to their canonical tasks; `public_api.validate_and_canonicalize_tasks` de-duplicates. This is a code inspection concern only, not evidence that runs duplicate stages.

## Checkpoint safety and next gates

1. Preserve the local dirty checkout **without** `git reset`, `git clean`, `stash drop`, forced checkout or force-push; store a `git bundle` and **all** staged/unstaged/untracked changed working-tree bytes outside the repository, verify hashes.
2. Preserve original `_01`, `_91`–`_96` run directories. Do not create `_97` or edit frozen FITS.
3. Write **new isolated** N2146sat role manifest, preflight-only; verify loader/engine resolve 6 OBJ + 1 SKY + 20 BIAS and exact ARC/FLAT list, no source overlap, all source SHA guards.
4. Add regression tests for a misleading `IMAGETYP=obj` header that is explicitly observer-assigned to SKY, and role disjointness across Cosmics, Sky, Linearize, Stack2D and existing product inventory.
5. Confirm celestial slit PA / y-registration for the six object frames from observation log or source geometry. Mechanical `ROTANGLE` alone is insufficient.
6. Independently validate wavelength TRAIN vs true HOLDOUT, sky [N I] doublet/line ratio, ARC/sky LSF and uncertainty coverage before enabling any science data products.
7. Run fresh targeted Windows checks and E2E on a **new** isolated worktree/run only after role and scientific gates. Distinguish `ENGINEERING_OK` and `SCIENCE_VALIDATED` explicitly.

## Evidence and full recoverability

- [Machine-readable checkpoint metadata](SOURCE_LOCKS.json), local configuration hashes, frozen FITS SHA256 and run-status matrix. This metadata was transcribed from observer-submitted read-only inventory and existing GitHub science reports; it is **not** an upload of the original 2024 FITS data.
- [Recovery and unfinished work](RECOVERY.md) explains the safe local backup and recovery boundaries.
- GitHub history: [PR #178](https://github.com/frigidable/Scorpipe/pull/178), [original pilot](../../qa/SC1_N2146SAT_ENGINEERING_PILOT.md), [LineID26f](../SCORPIO1_N2146SAT_LINEID26F_CONFIRMED_OFFSET_SKY_AND_BIAS_GATE.md), [LSF96](../SCORPIO1_N2146SAT_LSF96_UPLOAD_AUDIT.md).
- Observer's detailed local inventory file SHA256: `307fe97898e84f6f837bf9387719afb82a7f4ce0a8b9d25657ba4ec0e0c8ca25`; private attached evidence, **not** stored in this public GitHub tree.
- The branch contains a **snapshot of committed source only**. Source 2024 FITS files, local work products, external NIST downloads and unstaged edits must be preserved independently, with hashes.

**Release status: SCIENCE HOLD.** These are deliberately durable descriptions of achieved work and blockers, not a bypass of any scientific verification.
