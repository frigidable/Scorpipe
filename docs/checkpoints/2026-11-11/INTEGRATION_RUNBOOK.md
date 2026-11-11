# Scorpipe vNext integration runbook — 2026-11-11

**Status: DRAFT / SCIENCE HOLD.** This runbook does **not** authorize merging into `main` or processing new science FITS.

## Canonical governance

- [PR #180](https://github.com/frigidable/Scorpipe/pull/180): one review surface against protected `main`; initially branched from PR #178 after docs-checkpoint PR #179 was fast-forward merged.
- [Issue #181](https://github.com/frigidable/Scorpipe/issues/181): 26 divergent branch tip review queue. Each item needs code-equivalence analysis, targeted tests and a disposition.
- [BRANCH_CONSOLIDATION_REGISTRY.json](BRANCH_CONSOLIDATION_REGISTRY.json): all 84 remotely listed branch SHA pointers (at snapshot time).
- [SOURCE_LOCKS.json](SOURCE_LOCKS.json): frozen original source and science-product checksums.
- [README.md](README.md): actual verified scientific status and unresolved model gates.
- [RECOVERY.md](RECOVERY.md): local backup and disaster recovery instructions.

The observer successfully ran a separate private backup of the dirty Windows worktree: reported `BACKUP_VERIFIED` with 5,789 Git objects, 4 modified tracked, 83 untracked files and 139 ZIP members including 52 historical evidence files. Neither the private ZIP nor the full original FITS archive is part of this GitHub repository. Do not reset or delete it.

## Exact project changes made on the integration branch

1. `project_manifest.py`: reject a physical FITS used in both dedicated `OBJECT_FRAMES` and `SKY_FRAMES`, recognizing relative/absolute representations; raise **before** delivering frames to Cosmics/Stack2D.
2. `tests/test_n2146sat_offset_sky_manifest_contract.py`: verify historical 7-OBJ config can be intentionally reassigned to 6 SCIENCE + OFF04 SKY, without dropping existing BIAS20/ARC2/FLAT3, and reject role collision.
3. `docs/examples/N2146sat_20241206_offset_sky.project_manifest.yaml`: **example only**, not an active run file. Override only OBJ and SKY, retain other original calibration roles.
4. `recipes.py`: remove duplicate canonical aliases `biascorr` / `arclineid`, and run Cosmics before Flatfield in relevant science recipes to avoid first-run/rerun differences from opportunistically consumed CLEAN products.
5. `tests/test_recipe_science_dependencies.py`: exercise canonical task uniqueness and prerequisite ordering (adapted from standalone PR #85).
6. `tests/test_barycentric_wavelength_convention.py`: validate the already-present positive Astropy optical Doppler multiplication and invalid-input fail-closed behavior (based on independent PR #84 and the Astropy official optical convention). No barycentric source changes in this branch.

Changes **must be tested on the final SHA** before considering them accepted. They are not scientific release approval.

## Branch consolidation policy

- Already ancestors: Do **not** merge again. Their commits and history are already reachable via PR #180. When PR #180 is validated, resolve their old PRs with clear links; do not force rewrite historical tips.
- Divergent: Review minimal patch against `main` and compare actual source/testing semantics against PR #180. Transplant only needed changes into small reviewable commits; do not blindly merge a stale branch that diverged by hundreds of commits.
- `audit/barycentric-optical-wavelength-factor-20261103`: current code already multiplies wavelengths correctly; regression added in PR #180; original PR #84 is superseded on scientific behavior once tests pass.
- `audit/recipe-dependency-order-20261103`: unique canonical stage and Cosmics-before-Flatfield fix adopted; original PR #85 is superseded once tests pass.
- Dependabot/release: update dependencies using separate security/release review; DO NOT merge stale release-please tags or dependency pins in bulk.
- Experimental frozen covariance and Sky/LSF: preserve their flags, tests and provenance. Their presence in branch history does not certify science.

## Required gates before main

- GitHub Actions `tests_smoke`, `tests_fast`, `meta_validate`, `dependency_review`, `build_check`, semantic PR title all green on a consistent final HEAD.
- Targeted Windows checks on exact SHA with original source environment. Old 108/108 or 17/17 results are not transferable to later commits.
- Test effective `project_manifest` frame selections using the actual Scorpipe config loader; do not mutate historical run directories.
- Confirm sky OFF04 spatial contamination mask; zenith versus ROTANGLE versus sky slit PA; Wave2D genuinely independent holdouts, LSF support and [N I]5199 identity/ratio before full science.
- Merge only with explicit scientific release review; `main` protected. If tests fail, fix on integration branch, retain Draft.
- Keep original `_01/_91.._96` and locked raw FITS immutable. Do not manufacture `_97`.

## If CI fails

Read the individual GitHub Actions job logs and first failing test, preserve the failure in issue #181 and make a minimal correction. Do not mark jobs as passed from run existence or from old commit status. Check the conventional PR title before re-running the semantic-title job.
