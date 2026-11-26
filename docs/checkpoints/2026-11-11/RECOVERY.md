# Recovery procedure and unfinished work — 2026-11-11

**Scope:** recover committed Git history, local uncommitted changes, and protected astronomical evidence independently. The checkpoints **do not** modify `main`, PR #178, the observer's checkout, or any scientific FITS.

## Remote recovery / GitHub refs

1. Committed PR source at checkpoint is pinned by `archive/pr178-20261111-e1c1e3d2` → `e1c1e3d2072b7df953128e4e7f046d0efec91fc4`.
2. Observer's committed local checkout base was `d145f171d057ce56516e0ae755a3afe9a9d80c95`; independent archived ref `archive/local-base-20261111-d145f171`.
3. Documentation and lock files live on `checkpoint/scorpipe-20261111-science-hold`; its own tip SHA is distinct because this branch contains the checkpoint docs.
4. PR #178 is a **Draft** against `qa/sc1-flat-profile-shift-and-route-20261106`. It is **not** an endorsement to merge its branch into `main`. The checkpoint should be reviewed as docs-only, with an independent PR against the scientific development branch if desired.

If the active PR branch is later rewritten, the two archive refs preserve prior **committed** reachable objects, unless administrators delete/force-update those archive refs. GitHub branches are **not** protected immutable tags: maintainers must preserve them intentionally.

## Local uncommitted changes: CRITICAL

The submitted observer inventory reports dirty local branch `fix/barycentric-velocity-conventions` with `CHANGELOG.md`, `barycentric.py`, `schema.py`, `linearize.py`, a new velocity-correction audit, multiple untracked ZIPs/README and an untracked pytest directory. Only the first 20 working-tree entries were collected. None of these staged/unstaged/untracked bytes is in the current GitHub snapshot.

**Before any checkout, pull, merge, rebase, branch cleanup or local Git operation that might change the worktree**, run the read-only-source backup helper provided in the checkpoint conversation, in the existing checkout, to an outside destination. The helper should:
- run `git bundle create ... --all` to save all locally reachable committed refs;
- export `git diff HEAD --binary`, separately `git diff --cached --binary` and `git diff --binary`;
- archive every **untracked non-ignored** file and the current contents of tracked changed files (with relative paths) into a ZIP outside the repo;
- save complete `git status --porcelain=v1 -z`, `git branch -avv`, commit HEAD and checksums, plus run configurations/evidence;
- test ZIP CRC and `git bundle verify`, with a post-creation SHA-256 manifest.

**Do not use** `git reset --hard`, `git clean -fdx`, `git stash drop`, `git checkout -f`, `git push --force`. Do not add raw observatory FITS or user-specific temporary files to this **public** GitHub repository.

`git bundle` does **not** contain dirty working-tree files or ignored files. A changed-files ZIP does **not** include `.git` commit history or ignored artifacts. Both are needed. Large binary RAW and prior run directories need a separate backed-up copy (external disk/Drive etc.) with independently verified SHA256.

## Existing science runs: historical evidence, not redo targets

- `_01`: original run, old 20-BIAS and 7-OBJ selection.
- `_91`: prior Cosmics fail (engineering evidence).
- `_92`: prior LineID fail (engineering evidence).
- `_93`: operational Wave2D success despite physically invalid line tracks.
- `_94`: Wave2D fails closed on missing/poor line tracks.
- `_95`: frozen engineering Wave2D `lambda_map.fits`; SHA `37808e73...`.
- `_96`: frozen identical wavelength map and engineering LSF; LSF SHA `45f219f0...`, `science_lsf_validated=false`.

Do **not** overwrite any of `_01`, `_91`–`_96`. Do not fabricate `_97` to signal progress. The seven-object historical selections were later shown to include confirmed OFF sky frame 04. Preserve this provenance; correct the role only in a new, isolated manifest/run.

## P0 work after this checkpoint

**P0A — role consistency (preflight only):** exact raw inventory / project manifest with OBJ07–12 vs SKY04, and previous 20 BIAS / 2 ARC / 3 FLAT. Check loaded `frames` and hashes using the actual Scorpipe loader. Do not rely on an edited YAML alone. Verify no overlap of OBJ/SKY, and that `cosmics`, `sky`, `linearize`, `stack2d` consume the intended set. Do not enable `sky_scale_raw` until the physical sky template is qualified.

**P0B — integration regression:** automated regression test where OBJ04 has `IMAGETYP=obj` but explicit observer `SKY_FRAMES` assignment overrides it. Guard the two roles, readout and dropped inputs; protect the six-object science stack from the offset exposure. Verify that alias task names are canonicalized once.

**P0C — geometry and absolute spectra:** verify the celestial slit PA, acquisition-to-slit mapping and on-target y-registration; readout-matched bias provenance remains existing, not a new project. Re-evaluate NIST HOLDOUT train/test leakage and conditional [N I]5199 model/airglow before any Wave2D/LSF/sky science release.

**P0D — CI and staged deployment:** run targeted and full suites at exact pinned HEAD in separate clean worktree after role-only changes. Document what actually ran on Windows/Linux; no extrapolation from old test counts. Any failure remains recorded and blocks downstream stages.

## Quick GitHub verification

Use the GitHub UI for archived branch tip, commit tree, and the docs-only checkpoint PR. To audit via PowerShell **without changing the worktree**, inside an existing repository:

```powershell
git status --short
git rev-parse HEAD
git branch --show-current
git remote -v
git show --no-patch --oneline e1c1e3d2072b7df953128e4e7f046d0efec91fc4
git show --no-patch --oneline d145f171d057ce56516e0ae755a3afe9a9d80c95
```

The ability to `git show` a commit does not imply local files are backed up or science calibration accepted.
