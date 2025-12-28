# GitHub Rulesets (Branch Protection)

This repository uses a **Ruleset** (instead of the older "Branch protection rules") to protect the default branch.

## What we enforce

For the default branch (`~DEFAULT_BRANCH`) we enforce:

- No direct push / no force-push
- Changes must come through a pull request
- Code owner review (only for files that have CODEOWNERS entries)
- Required status checks:
  - `CI / gate`
  - `CodeQL / analyze`

Ruleset payload lives in:

- `.github/rulesets/main-protection.json`

## Apply / update ruleset

You can apply (create or update) the ruleset via script:

```bash
bash scripts/github/apply_rulesets.sh OWNER/REPO
```

To use a different JSON file:

```bash
bash scripts/github/apply_rulesets.sh OWNER/REPO path/to/ruleset.json
```

## Notes on CODEOWNERS

If you enable "Require code owner review" but your `CODEOWNERS` file points to non-existing users/teams,
GitHub may block merges. Keep `.github/CODEOWNERS` valid and up to date.
