# GitHub rulesets & branch protection

This project is intended to be run with **protected `main`** to prevent accidental direct pushes and to guarantee that every merge is reviewed and CI-tested.

## What we enforce

Recommended rules for `main`:

- Require pull requests (no direct push to `main`)
- Require status checks:
  - `CI / gate`
  - `CodeQL / analyze`
- Require CODEOWNERS review
- Require all PR review threads resolved
- Block force-pushes
- Linear history (prefer squash merges)

These are exactly the settings encoded in:

- `.github/CODEOWNERS`
- `.github/rulesets/main-protection.json`

## Apply via GitHub UI (manual)

GitHub → Repository → **Settings** → **Rules** → **Rulesets**:

1. Create a **branch ruleset**
2. Target: `main` (or “Default branch”)
3. Enable the rules listed above
4. Save & enable

GitHub Docs:
- About rulesets
- Creating and managing rulesets
- About CODEOWNERS and “Require review from Code Owners”

## Apply via GitHub CLI (as code)

If you have admin access, you can apply the same ruleset from the repository files:

```bash
# from repo root
scripts/github/apply_rulesets.sh OWNER/REPO
# or (auto-detect current repo)
scripts/github/apply_rulesets.sh
```

This uses the official GitHub REST API endpoint:
`POST /repos/{owner}/{repo}/rulesets` and `PUT /repos/{owner}/{repo}/rulesets/{ruleset_id}`.

Note: rulesets availability depends on the repository plan/visibility.
