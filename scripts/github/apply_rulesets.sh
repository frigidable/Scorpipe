#!/usr/bin/env bash
set -euo pipefail

# Apply (create or update) the repository ruleset from `.github/rulesets/main-protection.json`.
#
# Requirements:
#   - GitHub CLI (`gh`) authenticated with a user that has admin permissions on the repo.
#
# Usage:
#   scripts/github/apply_rulesets.sh [OWNER/REPO]
#
# Example:
#   scripts/github/apply_rulesets.sh scorpio-pipe/Scorpipe

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
OWNER="${REPO%/*}"
NAME="${REPO#*/}"

RULESET_NAME="main-protection"
RULESET_FILE=".github/rulesets/main-protection.json"

if [[ ! -f "${RULESET_FILE}" ]]; then
  echo "ERROR: ${RULESET_FILE} not found (run from repo root)." >&2
  exit 1
fi

# Find existing ruleset id (if any)
EXISTING_ID="$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" "/repos/${OWNER}/${NAME}/rulesets" --jq ".[] | select(.name==\"${RULESET_NAME}\") | .id" || true)"

if [[ -n "${EXISTING_ID}" ]]; then
  echo "Updating ruleset '${RULESET_NAME}' (id=${EXISTING_ID}) in ${REPO}"
  gh api -X PUT     -H "Accept: application/vnd.github+json"     -H "X-GitHub-Api-Version: 2022-11-28"     "/repos/${OWNER}/${NAME}/rulesets/${EXISTING_ID}"     --input "${RULESET_FILE}" >/dev/null
else
  echo "Creating ruleset '${RULESET_NAME}' in ${REPO}"
  gh api -X POST     -H "Accept: application/vnd.github+json"     -H "X-GitHub-Api-Version: 2022-11-28"     "/repos/${OWNER}/${NAME}/rulesets"     --input "${RULESET_FILE}" >/dev/null
fi

echo "OK: ruleset applied."
