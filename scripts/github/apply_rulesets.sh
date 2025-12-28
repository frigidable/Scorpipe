#!/usr/bin/env bash
set -euo pipefail

# Apply (create or update) repository rulesets via GitHub REST API.
#
# Usage:
#   scripts/github/apply_rulesets.sh OWNER/REPO [ruleset-json]
#
# Default ruleset JSON:
#   .github/rulesets/main-protection.json
#
# Requirements:
#   - gh (GitHub CLI) authenticated with a token that has repo admin rights
#
# Notes:
#   - This script is idempotent: it updates the ruleset if it already exists.
#   - GitHub API version used: 2022-11-28 (docs "REST API endpoints for rules").

REPO="${1:-}"
RULESET_FILE="${2:-.github/rulesets/main-protection.json}"

if [[ -z "${REPO}" ]]; then
  echo "ERROR: missing OWNER/REPO argument" >&2
  exit 2
fi

if [[ ! -f "${RULESET_FILE}" ]]; then
  echo "ERROR: ruleset file not found: ${RULESET_FILE}" >&2
  exit 2
fi

command -v gh >/dev/null 2>&1 || { echo "ERROR: 'gh' not found in PATH" >&2; exit 2; }

RULESET_NAME="$(python - "${RULESET_FILE}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["name"])
PY
)"

echo "Repo:         ${REPO}"
echo "Ruleset:      ${RULESET_NAME}"
echo "Ruleset file: ${RULESET_FILE}"
echo

# Look up existing ruleset by name
existing_id="$(gh api   -H "Accept: application/vnd.github+json"   -H "X-GitHub-Api-Version: 2022-11-28"   "repos/${REPO}/rulesets"   --paginate   --jq ".[] | select(.name == \"${RULESET_NAME}\") | .id"   | head -n 1 || true)"

if [[ -n "${existing_id}" ]]; then
  echo "Updating existing ruleset id=${existing_id} ..."
  gh api     -X PUT     -H "Accept: application/vnd.github+json"     -H "X-GitHub-Api-Version: 2022-11-28"     "repos/${REPO}/rulesets/${existing_id}"     --input "${RULESET_FILE}" >/dev/null
else
  echo "Creating new ruleset ..."
  gh api     -X POST     -H "Accept: application/vnd.github+json"     -H "X-GitHub-Api-Version: 2022-11-28"     "repos/${REPO}/rulesets"     --input "${RULESET_FILE}" >/dev/null
fi

echo
echo "Done. Review in:"
echo "  https://github.com/${REPO}/settings/rules"
