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
#   - jq
#
# Notes:
#   - This script is idempotent: it updates the ruleset if it already exists.
#   - GitHub API version used: 2022-11-28 (docs "REST API endpoints for rules").

REPO="${1:-}"

if [[ -z "${REPO}" ]]; then
  echo "ERROR: missing OWNER/REPO argument" >&2
  exit 2
fi

command -v gh >/dev/null 2>&1 || { echo "ERROR: 'gh' not found in PATH" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "ERROR: 'jq' not found in PATH" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_RULESET_FILE="${REPO_ROOT}/.github/rulesets/main-protection.json"
RULESET_FILE="${2:-${DEFAULT_RULESET_FILE}}"

if [[ ! -f "${RULESET_FILE}" ]]; then
  echo "ERROR: ruleset file not found: ${RULESET_FILE}" >&2
  exit 2
fi

# Ensure gh is authenticated early (gives a clearer error than a failed API call).
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 2
fi

RULESET_NAME="$(jq -r '.name' "${RULESET_FILE}")"
if [[ -z "${RULESET_NAME}" || "${RULESET_NAME}" == "null" ]]; then
  echo "ERROR: ruleset JSON must include a non-empty 'name' field" >&2
  exit 2
fi

echo "Repo:         ${REPO}"
echo "Ruleset:      ${RULESET_NAME}"
echo "Ruleset file: ${RULESET_FILE}"
echo

API_HEADERS=(
  -H "Accept: application/vnd.github+json"
  -H "X-GitHub-Api-Version: 2022-11-28"
)

# Look up existing ruleset by name (GET list -> jq filter)
existing_id="$(
  gh api "repos/${REPO}/rulesets" --paginate "${API_HEADERS[@]}" \
    | jq -r --arg name "${RULESET_NAME}" '.[] | select(.name == $name) | .id' \
    | head -n 1
)"

if [[ -n "${existing_id}" && "${existing_id}" != "null" ]]; then
  echo "Updating existing ruleset id=${existing_id} ..."
  gh api "repos/${REPO}/rulesets/${existing_id}" \
    --method PUT \
    "${API_HEADERS[@]}" \
    --input "${RULESET_FILE}" >/dev/null
else
  echo "Creating new ruleset ..."
  gh api "repos/${REPO}/rulesets" \
    --method POST \
    "${API_HEADERS[@]}" \
    --input "${RULESET_FILE}" >/dev/null
fi

echo
echo "Done. Review in:"
echo "  https://github.com/${REPO}/settings/rules"
