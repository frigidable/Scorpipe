#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cleanup_dependabot_actions_prs.sh [OWNER/REPO] [--yes]

By default this script prints open Dependabot PRs for GitHub Actions updates
and suggests which ones to close (keeping the most recently updated PR).

With --yes it will close the duplicates automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "[ERROR] gh is not installed." >&2
  exit 127
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is not installed." >&2
  exit 127
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "[ERROR] gh is not authenticated. Run: gh auth login" >&2
  exit 2
fi

REPO=""
YES="false"
for arg in "$@"; do
  if [[ "${arg}" == "--yes" ]]; then
    YES="true"
  elif [[ -z "${REPO}" ]]; then
    REPO="${arg}"
  fi
done

if [[ -z "${REPO}" ]]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi

echo "[INFO] Listing open Dependabot PRs (github-actions) in ${REPO}..."
PRS_JSON="$(gh pr list -R "${REPO}" \
  --search 'is:pr is:open author:app/dependabot label:github_actions' \
  --json number,title,updatedAt,url)"

COUNT="$(echo "${PRS_JSON}" | jq 'length')"
if [[ "${COUNT}" -le 1 ]]; then
  echo "[OK] ${COUNT} PR found. Nothing to clean up."
  echo "${PRS_JSON}" | jq -r '.[] | "#\(.number)  \(.title)  (\(.url))"'
  exit 0
fi

echo "[INFO] Found ${COUNT} PRs:"
echo "${PRS_JSON}" | jq -r '.[] | "#\(.number)  \(.title)  updated:\(.updatedAt)"'

# Keep the most recently updated PR; close others (safe default)
KEEP="$(echo "${PRS_JSON}" | jq -r 'sort_by(.updatedAt) | last | .number')"
CLOSE_LIST="$(echo "${PRS_JSON}" | jq -r --arg keep "${KEEP}" '.[] | select((.number|tostring) != $keep) | .number')"

echo
echo "[PLAN] Keep PR #${KEEP} (most recently updated), close:"
echo "${CLOSE_LIST}" | sed 's/^/  - #/'

if [[ "${YES}" != "true" ]]; then
  echo
  echo "[DRY-RUN] Re-run with --yes to close these PRs automatically:"
  echo "  bash scripts/github/cleanup_dependabot_actions_prs.sh ${REPO} --yes"
  exit 0
fi

echo
echo "[ACTION] Closing duplicates..."
while read -r num; do
  [[ -z "${num}" ]] && continue
  echo "  - closing #${num}"
  gh pr close -R "${REPO}" "${num}" --comment "Closing in favor of the most recent grouped/active Dependabot GitHub Actions update PR (#${KEEP})."
done <<< "${CLOSE_LIST}"

echo "[OK] Done."
