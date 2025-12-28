#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [OWNER/REPO]"
  echo "Enables 'Allow auto-merge' for the repository using GitHub CLI."
}

if ! command -v gh >/dev/null 2>&1; then
  echo "[ERROR] gh is not installed. Install GitHub CLI first." >&2
  exit 127
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "[ERROR] gh is not authenticated. Run: gh auth login" >&2
  exit 2
fi

REPO="${1:-}"
if [[ -z "${REPO}" ]]; then
  # Use current repository context (from git remote / current dir)
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi

echo "[INFO] Enabling auto-merge for ${REPO}"
gh repo edit "${REPO}" --enable-auto-merge

echo "[OK] Auto-merge enabled. Now also check:"
echo "  Settings → Actions → General → Workflow permissions:"
echo "    - Read and write permissions"
echo "    - Allow GitHub Actions to create and approve pull requests"
