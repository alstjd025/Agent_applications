#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"

echo "[safe-push] Repo: $(pwd)"
echo "[safe-push] Remote: ${REMOTE}"
echo "[safe-push] Branch: ${BRANCH}"
echo "[safe-push] Clearing cached GitHub HTTPS credentials for this shared account..."

printf "protocol=https\nhost=github.com\n" | git credential reject || true

echo "[safe-push] Pushing..."
git push -u "${REMOTE}" "${BRANCH}"

echo "[safe-push] Done."
