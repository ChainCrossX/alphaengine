#!/usr/bin/env bash
# One-shot: initialize git, commit, and push to a new private GitHub repo.
# Requires the GitHub CLI (`gh`). Install: https://cli.github.com/
#
# Usage:
#   ./push_to_github.sh                  # repo name defaults to "alphaengine"
#   ./push_to_github.sh my-trading-bot   # custom repo name
#
# What it does:
#   1. Verifies gh is installed and you are signed in
#   2. Initializes git if needed
#   3. Commits all files
#   4. Creates a PRIVATE GitHub repo and pushes main

set -euo pipefail

REPO_NAME="${1:-alphaengine}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }
ok() { printf "  \033[32mok\033[0m %s\n" "$1"; }
err() { printf "  \033[31mxx\033[0m %s\n" "$1"; }

bold "1. GitHub CLI"
if ! command -v gh >/dev/null 2>&1; then
  err "gh not found. Install: https://cli.github.com/  then re-run."
  exit 1
fi
ok "gh installed"

bold "2. GitHub auth"
if ! gh auth status >/dev/null 2>&1; then
  err "Not signed in. Run: gh auth login   then re-run this script."
  exit 1
fi
ok "signed in as $(gh api user --jq .login)"

bold "3. Git init"
if [ ! -d .git ]; then
  git init -b main
  ok "git initialized"
else
  ok "git already initialized"
fi

bold "4. Stage and commit"
git add .
if git diff --cached --quiet; then
  ok "no changes to commit"
else
  git commit -m "AlphaEngine snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ok "commit created"
fi

bold "5. Create private repo and push"
if git remote get-url origin >/dev/null 2>&1; then
  ok "origin already set; pushing"
  git push -u origin main
else
  gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
  ok "repo created and pushed"
fi

URL=$(gh repo view --json url --jq .url)
bold "Done."
echo "  Repo URL: $URL"
echo
echo "Next steps:"
echo "  Render:  https://render.com/deploy and point at this repo"
echo "  Railway: https://railway.app/new from GitHub repo"
echo "  Fly.io:  flyctl launch --copy-config"
