#!/usr/bin/env bash
# Creates / refreshes the orphan `submission` branch with only the files the
# Rinha evaluator needs:
#   docker-compose.yml — pinned to the immutable image tag
#   docker/nginx.conf
#   info.json
#   LICENSE
#
# Uses `git worktree` so the main working tree is left alone. Run from any
# branch you like.
#
# Usage:
#   ./scripts/make-submission.sh <git-sha-of-image>
set -euo pipefail

SHA="${1:?usage: $0 <git-sha-of-image>}"
IMAGE="lucasraziel/rinha2026-api:${SHA}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WT_DIR="$(mktemp -d -t rinha-submission-XXXXXX)"
echo "→ creating worktree at ${WT_DIR}"

# does the submission branch already exist locally?
if git show-ref --verify --quiet refs/heads/submission; then
    git worktree add "${WT_DIR}" submission
    pushd "${WT_DIR}" >/dev/null
    # wipe everything tracked so we start from a clean tree
    git rm -rf . >/dev/null 2>&1 || true
    git clean -fdx
    popd >/dev/null
else
    # create new orphan branch in the worktree
    git worktree add --detach "${WT_DIR}"
    pushd "${WT_DIR}" >/dev/null
    git checkout --orphan submission
    git rm -rf . >/dev/null 2>&1 || true
    git clean -fdx
    popd >/dev/null
fi

# stage submission files
mkdir -p "${WT_DIR}/docker"
cp LICENSE info.json "${WT_DIR}/"
cp docker/nginx.conf "${WT_DIR}/docker/"

# render docker-compose with the pinned image
cat > "${WT_DIR}/docker-compose.yml" <<EOF
## Rinha 2026 — submission
## total budget: 1.0 CPU + 350 MB RAM
## image: ${IMAGE}

services:
  api1:
    image: ${IMAGE}
    environment:
      PORT: "9000"
      NPROBE: "16"
      USE_IVF: "false"
    networks:
      - rinha
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "160M"

  api2:
    image: ${IMAGE}
    environment:
      PORT: "9000"
      NPROBE: "16"
      USE_IVF: "false"
    networks:
      - rinha
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "160M"
    depends_on:
      - api1

  nginx:
    image: nginx:1.27-alpine
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "9999:9999"
    networks:
      - rinha
    depends_on:
      - api1
      - api2
    deploy:
      resources:
        limits:
          cpus: "0.10"
          memory: "25M"

networks:
  rinha:
    driver: bridge
EOF

pushd "${WT_DIR}" >/dev/null
echo
echo "→ submission tree:"
find . -maxdepth 2 -type f -not -path './.git/*' | sort
echo
git add -A
git status -s

cat <<EOF

→ Worktree at:
    ${WT_DIR}

→ Now (manually, GPG signing needs a TTY):
    cd ${WT_DIR}
    git commit -m "submission: pin image to ${SHA:0:7}"
    git push -u origin submission
    cd -
    git worktree remove ${WT_DIR}

Then on the official Rinha repo (https://github.com/zanfranceschi/rinha-de-backend-2026):
  1. Fork it.
  2. Add participants/lucasraziel.json:
EOF
cat "${ROOT}/participants-snippet/lucasraziel.json"
echo "  3. Open a PR."
popd >/dev/null
