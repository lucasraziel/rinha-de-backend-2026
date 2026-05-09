#!/usr/bin/env bash
# Creates / refreshes the orphan `submission` branch with only the files the
# Rinha evaluator needs:
#   docker-compose.yml — pinned to the immutable image tag
#   nginx.conf
#   info.json
#   LICENSE
#
# Usage:
#   ./scripts/make-submission.sh <git-sha-of-image>
#
# The <git-sha> must match an image already pushed to Docker Hub by the
# release workflow (lucasraziel/rinha2026-api:<sha>).
set -euo pipefail

SHA="${1:?usage: $0 <git-sha-of-image>}"
IMAGE="lucasraziel/rinha2026-api:${SHA}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# verify the image exists on Docker Hub before going through the trouble
echo "→ checking ${IMAGE} exists on Docker Hub"
TOKEN=$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lucasraziel/rinha2026-api:pull" | jq -r .token)
if ! curl -fsS -o /dev/null -w '%{http_code}' \
     -H "Authorization: Bearer ${TOKEN}" \
     -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.docker.distribution.manifest.list.v2+json' \
     "https://registry-1.docker.io/v2/lucasraziel/rinha2026-api/manifests/${SHA}" \
     | grep -q '^200$'; then
    echo "❌ image not found: ${IMAGE}"
    echo "   wait for the release workflow to finish, or push manually"
    exit 1
fi
echo "  ✓ image exists"

# stage submission files in a temp dir
TMP=$(mktemp -d)
trap "rm -rf ${TMP}" EXIT

cp LICENSE info.json "${TMP}/"
mkdir -p "${TMP}/docker"
cp docker/nginx.conf "${TMP}/docker/"

# render docker-compose with the pinned image
cat > "${TMP}/docker-compose.yml" <<EOF
## Rinha 2026 — submission
## total budget: 1.0 CPU + 350 MB RAM
## image: ${IMAGE}

services:
  api1:
    image: ${IMAGE}
    environment:
      PORT: "9000"
      NPROBE: "16"
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

echo "→ submission tree:"
(cd "${TMP}" && find . -type f | sort)

# create / reset the submission branch from the staged tree
echo "→ creating orphan submission branch"
git switch --orphan submission 2>/dev/null || git switch submission
git rm -rf . >/dev/null 2>&1 || true
git clean -fdx
cp -R "${TMP}/." .

git add -A
git status -s

cat <<EOF

→ Now (manually, since GPG signing needs a TTY):
    git commit -m "submission: pin image to ${SHA:0:7}"
    git push -u origin submission

Then on the Rinha repo (https://github.com/zanfranceschi/rinha-de-backend-2026):
  1. Fork it.
  2. Add participants/lucasraziel.json with:
        [{"id": "rinha2026-api", "repo": "https://github.com/lucasraziel/rinha-de-backend-2026"}]
  3. Open a PR.
EOF
