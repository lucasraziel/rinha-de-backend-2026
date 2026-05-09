#!/usr/bin/env bash
# Runs the official k6 script against a locally-running compose stack.
# Requires k6 (`brew install k6`) and a fresh `docker compose up -d`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "→ docker compose up"
docker compose up -d --build

echo "→ waiting for /ready"
for i in $(seq 1 30); do
    if curl -fsS -o /dev/null http://localhost:9999/ready; then
        echo "  ready after ${i}s"
        break
    fi
    sleep 1
done

echo "→ k6 run"
k6 run bench/test.js

echo "→ docker compose down"
docker compose down
