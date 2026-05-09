#!/usr/bin/env bash
# Downloads the read-only resources distributed by the Rinha 2026 challenge
# into ./resources/. We never commit references.json.gz (it's ~50 MB).
set -euo pipefail

BASE="https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/resources"
mkdir -p "$DIR"

if [[ ! -f "$DIR/references.json.gz" ]]; then
    echo "fetching references.json.gz (~50 MB)…"
    curl -fsSL "$BASE/references.json.gz" -o "$DIR/references.json.gz"
else
    echo "references.json.gz already present"
fi

echo "done. resources are in $DIR"
