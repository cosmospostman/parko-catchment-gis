#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
python "$REPO_ROOT/analysis/fetch_ala_occurrences.py"
python "$REPO_ROOT/analysis/export_sightings_geojson.py"
echo "Sightings written to outputs/ala_cache/ala_sightings.geojson"
