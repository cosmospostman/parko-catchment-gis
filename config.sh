#!/usr/bin/env bash
# config.sh — source this before any Python subprocess
# Usage: source config.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# BASE_DIR is the root for all operational data — set this to a location outside the repo.
export BASE_DIR="${BASE_DIR:-/data/mrc-parko}"
export CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/cache}"
export WORKING_DIR="${WORKING_DIR:-${BASE_DIR}/working}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-${BASE_DIR}/outputs}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
export CATCHMENT_GEOJSON="${CATCHMENT_GEOJSON:-${BASE_DIR}/mitchell_catchment.geojson}"
export CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"

# PROJ_DATA — when rasterio and pyproj bundle different libproj versions, point
# PROJ_DATA at rasterio's proj_data so the minor version matches the libproj
# that wins at runtime (rasterio's is newer and takes precedence).
if python3 -c "import rasterio" 2>/dev/null; then
    export PROJ_DATA="${PROJ_DATA:-$(python3 -c "import os, rasterio; print(os.path.join(os.path.dirname(rasterio.__file__), 'proj_data'))")}"
fi
