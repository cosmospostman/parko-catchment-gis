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
