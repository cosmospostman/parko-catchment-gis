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
# Use 'python' (not 'python3') so this works inside an activated venv on EC2.
_PYTHON="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
_PYTHON="${_PYTHON:-$(command -v python 2>/dev/null || command -v python3 2>/dev/null)}"
if [[ -n "${_PYTHON}" ]] && "${_PYTHON}" -c "import rasterio" 2>/dev/null; then
    _RASTERIO_PROJ_DATA="$("${_PYTHON}" -c "import os, rasterio; p=os.path.join(os.path.dirname(rasterio.__file__),'proj_data'); print(p if os.path.isdir(p) else '')" 2>/dev/null || echo "")"
    if [[ -n "${_RASTERIO_PROJ_DATA}" ]]; then
        export PROJ_DATA="${PROJ_DATA:-${_RASTERIO_PROJ_DATA}}"
    else
        # Fall back to pyproj's bundled data directory
        _PYPROJ_DATA="$("${_PYTHON}" -c "from pyproj.datadir import get_data_dir; print(get_data_dir())" 2>/dev/null || echo "")"
        [[ -n "${_PYPROJ_DATA}" ]] && export PROJ_DATA="${PROJ_DATA:-${_PYPROJ_DATA}}"
    fi
fi
unset _PYTHON _RASTERIO_PROJ_DATA _PYPROJ_DATA

# Composite seasonal window — late dry season to capture vegetation stress
# (post-wet flush ends ~July; wet season breaks ~December in Gulf Country)
export COMPOSITE_START="${COMPOSITE_START:-08-01}"
export COMPOSITE_END="${COMPOSITE_END:-11-30}"

# Auto-activate local S2 cache if the EBS volume is mounted at /mnt/ebs/s2cache
export LOCAL_S2_ROOT="${LOCAL_S2_ROOT:-$([ -d /mnt/ebs/s2cache ] && echo /mnt/ebs/s2cache || echo '')}"

# Auto-activate local S1 cache if the EBS volume is mounted at /mnt/ebs/s1cache
export LOCAL_S1_ROOT="${LOCAL_S1_ROOT:-$([ -d /mnt/ebs/s1cache ] && echo /mnt/ebs/s1cache || echo '')}"
