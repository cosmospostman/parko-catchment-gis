#!/usr/bin/env bash
# run.sh — Pipeline orchestrator for Parkinsonia catchment GIS analysis
# Usage: ./run.sh YEAR [--composite-start MM-DD] [--composite-end MM-DD]
#                      [--from-step N] [--only-step N] [--dry-run]
#                      [--rebuild-baseline] [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 YEAR [--composite-start MM-DD] [--composite-end MM-DD]" >&2
    echo "            [--from-step N] [--only-step N] [--dry-run]" >&2
    echo "            [--rebuild-baseline] [--force]" >&2
    exit 3
fi

YEAR="$1"; shift

# Validate year is a 4-digit integer
if ! [[ "${YEAR}" =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: YEAR must be a 4-digit integer, got: ${YEAR}" >&2
    exit 3
fi

COMPOSITE_START="05-01"
COMPOSITE_END="10-31"
FROM_STEP=1
ONLY_STEP=""
DRY_RUN=false
REBUILD_BASELINE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --composite-start) COMPOSITE_START="$2"; shift 2 ;;
        --composite-end)   COMPOSITE_END="$2";   shift 2 ;;
        --from-step)       FROM_STEP="$2";        shift 2 ;;
        --only-step)       ONLY_STEP="$2";        shift 2 ;;
        --dry-run)         DRY_RUN=true;          shift   ;;
        --rebuild-baseline) REBUILD_BASELINE=true; shift  ;;
        --force)           FORCE=true;            shift   ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 3 ;;
    esac
done

export YEAR COMPOSITE_START COMPOSITE_END
export REBUILD_BASELINE
export CODE_DIR
export PIPELINE_RUN=1
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Logging setup ─────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}" 2>/dev/null || true
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${YEAR}_${TIMESTAMP}.log"
# Redirect all output through tee to log file (fall back to no log if dir unwritable)
if [[ -w "${LOG_DIR}" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo ""
printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"
printf "${BOLD}  Parkinsonia GIS Pipeline — Year %s${RESET}\n" "${YEAR}"
printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"
echo "  Log:              ${LOG_FILE}"
echo "  Composite window: ${COMPOSITE_START} → ${COMPOSITE_END}"
echo "  Rebuild baseline: ${REBUILD_BASELINE}"
echo "  Dry run:          ${DRY_RUN}"
[[ -n "${ONLY_STEP}" ]] && echo "  Only step:        ${ONLY_STEP}"
[[ "${FROM_STEP}" != "1" ]] && echo "  From step:        ${FROM_STEP}"
[[ "${FORCE}" == "true" ]] && echo "  Force:            yes (clearing sentinels)"
echo ""

# ── Git SHA for sentinel naming ───────────────────────────────────────────────
GIT_SHA="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "nogit")"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
preflight_failed=false

preflight_check() {
    local desc="$1"
    local ok="$2"   # "true" or "false"
    if [[ "${ok}" == "true" ]]; then
        printf "  ${GREEN}✓${RESET} %s\n" "${desc}"
    else
        printf "  ${RED}✗${RESET} %s\n" "${desc}"
        preflight_failed=true
    fi
}

echo "Pre-flight checks:"

# 1. Catchment GeoJSON
if [[ -f "${CATCHMENT_GEOJSON}" ]]; then
    preflight_check "mitchell_catchment.geojson exists" "true"
else
    preflight_check "mitchell_catchment.geojson exists at ${CATCHMENT_GEOJSON}" "false"
fi

# 2. Writable dirs
mkdir -p "${WORKING_DIR}" "${OUTPUTS_DIR}" 2>/dev/null || true
if [[ -w "${WORKING_DIR}" ]]; then
    preflight_check "WORKING_DIR is writable (${WORKING_DIR})" "true"
else
    preflight_check "WORKING_DIR is writable (${WORKING_DIR})" "false"
fi
if [[ -w "${OUTPUTS_DIR}" ]]; then
    preflight_check "OUTPUTS_DIR is writable (${OUTPUTS_DIR})" "true"
else
    preflight_check "OUTPUTS_DIR is writable (${OUTPUTS_DIR})" "false"
fi

# 3. Python dependencies
if python -c "import stackstac, odc, rioxarray, sklearn, geopandas" 2>/dev/null; then
    preflight_check "Python dependencies importable" "true"
else
    preflight_check "Python dependencies importable (stackstac, odc, rioxarray, sklearn, geopandas)" "false"
fi

# 4. Composite window is valid (start < end, both parseable as MM-DD dates)
_cs_days=$(python -c "
import sys
from datetime import datetime
try:
    s = datetime.strptime('2000-${COMPOSITE_START}', '%Y-%m-%d')
    e = datetime.strptime('2000-${COMPOSITE_END}',   '%Y-%m-%d')
    sys.exit(0 if s < e else 1)
except ValueError:
    sys.exit(1)
" 2>/dev/null; echo $?)
if [[ "${_cs_days}" == "0" ]]; then
    preflight_check "Composite window valid (${COMPOSITE_START} → ${COMPOSITE_END})" "true"
else
    preflight_check "Composite window valid — start must be before end and in MM-DD format (got ${COMPOSITE_START} → ${COMPOSITE_END})" "false"
fi

# 5. YEAR is within sensible range (>= 2015 Sentinel-2 launch, <= current year)
_current_year=$(date +%Y)
if [[ ${YEAR} -ge 2015 && ${YEAR} -le ${_current_year} ]]; then
    preflight_check "YEAR ${YEAR} is within valid range (2015 – ${_current_year})" "true"
else
    preflight_check "YEAR ${YEAR} is out of range — Sentinel-2 launched 2015, future years not supported" "false"
fi

# 6. TARGET_CRS covers the catchment bbox (catches wrong zone / unprojected CRS)
_crs_check=$(python -c "
import sys
sys.path.insert(0, '${CODE_DIR}')
import config
import geopandas as gpd
from pyproj import CRS, Transformer

try:
    crs = CRS.from_string(config.TARGET_CRS)
    if not crs.is_projected:
        print('not projected')
        sys.exit(1)
    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = catchment.to_crs('EPSG:4326').total_bounds  # minx miny maxx maxy
    t = Transformer.from_crs('EPSG:4326', config.TARGET_CRS, always_xy=True)
    xs, ys = t.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    import math
    if any(math.isinf(v) or math.isnan(v) for v in xs + ys):
        print('inf/nan bounds')
        sys.exit(1)
    if xs[1] <= xs[0] or ys[1] <= ys[0]:
        print('degenerate bounds')
        sys.exit(1)
    print('ok')
except Exception as e:
    print(str(e))
    sys.exit(1)
" 2>/dev/null || echo "error")
if [[ "${_crs_check}" == "ok" ]]; then
    preflight_check "TARGET_CRS covers catchment bbox" "true"
else
    preflight_check "TARGET_CRS covers catchment bbox — ${_crs_check}" "false"
fi

# 7. STAC endpoint reachable (soft warning only — network may be unavailable)
_stac_url=$(python -c "import sys; sys.path.insert(0,'${CODE_DIR}'); import config; print(config.STAC_ENDPOINT_ELEMENT84)" 2>/dev/null || echo "")
if [[ -n "${_stac_url}" ]] && curl --silent --max-time 5 --head "${_stac_url}" -o /dev/null 2>/dev/null; then
    preflight_check "STAC endpoint reachable (${_stac_url})" "true"
else
    printf "  ${YELLOW}⚠${RESET} STAC endpoint unreachable (%s) — Steps 01–04 will fail if network is unavailable\n" "${_stac_url}"
fi

# 8. Prior year probability raster (only for year > baseline)
PRIOR_YEAR=$(( YEAR - 1 ))
if [[ ${YEAR} -gt 2020 ]]; then
    PRIOR_PROB="$(python -c "
import sys, os
sys.path.insert(0, '${CODE_DIR}')
import config
print(config.probability_raster_path(${PRIOR_YEAR}))
" 2>/dev/null || echo "")"
    if [[ -n "${PRIOR_PROB}" && -f "${PRIOR_PROB}" ]]; then
        preflight_check "Prior year probability raster exists (${PRIOR_YEAR})" "true"
    else
        # Not a hard failure for year 1 of a new run, just warn
        printf "  ${YELLOW}⚠${RESET} Prior year probability raster not found (%s) — Step 07 will skip change detection\n" "${PRIOR_YEAR}"
    fi
fi

if [[ "${preflight_failed}" == "true" ]]; then
    echo ""
    printf "${RED}${BOLD}Pre-flight checks failed. Aborting.${RESET}\n"
    exit 3
fi

echo ""

# ── Force: clear sentinels ────────────────────────────────────────────────────
if [[ "${FORCE}" == "true" ]]; then
    printf "${YELLOW}Clearing all sentinels for year %s...${RESET}\n" "${YEAR}"
    rm -f "${WORKING_DIR}"/.step_*_complete_${YEAR}_* 2>/dev/null || true
    echo ""
fi

# ── Dry run ───────────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run — no scripts will be executed."
    echo "Steps that would run:"
    for step_num in 1 2 3 4 5 6 7; do
        printf "  Step %02d\n" "${step_num}"
    done
    exit 0
fi

# ── Step timing tracking ──────────────────────────────────────────────────────
declare -a STEP_NAMES
declare -a STEP_DURATIONS
declare -a STEP_STATUSES

# ── Step runner ───────────────────────────────────────────────────────────────
# run_step STEP_NUM ANALYSIS_NAME VERIFY_NAME
run_step() {
    local step_num="$1"
    local analysis_name="$2"
    local verify_name="$3"
    local step_nn
    step_nn="$(printf '%02d' "${step_num}")"

    STEP_NAMES+=("${analysis_name}")

    # Apply --only-step filter
    if [[ -n "${ONLY_STEP}" && "${step_num}" != "${ONLY_STEP}" ]]; then
        printf "${CYAN}[SKIP]${RESET} Step %s — not selected (--only-step %s)\n" "${step_nn}" "${ONLY_STEP}"
        STEP_STATUSES+=("SKIP")
        STEP_DURATIONS+=("—")
        return 0
    fi

    # Apply --from-step filter
    if [[ "${step_num}" -lt "${FROM_STEP}" ]]; then
        printf "${CYAN}[SKIP]${RESET} Step %s — before --from-step %s\n" "${step_nn}" "${FROM_STEP}"
        STEP_STATUSES+=("SKIP")
        STEP_DURATIONS+=("—")
        return 0
    fi

    # Check sentinel (auto-resume), unless --from-step overrides
    local sentinel="${WORKING_DIR}/.step_${step_nn}_complete_${YEAR}_${GIT_SHA}"
    if [[ -f "${sentinel}" && "${step_num}" -ge "${FROM_STEP}" && "${FROM_STEP}" == "1" ]]; then
        printf "${CYAN}[SKIP]${RESET} Step %s — already complete (sentinel found)\n" "${step_nn}"
        STEP_STATUSES+=("SKIP")
        STEP_DURATIONS+=("cached")
        return 0
    fi

    local analysis_script="${CODE_DIR}/analysis/${analysis_name}.py"
    local verify_script="${CODE_DIR}/verify/${verify_name}.py"

    printf "\n${BOLD}${CYAN}── Step %s: %s ──────────────────────────────────────────${RESET}\n" "${step_nn}" "${analysis_name}"

    local t_start
    t_start="$(date +%s)"

    # Run analysis script
    local stderr_file
    stderr_file="$(mktemp)"
    local exit_code=0

    if ! python "${analysis_script}" 2>"${stderr_file}"; then
        exit_code=$?
        echo ""
        printf "${RED}${BOLD}FAILED: analysis/${analysis_name}.py (exit ${exit_code})${RESET}\n"
        echo "Last 20 lines of stderr:"
        printf "${RED}"
        tail -20 "${stderr_file}"
        printf "${RESET}"
        rm -f "${stderr_file}"
        STEP_STATUSES+=("FAIL")
        local t_end
        t_end="$(date +%s)"
        STEP_DURATIONS+=("$(( t_end - t_start ))s")
        return 1
    fi
    rm -f "${stderr_file}"

    # Run verify script
    stderr_file="$(mktemp)"
    local verify_exit=0

    if ! python "${verify_script}" 2>"${stderr_file}"; then
        verify_exit=$?
        echo ""
        printf "${RED}${BOLD}FAILED: verify/${verify_name}.py (exit ${verify_exit}) — science checks did not pass${RESET}\n"
        echo "Last 20 lines of stderr:"
        printf "${RED}"
        tail -20 "${stderr_file}"
        printf "${RESET}"
        rm -f "${stderr_file}"
        STEP_STATUSES+=("VERIFY_FAIL")
        local t_end
        t_end="$(date +%s)"
        STEP_DURATIONS+=("$(( t_end - t_start ))s")
        return 2
    fi
    rm -f "${stderr_file}"

    # Write sentinel
    touch "${sentinel}"

    local t_end
    t_end="$(date +%s)"
    local duration=$(( t_end - t_start ))
    STEP_DURATIONS+=("${duration}s")
    STEP_STATUSES+=("PASS")

    printf "${GREEN}[PASS]${RESET} Step %s completed in %ds\n" "${step_nn}" "${duration}"
    return 0
}

# ── Summary table ─────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"
    printf "${BOLD}  Run Summary — Year %s${RESET}\n" "${YEAR}"
    printf "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"
    printf "  %-5s  %-35s  %-10s  %s\n" "Step" "Name" "Duration" "Status"
    printf "  %-5s  %-35s  %-10s  %s\n" "-----" "-----------------------------------" "----------" "------"

    for i in "${!STEP_NAMES[@]}"; do
        status="${STEP_STATUSES[$i]}"
        name="${STEP_NAMES[$i]}"
        dur="${STEP_DURATIONS[$i]}"
        step_n=$(( i + 1 ))
        nn="$(printf '%02d' "${step_n}")"
        case "${status}" in
            PASS)        colour="${GREEN}" ;;
            FAIL|VERIFY_FAIL) colour="${RED}" ;;
            SKIP)        colour="${CYAN}" ;;
            *)           colour="${RESET}" ;;
        esac
        printf "  %-5s  %-35s  %-10s  ${colour}%s${RESET}\n" "${nn}" "${name}" "${dur}" "${status}"
    done

    echo ""
    printf "  Log: %s\n" "${LOG_FILE}"
    echo ""

    if [[ "${OVERALL_EXIT}" -eq 0 ]]; then
        printf "${GREEN}${BOLD}Pipeline completed successfully.${RESET}\n"
    elif [[ "${OVERALL_EXIT}" -eq 2 ]]; then
        printf "${RED}${BOLD}Pipeline failed: science verification checks did not pass (exit 2).${RESET}\n"
    else
        printf "${RED}${BOLD}Pipeline failed: analysis script crashed (exit 1).${RESET}\n"
    fi
}

# ── Run all steps ─────────────────────────────────────────────────────────────
OVERALL_EXIT=0

run_step_or_abort() {
    local code=0
    run_step "$@" || code=$?
    if [[ $code -ne 0 ]]; then
        OVERALL_EXIT=$code
        print_summary
        exit "${OVERALL_EXIT}"
    fi
}

run_step_or_abort 1 "01_ndvi_composite"    "01_verify_ndvi_composite"
run_step_or_abort 2 "02_ndvi_anomaly"      "02_verify_ndvi_anomaly"
run_step_or_abort 3 "03_flowering_index"   "03_verify_flowering_index"
run_step_or_abort 4 "04_flood_extent"      "04_verify_flood_extent"
run_step_or_abort 5 "05_classifier"        "05_verify_classifier"
run_step_or_abort 6 "06_priority_patches"  "06_verify_priority_patches"
run_step_or_abort 7 "07_change_detection"  "07_verify_change_detection"

print_summary

# Git tag suggestion (only on success)
if [[ "${OVERALL_EXIT}" -eq 0 ]]; then
    SAFE_YEAR="${YEAR}"
    echo ""
    printf "${YELLOW}Suggested git tag (not auto-applied):${RESET}\n"
    printf "  git tag -a v%s-run-%s -m 'Pipeline run for %s'\n" "${SAFE_YEAR}" "${TIMESTAMP}" "${SAFE_YEAR}"
    echo ""
fi

exit "${OVERALL_EXIT}"
