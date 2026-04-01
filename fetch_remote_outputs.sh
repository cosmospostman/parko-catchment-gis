#!/usr/bin/env bash
# fetch_remote_outputs.sh — copy stage 1–4 outputs from remote hosts to this machine.
#
# Each remote host may have run any subset of stages 1–4. This script attempts to
# copy whatever outputs exist, skipping gracefully when files are absent.
#
# Usage:
#   ./fetch_remote_outputs.sh YEAR HOST1 KEY1 [HOST2 KEY2 ...]
#
# Examples:
#   ./fetch_remote_outputs.sh 2024 ec2-1.example.com ~/.ssh/id_stage1 ec2-2.example.com ~/.ssh/id_stage4
#
# Environment overrides (optional, mirror config.sh):
#   BASE_DIR         local destination root  (default: /data/mrc-parko)
#   REMOTE_BASE_DIR  remote source root      (default: same as BASE_DIR)
#   SSH_USER         remote username         (default: ubuntu)
#   SSH_PORT         remote SSH port         (default: 22)
#   SCP_OPTS         extra scp options       (default: -q)

set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 YEAR HOST1 KEY1 [HOST2 KEY2 ...]" >&2
    echo "  YEAR must be a 4-digit year (e.g. 2024)" >&2
    echo "  HOST/KEY pairs are processed left-to-right" >&2
    exit 1
}

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { log "INFO  $*"; }
warn() { log "WARN  $*" >&2; }
err()  { log "ERROR $*" >&2; }

# Try to copy a single remote path; returns 0 on success, 1 if not found, 2 on error.
try_scp() {
    local host="$1" key="$2" remote_path="$3" local_dest="$4"
    # Use user@host as-is if @ is present, otherwise prepend SSH_USER
    local target
    [[ "${host}" == *@* ]] && target="${host}" || target="${SSH_USER}@${host}"
    # shellcheck disable=SC2086
    if scp ${SCP_OPTS} -i "${key}" -P "${SSH_PORT}" \
           -r "${target}:${remote_path}" "${local_dest}" 2>/tmp/_scp_err; then
        return 0
    fi
    local msg
    msg=$(cat /tmp/_scp_err)
    # scp exits non-zero for both "file not found" and real errors; distinguish them.
    if echo "${msg}" | grep -qiE 'No such file|not found|no such'; then
        return 1  # simply absent — not an error
    fi
    warn "scp error for ${host}:${remote_path}: ${msg}"
    return 2
}

# ── argument parsing ──────────────────────────────────────────────────────────

[[ $# -lt 3 ]] && usage

YEAR="$1"; shift
[[ "${YEAR}" =~ ^[0-9]{4}$ ]] || { err "YEAR must be 4 digits, got '${YEAR}'"; usage; }

# Collect (host, key) pairs
declare -a HOSTS=()
declare -a KEYS=()
while [[ $# -ge 2 ]]; do
    HOSTS+=("$1")
    KEYS+=("$2")
    shift 2
done
[[ $# -gt 0 ]] && { err "Odd number of HOST/KEY arguments"; usage; }
[[ ${#HOSTS[@]} -eq 0 ]] && usage

# ── configuration ─────────────────────────────────────────────────────────────

BASE_DIR="${BASE_DIR:-/data/mrc-parko}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-${BASE_DIR}}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SCP_OPTS="${SCP_OPTS:--q}"

LOCAL_OUTPUTS="${BASE_DIR}/outputs/${YEAR}"
LOCAL_CACHE="${BASE_DIR}/cache"

# Ensure local destination directories exist
mkdir -p "${LOCAL_OUTPUTS}" "${LOCAL_CACHE}"

# ── file lists per stage ──────────────────────────────────────────────────────
#
# Each entry is a remote path relative to REMOTE_BASE_DIR.
# Quicklook PNGs are included; sentinel/working files are not needed to run
# stages 5-7 locally.

STAGE1_FILES=(
    "outputs/${YEAR}/ndvi_median_${YEAR}.tif"
    "outputs/${YEAR}/ndvi_median_${YEAR}_quicklook.png"
)

STAGE2_FILES=(
    "outputs/${YEAR}/ndvi_anomaly_${YEAR}.tif"
    "outputs/${YEAR}/ndvi_anomaly_${YEAR}_quicklook.png"
    "cache/ndvi_baseline_median.tif"     # reusable; copy if present
)

STAGE3_FILES=(
    "outputs/${YEAR}/flowering_index_${YEAR}.tif"
    "outputs/${YEAR}/flowering_index_${YEAR}_quicklook.png"
)

STAGE4_FILES=(
    "outputs/${YEAR}/flood_extent_${YEAR}.gpkg"
    "outputs/${YEAR}/hand_${YEAR}.tif"
    "outputs/${YEAR}/flood_obs_count_${YEAR}.tif"
    "outputs/${YEAR}/flood_extent_${YEAR}_quicklook.png"
)

# Also fetch the catchment boundary if present (needed by all stages)
COMMON_FILES=(
    "mitchell_catchment.geojson"
)

# ── per-host fetch ────────────────────────────────────────────────────────────

total_copied=0
total_skipped=0
total_errors=0

# Track which files were never obtained from any host (1 = still missing)
declare -A still_missing
for _f in "${COMMON_FILES[@]}" "${STAGE1_FILES[@]}" "${STAGE2_FILES[@]}" "${STAGE3_FILES[@]}" "${STAGE4_FILES[@]}"; do
    still_missing["${_f}"]=1
done
unset _f

for idx in "${!HOSTS[@]}"; do
    HOST="${HOSTS[$idx]}"
    KEY="${KEYS[$idx]}"

    info "─── Host ${HOST} (key: ${KEY}) ───────────────────────────────────────"

    # Validate key file exists
    if [[ ! -f "${KEY}" ]]; then
        err "SSH key not found: ${KEY} — skipping host ${HOST}"
        (( total_errors++ )) || true
        continue
    fi

    host_copied=0
    host_skipped=0
    host_errors=0

    # Build full list: common files + all stage files
    ALL_FILES=("${COMMON_FILES[@]}" "${STAGE1_FILES[@]}" "${STAGE2_FILES[@]}" "${STAGE3_FILES[@]}" "${STAGE4_FILES[@]}")

    for rel_path in "${ALL_FILES[@]}"; do
        remote_full="${REMOTE_BASE_DIR}/${rel_path}"
        local_dir="${BASE_DIR}/$(dirname "${rel_path}")"
        mkdir -p "${local_dir}"
        local_file="${BASE_DIR}/${rel_path}"

        if [[ -f "${local_file}" ]]; then
            info "  ${rel_path}  →  exists (skipped)"
            (( host_skipped++ )) || true
            # Mark as satisfied so it doesn't appear in the missing report
            still_missing["${rel_path}"]=0
            continue
        fi

        info "  ${rel_path}  →  copying..."
        result=0
        try_scp "${HOST}" "${KEY}" "${remote_full}" "${local_dir}/" || result=$?

        case ${result} in
            0)
                info "  ${rel_path}  →  done"
                still_missing["${rel_path}"]=0
                (( host_copied++ )) || true
                ;;
            1)
                info "  ${rel_path}  →  not on this host"
                (( host_skipped++ )) || true
                ;;
            2)
                warn "  ${rel_path}  →  ERROR"
                (( host_errors++ )) || true
                ;;
        esac
    done

    info "  Host ${HOST} summary: copied=${host_copied} skipped=${host_skipped} errors=${host_errors}"
    (( total_copied  += host_copied  )) || true
    (( total_skipped += host_skipped )) || true
    (( total_errors  += host_errors  )) || true
done

# ── summary ───────────────────────────────────────────────────────────────────

echo ""
info "═══════════════════════════════════════════════════════"
info "Fetch complete for year ${YEAR}"
info "  Hosts processed : ${#HOSTS[@]}"
info "  Files copied    : ${total_copied}"
info "  Skipped/exists  : ${total_skipped}"
info "  Errors          : ${total_errors}"
info "  Local outputs   : ${LOCAL_OUTPUTS}"

missing_list=()
for rel_path in "${!still_missing[@]}"; do
    [[ "${still_missing[$rel_path]}" -ne 0 ]] && missing_list+=("  ${rel_path}")
done
if [[ ${#missing_list[@]} -gt 0 ]]; then
    info "  Still missing from all hosts:"
    for m in "${missing_list[@]}"; do info "${m}"; done
fi
info "═══════════════════════════════════════════════════════"

if [[ ${total_errors} -gt 0 ]]; then
    warn "Some files failed to copy. Check warnings above."
    exit 1
fi
