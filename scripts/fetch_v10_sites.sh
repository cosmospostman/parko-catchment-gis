#!/usr/bin/env bash
# Fetch v10 training regions site by site.
# Continues on failure; writes a summary log at the end.

set -euo pipefail
PYTHON=".venv/bin/python"
CLI="cli/location.py"
LOG="outputs/fetch_v10_sites.log"

mkdir -p outputs

declare -A SITES
SITES[landsend]="landsend_presence_1 landsend_presence_2 landsend_presence_3 landsend_presence_4 landsend_presence_5 landsend_presence_6 landsend_presence_7 landsend_presence_8 landsend_sparse_presence_1 landsend_sparse_presence_2 landsend_sparse_presence_3 landsend_sparse_presence_4 landsend_sparse_presence_5 landsend_absence_1 landsend_absence_2 landsend_absence_3 landsend_absence_4 landsend_absence_5 landsend_absence_grass_1 landsend_absence_grass_2 landsend_absence_riverbed_1 landsend_absence_riverbed_2 landsend_absence_riverbed_3"
SITES[norman_road]="norman_road_presence_1 norman_road_presence_2 norman_road_presence_3 norman_road_presence_4 norman_road_presence_5 norman_road_presence_6 norman_road_presence_7 norman_road_presence_8 norman_road_presence_9 norman_road_absence_1 norman_road_absence_2 norman_road_absence_3 norman_road_absence_4 norman_road_absence_5 norman_road_absence_7"
SITES[quaids]="quaids_absence_1 quaids_absence_3 quaids_absence_5 quaids_absence_7 quaids_absence_9 quaids_val_absence_2 quaids_val_absence_4 quaids_val_absence_6 quaids_val_absence_8a quaids_val_absence_8b quaids_val_absence_10"
SITES[roper]="roper_presence_1 roper_presence_2 roper_presence_3 roper_presence_4 roper_absence_1 roper_absence_2 roper_absence_3"
SITES[rupert_ck]="rupert_ck_presence_1 rupert_ck_presence_2 rupert_ck_presence_3 rupert_ck_presence_sparse_1 rupert_ck_absence_1 rupert_ck_absence_2 rupert_ck_absence_3 rupert_ck_val_presence_1 rupert_ck_val_absence_1"
SITES[etna]="etna_presence_1 etna_presence_2 etna_presence_3 etna_presence_4 etna_presence_5 etna_presence_6 etna_presence_7 etna_presence_8 etna_presence_9 etna_absence_1 etna_absence_2 etna_absence_3 etna_absence_4 etna_absence_5 etna_absence_6 etna_absence_7 etna_absence_8 etna_absence_9 etna_absence_10 etna_absence_11 etna_absence_12"

ORDERED=(landsend norman_road quaids roper rupert_ck etna)

declare -A RESULTS
START_ALL=$(date +%s)

echo "===== fetch_v10_sites $(date '+%Y-%m-%d %H:%M:%S') =====" | tee "$LOG"

for site in "${ORDERED[@]}"; do
    regions="${SITES[$site]}"
    n=$(echo "$regions" | wc -w)
    echo "" | tee -a "$LOG"
    echo "--- $site ($n regions) ---" | tee -a "$LOG"
    t0=$(date +%s)

    if $PYTHON "$CLI" training fetch --regions $regions 2>&1 | tee -a "$LOG"; then
        elapsed=$(( $(date +%s) - t0 ))
        RESULTS[$site]="OK  (${elapsed}s)"
        echo "[OK]  $site completed in ${elapsed}s" | tee -a "$LOG"
    else
        elapsed=$(( $(date +%s) - t0 ))
        RESULTS[$site]="FAIL (${elapsed}s)"
        echo "[FAIL] $site failed after ${elapsed}s — continuing" | tee -a "$LOG"
    fi
done

ELAPSED_ALL=$(( $(date +%s) - START_ALL ))

echo "" | tee -a "$LOG"
echo "===== SUMMARY (total ${ELAPSED_ALL}s) =====" | tee -a "$LOG"
for site in "${ORDERED[@]}"; do
    printf "  %-16s %s\n" "$site" "${RESULTS[$site]}" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
