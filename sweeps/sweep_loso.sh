#!/usr/bin/env bash
# Leave-one-site-out sweep for v6_spectral, followed by a train-on-all
# per-site evaluation pass.
#
# Phase 1 — LOSO (12 runs):
#   Hold out each site with both presence and absence in turn as the val set.
#   Hyperparams fixed at best from hyperparameter sweep: lr=2e-6, wd=0.20, dropout=0.7
#   Excluded from LOSO (absence-only or commented out of experiment):
#     quaids, mitchell_river, muttaburra, barkly, ranken_river
#
# Phase 2 — train-on-all (1 training run + 1 eval pass over all sites):
#   Train once with no site held out (spatial split val).
#   Evaluate each site's pixels against the saved checkpoint via eval_per_site.py.

set -euo pipefail

EXPERIMENT="v6_spectral"
BASE_OUT="outputs/models/sweep_loso"
LOG_DIR="$BASE_OUT/logs"
mkdir -p "$LOG_DIR"

LR=2e-6
WD=0.20
DROPOUT=0.7

# Sites with both presence and absence active in v6_spectral
LOSO_SITES=(
    barcoorah
    frenchs
    lake_mueller
    maria_downs
    norman_road
    nassau
    stockholm
    pormpuraaw
    wongalee
    moroak
    roper
    alexandria
)

# All sites in the experiment including absence-only, for per-site eval
ALL_SITES=(
    barcoorah
    frenchs
    lake_mueller
    mitchell_river
    maria_downs
    norman_road
    quaids
    nassau
    muttaburra
    stockholm
    pormpuraaw
    wongalee
    moroak
    roper
    alexandria
)

# ---------------------------------------------------------------------------
# Phase 1 — LOSO
# ---------------------------------------------------------------------------

LOSO_SUMMARY="$BASE_OUT/loso_summary.txt"
printf "%-24s  %-12s  %-12s  %s\n" "held_out_site" "best_val_auc" "train_auc" "n_val_px" > "$LOSO_SUMMARY"

echo "========================================"
echo "Phase 1: Leave-one-site-out (${#LOSO_SITES[@]} runs)"
echo "========================================"

for site in "${LOSO_SITES[@]}"; do

    out_dir="$BASE_OUT/loso_$site"
    log_file="$LOG_DIR/loso_${site}.log"

    echo "----------------------------------------"
    echo "Held-out: $site"

    python -m tam.pipeline train \
        --experiment "$EXPERIMENT" \
        --output-dir "$out_dir" \
        --lr "$LR" \
        --weight-decay "$WD" \
        --dropout "$DROPOUT" \
        --val-sites "$site" \
        2>&1 | tee "$log_file"

    best_line=$(grep "  \*$" "$log_file" | tail -1 || true)
    best=$(echo "$best_line"  | grep -oP 'val_auc=\K[0-9.]+'   || echo "n/a")
    train=$(echo "$best_line" | grep -oP 'train_auc=\K[0-9.]+' || echo "n/a")
    n_val=$(grep -oP 'VAL TOTAL\s+\K[0-9,]+' "$log_file" | tr -d ',' | tail -1 || echo "n/a")

    printf "%-24s  %-12s  %-12s  %s\n" "$site" "$best" "$train" "$n_val" >> "$LOSO_SUMMARY"
    echo "  -> val_auc=$best  train_auc=$train  n_val_px=$n_val"
    echo ""

done

echo "========================================"
echo "Phase 1 complete. LOSO results (sorted by val_auc):"
echo ""
{ head -1 "$LOSO_SUMMARY"; tail -n +2 "$LOSO_SUMMARY" | sort -k2 -rn; }
echo ""

# ---------------------------------------------------------------------------
# Phase 2 — train on all, evaluate per site
# ---------------------------------------------------------------------------

ALL_OUT="$BASE_OUT/train_all"
ALL_LOG="$LOG_DIR/train_all.log"
PERSITE_SUMMARY="$BASE_OUT/persite_summary.txt"

echo "========================================"
echo "Phase 2: Train on all sites (spatial val split)"
echo "========================================"

python -m tam.pipeline train \
    --experiment "$EXPERIMENT" \
    --output-dir "$ALL_OUT" \
    --lr "$LR" \
    --weight-decay "$WD" \
    --dropout "$DROPOUT" \
    2>&1 | tee "$ALL_LOG"

echo ""
echo "Running per-site evaluation..."
echo ""

python -m tam.eval_per_site \
    --checkpoint "$ALL_OUT" \
    --experiment "$EXPERIMENT" \
    --sites "${ALL_SITES[@]}" \
    2>&1 | tee "$PERSITE_SUMMARY"

echo ""
echo "========================================"
echo "Phase 2 complete. Per-site results:"
cat "$PERSITE_SUMMARY"
echo ""
echo "Summaries:"
echo "  LOSO:     $LOSO_SUMMARY"
echo "  Per-site: $PERSITE_SUMMARY"
