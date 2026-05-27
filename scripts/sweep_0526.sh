#!/usr/bin/env bash
# sweep_0526.sh — overnight v10 hyperparameter sweep
#
# Runs 7 training jobs in sequence. Each writes its checkpoint to
# outputs/sweep-0526/<run_name>/. The pixel_df cache is shared via the
# canonical cache dir (outputs/pixel_cache/) — all runs use the same
# feature set and region IDs so the cache key will match.
#
# Expected wall time: ~7 × 60–80 min = 7–9 hours.
#
# Outputs:
#   outputs/sweep-0526/sweep.log     — full training console output
#   outputs/sweep-0526/results.txt   — summary table only
#
# Usage:
#   bash scripts/sweep_0526.sh

set -euo pipefail

SWEEP_DIR="outputs/sweep-0526"
mkdir -p "$SWEEP_DIR"

LOG="$SWEEP_DIR/sweep.log"
RESULTS="$SWEEP_DIR/results.txt"

TRAIN="python -m tam.pipeline train --experiment v10 --batch-size 4096 --max-seq-len 64"

log()  { echo "$@" | tee -a "$LOG"; }
logf() { echo "$@" >> "$LOG"; }  # log-only (no console)

log "=========================================="
log "sweep_0526.sh  started $(date)"
log "=========================================="

run() {
    local name="$1"; shift
    log ""
    log "--- $name  $(date) ---"
    $TRAIN --output-dir "$SWEEP_DIR/$name" "$@" 2>&1 | tee -a "$LOG"
    log "--- $name  done $(date) ---"
}

# 1. Baseline: current experiment defaults (seq=128, p_gate=0.3, dropout=0.5)
#    This is the new reference — different from the 0.860 run (seq=64, no gate).
run baseline

# 2–3. Dropout sweep
run dropout_0.4  --dropout 0.4
run dropout_0.3  --dropout 0.3

# 4. Depth
run layers_4     --n-layers 4

# 5–6. Learning rate
run lr_1e4       --lr 1e-4
run lr_2e5       --lr 2e-5

# 7. Gate augmentation with T_gate=16 (baseline uses T_gate=8)
run gate_aug_t16 --t-gate 16

log ""
log "=========================================="
log "sweep_0526.sh  finished $(date)"
log "=========================================="

# Write results summary to its own file
{
    echo "sweep_0526 results  $(date)"
    echo ""
    printf "%-20s  %s\n" "run" "CVaR25"
    printf "%-20s  %s\n" "--------------------" "------"
    for d in "$SWEEP_DIR"/*/; do
        name=$(basename "$d")
        cfg="$d/tam_config.json"
        if [ -f "$cfg" ]; then
            auc=$(python -c "import json; d=json.load(open('$cfg')); print(f\"{d.get('best_val_auc',0):.4f}\")" 2>/dev/null || echo "?")
        else
            auc="no checkpoint"
        fi
        printf "%-20s  %s\n" "$name" "$auc"
    done
} | tee "$RESULTS"
