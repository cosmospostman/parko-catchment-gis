#!/usr/bin/env bash
# Overnight hyperparameter sweep for v6_spectral.
# Grid: lr × weight_decay × dropout  (2 × 3 × 2 = 12 runs)

set -euo pipefail

EXPERIMENT="v6_spectral"
BASE_OUT="outputs/models/sweep"
LOG_DIR="$BASE_OUT/logs"
mkdir -p "$LOG_DIR"

LRS=(2e-6 5e-6)
WDS=(0.05 0.12 0.20)
DROPOUTS=(0.5 0.7)

SUMMARY="$BASE_OUT/summary.txt"
echo "run_id                          lr        wd     dropout   best_val_auc" > "$SUMMARY"

for lr in "${LRS[@]}"; do
for wd in "${WDS[@]}"; do
for do_ in "${DROPOUTS[@]}"; do

    run_id="lr${lr}_wd${wd}_do${do_}"
    out_dir="$BASE_OUT/$run_id"
    log_file="$LOG_DIR/${run_id}.log"

    echo "========================================"
    echo "Starting: $run_id"
    echo "  lr=$lr  weight_decay=$wd  dropout=$do_"
    echo "  output -> $out_dir"
    echo "  log    -> $log_file"
    echo "========================================"

    python -m tam.pipeline train \
        --experiment "$EXPERIMENT" \
        --output-dir "$out_dir" \
        --lr "$lr" \
        --weight-decay "$wd" \
        --dropout "$do_" \
        2>&1 | tee "$log_file"

    # Extract best val_auc from log (lines marked with *)
    best=$(grep "  \*$" "$log_file" | grep -oP 'val_auc=\K[0-9.]+' | tail -1 || true)
    best=${best:-"n/a"}

    printf "%-32s  %-8s  %-6s  %-8s  %s\n" \
        "$run_id" "$lr" "$wd" "$do_" "$best" >> "$SUMMARY"

    echo "  -> best val_auc: $best"
    echo ""

done
done
done

echo "========================================"
echo "Sweep complete. Results:"
echo ""
sort -k5 -rn "$SUMMARY"
echo ""
echo "Full summary: $SUMMARY"
