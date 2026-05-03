"""run_overnight.py — Run all v8 S1 experiments in logical order overnight.

Skips any experiment whose output directory already contains tam_model.pt.
After each run, writes attention plots and appends a one-line result to
outputs/overnight_results.tsv so progress is readable at a glance.

Usage:
    python run_overnight.py
    python run_overnight.py --dry-run      # print what would run, don't train
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "overnight.log"),
    ],
)
logger = logging.getLogger("overnight")

# ---------------------------------------------------------------------------
# Experiment order — logical sequence from Phase 1 baselines to multisite.
# Each entry: (experiment_name, note)
# ---------------------------------------------------------------------------
EXPERIMENTS: list[tuple[str, str]] = [
    # Phase 1 — single-site baselines (NR + CC)
    ("v8_s1_only",              "baseline temporal-only"),
    ("v8_s1_only_v2",           "tuned hyperparams"),
    ("v8_s1_only_phase",        "phase-shift augmentation"),
    ("v8_s1_only_phase_v2",     "phase-shift stabilised — ALREADY DONE"),
    # Phase 2 — global features
    ("v8_s1_pure",              "S1 temporal + global stats"),
    ("v8_s1_temporal_only",     "S1+S2 temporal, no globals (ablation)"),
    # Phase 3 — arid zone (NR + CC + Lake Mueller)
    ("v8_s1_only_nr_lm",        "baseline + Lake Mueller"),
    ("v8_s1_only_nr_lm_phase",  "phase-shift + Lake Mueller"),
    ("v8_s1_pure_nr_lm",        "pure SAR globals + Lake Mueller"),
    # Phase 4 — cross-zone transfer (holdout)
    ("v8_s1_only_lm",           "LM zero-shot holdout"),
    ("v8_s1_only_bc",           "Barcoorah holdout"),
    ("v8_s1_only_nr_lm_phase_bc", "phase-shift + BC holdout"),
    ("v8_s1_pure_lm",           "pure SAR globals + LM holdout"),
    ("v8_s1_pure_bc",           "pure SAR globals + BC holdout"),
    # Phase 5 — z-score normalisation
    ("v8_s1_zscore_nr",         "z-score baseline NR+CC"),
    ("v8_s1_zscore",            "z-score multisite"),
    # Phase 6 — multisite expansion
    ("v8_s1_phase_multisite",   "4-site phase-shift"),
    ("v8_s1_phase_multisite_v2","5-site phase-shift + Stockholm"),
]

RESULTS_TSV = PROJECT_ROOT / "outputs" / "overnight_results.tsv"


def checkpoint_exists(exp_name: str) -> bool:
    return (PROJECT_ROOT / "outputs" / f"tam-{exp_name}" / "tam_model.pt").exists()


def read_best_auc(exp_name: str) -> str:
    """Scrape best val AUC from the training log in overnight.log."""
    log = PROJECT_ROOT / "outputs" / "overnight.log"
    if not log.exists():
        return "?"
    needle = f"Best val AUC"
    # Read last 500 lines and find the most recent entry for this experiment
    lines = log.read_text().splitlines()
    auc = "?"
    in_exp = False
    for line in lines:
        if f"START {exp_name}" in line:
            in_exp = True
        if in_exp and needle in line:
            try:
                auc = line.split("Best val AUC:")[1].split("—")[0].strip()
            except Exception:
                pass
        if in_exp and f"END {exp_name}" in line:
            in_exp = False
    return auc


def run_experiment(exp_name: str, dry_run: bool) -> bool:
    """Train one experiment. Returns True on success."""
    out_dir = PROJECT_ROOT / "outputs" / f"tam-{exp_name}"

    if checkpoint_exists(exp_name):
        logger.info("SKIP %s — checkpoint already exists", exp_name)
        return True

    logger.info("=" * 70)
    logger.info("START %s", exp_name)
    logger.info("=" * 70)

    if dry_run:
        logger.info("DRY RUN — would run: python -m tam.pipeline train --experiment %s", exp_name)
        return True

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "tam.pipeline", "train", "--experiment", exp_name],
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("FAILED %s (exit %d) after %.0fs", exp_name, result.returncode, elapsed)
        _append_result(exp_name, "FAILED", elapsed)
        return False

    logger.info("END %s  elapsed=%.0fs", exp_name, elapsed)
    return True


def run_attention(exp_name: str, dry_run: bool) -> None:
    out_dir = PROJECT_ROOT / "outputs" / f"tam-{exp_name}"
    attn_dir = out_dir / "attention"
    if attn_dir.exists() and any(attn_dir.glob("*.png")):
        logger.info("SKIP attention %s — already exists", exp_name)
        return

    if dry_run:
        logger.info("DRY RUN — would run viz_attention for %s", exp_name)
        return

    logger.info("Running attention viz for %s", exp_name)
    subprocess.run(
        [sys.executable, "-m", "tam.viz_attention",
         "--checkpoint", str(out_dir),
         "--experiment", exp_name],
        cwd=PROJECT_ROOT,
    )


def _append_result(exp_name: str, status: str, elapsed: float, auc: str = "?") -> None:
    if not RESULTS_TSV.parent.exists():
        RESULTS_TSV.parent.mkdir(parents=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("experiment\tstatus\tbest_val_auc\telapsed_s\tnote\n")
    note = next((n for e, n in EXPERIMENTS if e == exp_name), "")
    with RESULTS_TSV.open("a") as f:
        f.write(f"{exp_name}\t{status}\t{auc}\t{elapsed:.0f}\t{note}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)

    logger.info("Overnight run starting — %d experiments queued", len(EXPERIMENTS))
    if args.dry_run:
        logger.info("DRY RUN MODE")

    total_t0 = time.time()

    for exp_name, note in EXPERIMENTS:
        already_done = checkpoint_exists(exp_name)

        t0 = time.time()
        ok = run_experiment(exp_name, args.dry_run)
        elapsed = time.time() - t0

        if ok and not args.dry_run:
            run_attention(exp_name, args.dry_run)
            auc = read_best_auc(exp_name)
            status = "SKIP" if already_done else "OK"
            _append_result(exp_name, status, elapsed, auc)

    total_elapsed = time.time() - total_t0
    logger.info("All done. Total elapsed: %.0f min", total_elapsed / 60)
    logger.info("Results: %s", RESULTS_TSV)

    # Print the results table to stdout for easy reading
    if RESULTS_TSV.exists():
        print("\n" + "=" * 70)
        print("OVERNIGHT RESULTS")
        print("=" * 70)
        print(RESULTS_TSV.read_text())


if __name__ == "__main__":
    main()
