#!/usr/bin/env python3
"""Repeatedly trains v4_spectral overnight, writing logs to outputs/train-loop/<iteration>/."""

import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

OUT_ROOT = Path("outputs/train-loop")
SUMMARY_LOG = OUT_ROOT / "summary.log"

# Search ranges (log-uniform for lr, uniform for the rest)
LR_RANGE      = (3e-5, 3e-4)
DROPOUT_RANGE = (0.1, 0.5)
N_LAYERS_OPTS = [1, 2, 3]


def sample_hparams() -> dict:
    log_lo, log_hi = [__import__("math").log(x) for x in LR_RANGE]
    lr = __import__("math").exp(random.uniform(log_lo, log_hi))
    return {
        "lr":       round(lr, 6),
        "dropout":  round(random.uniform(*DROPOUT_RANGE), 2),
        "n_layers": random.choice(N_LAYERS_OPTS),
    }


def parse_best_auc(log_path: Path) -> str:
    """Extract best val AUC from the training log, or '?' if not found."""
    try:
        text = log_path.read_text()
        matches = re.findall(r"Best val AUC:\s*([\d.]+)", text)
        return matches[-1] if matches else "?"
    except OSError:
        return "?"


def parse_stopped_epoch(log_path: Path) -> str:
    """Return the epoch training stopped at, or '?' if not found."""
    try:
        text = log_path.read_text()
        # Early stopping line
        m = re.search(r"Early stopping at epoch (\d+)", text)
        if m:
            return m.group(1)
        # Last epoch line: "epoch  N/20"
        matches = re.findall(r"epoch\s+(\d+)/\d+", text)
        return matches[-1] if matches else "?"
    except OSError:
        return "?"


def run_iteration(iteration: int) -> None:
    out_dir = OUT_ROOT / str(iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    hparams = sample_hparams()
    cmd = [
        sys.executable, "-m", "tam.pipeline", "train",
        "--experiment", "v4_spectral",
        "--output-dir", str(out_dir),
        "--lr",       str(hparams["lr"]),
        "--dropout",  str(hparams["dropout"]),
        "--n-layers", str(hparams["n_layers"]),
    ]

    started = datetime.now().isoformat(timespec="seconds")
    hparam_str = f"lr={hparams['lr']}  dropout={hparams['dropout']}  n_layers={hparams['n_layers']}"
    print(f"[{started}] iteration {iteration}  {hparam_str}", flush=True)

    with log_path.open("w") as log:
        log.write(f"started: {started}\ncmd: {' '.join(cmd)}\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    finished = datetime.now().isoformat(timespec="seconds")
    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"

    with log_path.open("a") as log:
        log.write(f"\nfinished: {finished}  status: {status}\n")

    best_auc   = parse_best_auc(log_path)
    stopped_at = parse_stopped_epoch(log_path)

    summary_line = (
        f"{started}  iter={iteration:>3}  "
        f"lr={hparams['lr']:<10}  dropout={hparams['dropout']:<5}  n_layers={hparams['n_layers']}  "
        f"best_auc={best_auc:<7}  stopped_epoch={stopped_at:<4}  status={status}\n"
    )
    SUMMARY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_LOG.open("a") as f:
        f.write(summary_line)

    print(f"[{finished}] iteration {iteration}  best_auc={best_auc}  stopped_epoch={stopped_at}  {status}", flush=True)


def main() -> None:
    iteration = 0
    while True:
        run_iteration(iteration)
        iteration += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
