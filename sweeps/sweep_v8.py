"""sweep_v8.py — Sweep over S1 feature variants for v8 Norman Road experiment.

Grid explores:
  - use_s1: True / False          (S1 time series on/off)
  - s1_globals: True / False      (S1 global features on/off)
  - lr: 1e-5 / 5e-5               (two learning rates)

= 8 runs total.

Usage:
    python sweep_v8.py
    python sweep_v8.py --out outputs/models/sweep_v8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sweep_v8")


def run_one(
    run_id: str,
    out_dir: Path,
    use_s1: bool,
    s1_globals: bool,
    lr: float,
) -> float | None:
    """Train one variant and return best val_auc, or None on failure."""
    import importlib
    import torch
    import pandas as pd

    from tam.core.config import TAMConfig
    from tam.core.dataset import ALL_FEATURE_COLS, S1_FEATURE_COLS
    from tam.core.train import train_tam
    from tam.core.global_features import GLOBAL_FEATURE_NAMES
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    exp_mod = importlib.import_module("tam.experiments.v8")
    exp = exp_mod.EXPERIMENT

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    feature_cols = ALL_FEATURE_COLS + S1_FEATURE_COLS if use_s1 else ALL_FEATURE_COLS
    n_bands = len(feature_cols)  # 17 or 13

    # S1 globals: last 4 of GLOBAL_FEATURE_NAMES; S2 globals: first 5
    n_global = 9 if s1_globals else 5

    cfg = TAMConfig(
        # Architecture — match v8 defaults
        d_model=64,
        n_layers=2,
        dropout=0.5,
        n_bands=n_bands,
        n_global_features=n_global,
        use_s1=use_s1,
        # Training
        lr=lr,
        weight_decay=0.1,
        n_epochs=60,
        patience=10,
        band_noise_std=0.03,
        obs_dropout_min=6,
        doy_density_norm=True,
    )

    logger.info("=== %s: use_s1=%s s1_globals=%s lr=%s ===", run_id, use_s1, s1_globals, lr)

    # Load pixels
    import pyarrow.parquet as pq
    import pandas as pd

    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.error("Missing tile parquet: %s", path)
            return None
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        s1_cols = [c for c in ("source", "vh", "vv") if c in available]
        base = ["point_id", "lon", "lat", "date", "scl_purity", "scl"]
        read_cols = base + [c for c in feature_cols if c in available] + s1_cols
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year
    pixel_df["doy"] = pd.to_datetime(pixel_df["date"]).dt.day_of_year

    # Labels
    from tam.utils import label_pixels

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    labelled = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    labels = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    # Redirect logging to file for this run
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        model = train_tam(
            pixel_df=pixel_df,
            labels=labels,
            pixel_coords=pixel_coords,
            out_dir=out_dir,
            cfg=cfg,
        )
    except Exception as exc:
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        return None
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    # Extract best val_auc from log
    best_val = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if "val_auc=" in line and "*" in line:
                try:
                    best_val = float(line.split("val_auc=")[1].split()[0])
                except Exception:
                    pass

    logger.info("=== %s done: best_val_auc=%s ===", run_id, best_val)
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/models/sweep_v8")
    args = parser.parse_args()

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    runs = [
        # (use_s1, s1_globals, lr)
        (False, False, 1e-5),   # baseline: S2 only, no S1 globals
        (False, False, 5e-5),   # baseline: higher lr
        (True,  False, 1e-5),   # S1 time series only, no S1 globals
        (True,  False, 5e-5),
        (False, True,  1e-5),   # S1 globals only, no S1 time series
        (False, True,  5e-5),
        (True,  True,  1e-5),   # S1 time series + globals (full v8)
        (True,  True,  5e-5),
    ]

    summary_path = base_out / "summary.txt"
    header = f"{'run_id':<40}  {'use_s1':<7}  {'s1_glob':<8}  {'lr':<8}  {'val_auc'}"
    print(header)
    with open(summary_path, "w") as fh:
        fh.write(header + "\n")

    for use_s1, s1_globals, lr in runs:
        run_id = f"s1ts{'1' if use_s1 else '0'}_s1g{'1' if s1_globals else '0'}_lr{lr:.0e}"
        out_dir = base_out / run_id
        val_auc = run_one(run_id, out_dir, use_s1, s1_globals, lr)
        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id:<40}  {str(use_s1):<7}  {str(s1_globals):<8}  {lr:<8.0e}  {val_str}"
        print(row)
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
