"""sweep_multisite_phase.py — Hyperparameter sweep for multi-site phase-shift S1 model.

Grid:
  lr:      1e-5, 5e-6, 2e-6
  dropout: 0.5, 0.6, 0.7
  d_model: 64, 128

= 18 runs total. All runs use:
  - NR + Cloncurry + Barcoorah + Stockholm training
  - Lake Mueller holdout (val_sites)
  - doy_phase_shift=True
  - S1-only temporal (4 bands, no globals)
  - patience=20, n_epochs=80

Results written to outputs/models/sweep_multisite_phase/summary.csv

Usage:
    python sweep_multisite_phase.py
    python sweep_multisite_phase.py --out outputs/models/sweep_multisite_phase
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sweep_multisite_phase")

REGION_IDS = [
    "norman_road_presence_1", "norman_road_presence_2",
    "norman_road_presence_3", "norman_road_presence_4",
    "norman_road_presence_5", "norman_road_presence_6",
    "norman_road_presence_7", "norman_road_presence_8",
    "norman_road_presence_9",
    "norman_road_absence_1", "norman_road_absence_2",
    "norman_road_absence_3", "norman_road_absence_4",
    "norman_road_absence_5", "norman_road_absence_7",
    "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
    "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
    "cloncurry_absence_7",
    "barcoorah_presence", "barcoorah_presence_2", "barcoorah_presence_3",
    "barcoorah_absence_lake", "barcoorah_absence_woodland",
    "barcoorah_absence_2", "barcoorah_absence_3",
    "stockholm_presence_1", "stockholm_presence_2",
    "stockholm_absence_1",
    "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
    "lake_mueller_absence", "lake_mueller_absence_2",
]


def load_pixels(region_ids: list[str]) -> tuple:
    import pyarrow.parquet as pq
    import pandas as pd
    from tam.core.dataset import S1_FEATURE_COLS
    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    regions = select_regions(region_ids)
    tile_ids = tile_ids_for_regions(region_ids)

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            raise FileNotFoundError(f"Missing tile parquet: {path}")
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        s1_cols = [c for c in ("source", "vh", "vv") if c in available]
        base = ["point_id", "lon", "lat", "date", "scl_purity", "scl"]
        read_cols = base + [c for c in S1_FEATURE_COLS if c in available] + s1_cols
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    labelled = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    labels = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    return pixel_df, labels, pixel_coords


def run_one(
    run_id: str,
    out_dir: Path,
    pixel_df,
    labels,
    pixel_coords,
    lr: float,
    dropout: float,
    d_model: int,
) -> float | None:
    from tam.core.config import TAMConfig
    from tam.core.train import train_tam

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    d_ff = d_model  # keep d_ff == d_model for simplicity

    cfg = TAMConfig(
        d_model=d_model,
        n_layers=2,
        dropout=dropout,
        d_ff=d_ff,
        n_bands=4,
        n_global_features=0,
        use_s1="s1_only",
        lr=lr,
        weight_decay=0.1,
        n_epochs=80,
        patience=20,
        band_noise_std=0.0,
        obs_dropout_min=4,
        doy_density_norm=True,
        doy_phase_shift=True,
        val_sites=("lake_mueller",),
    )

    logger.info("=== %s: lr=%.0e dropout=%.1f d_model=%d ===", run_id, lr, dropout, d_model)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        train_tam(
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

    best_val = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if "Best val AUC" in line:
                try:
                    best_val = float(line.split("Best val AUC:")[1].split()[0])
                except Exception:
                    pass

    logger.info("=== %s done: best_val_auc=%s ===", run_id, best_val)
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/models/sweep_multisite_phase")
    args = parser.parse_args()

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pixels once for all runs...")
    pixel_df, labels, pixel_coords = load_pixels(REGION_IDS)
    logger.info("Pixels loaded: %d observations, %d labelled pixels", len(pixel_df), len(labels))

    lrs      = [1e-5, 5e-6, 2e-6]
    dropouts = [0.5, 0.6, 0.7]
    d_models = [64, 128]

    grid = list(product(lrs, dropouts, d_models))
    logger.info("Total runs: %d", len(grid))

    summary_path = base_out / "summary.csv"
    fieldnames = ["run_id", "lr", "dropout", "d_model", "val_auc"]
    with open(summary_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    for lr, dropout, d_model in grid:
        run_id = f"lr{lr:.0e}_do{dropout:.1f}_dm{d_model}"
        out_dir = base_out / run_id
        val_auc = run_one(run_id, out_dir, pixel_df, labels, pixel_coords, lr, dropout, d_model)
        row = {"run_id": run_id, "lr": lr, "dropout": dropout, "d_model": d_model,
               "val_auc": f"{val_auc:.4f}" if val_auc is not None else "FAILED"}
        with open(summary_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)
        logger.info("Summary updated: %s → %s", run_id, row["val_auc"])

    logger.info("Sweep complete. Results: %s", summary_path)


if __name__ == "__main__":
    main()
