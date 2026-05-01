"""sweep_zscore_etna_landsend.py — Hyperparam sweep over v8_s1_zscore_nr_etna_landsend.

Grid:
  lr:      1e-5, 5e-5
  d_model: 64, 128

= 4 runs total. All other settings identical to v8_s1_zscore_nr_etna_landsend.

Usage:
    python sweep_zscore_etna_landsend.py
    python sweep_zscore_etna_landsend.py --out outputs/sweep_zscore_etna_landsend
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sweep_zscore")


def run_one(run_id: str, out_dir: Path, lr: float, d_model: int) -> float | None:
    """Train one variant and return best val_auc, or None on failure."""
    import importlib
    import pyarrow.parquet as pq
    import pandas as pd

    from tam.core.config import TAMConfig
    from tam.core.train import train_tam
    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    exp = importlib.import_module("tam.experiments.v8_s1_zscore_nr_etna_landsend").EXPERIMENT

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cfg = TAMConfig(
        d_model=d_model,
        n_layers=2,
        dropout=0.5,
        n_bands=4,
        n_global_features=0,
        lr=lr,
        weight_decay=0.1,
        n_epochs=60,
        patience=15,
        band_noise_std=0.0,
        obs_dropout_min=4,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1="s1_only",
    )

    logger.info("=== %s: lr=%s d_model=%d ===", run_id, lr, d_model)

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
        s1_cols = [c for c in ("source", "vh", "vv", "scl") if c in available]
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    dates = pd.to_datetime(pixel_df["date"])
    pixel_df["year"] = dates.dt.year
    pixel_df["doy"]  = dates.dt.day_of_year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    labels = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    # Capture training logs to file so we can extract val_auc
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        train_tam(
            pixel_df=pixel_df[pixel_df["point_id"].isin(labels.index)],
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
    for line in log_path.read_text().splitlines():
        if "Best val AUC:" in line:
            try:
                best_val = float(line.split("Best val AUC:")[1].split("—")[0].strip())
            except Exception:
                pass

    logger.info("=== %s done: best_val_auc=%s ===", run_id, best_val)
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/sweep_zscore_etna_landsend")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    # Add file logging for the sweep itself
    logging.getLogger().addHandler(logging.FileHandler(base_out / "sweep.log"))

    grid = [
        # (lr,   d_model)
        (1e-5,  64),
        (5e-5,  64),
        (1e-5, 128),
        (5e-5, 128),
    ]

    summary_path = base_out / "summary.tsv"
    header = "run_id\tlr\td_model\tval_auc"
    print(header.replace("\t", "  "))
    with open(summary_path, "w") as fh:
        fh.write(header + "\n")

    for lr, d_model in grid:
        run_id = f"lr{lr:.0e}_dm{d_model}"
        out_dir = base_out / run_id

        if args.dry_run:
            logger.info("DRY RUN — would run %s", run_id)
            continue

        val_auc = run_one(run_id, out_dir, lr, d_model)
        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id}\t{lr:.0e}\t{d_model}\t{val_str}"
        print(row.replace("\t", "  "))
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")

    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
