"""train_one.py — Single training run with explicit hyperparams.

Usage
-----
    python sweeps/train_one.py --out outputs/train-lr5e5 --lr 5e-5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--obs-dropout-min", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log"),
        ],
        force=True,
    )
    log = logging.getLogger("train_one")

    import importlib
    import pyarrow.parquet as pq
    from tam.core.config import TAMConfig
    from tam.core.dataset import V9_FEATURE_COLS
    from tam.core.train import train_tam
    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    exp = importlib.import_module("tam.experiments.v9_spectral").EXPERIMENT
    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            log.error("Missing tile parquet: %s", path)
            sys.exit(1)
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        extra = [c for c in ("source", "scl", "scl_purity") if c in available]
        base = ["point_id", "lon", "lat", "date"]
        read_cols = list(dict.fromkeys(
            base + [c for c in exp.feature_cols if c in available] + extra
        ))
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pl.from_arrow(pf.read_row_group(rg, columns=read_cols)))

    pixel_df = pl.concat(chunks).with_columns([
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
    ])

    pixel_coords = pixel_df.select(["point_id", "lon", "lat"]).unique("point_id")
    labelled = label_pixels(pixel_coords, regions).filter(pl.col("is_presence").is_not_null())
    labels = {
        row["point_id"]: 1.0 if row["is_presence"] else 0.0
        for row in labelled.iter_rows(named=True)
    }

    log.info(
        "lr=%.0e  d_model=%d  n_layers=%d  dropout=%.1f  wd=%.1f  obs_dropout_min=%d",
        args.lr, args.d_model, args.n_layers, args.dropout, args.weight_decay,
        args.obs_dropout_min,
    )

    cfg = TAMConfig(
        n_heads=4,
        d_ff=64,
        n_bands=len(V9_FEATURE_COLS),
        n_annual_features=0,
        n_epochs=60,
        patience=15,
        band_noise_std=0.03,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1=False,
        val_region_ids=tuple(exp.val_region_ids),
        feature_cols_override=tuple(V9_FEATURE_COLS),
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        obs_dropout_min=args.obs_dropout_min,
    )

    _, best_val_auc = train_tam(
        pixel_df=pixel_df.filter(pl.col("point_id").is_in(set(labels))),
        labels=labels,
        pixel_coords=pixel_coords,
        out_dir=out_dir,
        cfg=cfg,
        device=args.device,
    )
    log.info("best_val_auc=%.4f", best_val_auc)


if __name__ == "__main__":
    main()
