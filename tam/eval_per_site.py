"""Evaluate a saved TAM checkpoint against each site's pixels individually.

Usage:
    python -m tam.eval_per_site --checkpoint outputs/models/sweep_loso/train_all \
        --experiment v6_spectral --sites frenchs barcoorah ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.core.config import TAMConfig
from tam.core.dataset import TAMDataset, collate_fn
from tam.core.train import load_tam
from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--sites",       nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path(args.checkpoint)
    exp = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT

    # --- Load pixels ----------------------------------------------------------
    regions  = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks: list[pl.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.warning("Missing tile: %s", path)
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(
                rg, columns=["point_id", "lon", "lat", "date", "scl_purity"] + exp.feature_cols
            )
            chunks.append(pl.from_arrow(tbl))

    pixel_df = pl.concat(chunks).with_columns([
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
    ])

    pixel_coords = pixel_df.select(["point_id", "lon", "lat"]).unique("point_id")
    labelled     = label_pixels(pixel_coords, regions).filter(pl.col("is_presence").is_not_null())
    all_labels   = {
        row["point_id"]: 1.0 if row["is_presence"] else 0.0
        for row in labelled.iter_rows(named=True)
    }

    # --- Load checkpoint ------------------------------------------------------
    model, band_mean, band_std, *_ = load_tam(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(out_dir / "tam_config.json") as fh:
        cfg = TAMConfig.from_dict(json.load(fh))

    annual_feat_df: pl.DataFrame | None = None
    cache_path = out_dir / "annual_features_cache.parquet"
    if cache_path.exists():
        annual_feat_df = pl.read_parquet(cache_path)

    # --- Per-site eval --------------------------------------------------------
    print(f"{'site':<24}  {'val_auc':<10}  {'n_px'}")

    for site in args.sites:
        site_labels = {pid: v for pid, v in all_labels.items() if point_site(pid) == site}
        n_px = len(site_labels)

        if n_px == 0:
            print(f"{site:<24}  {'n/a':<10}  0  (no pixels)")
            continue

        unique_vals = set(site_labels.values())
        if len(unique_vals) < 2:
            only = "presence-only" if 1.0 in unique_vals else "absence-only"
            print(f"{site:<24}  {'n/a':<10}  {n_px}  ({only})")
            continue

        site_pixel_df = pixel_df.filter(pl.col("point_id").is_in(set(site_labels)))

        ds = TAMDataset(
            site_pixel_df, site_labels,
            band_mean=band_mean, band_std=band_std,
            scl_purity_min=cfg.scl_purity_min,
            min_obs_per_year=cfg.min_obs_per_year,
            doy_jitter=0,
            annual_features_df=annual_feat_df,
        )

        if len(ds) == 0:
            print(f"{site:<24}  {'n/a':<10}  {n_px}  (all filtered)")
            continue

        loader = DataLoader(ds, batch_size=4096, shuffle=False, collate_fn=collate_fn)
        probs, gt = [], []
        with torch.no_grad():
            for batch in loader:
                p, _ = model(
                    batch["bands"].to(device),
                    batch["doy"].to(device),
                    batch["mask"].to(device),
                    batch["n_obs"].to(device),
                    batch["annual_feats"].to(device),
                )
                probs.extend(torch.sigmoid(p).cpu().numpy().tolist())
                gt.extend(batch["label"].numpy().tolist())

        probs_arr, gt_arr = np.array(probs), np.array(gt)
        finite = np.isfinite(probs_arr)
        auc = roc_auc_score(gt_arr[finite], probs_arr[finite]) if len(set(gt_arr[finite].tolist())) > 1 else float("nan")
        print(f"{site:<24}  {auc:<10.3f}  {n_px}")


if __name__ == "__main__":
    main()
