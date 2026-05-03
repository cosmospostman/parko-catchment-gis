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
import pandas as pd
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

    chunks: list[pd.DataFrame] = []
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
            chunks.append(tbl.to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    dates = pd.to_datetime(pixel_df["date"])
    pixel_df["year"] = dates.dt.year
    pixel_df["doy"]  = dates.dt.day_of_year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled      = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels    = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    # --- Load checkpoint ------------------------------------------------------
    model, band_mean, band_std = load_tam(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(out_dir / "tam_config.json") as fh:
        cfg = TAMConfig.from_dict(json.load(fh))

    global_feat_df: pd.DataFrame | None = None
    cache_path = out_dir / "global_features_cache.parquet"
    if cache_path.exists():
        global_feat_df = pd.read_parquet(cache_path)

    # --- Per-site eval --------------------------------------------------------
    print(f"{'site':<24}  {'val_auc':<10}  {'n_px'}")

    for site in args.sites:
        site_labels = all_labels[all_labels.index.map(point_site) == site]
        n_px = len(site_labels)

        if n_px == 0:
            print(f"{site:<24}  {'n/a':<10}  0  (no pixels)")
            continue

        if site_labels.nunique() < 2:
            only = "presence-only" if site_labels.iloc[0] == 1 else "absence-only"
            print(f"{site:<24}  {'n/a':<10}  {n_px}  ({only})")
            continue

        site_pixel_df = pixel_df[pixel_df["point_id"].isin(site_labels.index)]

        ds = TAMDataset(
            site_pixel_df, site_labels,
            band_mean=band_mean, band_std=band_std,
            scl_purity_min=cfg.scl_purity_min,
            min_obs_per_year=cfg.min_obs_per_year,
            doy_jitter=0,
            global_features_df=global_feat_df,
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
                    batch["global_feats"].to(device),
                )
                probs.extend(torch.sigmoid(p).cpu().numpy().tolist())
                gt.extend(batch["label"].numpy().tolist())

        probs_arr, gt_arr = np.array(probs), np.array(gt)
        finite = np.isfinite(probs_arr)
        auc = roc_auc_score(gt_arr[finite], probs_arr[finite]) if len(set(gt_arr[finite].tolist())) > 1 else float("nan")
        print(f"{site:<24}  {auc:<10.3f}  {n_px}")


if __name__ == "__main__":
    main()
