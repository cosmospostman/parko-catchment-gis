"""Plot observation density by DOY for presence and absence pixels per site.

Used to check whether attention peaks in viz_attention correspond to
observation density artefacts (cloud cover) rather than phenological signal.

Usage:
    python -m tam.viz_obs_density --experiment v7_norman_road_only \
        --out outputs/obs_density_norman_road.png
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

DOY_BINS = np.arange(1, 366, 7)

def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sites",      nargs="+", default=None)
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    exp = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT

    regions  = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id", "lon", "lat", "date", "scl_purity"])
            chunks.append(tbl.to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["doy"] = pd.to_datetime(pixel_df["date"]).dt.day_of_year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    all_sites = sorted({point_site(pid) for pid in all_labels.index})
    sites = args.sites if args.sites else all_sites

    bin_centres = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2
    month_doys   = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    n_sites = len(sites)
    fig, axes = plt.subplots(n_sites, 2, figsize=(12, max(3, 2.5 * n_sites)),
                             sharex=True, squeeze=False)
    fig.suptitle("Observation density by day-of-year", fontsize=13, y=1.01)

    for row, site in enumerate(sites):
        site_labels = all_labels[all_labels.index.map(point_site) == site]

        for col, cls in enumerate([1.0, 0.0]):
            ax = axes[row][col]
            cls_name = "presence" if cls == 1.0 else "absence"
            color    = "steelblue" if cls == 1.0 else "coral"
            cls_pids = set(site_labels[site_labels == cls].index)

            obs = pixel_df[pixel_df["point_id"].isin(cls_pids)]["doy"].values
            if len(obs) == 0:
                ax.text(0.5, 0.5, "no observations", ha="center", va="center", transform=ax.transAxes)
            else:
                counts, _ = np.histogram(obs, bins=DOY_BINS)
                # Normalise by number of pixels so sites with more pixels don't dominate
                n_px = len(cls_pids)
                ax.bar(bin_centres, counts / n_px, width=6, color=color, alpha=0.8)

            ax.set_title(f"{site} — {cls_name}  ({len(cls_pids)} px)", fontsize=9)
            ax.set_xticks(month_doys)
            ax.set_xticklabels(month_labels, fontsize=7, rotation=45)
            ax.set_ylabel("obs / pixel", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Day of year", fontsize=9)

    plt.tight_layout()
    out_path = Path(args.out) if args.out else Path(f"outputs/obs_density_{args.experiment}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
