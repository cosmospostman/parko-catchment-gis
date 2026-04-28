"""Inter-annual consistency diagnostic.

For each pixel, computes the mean and std of annual-mean band values across years.
Plots:
  1. Mean spectral value by DOY, averaged across all years (presence vs absence)
  2. Inter-annual std of annual-mean values — lower std = more consistent signal

If Parkinsonia has a stable phenological fingerprint, presence pixels should show
lower inter-annual variance than absence pixels in the discriminative bands/seasons.

Usage:
    python -m tam.viz_interannual --experiment v7_norman_road_only \
        --out outputs/interannual_norman_road
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

DEFAULT_BANDS = ["NDVI", "EVI", "B08", "B11"]
DOY_BINS      = np.arange(1, 366, 7)
MONTH_DOYS    = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def bin_by_doy(df: pd.DataFrame, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, stds) per DOY bin for a single band."""
    bin_idx = np.searchsorted(DOY_BINS, df["doy"].values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(DOY_BINS) - 2)
    n_bins  = len(DOY_BINS) - 1
    means   = np.full(n_bins, np.nan)
    stds    = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = df.loc[bin_idx == b, band].dropna().values
        if len(vals) >= 3:
            means[b] = np.mean(vals)
            stds[b]  = np.std(vals)
    return means, stds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sites",      nargs="+", default=None)
    parser.add_argument("--bands",      nargs="+", default=DEFAULT_BANDS)
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    exp     = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT
    out_dir = Path(args.out) if args.out else Path(f"outputs/interannual_{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = [b for b in args.bands if b in exp.feature_cols]

    regions  = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(
                rg, columns=["point_id", "lon", "lat", "date", "scl_purity"] + exp.feature_cols
            )
            chunks.append(tbl.to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year

    pixel_coords = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    pixel_df = pixel_df[pixel_df["point_id"].isin(all_labels.index)].copy()
    pixel_df["label"] = pixel_df["point_id"].map(all_labels)
    pixel_df["site"]  = pixel_df["point_id"].map(point_site)

    sites = args.sites or sorted(pixel_df["site"].unique())
    bin_centres = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2

    summary_lines = [
        f"Inter-annual consistency — {args.experiment}",
        f"Bands: {', '.join(bands)}",
        "",
        "Max separation by band (mean signal):",
        f"{'site':<20} {'band':<8} {'month':<8} {'sep':>8}  {'direction':<12}  {'pres_iannual_std':>18}  {'abs_iannual_std':>16}",
        "-" * 100,
    ]

    for site in sites:
        site_df = pixel_df[pixel_df["site"] == site]
        n_bands = len(bands)

        # --- Plot 1: mean spectral by DOY (all years pooled) ------------------
        fig1, axes1 = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands), sharex=True)
        if n_bands == 1:
            axes1 = [axes1]
        fig1.suptitle(f"{site} — mean spectral by DOY (all years pooled)", fontsize=11)

        # --- Plot 2: inter-annual std per pixel, then mean across pixels ------
        fig2, axes2 = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands), sharex=True)
        if n_bands == 1:
            axes2 = [axes2]
        fig2.suptitle(f"{site} — inter-annual std of annual-mean values", fontsize=11)

        for ax1, ax2, band in zip(axes1, axes2, bands):
            for cls, cls_name, color in [(0.0, "absence", "coral"), (1.0, "presence", "steelblue")]:
                cls_df = site_df[site_df["label"] == cls].dropna(subset=[band])
                if cls_df.empty:
                    continue

                # Plot 1: pooled mean ± std by DOY
                means, stds = bin_by_doy(cls_df, band)
                valid = ~np.isnan(means)
                ax1.plot(bin_centres[valid], means[valid], color=color, label=cls_name, linewidth=1.5)
                ax1.fill_between(bin_centres[valid],
                                 (means - stds)[valid], (means + stds)[valid],
                                 color=color, alpha=0.15)

                # Plot 2: per-pixel annual mean, then std across years
                # annual_mean[pid, year] → std across years → mean across pixels
                ann = cls_df.groupby(["point_id", "year"])[band].mean().reset_index()
                pix_std = ann.groupby("point_id")[band].std().dropna()
                pix_mean_of_std = pix_std.mean()
                pix_mean        = ann.groupby("point_id")[band].mean()

                # Scatter: per-pixel mean vs inter-annual std
                ax2.scatter(pix_mean, pix_std, color=color, alpha=0.3, s=8, label=cls_name)

                # Summary stat
                if cls == 1.0:
                    pres_iannual_std = pix_mean_of_std
                else:
                    abs_iannual_std = pix_mean_of_std

            ax1.set_ylabel(band, fontsize=9)
            ax1.set_xticks(MONTH_DOYS)
            ax1.set_xticklabels(MONTH_LBLS, fontsize=8)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.set_ylabel(f"{band}\ninter-annual std", fontsize=9)
            ax2.set_xlabel(f"{band} mean", fontsize=8)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # Text summary
            pres_df = site_df[(site_df["label"] == 1.0)][["doy", band]].dropna()
            abs_df  = site_df[(site_df["label"] == 0.0)][["doy", band]].dropna()
            if not pres_df.empty and not abs_df.empty:
                pm, _ = bin_by_doy(pres_df, band)
                am, _ = bin_by_doy(abs_df,  band)
                sep   = pm - am
                valid_sep = ~np.isnan(sep)
                if valid_sep.any():
                    max_b     = np.nanargmax(np.abs(sep))
                    max_sep   = sep[max_b]
                    doy_peak  = bin_centres[max_b]
                    month_idx = max(0, np.searchsorted(MONTH_DOYS, doy_peak, side="right") - 1)
                    month     = MONTH_LBLS[month_idx]
                    direction = "pres>abs" if max_sep > 0 else "abs>pres"
                    summary_lines.append(
                        f"{site:<20} {band:<8} {month:<8} {abs(max_sep):>8.4f}  {direction:<12}  "
                        f"{pres_iannual_std:>18.4f}  {abs_iannual_std:>16.4f}"
                    )

        axes1[-1].set_xlabel("Day of year", fontsize=9)
        plt.figure(fig1.number)
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_{site}.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)

        plt.figure(fig2.number)
        plt.tight_layout()
        plt.savefig(out_dir / f"interannual_std_{site}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

        print(f"Saved plots for {site}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
