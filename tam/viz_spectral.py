"""Plot mean spectral band and index values by DOY for presence vs absence pixels.

Shows whether any band/index has visible separation between presence and absence
at any time of year — a prerequisite for the temporal model to learn discrimination.

Usage:
    python -m tam.viz_spectral --experiment v7_norman_road_only \
        --out outputs/spectral_norman_road

    python -m tam.viz_spectral --experiment v7_norman_road_only \
        --sites norman_road --bands NDVI B08 B11
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

# Key bands to plot by default — full S2 set plus indices
DEFAULT_BANDS = ["NDVI", "EVI", "B08", "B11", "B8A", "B04"]

DOY_BINS   = np.arange(1, 366, 7)
MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sites",      nargs="+", default=None)
    parser.add_argument("--bands",      nargs="+", default=DEFAULT_BANDS,
                        help="Bands/indices to plot")
    parser.add_argument("--out",        default=None,
                        help="Output directory (default: outputs/spectral_<experiment>)")
    args = parser.parse_args()

    exp     = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT
    out_dir = Path(args.out) if args.out else Path(f"outputs/spectral_{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure requested bands are in feature_cols
    bands = [b for b in args.bands if b in exp.feature_cols]
    missing = set(args.bands) - set(bands)
    if missing:
        print(f"Warning: bands not in experiment feature_cols, skipping: {missing}")

    # --- Load pixels ----------------------------------------------------------
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
        f"Spectral summary — {args.experiment}",
        f"Bands: {', '.join(bands)}",
        "",
    ]

    for site in sites:
        site_df = pixel_df[pixel_df["site"] == site]
        n_bands = len(bands)
        fig, axes = plt.subplots(
            n_bands, 1,
            figsize=(12, 3 * n_bands),
            sharex=True,
        )
        if n_bands == 1:
            axes = [axes]
        fig.suptitle(f"{site} — mean spectral values by DOY", fontsize=12)

        summary_lines.append(f"=== {site} ===")
        summary_lines.append(f"{'band':<8}  {'max_sep_month':<14}  {'max_sep_value':<14}  {'direction'}")
        summary_lines.append("-" * 60)

        for ax, band in zip(axes, bands):
            for cls, label, color in [(1.0, "presence", "steelblue"), (0.0, "absence", "coral")]:
                cls_df = site_df[site_df["label"] == cls]
                if cls_df.empty:
                    continue

                # Bin observations by DOY and compute mean ± std
                obs = cls_df[["doy", band]].dropna()
                bin_idx = np.searchsorted(DOY_BINS, obs["doy"].values, side="right") - 1
                bin_idx = np.clip(bin_idx, 0, len(DOY_BINS) - 2)

                means = np.full(len(bin_centres), np.nan)
                stds  = np.full(len(bin_centres), np.nan)
                for b in range(len(bin_centres)):
                    vals = obs.loc[bin_idx == b, band].values
                    if len(vals) >= 3:
                        means[b] = np.mean(vals)
                        stds[b]  = np.std(vals)

                valid = ~np.isnan(means)
                ax.plot(bin_centres[valid], means[valid], color=color, label=label, linewidth=1.5)
                ax.fill_between(
                    bin_centres[valid],
                    (means - stds)[valid],
                    (means + stds)[valid],
                    color=color, alpha=0.15,
                )

            ax.set_ylabel(band, fontsize=9)
            ax.set_xticks(MONTH_DOYS)
            ax.set_xticklabels(MONTH_LBLS, fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Text summary: find DOY bin with largest presence-absence separation
            pres_df = site_df[site_df["label"] == 1.0][["doy", band]].dropna()
            abs_df  = site_df[site_df["label"] == 0.0][["doy", band]].dropna()
            if not pres_df.empty and not abs_df.empty:
                p_means = np.full(len(bin_centres), np.nan)
                a_means = np.full(len(bin_centres), np.nan)
                for b in range(len(bin_centres)):
                    p_vals = pres_df.loc[np.searchsorted(DOY_BINS, pres_df["doy"].values, side="right") - 1 == b, band].values
                    a_vals = abs_df.loc[np.searchsorted(DOY_BINS, abs_df["doy"].values, side="right") - 1 == b, band].values
                    if len(p_vals) >= 3:
                        p_means[b] = np.mean(p_vals)
                    if len(a_vals) >= 3:
                        a_means[b] = np.mean(a_vals)

                sep = p_means - a_means
                valid_sep = ~np.isnan(sep)
                if valid_sep.any():
                    max_b   = np.nanargmax(np.abs(sep))
                    max_sep = sep[max_b]
                    doy_peak = bin_centres[max_b]
                    month_idx = max(0, np.searchsorted(MONTH_DOYS, doy_peak, side="right") - 1)
                    month = MONTH_LBLS[month_idx]
                    direction = "pres>abs" if max_sep > 0 else "abs>pres"
                    summary_lines.append(f"{band:<8}  {month:<14}  {abs(max_sep):<14.4f}  {direction}")

        axes[-1].set_xlabel("Day of year", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"spectral_{site}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir}/spectral_{site}.png")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
