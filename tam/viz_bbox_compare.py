"""Compare spectral/temporal profiles between two bboxes or region IDs.

Use this before committing new presence/absence regions to training —
verify there is visible spectral separation worth learning from.

Usage:
    # Compare two region IDs already in training.yaml
    python -m tam.viz_bbox_compare \
        --region-a norman_road_presence_1 \
        --region-b norman_road_absence_3 \
        --out outputs/compare_norman_road_p1_a3.png

    # Compare arbitrary bboxes (lon_min lat_min lon_max lat_max)
    python -m tam.viz_bbox_compare \
        --bbox-a 141.647945 -20.397868 141.649492 -20.396426 \
        --bbox-b 141.524279 -20.236115 141.527576 -20.232487 \
        --year 2025 \
        --label-a "presence candidate" \
        --label-b "absence candidate" \
        --out outputs/compare_candidate.png
"""

from __future__ import annotations

import argparse
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

from utils.training_collector import tile_ids_for_regions, tile_parquet_path
from utils.regions import select_regions

BANDS   = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
INDICES = ["NDVI","EVI","NDWI"]
ALL_COLS = BANDS + INDICES

DOY_BINS   = np.arange(1, 366, 7)
MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def pixels_for_region(region_id: str) -> pd.DataFrame:
    """Load pixel observations for a single region ID from training.yaml."""
    regions  = select_regions([region_id])
    tile_ids = tile_ids_for_regions([region_id])
    chunks   = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            print(f"Warning: missing tile {path}")
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id","lon","lat","date","scl_purity"] + ALL_COLS)
            chunks.append(tbl.to_pandas())
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    # Filter to bbox
    r = regions[0]
    lon_min, lat_min, lon_max, lat_max = r.bbox
    df = df[(df["lon"].between(lon_min, lon_max)) & (df["lat"].between(lat_min, lat_max))]
    if r.year is not None:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df = df[df["year"].between(r.year - 5, r.year)]
    return df


def pixels_for_bbox(bbox: list[float], year: int | None) -> pd.DataFrame:
    """Load pixel observations for an arbitrary bbox by scanning all tile parquets."""
    from utils.training_collector import TILES_DIR
    lon_min, lat_min, lon_max, lat_max = bbox
    chunks = []
    tiles_dir = Path(TILES_DIR) if hasattr(__import__("utils.training_collector", fromlist=["TILES_DIR"]), "TILES_DIR") else PROJECT_ROOT / "data" / "training" / "tiles"
    for path in sorted(tiles_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id","lon","lat","date","scl_purity"] + ALL_COLS)
            df_chunk = tbl.to_pandas()
            df_chunk = df_chunk[
                df_chunk["lon"].between(lon_min, lon_max) &
                df_chunk["lat"].between(lat_min, lat_max)
            ]
            if not df_chunk.empty:
                chunks.append(df_chunk)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    if year is not None:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df = df[df["year"].between(year - 5, year)]
    return df


def bin_by_doy(df: pd.DataFrame, band: str):
    bin_idx = np.clip(np.searchsorted(DOY_BINS, df["doy"].values, side="right") - 1, 0, len(DOY_BINS) - 2)
    n_bins  = len(DOY_BINS) - 1
    means   = np.full(n_bins, np.nan)
    stds    = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = df.loc[bin_idx == b, band].dropna().values
        if len(vals) >= 2:
            means[b] = np.mean(vals)
            stds[b]  = np.std(vals)
    return means, stds


def plot_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame,
                    label_a: str, label_b: str,
                    bands: list[str], out_path: Path) -> None:

    bin_centres = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2
    n_bands = len(bands)

    fig, axes = plt.subplots(n_bands, 1, figsize=(13, 3 * n_bands), sharex=True)
    if n_bands == 1:
        axes = [axes]
    fig.suptitle(f"Spectral comparison: {label_a}  vs  {label_b}", fontsize=12)

    summary_lines = [
        f"Spectral comparison: {label_a} vs {label_b}",
        f"n_pixels  {label_a}: {df_a['point_id'].nunique()}  |  {label_b}: {df_b['point_id'].nunique()}",
        f"n_obs     {label_a}: {len(df_a)}  |  {label_b}: {len(df_b)}",
        "",
        f"{'band':<8}  {'peak_sep_month':<16}  {'max_sep':>8}  direction",
        "-" * 55,
    ]

    for ax, band in zip(axes, bands):
        for df, label, color in [(df_a, label_a, "steelblue"), (df_b, label_b, "coral")]:
            sub = df[["doy", band]].dropna()
            if sub.empty:
                continue
            means, stds = bin_by_doy(sub, band)
            valid = ~np.isnan(means)
            ax.plot(bin_centres[valid], means[valid], color=color, label=label, linewidth=1.5)
            ax.fill_between(bin_centres[valid],
                            (means - stds)[valid], (means + stds)[valid],
                            color=color, alpha=0.15)

        ax.set_ylabel(band, fontsize=9)
        ax.set_xticks(MONTH_DOYS)
        ax.set_xticklabels(MONTH_LBLS, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Summary: max separation
        sub_a = df_a[["doy", band]].dropna()
        sub_b = df_b[["doy", band]].dropna()
        if not sub_a.empty and not sub_b.empty:
            ma, _ = bin_by_doy(sub_a, band)
            mb, _ = bin_by_doy(sub_b, band)
            sep   = ma - mb
            valid_sep = ~np.isnan(sep)
            if valid_sep.any():
                max_b     = np.nanargmax(np.abs(sep))
                max_sep   = sep[max_b]
                doy_peak  = bin_centres[max_b]
                month_idx = max(0, np.searchsorted(MONTH_DOYS, doy_peak, side="right") - 1)
                month     = MONTH_LBLS[month_idx]
                direction = f"{label_a}>{label_b}" if max_sep > 0 else f"{label_b}>{label_a}"
                summary_lines.append(f"{band:<8}  {month:<16}  {abs(max_sep):>8.4f}  {direction}")

    axes[-1].set_xlabel("Day of year", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_text = "\n".join(summary_lines)
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(summary_text)
    print(summary_text)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    # Region ID mode
    parser.add_argument("--region-a", default=None, help="Region ID from training.yaml for A")
    parser.add_argument("--region-b", default=None, help="Region ID from training.yaml for B")

    # Bbox mode
    parser.add_argument("--bbox-a",   nargs=4, type=float, metavar=("LON_MIN","LAT_MIN","LON_MAX","LAT_MAX"))
    parser.add_argument("--bbox-b",   nargs=4, type=float, metavar=("LON_MIN","LAT_MIN","LON_MAX","LAT_MAX"))
    parser.add_argument("--year",     type=int, default=None, help="Year for bbox mode (loads year-5 to year)")
    parser.add_argument("--label-a",  default="A")
    parser.add_argument("--label-b",  default="B")

    parser.add_argument("--bands",    nargs="+", default=["NDVI","B8A","B07","B08","B06","B11","B12"],
                        help="Bands to plot")
    parser.add_argument("--out",      default="outputs/bbox_compare.png")
    args = parser.parse_args()

    # Load data
    if args.region_a and args.region_b:
        df_a   = pixels_for_region(args.region_a)
        df_b   = pixels_for_region(args.region_b)
        label_a = args.region_a
        label_b = args.region_b
    elif args.bbox_a and args.bbox_b:
        df_a   = pixels_for_bbox(args.bbox_a, args.year)
        df_b   = pixels_for_bbox(args.bbox_b, args.year)
        label_a = args.label_a
        label_b = args.label_b
    else:
        parser.error("Provide either --region-a/--region-b or --bbox-a/--bbox-b")

    if df_a.empty or df_b.empty:
        print("Error: no pixels found for one or both regions. Check tile data is collected.")
        sys.exit(1)

    df_a["doy"] = pd.to_datetime(df_a["date"]).dt.day_of_year
    df_b["doy"] = pd.to_datetime(df_b["date"]).dt.day_of_year

    n_a = df_a["point_id"].nunique()
    n_b = df_b["point_id"].nunique()
    print(f"{label_a}: {n_a} pixels, {len(df_a)} observations")
    print(f"{label_b}: {n_b} pixels, {len(df_b)} observations")

    bands = [b for b in args.bands if b in ALL_COLS]
    plot_comparison(df_a, df_b, label_a, label_b, bands, Path(args.out))


if __name__ == "__main__":
    main()
