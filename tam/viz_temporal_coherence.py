"""Temporal coherence diagnostic.

For each pixel in a region, computes mean pairwise temporal correlation with
its k nearest neighbours across all bands. Plots the distribution of coherence
scores for presence vs absence pixels per site.

If Parkinsonia forms spectrally coherent monoculture patches, presence pixels
should show higher neighbour correlation than absence pixels — independent of
absolute spectral values.

Usage:
    python -m tam.viz_temporal_coherence --experiment v7_frenchs_only \
        --k 8 --out outputs/temporal-coherence/frenchs

    python -m tam.viz_temporal_coherence --experiment v4_spectral_ref \
        --sites frenchs norman_road --k 8
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
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","NDVI","NDWI","EVI"]

def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def compute_coherence(pixel_df: pd.DataFrame, all_labels: pd.Series, k: int, bands: list[str]) -> pd.DataFrame:
    """Compute mean neighbour temporal correlation per pixel.

    For each pixel, finds k nearest spatial neighbours (by lat/lon),
    computes Pearson correlation of each band's time series between
    target and neighbour, averages across neighbours and bands.

    Returns DataFrame with columns: point_id, mean_coherence, per-band coherences.
    """
    # Build pivot: mean band value per (point_id, doy_bin) — weekly bins
    doy_bins = np.arange(1, 366, 14)  # fortnightly — enough temporal resolution, fewer NaNs
    pixel_df = pixel_df.copy()
    pixel_df["doy_bin"] = np.searchsorted(doy_bins, pixel_df["doy"].values, side="right")

    # Per-pixel mean per doy_bin per band
    ts = (
        pixel_df.groupby(["point_id", "doy_bin"])[bands]
        .mean()
        .reset_index()
    )

    # Unique pixels with coords
    coords_df = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").set_index("point_id")
    pids = list(coords_df.index)
    if len(pids) < k + 1:
        return pd.DataFrame()

    # KD-tree for spatial lookup
    coords = coords_df.loc[pids, ["lat","lon"]].values
    tree   = cKDTree(coords)

    # For each pixel, query k+1 neighbours (includes self), skip self
    dists, idxs = tree.query(coords, k=min(k + 1, len(pids)))

    results = []
    # Pivot time series per pixel per band for fast correlation
    ts_pivot: dict[str, pd.DataFrame] = {}
    for band in bands:
        piv = ts.pivot(index="point_id", columns="doy_bin", values=band)
        ts_pivot[band] = piv

    for i, pid in enumerate(pids):
        neighbour_idxs = [j for j in idxs[i] if pids[j] != pid][:k]
        if not neighbour_idxs:
            continue

        band_coherences: dict[str, float] = {}
        for band in bands:
            piv = ts_pivot[band]
            if pid not in piv.index:
                continue
            target_ts = piv.loc[pid].values.astype(float)

            corrs = []
            for ni in neighbour_idxs:
                npid = pids[ni]
                if npid not in piv.index:
                    continue
                nb_ts = piv.loc[npid].values.astype(float)
                # Only use bins where both have data
                valid = ~(np.isnan(target_ts) | np.isnan(nb_ts))
                if valid.sum() < 4:
                    continue
                r = np.corrcoef(target_ts[valid], nb_ts[valid])[0, 1]
                if np.isfinite(r):
                    corrs.append(r)

            if corrs:
                band_coherences[f"tc_{band}"] = float(np.mean(corrs))

        if band_coherences:
            row = {"point_id": pid, **band_coherences}
            row["tc_mean"] = float(np.mean(list(band_coherences.values())))
            results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("point_id")
    df["label"] = all_labels.reindex(df.index)
    df["site"]  = pd.Series({p: point_site(p) for p in df.index})
    return df.dropna(subset=["label"])


def coherence_for_region(region_id: str, k: int, bands: list[str]) -> pd.Series:
    """Compute mean neighbour coherence for a single region, returned as a Series of tc_mean values."""
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path
    from tam.utils import label_pixels

    regions  = select_regions([region_id])
    tile_ids = tile_ids_for_regions([region_id])

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id","lon","lat","date","scl_purity"] + bands)
            chunks.append(tbl.to_pandas())

    if not chunks:
        return pd.Series(dtype=float)

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year

    pixel_coords = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").reset_index(drop=True)

    # Filter to bbox
    r = regions[0]
    lon_min, lat_min, lon_max, lat_max = r.bbox
    in_bbox = (
        pixel_df["lon"].between(lon_min, lon_max) &
        pixel_df["lat"].between(lat_min, lat_max)
    )
    pixel_df = pixel_df[in_bbox].copy()
    pixel_coords = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").reset_index(drop=True)

    if r.year is not None:
        pixel_df = pixel_df[pixel_df["year"].between(r.year - 5, r.year)]

    labelled   = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})
    pixel_df   = pixel_df[pixel_df["point_id"].isin(all_labels.index)]

    coh_df = compute_coherence(pixel_df, all_labels, k, bands)
    if coh_df.empty:
        return pd.Series(dtype=float)
    return coh_df["tc_mean"].dropna()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",  default=None)
    parser.add_argument("--region-ids",  nargs="+", default=None,
                        help="Individual region IDs to compare on a single chart")
    parser.add_argument("--region-labels", nargs="+", default=None,
                        help="Display labels for --region-ids (same order)")
    parser.add_argument("--sites",       nargs="+", default=None)
    parser.add_argument("--k",           type=int, default=8)
    parser.add_argument("--bands",       nargs="+", default=BANDS)
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    if args.region_ids:
        # --- Per-region mode: one violin per region, all on one chart --------
        bands   = [b for b in args.bands if b in BANDS]
        out_dir = Path(args.out) if args.out else Path("outputs/temporal-coherence/region-compare")
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = args.region_labels if args.region_labels else args.region_ids
        data   = []
        summary_lines = [
            f"Temporal coherence — per-region comparison  k={args.k}",
            "",
            f"{'region':<40} {'n_px':>6}  {'mean_tc':>8}  {'std_tc':>8}",
            "-" * 65,
        ]

        for region_id in args.region_ids:
            print(f"Computing coherence for {region_id}...")
            tc = coherence_for_region(region_id, args.k, bands)
            data.append(tc.values)
            summary_lines.append(
                f"{region_id:<40} {len(tc):>6}  {tc.mean():>8.4f}  {tc.std():>8.4f}"
            )

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(args.region_ids)), 5))
        fig.suptitle(f"Temporal coherence by region  (k={args.k})", fontsize=12)

        colors = plt.cm.tab10(np.linspace(0, 1, len(args.region_ids)))
        parts  = ax.violinplot(
            [d for d in data if len(d) > 1],
            positions=range(1, len(data) + 1),
            showmedians=True, showextrema=True,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("mean neighbour correlation", fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = out_dir / "coherence_by_region.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_text = "\n".join(summary_lines)
        (out_dir / "summary.txt").write_text(summary_text)
        print(summary_text)
        print(f"\nAll outputs written to: {out_dir}")
        return

    if not args.experiment:
        parser.error("Provide either --experiment or --region-ids")

    exp     = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT
    out_dir = Path(args.out) if args.out else Path(f"outputs/temporal-coherence/{args.experiment}")
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
                rg, columns=["point_id","lon","lat","date","scl_purity"] + bands
            )
            chunks.append(tbl.to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year

    pixel_coords = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    pixel_df = pixel_df[pixel_df["point_id"].isin(all_labels.index)].copy()
    pixel_df["site"] = pixel_df["point_id"].map(point_site)

    sites = args.sites or sorted(pixel_df["site"].unique())

    summary_lines = [
        f"Temporal coherence diagnostic — {args.experiment}",
        f"k={args.k}  bands={', '.join(bands)}",
        "",
        f"{'site':<20} {'class':<10} {'n_px':>6}  {'mean_tc':>8}  {'std_tc':>8}  {'p_value':>10}  {'separable'}",
        "-" * 80,
    ]

    n_sites = len(sites)
    fig, axes = plt.subplots(n_sites, 1, figsize=(10, 4 * n_sites), squeeze=False)
    fig.suptitle(f"Temporal coherence distributions — {args.experiment}  (k={args.k})", fontsize=12)

    for row, site in enumerate(sites):
        ax = axes[row][0]
        site_df = pixel_df[pixel_df["site"] == site]

        print(f"Computing coherence for {site} ({site_df['point_id'].nunique()} pixels)...")
        coh_df = compute_coherence(site_df, all_labels, args.k, bands)

        if coh_df.empty:
            ax.text(0.5, 0.5, f"{site}: insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(site, fontsize=10)
            continue

        pres = coh_df.loc[coh_df["label"] == 1.0, "tc_mean"].dropna()
        abs_ = coh_df.loc[coh_df["label"] == 0.0, "tc_mean"].dropna()

        # Mann-Whitney U test
        p_val = np.nan
        if len(pres) >= 3 and len(abs_) >= 3:
            _, p_val = mannwhitneyu(pres, abs_, alternative="two-sided")

        separable = "YES" if p_val < 0.05 else "no" if np.isfinite(p_val) else "n/a"

        # Plot violin / histogram
        data_to_plot = []
        labels_plot  = []
        if len(pres) >= 2:
            data_to_plot.append(pres.values)
            labels_plot.append(f"presence\n(n={len(pres)}, μ={pres.mean():.3f})")
        if len(abs_) >= 2:
            data_to_plot.append(abs_.values)
            labels_plot.append(f"absence\n(n={len(abs_)}, μ={abs_.mean():.3f})")

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, showmedians=True, showextrema=True)
            colors = ["steelblue", "coral"]
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax.set_xticks(range(1, len(labels_plot) + 1))
            ax.set_xticklabels(labels_plot, fontsize=9)

        ax.set_title(f"{site}  —  p={p_val:.4f}  {separable}", fontsize=10)
        ax.set_ylabel("mean neighbour correlation", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

        # Per-band summary
        band_cols = [c for c in coh_df.columns if c.startswith("tc_") and c != "tc_mean"]
        for cls, cls_name, val in [(1.0, "presence", pres.mean() if len(pres) else np.nan),
                                    (0.0, "absence",  abs_.mean() if len(abs_) else np.nan)]:
            summary_lines.append(
                f"{site:<20} {cls_name:<10} {int(len(pres) if cls==1.0 else len(abs_)):>6}  "
                f"{val:>8.4f}  "
                f"{(pres.std() if cls==1.0 else abs_.std()):>8.4f}  "
                f"{p_val:>10.4f}  {separable}"
            )

        # Per-band breakdown
        if band_cols:
            summary_lines.append(f"  {'band':<10}  {'pres_mean':>10}  {'abs_mean':>10}  {'diff':>8}")
            for bc in band_cols:
                pm = coh_df.loc[coh_df["label"]==1.0, bc].mean()
                am = coh_df.loc[coh_df["label"]==0.0, bc].mean()
                summary_lines.append(f"  {bc:<10}  {pm:>10.4f}  {am:>10.4f}  {pm-am:>8.4f}")

        summary_lines.append("")

        # Save per-site CSV for further analysis
        coh_df.to_csv(out_dir / f"coherence_{site}.csv")

    plt.tight_layout()
    plot_path = out_dir / "coherence_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
