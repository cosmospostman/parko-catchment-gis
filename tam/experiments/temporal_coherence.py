"""tam/experiments/temporal_coherence.py — Temporal coherence diagnostic.

For each labelled bbox, computes per-pixel 95th-percentile Pearson correlation
with neighbours within a 5×5 window (r=2), per band. Only interior pixels
(at least 2px from the bbox boundary) are scored so every pixel has a full
neighbourhood of same-class pixels.

This is a pre-buffer-zone approximation: all neighbours are labelled pixels
from the same bbox, so coherence measures within-class homogeneity rather than
real landscape neighbourhood structure. The test verifies that the feature fires
for Parkinsonia presence pixels before committing to the buffer zone extension.

Usage:
    python -m tam.experiments.temporal_coherence --experiment v7_frenchs_only
    python -m tam.experiments.temporal_coherence --experiment v4_spectral_ref --sites norman_road frenchs
    python -m tam.experiments.temporal_coherence --region-ids norman_road_presence_1 norman_road_absence_5
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
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "EVI"]

# Default radius in pixels — 5×5 window
_RADIUS = 2

# Approximate S2 10m pixel size in degrees (used for interior filter only)
_PIX_DEG = 9.2e-5


def _point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def _interior_mask(coords_df: pd.DataFrame, bbox: tuple, radius_px: int) -> pd.Series:
    """Return boolean mask: True for pixels at least radius_px from the bbox edge.

    Uses the bbox bounds shrunk inward by radius_px * pixel_size degrees.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    margin = radius_px * _PIX_DEG
    return (
        (coords_df["lon"] >= lon_min + margin) &
        (coords_df["lon"] <= lon_max - margin) &
        (coords_df["lat"] >= lat_min + margin) &
        (coords_df["lat"] <= lat_max - margin)
    )



def _score_pixel_chunk(
    chunk_pids: list[str],
    interior_set: set[str],
    pid_index: dict[str, int],
    neighbour_map: dict[str, list[int]],
    ts_arrays: dict[str, np.ndarray],  # band -> (N_pids, T) array
    bands: list[str],
) -> list[dict]:
    results = []
    for pid in chunk_pids:
        if pid not in interior_set or pid not in pid_index:
            continue
        i = pid_index[pid]
        nb_idxs = neighbour_map.get(pid, [])
        if not nb_idxs:
            continue

        band_coherences: dict[str, float] = {}
        for band in bands:
            arr = ts_arrays[band]
            target = arr[i]
            neighbours = arr[nb_idxs]

            corrs = []
            valid_t = ~np.isnan(target)
            for nb_row in neighbours:
                valid = valid_t & ~np.isnan(nb_row)
                if valid.sum() < 4:
                    continue
                av = target[valid] - target[valid].mean()
                bv = nb_row[valid] - nb_row[valid].mean()
                denom = np.sqrt((av * av).sum() * (bv * bv).sum())
                if denom < 1e-10:
                    continue
                r = float((av * bv).sum() / denom)
                if np.isfinite(r):
                    corrs.append(r)

            if corrs:
                band_coherences[f"tc_p95_{band}"] = float(np.percentile(corrs, 95))

        if band_coherences:
            row = {"point_id": pid, **band_coherences}
            row["tc_p95_mean"] = float(np.mean(list(band_coherences.values())))
            results.append(row)
    return results


def compute_coherence(
    pixel_df: pd.DataFrame,
    interior_pids: set[str],
    bands: list[str],
    n_jobs: int = -1,
    radius: int = _RADIUS,
) -> pd.DataFrame:
    """Compute 95th-percentile neighbour temporal correlation for interior pixels.

    For each interior pixel, builds a (2r+1)×(2r+1) spatial neighbourhood from all
    available pixels, computes per-band Pearson correlation against each neighbour
    over the full multi-year time series, then takes the 95th percentile of the
    resulting correlation distribution.

    Returns DataFrame indexed by point_id with columns tc_p95_<band> and tc_p95_mean.
    """
    import os
    from joblib import Parallel, delayed

    # Build fortnightly-binned mean time series per pixel per band
    pixel_df = pixel_df.copy()
    doy_bins = np.arange(1, 366, 14)
    pixel_df["doy_bin"] = np.searchsorted(doy_bins, pixel_df["doy"].values, side="right")

    ts = (
        pixel_df.groupby(["point_id", "doy_bin"])[bands]
        .mean()
        .reset_index()
    )

    # Pixel grid coordinates
    coords_df = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .set_index("point_id")
    )
    all_pids = list(coords_df.index)
    pid_index = {pid: i for i, pid in enumerate(all_pids)}

    # Pivot time series per band into dense (N_pids, T) arrays
    n_bins = int(pixel_df["doy_bin"].max()) + 1
    ts_arrays: dict[str, np.ndarray] = {}
    for band in bands:
        piv = ts.pivot(index="point_id", columns="doy_bin", values=band)
        arr = np.full((len(all_pids), n_bins), np.nan, dtype=np.float32)
        for pid, row in piv.iterrows():
            if pid in pid_index:
                for col, val in row.items():
                    if col < n_bins:
                        arr[pid_index[pid], col] = val
        ts_arrays[band] = arr

    # Build neighbour map once: for each interior pixel, indices of spatial neighbours
    radius_deg = radius * _PIX_DEG * 1.5
    lons = coords_df["lon"].values
    lats = coords_df["lat"].values

    neighbour_map: dict[str, list[int]] = {}
    for pid in interior_pids:
        if pid not in pid_index:
            continue
        i = pid_index[pid]
        dx = np.abs(lons - lons[i])
        dy = np.abs(lats - lats[i])
        nb_idxs = [j for j in np.where((dx <= radius_deg) & (dy <= radius_deg))[0] if j != i]
        if nb_idxs:
            neighbour_map[pid] = nb_idxs

    # Split interior pixels into chunks for parallel processing
    interior_list = [p for p in interior_pids if p in neighbour_map]
    if not interior_list:
        return pd.DataFrame()

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    chunks = np.array_split(interior_list, n_workers)

    nested = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_score_pixel_chunk)(
            list(chunk), interior_pids, pid_index, neighbour_map, ts_arrays, bands
        )
        for chunk in chunks if len(chunk)
    )
    results = [row for rows in nested for row in rows]

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("point_id")


def run_experiment(pixel_df: pd.DataFrame, regions: list, bands: list[str], out_dir: Path, label: str, radius: int = _RADIUS) -> None:
    """Compute coherence per bbox, write a single horizontal violin plot and summary."""
    pixel_df = pixel_df.copy()
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})
    pixel_df     = pixel_df[pixel_df["point_id"].isin(all_labels.index)].copy()

    summary_lines = [
        f"Temporal coherence diagnostic — {label}",
        f"radius=r{radius} ({2*radius+1}×{2*radius+1} window)  percentile=p95  interior_padding={radius}px",
        "",
        f"{'region':<40} {'class':<10} {'n_px':>6}  {'mean':>8}  {'std':>8}",
        "-" * 80,
    ]

    # Collect per-region results for the combined plot
    plot_data:   list[np.ndarray] = []
    plot_labels: list[str]        = []
    plot_colors: list[str]        = []

    all_coh: list[pd.DataFrame] = []

    for region in regions:
        # Pixels belonging to this bbox
        region_pids = set(
            pixel_df[pixel_df["point_id"].str.startswith(region.id + "_")]["point_id"].tolist()
        )
        if not region_pids:
            continue

        region_df     = pixel_df[pixel_df["point_id"].isin(region_pids)]
        region_coords = pixel_coords[pixel_coords["point_id"].isin(region_pids)]

        interior_mask = _interior_mask(region_coords, tuple(region.bbox), radius)
        interior_pids = set(region_coords[interior_mask]["point_id"].tolist())
        if len(interior_pids) < 5:
            interior_pids = region_pids  # fall back if bbox too small

        print(f"  {region.id}: {len(interior_pids)} interior pixels ...")
        coh_df = compute_coherence(region_df, interior_pids, bands, radius=radius)
        if coh_df.empty:
            continue

        coh_df["label"]  = all_labels.reindex(coh_df.index)
        coh_df["region"] = region.id
        coh_df = coh_df.dropna(subset=["label"])
        all_coh.append(coh_df)

        vals = coh_df["tc_p95_mean"].dropna()
        if len(vals) < 2:
            continue

        color = "steelblue" if region.is_presence else "coral"
        cls_name = "presence" if region.is_presence else "absence"
        plot_data.append(vals.values)
        plot_labels.append(f"{region.id}  (n={len(vals)}, μ={vals.mean():.3f})")
        plot_colors.append(color)

        summary_lines.append(
            f"{region.id:<40} {cls_name:<10} {len(vals):>6}  {vals.mean():>8.4f}  {vals.std():>8.4f}"
        )

    # Save combined CSV
    if all_coh:
        combined = pd.concat(all_coh)
        combined.to_csv(out_dir / "coherence_all.csv")
    else:
        combined = pd.DataFrame()

    # One plot per band (plus one for the mean across bands)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.75, label="presence"),
        Patch(facecolor="coral",     alpha=0.75, label="absence"),
    ]

    band_cols = [f"tc_p95_{b}" for b in bands if not combined.empty and f"tc_p95_{b}" in combined.columns]
    plot_cols = band_cols + (["tc_p95_mean"] if "tc_p95_mean" in combined.columns else [])

    # Build ordered region list matching plot_labels
    ordered_regions = []
    region_by_id = {r.id: r for r in regions}
    for lbl in plot_labels:
        rid = lbl.split("  ")[0].strip()
        if rid in region_by_id:
            ordered_regions.append(region_by_id[rid])

    for col in plot_cols:
        band_name = col.replace("tc_p95_", "")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_labels))))
        fig.suptitle(f"Temporal coherence (p95, r=2) — {band_name} — {label}", fontsize=11)

        band_data    = []
        band_colors  = []
        valid_labels = []

        for region, lbl, color in zip(ordered_regions, plot_labels, plot_colors):
            if combined.empty or col not in combined.columns:
                continue
            vals = combined[combined["region"] == region.id][col].dropna()
            if len(vals) < 2:
                continue
            band_data.append(vals.values)
            band_colors.append(color)
            valid_labels.append(lbl)

        if band_data:
            positions = list(range(len(band_data), 0, -1))
            parts = ax.violinplot(band_data, positions=positions, vert=False,
                                  showmedians=True, showextrema=True)
            for pc, color in zip(parts["bodies"], band_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.8)
            ax.set_yticks(positions)
            ax.set_yticklabels(valid_labels, fontsize=7)

        ax.set_xlabel(f"tc_p95_{band_name}", fontsize=9)
        ax.set_xlim(0.9, 1.01)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(out_dir / f"coherence_{band_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved coherence_{band_name}.png")

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nOutputs written to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",         action="store_true", help="Run on all regions in training.yaml")
    parser.add_argument("--experiment",  default=None, help="Experiment module name (e.g. v7_frenchs_only)")
    parser.add_argument("--region-ids",  nargs="+",    default=None, help="Individual region IDs")
    parser.add_argument("--sites",       nargs="+",    default=None, help="Filter to these site prefixes")
    parser.add_argument("--bands",       nargs="+",    default=BANDS)
    parser.add_argument("--radius",      type=int,     default=_RADIUS, help="Neighbourhood radius in pixels (default: 2)")
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    if not args.all and not args.experiment and not args.region_ids:
        parser.error("Provide --all, --experiment, or --region-ids")

    bands = [b for b in args.bands if b in BANDS]

    if args.all:
        from utils.regions import load_regions
        regions_all = load_regions()
        region_ids = [r.id for r in regions_all]
        label = "all"
    elif args.region_ids:
        region_ids = args.region_ids
        label = "_".join(region_ids[:2])
    else:
        exp = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT
        region_ids = list(exp.region_ids)
        label = args.experiment
        bands = [b for b in bands if b in exp.feature_cols]

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "temporal-coherence" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    regions  = select_regions(region_ids)
    tile_ids = tile_ids_for_regions(region_ids)

    read_bands = [b for b in bands if b not in ("NDVI", "NDWI", "EVI")]
    read_cols  = ["point_id", "lon", "lat", "date", "scl_purity"] + read_bands

    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            print(f"Missing tile: {path}")
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())

    if not chunks:
        print("No data found.")
        sys.exit(1)

    pixel_df = pd.concat(chunks, ignore_index=True)

    # Add spectral indices if needed
    from analysis.constants import add_spectral_indices
    if any(b in bands for b in ("NDVI", "NDWI", "EVI")):
        pixel_df = add_spectral_indices(pixel_df)

    # Filter to requested sites if specified
    if args.sites:
        pixel_df = pixel_df[pixel_df["point_id"].map(_point_site).isin(args.sites)]
        regions  = [r for r in regions if any(r.id.startswith(s) for s in args.sites)]

    if args.sites:
        label = "_".join(args.sites)

    print(f"Loaded {len(pixel_df):,} observations, {pixel_df['point_id'].nunique():,} pixels")
    run_experiment(pixel_df, regions, bands, out_dir, label, radius=args.radius)


if __name__ == "__main__":
    main()
