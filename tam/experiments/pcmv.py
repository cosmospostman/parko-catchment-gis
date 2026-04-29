"""tam/experiments/pcmv.py — PCMV (Pseudo Cross Multivariate Variogram) diagnostic.

For each labelled bbox, computes per-pixel PCMV against neighbours within a
5×5 window (r=2). PCMV is the mean squared NDVI difference between a pixel and
its neighbour across all shared cloud-free observation dates:

    γ_ij = 1/N Σ_{t ∈ shared_dates} (NDVI_i(t) - NDVI_j(t))²

Per-pixel score is the **minimum** across all neighbours (most similar neighbour),
so that dense Parkinsonia patches score near-zero and heterogeneous native
vegetation scores higher.

Unlike Pearson coherence, PCMV retains the absolute variance σ²:
    γ(h) = σ²(1 − ρ(h))
Two pixels with the same phenological shape but different NDVI levels are
indistinguishable by Pearson but distinguishable by PCMV.

Usage:
    python -m tam.experiments.pcmv --all --sites norman_road
    python -m tam.experiments.pcmv --all
    python -m tam.experiments.pcmv --experiment v7_frenchs_only
    python -m tam.experiments.pcmv --region-ids norman_road_presence_1 norman_road_absence_5
    python -m tam.experiments.pcmv --all --bands NDVI B11 B12
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

# Bands supported — raw obs, so we use the full date-indexed time series
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "EVI"]

_RADIUS  = 2           # neighbourhood radius in pixels
_PIX_DEG = 9.2e-5      # approximate S2 10m pixel size in degrees


def _point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def _interior_mask(coords_df: pd.DataFrame, bbox: tuple, radius_px: int) -> pd.Series:
    lon_min, lat_min, lon_max, lat_max = bbox
    margin = radius_px * _PIX_DEG
    return (
        (coords_df["lon"] >= lon_min + margin) &
        (coords_df["lon"] <= lon_max - margin) &
        (coords_df["lat"] >= lat_min + margin) &
        (coords_df["lat"] <= lat_max - margin)
    )


def _pcmv_chunk_vectorised(
    chunk_idxs: np.ndarray,       # integer indices into obs_arrays rows
    chunk_pids: list[str],
    neighbour_map: dict[str, list[int]],
    obs_arrays: dict[str, np.ndarray],  # band -> (N_pids, T)
    bands: list[str],
) -> list[dict]:
    """Vectorised PCMV scoring for a chunk of pixels.

    For each pixel, stacks all neighbour rows into a (K, T) matrix,
    computes shared-obs mean squared diff against the target row,
    then takes the minimum across neighbours — all in numpy with no Python
    pixel loop.
    """
    results = []
    for i, pid in zip(chunk_idxs, chunk_pids):
        nb_idxs = neighbour_map.get(pid)
        if not nb_idxs:
            continue
        nb_arr = np.array(nb_idxs, dtype=np.intp)

        band_pcmv: dict[str, float] = {}
        for band in bands:
            arr   = obs_arrays[band]
            target = arr[i]                          # (T,)
            nbs    = arr[nb_arr]                     # (K, T)

            # valid mask: target has obs
            t_valid = ~np.isnan(target)              # (T,)
            # for each neighbour: valid where both have obs
            nb_valid = ~np.isnan(nbs)                # (K, T)
            shared   = t_valid[np.newaxis, :] & nb_valid  # (K, T)
            n_shared = shared.sum(axis=1)            # (K,)

            # zero out positions with no shared obs to avoid nan contamination
            diff = nbs - target[np.newaxis, :]       # (K, T)
            diff_sq = diff ** 2
            diff_sq[~shared] = 0.0

            good = n_shared >= 4
            if not good.any():
                continue

            pcmv_per_nb = np.where(
                good,
                diff_sq.sum(axis=1) / np.maximum(n_shared, 1),
                np.inf,
            )
            min_pcmv = float(pcmv_per_nb.min())
            if np.isfinite(min_pcmv):
                band_pcmv[f"pcmv_{band}"] = min_pcmv

        if band_pcmv:
            row = {"point_id": pid, **band_pcmv}
            row["pcmv_mean"] = float(np.mean(list(band_pcmv.values())))
            results.append(row)
    return results


def compute_pcmv(
    pixel_df: pd.DataFrame,
    interior_pids: set[str],
    bands: list[str],
    n_jobs: int = -1,
    radius: int = _RADIUS,
) -> pd.DataFrame:
    """Compute minimum-neighbour PCMV for interior pixels.

    Uses raw observation dates (not binned), preserving the full asynchrony
    signal. For each pixel pair, only dates where both pixels have a valid
    cloud-free observation contribute.

    Returns DataFrame indexed by point_id with columns pcmv_<band> and pcmv_mean.
    """
    from scipy.spatial import KDTree

    pixel_df = pixel_df.copy()
    pixel_df["date"] = pixel_df["date"].astype(str)

    # Pixel grid coordinates
    coords_df = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .reset_index(drop=True)
    )
    all_pids  = coords_df["point_id"].tolist()
    pid_index = {pid: k for k, pid in enumerate(all_pids)}
    N = len(all_pids)

    # Map point_id → integer index, then pivot each band into a dense (N, T) array
    pixel_df["_pid_idx"] = pixel_df["point_id"].map(pid_index)
    all_dates = sorted(pixel_df["date"].unique())
    date_index = {d: i for i, d in enumerate(all_dates)}
    pixel_df["_date_idx"] = pixel_df["date"].map(date_index)
    T = len(all_dates)

    print(f"    Building {N:,} × {T} obs arrays ...")
    obs_arrays: dict[str, np.ndarray] = {}
    for band in bands:
        arr = np.full((N, T), np.nan, dtype=np.float32)
        sub = pixel_df[["_pid_idx", "_date_idx", band]].dropna(subset=[band])
        arr[sub["_pid_idx"].values, sub["_date_idx"].values] = sub[band].values.astype(np.float32)
        obs_arrays[band] = arr

    # Build neighbour map via KDTree — O(N log N) instead of O(N²)
    print(f"    Building neighbour map (KDTree) ...")
    radius_deg = radius * _PIX_DEG * 1.5
    lons = coords_df["lon"].values
    lats = coords_df["lat"].values
    tree = KDTree(np.column_stack([lons, lats]))

    interior_idxs = np.array([pid_index[p] for p in interior_pids if p in pid_index])
    query_pts = np.column_stack([lons[interior_idxs], lats[interior_idxs]])
    neighbour_lists = tree.query_ball_point(query_pts, r=radius_deg, workers=-1)

    neighbour_map: dict[str, list[int]] = {}
    for pid_i, nb_list in zip(interior_idxs, neighbour_lists):
        nbs = [j for j in nb_list if j != pid_i]
        if nbs:
            neighbour_map[all_pids[pid_i]] = nbs

    interior_list = [p for p in interior_pids if p in neighbour_map]
    if not interior_list:
        return pd.DataFrame()

    interior_idxs_list = np.array([pid_index[p] for p in interior_list], dtype=np.intp)

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    # Use more chunks than workers for finer progress reporting
    n_chunks  = n_workers * 8
    idx_chunks = np.array_split(interior_idxs_list, n_chunks)
    pid_chunks = np.array_split(interior_list, n_chunks)
    total = len(interior_list)
    print(f"    Scoring {total:,} pixels across {n_workers} threads ({n_chunks} chunks) ...")

    import threading
    completed = [0]
    lock = threading.Lock()

    def _score_with_progress(idx_chunk, pid_chunk):
        result = _pcmv_chunk_vectorised(idx_chunk, list(pid_chunk), neighbour_map, obs_arrays, bands)
        with lock:
            completed[0] += len(idx_chunk)
            pct = 100 * completed[0] / total
            print(f"    {completed[0]:>9,} / {total:,}  ({pct:.1f}%)", end="\r", flush=True)
        return result

    nested = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(_score_with_progress)(idx_chunk, pid_chunk)
        for idx_chunk, pid_chunk in zip(idx_chunks, pid_chunks) if len(idx_chunk)
    )
    print()  # newline after \r progress
    results = [row for rows in nested for row in rows]

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("point_id")


def _make_violin_plot(
    band_data: list[np.ndarray],
    band_colors: list[str],
    valid_labels: list[str],
    band_name: str,
    label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(valid_labels))))
    fig.suptitle(f"PCMV (min-neighbour, r={_RADIUS}) — {band_name} — {label}", fontsize=11)

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

    ax.set_xlabel(f"pcmv_{band_name}  (mean squared NDVI diff, lower = more similar)", fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.75, label="presence"),
            Patch(facecolor="coral",     alpha=0.75, label="absence"),
        ],
        fontsize=8, loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_location_experiment(
    pixel_df: pd.DataFrame,
    location_id: str,
    bbox: list[float],
    bands: list[str],
    out_dir: Path,
    radius: int = _RADIUS,
) -> None:
    """Like run_experiment but for an unlabelled location — no presence/absence split.

    All pixels are treated as a single group and plotted together.
    """
    pixel_df = pixel_df.copy()
    pixel_df["date"] = pd.to_datetime(pixel_df["date"]).dt.strftime("%Y-%m-%d")

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    all_pids     = set(pixel_coords["point_id"].tolist())

    interior_mask = _interior_mask(pixel_coords, tuple(bbox), radius)
    interior_pids = set(pixel_coords[interior_mask]["point_id"].tolist())
    if len(interior_pids) < 5:
        interior_pids = all_pids

    print(f"  {location_id}: {len(interior_pids):,} interior pixels ...")
    pcmv_df = compute_pcmv(pixel_df, interior_pids, bands, radius=radius)
    if pcmv_df.empty:
        print("No PCMV results computed.")
        return

    pcmv_df.to_csv(out_dir / "pcmv_all.csv")

    summary_lines = [
        f"PCMV diagnostic — {location_id}",
        f"radius=r{radius} ({2*radius+1}×{2*radius+1} window)  aggregation=min-neighbour  interior_padding={radius}px",
        "",
        f"{'band':<20} {'n_px':>6}  {'mean_pcmv':>10}  {'std_pcmv':>10}  {'p50':>8}",
        "-" * 65,
    ]

    plot_cols = (
        [f"pcmv_{b}" for b in bands if f"pcmv_{b}" in pcmv_df.columns]
        + (["pcmv_mean"] if "pcmv_mean" in pcmv_df.columns else [])
    )

    for col in plot_cols:
        band_name = col.replace("pcmv_", "")
        vals = pcmv_df[col].dropna()
        if len(vals) < 2:
            continue

        summary_lines.append(
            f"{band_name:<20} {len(vals):>6}  {vals.mean():>10.4f}  {vals.std():>10.4f}  {vals.median():>8.4f}"
        )

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.suptitle(f"PCMV (min-neighbour, r={radius}) — {band_name} — {location_id}", fontsize=11)
        ax.hist(vals.values, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
        ax.set_xlabel(f"pcmv_{band_name}  (mean sq diff, lower = more similar)", fontsize=9)
        ax.set_ylabel("pixels", fontsize=9)
        ax.axvline(vals.median(), color="black", linewidth=1, linestyle="--", label=f"median={vals.median():.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        out_path = out_dir / f"pcmv_{band_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nOutputs written to: {out_dir}")


def run_experiment(
    pixel_df: pd.DataFrame,
    regions: list,
    bands: list[str],
    out_dir: Path,
    label: str,
    radius: int = _RADIUS,
) -> None:
    pixel_df = pixel_df.copy()
    pixel_df["date"] = pd.to_datetime(pixel_df["date"]).dt.strftime("%Y-%m-%d")

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})
    pixel_df     = pixel_df[pixel_df["point_id"].isin(all_labels.index)].copy()

    summary_lines = [
        f"PCMV diagnostic — {label}",
        f"radius=r{radius} ({2*radius+1}×{2*radius+1} window)  aggregation=min-neighbour  interior_padding={radius}px",
        "",
        f"{'region':<40} {'class':<10} {'n_px':>6}  {'mean_pcmv':>10}  {'std_pcmv':>10}",
        "-" * 85,
    ]

    plot_labels: list[str] = []
    plot_colors: list[str] = []
    all_pcmv: list[pd.DataFrame] = []

    for region in regions:
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
        pcmv_df = compute_pcmv(region_df, interior_pids, bands, radius=radius)
        if pcmv_df.empty:
            continue

        pcmv_df["label"]  = all_labels.reindex(pcmv_df.index)
        pcmv_df["region"] = region.id
        pcmv_df = pcmv_df.dropna(subset=["label"])
        all_pcmv.append(pcmv_df)

        vals = pcmv_df["pcmv_mean"].dropna() if "pcmv_mean" in pcmv_df.columns else pd.Series(dtype=float)
        if len(vals) < 2:
            continue

        color    = "steelblue" if region.is_presence else "coral"
        cls_name = "presence"  if region.is_presence else "absence"
        plot_labels.append(f"{region.id}  (n={len(vals)}, μ={vals.mean():.4f})")
        plot_colors.append(color)

        summary_lines.append(
            f"{region.id:<40} {cls_name:<10} {len(vals):>6}  {vals.mean():>10.4f}  {vals.std():>10.4f}"
        )

    if not all_pcmv:
        print("No data computed.")
        return

    combined = pd.concat(all_pcmv)
    combined.to_csv(out_dir / "pcmv_all.csv")

    region_by_id    = {r.id: r for r in regions}
    ordered_regions = []
    for lbl in plot_labels:
        rid = lbl.split("  ")[0].strip()
        if rid in region_by_id:
            ordered_regions.append(region_by_id[rid])

    plot_cols = (
        [f"pcmv_{b}" for b in bands if f"pcmv_{b}" in combined.columns]
        + (["pcmv_mean"] if "pcmv_mean" in combined.columns else [])
    )

    for col in plot_cols:
        band_name    = col.replace("pcmv_", "")
        band_data    = []
        band_colors  = []
        valid_labels = []

        for region, lbl, color in zip(ordered_regions, plot_labels, plot_colors):
            vals = combined[combined["region"] == region.id][col].dropna()
            if len(vals) < 2:
                continue
            band_data.append(vals.values)
            band_colors.append(color)
            valid_labels.append(lbl)

        if not band_data:
            continue

        out_path = out_dir / f"pcmv_{band_name}.png"
        _make_violin_plot(band_data, band_colors, valid_labels, band_name, label, out_path)
        print(f"  Saved {out_path.name}")

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nOutputs written to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",        action="store_true", help="Run on all regions in training.yaml")
    parser.add_argument("--experiment", default=None, help="Experiment module name (e.g. v7_frenchs_only)")
    parser.add_argument("--region-ids", nargs="+",   default=None, help="Individual region IDs")
    parser.add_argument("--sites",      nargs="+",   default=None, help="Filter to these site prefixes")
    parser.add_argument("--location",   default=None, help="Location ID from data/locations/<id>.yaml (unlabelled)")
    parser.add_argument("--bands",      nargs="+",   default=["NDVI"], help="Bands to compute PCMV for (default: NDVI)")
    parser.add_argument("--radius",     type=int,    default=_RADIUS, help="Neighbourhood radius in pixels (default: 2)")
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    if not args.all and not args.experiment and not args.region_ids and not args.location:
        parser.error("Provide --all, --experiment, --region-ids, or --location")

    bands = [b for b in args.bands if b in BANDS]
    if not bands:
        print(f"No valid bands in {args.bands}. Choose from {BANDS}")
        sys.exit(1)

    index_bands = {"NDVI", "NDWI", "EVI"}
    needs_indices = any(b in index_bands for b in bands)

    # --- Location mode: unlabelled site parquet ---
    if args.location:
        from utils.location import get as get_location
        loc     = get_location(args.location)
        out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "pcmv" / args.location
        out_dir.mkdir(parents=True, exist_ok=True)

        # Columns already present in location parquets (indices pre-computed)
        avail_cols = {"point_id", "lon", "lat", "date", "scl_purity",
                      "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
                      "NDVI", "NDWI", "EVI"}
        read_cols = ["point_id", "lon", "lat", "date", "scl_purity"] + [
            b for b in bands if b in avail_cols
        ]

        chunks: list[pd.DataFrame] = []
        for year, paths in loc.parquet_tile_paths().items():
            for path in paths:
                print(f"  Reading {path.name} ...")
                pf = pq.ParquetFile(path)
                avail = set(pf.schema_arrow.names)
                cols  = [c for c in read_cols if c in avail]
                for rg in range(pf.metadata.num_row_groups):
                    chunks.append(pf.read_row_group(rg, columns=cols).to_pandas())

        if not chunks:
            print("No data found.")
            sys.exit(1)

        pixel_df = pd.concat(chunks, ignore_index=True)

        if needs_indices and "NDVI" not in pixel_df.columns:
            from analysis.constants import add_spectral_indices
            pixel_df = add_spectral_indices(pixel_df)

        print(f"Loaded {len(pixel_df):,} observations, {pixel_df['point_id'].nunique():,} pixels")
        run_location_experiment(pixel_df, args.location, loc.bbox, bands, out_dir, radius=args.radius)
        return

    # --- Training region mode ---
    if args.all:
        from utils.regions import load_regions
        regions_all = load_regions()
        region_ids  = [r.id for r in regions_all]
        label = "all"
    elif args.region_ids:
        region_ids = args.region_ids
        label = "_".join(region_ids[:2])
    else:
        exp        = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT
        region_ids = list(exp.region_ids)
        label      = args.experiment
        bands      = [b for b in bands if b in exp.feature_cols]

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "pcmv" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    regions  = select_regions(region_ids)
    tile_ids = tile_ids_for_regions(region_ids)

    raw_for_indices = ["B02", "B03", "B04", "B08"] if needs_indices else []
    read_bands = sorted(set(
        [b for b in bands if b not in index_bands] + raw_for_indices
    ))
    read_cols  = ["point_id", "lon", "lat", "date", "scl_purity"] + read_bands

    chunks = []
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

    from analysis.constants import add_spectral_indices
    if needs_indices:
        pixel_df = add_spectral_indices(pixel_df)

    if args.sites:
        pixel_df = pixel_df[pixel_df["point_id"].map(_point_site).isin(args.sites)]
        regions  = [r for r in regions if any(r.id.startswith(s) for s in args.sites)]
        label    = "_".join(args.sites)

    print(f"Loaded {len(pixel_df):,} observations, {pixel_df['point_id'].nunique():,} pixels")
    run_experiment(pixel_df, regions, bands, out_dir, label, radius=args.radius)


if __name__ == "__main__":
    main()
