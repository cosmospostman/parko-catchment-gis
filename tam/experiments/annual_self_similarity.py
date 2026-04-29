"""tam/experiments/annual_self_similarity.py — Annual self-similarity diagnostic.

For each pixel, computes pairwise Pearson correlation between its annual NDVI
(or per-band) time series curves across years, then takes the mean of all
year-pair correlations as a self-similarity score.

The hypothesis: Parkinsonia stands are remarkably consistent year-on-year
(near-zero inter-bbox separation in Norman Road comparisons). Native vegetation
shows more inter-annual variation due to rainfall variability, species mixing,
and individual phenological responses.

This is a purely per-pixel feature — no spatial neighbourhood needed, no buffer
zone requirement. Can be computed from existing training data immediately.

Usage:
    python -m tam.experiments.annual_self_similarity --all --sites norman_road
    python -m tam.experiments.annual_self_similarity --all
    python -m tam.experiments.annual_self_similarity --experiment v7_frenchs_only
    python -m tam.experiments.annual_self_similarity --region-ids norman_road_presence_1 norman_road_absence_5
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
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "EVI"]

# Fortnightly DOY bins — enough temporal resolution, robust to missing obs
_DOY_BINS = np.arange(1, 366, 14)
_MIN_BINS  = 6   # minimum non-NaN bins required to use a year's curve
_MIN_YEARS = 2   # minimum years with valid curves to compute self-similarity


def _point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < _MIN_BINS:
        return None
    av = a[valid] - a[valid].mean()
    bv = b[valid] - b[valid].mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    if denom < 1e-10:
        return None
    r = float((av * bv).sum() / denom)
    return r if np.isfinite(r) else None


def _score_pixel(pid: str, annual_curves: dict[int, np.ndarray]) -> dict | None:
    """Compute mean pairwise inter-annual Pearson correlation for one pixel."""
    years = sorted(annual_curves.keys())
    if len(years) < _MIN_YEARS:
        return None

    corrs = []
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            r = _pearson(annual_curves[years[i]], annual_curves[years[j]])
            if r is not None:
                corrs.append(r)

    if not corrs:
        return None

    return {
        "point_id":        pid,
        "self_sim_mean":   float(np.mean(corrs)),
        "self_sim_p25":    float(np.percentile(corrs, 25)),
        "self_sim_min":    float(np.min(corrs)),
        "n_year_pairs":    len(corrs),
    }


def compute_self_similarity(
    pixel_df: pd.DataFrame,
    bands: list[str],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute inter-annual self-similarity per pixel per band.

    Returns DataFrame indexed by point_id with columns:
      ss_mean_<band>, ss_p25_<band>, ss_min_<band>, ss_mean_mean (average across bands).
    """
    pixel_df = pixel_df.copy()
    pixel_df["doy_bin"] = np.searchsorted(_DOY_BINS, pixel_df["doy"].values, side="right")

    # Fortnightly mean per (pixel, year, doy_bin, band)
    ts = (
        pixel_df.groupby(["point_id", "year", "doy_bin"])[bands]
        .mean()
        .reset_index()
    )

    pids = ts["point_id"].unique()
    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    def _score_band(band: str) -> pd.DataFrame:
        piv = ts.pivot_table(index=["point_id", "year"], columns="doy_bin", values=band)
        n_bins = piv.shape[1]

        results = []
        for pid in pids:
            if pid not in piv.index.get_level_values(0):
                continue
            pixel_years = piv.loc[pid]
            annual_curves = {
                int(yr): row.values.astype(float)
                for yr, row in pixel_years.iterrows()
                if (~np.isnan(row.values)).sum() >= _MIN_BINS
            }
            row = _score_pixel(pid, annual_curves)
            if row:
                results.append(row)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).set_index("point_id")
        return df.rename(columns={
            "self_sim_mean": f"ss_mean_{band}",
            "self_sim_p25":  f"ss_p25_{band}",
            "self_sim_min":  f"ss_min_{band}",
            "n_year_pairs":  f"ss_nyears_{band}",
        })

    band_dfs = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_score_band)(band) for band in bands
    )

    result = None
    for df in band_dfs:
        if df.empty:
            continue
        result = df if result is None else result.join(df, how="outer")

    if result is None or result.empty:
        return pd.DataFrame()

    mean_cols = [f"ss_mean_{b}" for b in bands if f"ss_mean_{b}" in result.columns]
    if mean_cols:
        result["ss_mean_mean"] = result[mean_cols].mean(axis=1)

    return result


def _make_violin_plot(
    plot_data: list[np.ndarray],
    plot_labels: list[str],
    plot_colors: list[str],
    col: str,
    label: str,
    out_path: Path,
) -> None:
    n = len(plot_data)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n)))
    band_name = col.replace("ss_mean_", "").replace("ss_p25_", "").replace("ss_min_", "")
    stat_name = "mean" if "ss_mean" in col else ("p25" if "ss_p25" in col else "min")
    fig.suptitle(f"Annual self-similarity ({stat_name}) — {band_name} — {label}", fontsize=11)

    if plot_data:
        positions = list(range(n, 0, -1))
        parts = ax.violinplot(plot_data, positions=positions, vert=False,
                              showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(plot_labels, fontsize=7)

    ax.set_xlabel(f"inter-annual Pearson r ({stat_name})", fontsize=9)
    ax.set_xlim(-0.1, 1.05)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="steelblue", alpha=0.75, label="presence"),
                 Patch(facecolor="coral",     alpha=0.75, label="absence")],
        fontsize=8, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    pixel_df: pd.DataFrame,
    regions: list,
    bands: list[str],
    out_dir: Path,
    label: str,
) -> None:
    pixel_df = pixel_df.copy()
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year

    pixel_coords = pixel_df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})
    pixel_df     = pixel_df[pixel_df["point_id"].isin(all_labels.index)].copy()

    summary_lines = [
        f"Annual self-similarity diagnostic — {label}",
        f"min_bins={_MIN_BINS}  min_years={_MIN_YEARS}",
        "",
        f"{'region':<40} {'class':<10} {'n_px':>6}  {'mean_ss':>8}  {'std_ss':>8}",
        "-" * 80,
    ]

    plot_labels: list[str] = []
    plot_colors: list[str] = []
    all_ss: list[pd.DataFrame] = []

    for region in regions:
        region_pids = set(
            pixel_df[pixel_df["point_id"].str.startswith(region.id + "_")]["point_id"].tolist()
        )
        if not region_pids:
            continue

        region_df = pixel_df[pixel_df["point_id"].isin(region_pids)]
        print(f"  {region.id}: {len(region_pids)} pixels ...")

        ss_df = compute_self_similarity(region_df, bands)
        if ss_df.empty:
            continue

        ss_df["label"]  = all_labels.reindex(ss_df.index)
        ss_df["region"] = region.id
        ss_df = ss_df.dropna(subset=["label"])
        all_ss.append(ss_df)

        vals = ss_df["ss_mean_mean"].dropna() if "ss_mean_mean" in ss_df.columns else pd.Series(dtype=float)
        if len(vals) < 2:
            continue

        color    = "steelblue" if region.is_presence else "coral"
        cls_name = "presence"  if region.is_presence else "absence"
        plot_labels.append(f"{region.id}  (n={len(vals)}, μ={vals.mean():.3f})")
        plot_colors.append(color)

        summary_lines.append(
            f"{region.id:<40} {cls_name:<10} {len(vals):>6}  {vals.mean():>8.4f}  {vals.std():>8.4f}"
        )

    if not all_ss:
        print("No data computed.")
        return

    combined = pd.concat(all_ss)
    combined.to_csv(out_dir / "self_similarity_all.csv")

    region_by_id    = {r.id: r for r in regions}
    ordered_regions = []
    for lbl in plot_labels:
        rid = lbl.split("  ")[0].strip()
        if rid in region_by_id:
            ordered_regions.append(region_by_id[rid])

    # One plot per band (ss_mean_<band>) plus the cross-band mean
    plot_cols = (
        [f"ss_mean_{b}" for b in bands if f"ss_mean_{b}" in combined.columns]
        + (["ss_mean_mean"] if "ss_mean_mean" in combined.columns else [])
    )

    for col in plot_cols:
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

        out_path = out_dir / f"self_sim_{col.replace('ss_mean_', '').replace('ss_', '')}.png"
        _make_violin_plot(band_data, valid_labels, band_colors, col, label, out_path)
        print(f"  Saved {out_path.name}")

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
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    if not args.all and not args.experiment and not args.region_ids:
        parser.error("Provide --all, --experiment, or --region-ids")

    bands = [b for b in args.bands if b in BANDS]

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

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "self-similarity" / label
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

    from analysis.constants import add_spectral_indices
    if any(b in bands for b in ("NDVI", "NDWI", "EVI")):
        pixel_df = add_spectral_indices(pixel_df)

    if args.sites:
        pixel_df = pixel_df[pixel_df["point_id"].map(_point_site).isin(args.sites)]
        regions  = [r for r in regions if any(r.id.startswith(s) for s in args.sites)]
        label    = "_".join(args.sites)

    print(f"Loaded {len(pixel_df):,} observations, {pixel_df['point_id'].nunique():,} pixels")
    run_experiment(pixel_df, regions, bands, out_dir, label)


if __name__ == "__main__":
    main()
