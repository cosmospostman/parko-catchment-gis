"""Dry-season NIR stability analysis over the Longreach Parkinsonia infestation.

For each of the 374 pixels in the 6-year archive (longreach_pixels.parquet), compute:
  - Per-year dry-season (Jun–Oct) median B08 (NIR)
  - nir_mean  — mean of those annual medians
  - nir_std   — standard deviation across years
  - nir_cv    — coefficient of variation (std / mean); lower = more stable year-to-year
  - n_years   — number of years with ≥ MIN_OBS qualifying observations

Outputs:
  outputs/longreach_dry_nir_stats.parquet
  outputs/longreach_dry_nir_map.png         (pixels coloured by nir_mean)
  outputs/longreach_dry_nir_cv_map.png      (pixels coloured by nir_cv)
  outputs/longreach_dry_nir_cv_hist.png     (histogram of nir_cv)

See research/LONGREACH-DRY-NIR.md for the full analysis plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_PATH  = PROJECT_ROOT / "data" / "longreach_pixels.parquet"
OUT_DIR       = PROJECT_ROOT / "outputs" / "longreach-dry-nir"

# Dry season: months 6–10 inclusive
DRY_MONTHS    = {6, 7, 8, 9, 10}

# Minimum qualifying observations per (pixel, year) to include that year
MIN_OBS       = 5

# Quality filter: retain rows where at least this fraction of SCL pixels are
# valid (not cloud / shadow / saturated)
SCL_PURITY_MIN = 0.5

# Original infestation patch bbox — annotated on maps, used for success criteria
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Full survey area (infestation + southern extension) — map and WMS extent
# Derived from the extended longreach_pixels.parquet fetch
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_filter(path: Path) -> pd.DataFrame:
    log(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log(f"  Pixels: {df['point_id'].nunique()}  |  Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    before = len(df)
    df = df[df["scl_purity"] >= SCL_PURITY_MIN].copy()
    dropped = before - len(df)
    log(f"  Quality filter (scl_purity ≥ {SCL_PURITY_MIN}): dropped {dropped:,} rows ({100*dropped/before:.1f}%), retained {len(df):,}")

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    dry_mask    = df["month"].isin(DRY_MONTHS)
    df_dry      = df[dry_mask].copy()
    log(f"  Dry-season filter (months {sorted(DRY_MONTHS)}): {len(df_dry):,} rows ({100*len(df_dry)/len(df):.1f}% of quality-filtered)")

    return df_dry


def per_year_medians(df_dry: pd.DataFrame) -> pd.DataFrame:
    log("\nComputing per-pixel per-year dry-season B08 medians...")

    grouped = df_dry.groupby(["point_id", "year"])
    obs_counts = grouped["B08"].count()
    log(f"  (pixel, year) groups: {len(obs_counts):,}")
    log(f"  Observations per group — min: {obs_counts.min()}  median: {obs_counts.median():.0f}  max: {obs_counts.max()}")

    # Only include (pixel, year) pairs with enough observations
    valid_groups = obs_counts[obs_counts >= MIN_OBS].index
    n_dropped_groups = len(obs_counts) - len(valid_groups)
    log(f"  Groups with < {MIN_OBS} obs (excluded): {n_dropped_groups} ({100*n_dropped_groups/len(obs_counts):.1f}%)")

    df_valid = df_dry.set_index(["point_id", "year"]).loc[valid_groups].reset_index()
    medians = (
        df_valid
        .groupby(["point_id", "year"])["B08"]
        .median()
        .rename("nir_median")
        .reset_index()
    )
    log(f"  Annual medians computed: {len(medians):,} (pixel × year)")
    return medians


def per_pixel_stats(medians: pd.DataFrame, df_dry: pd.DataFrame) -> pd.DataFrame:
    log("\nAggregating per-pixel summary statistics across years...")

    stats = (
        medians
        .groupby("point_id")["nir_median"]
        .agg(
            nir_mean="mean",
            nir_std="std",
            n_years="count",
        )
        .reset_index()
    )
    stats["nir_cv"] = stats["nir_std"] / stats["nir_mean"]

    # Attach lon/lat from the original data (stable per pixel)
    coords = df_dry[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    stats = stats.merge(coords, on="point_id", how="left")

    log(f"  Pixels in summary: {len(stats)}")
    log(f"  n_years per pixel — min: {stats['n_years'].min()}  median: {stats['n_years'].median():.0f}  max: {stats['n_years'].max()}")
    log(f"  nir_mean  — min: {stats['nir_mean'].min():.4f}  median: {stats['nir_mean'].median():.4f}  max: {stats['nir_mean'].max():.4f}")
    log(f"  nir_cv    — min: {stats['nir_cv'].min():.4f}  median: {stats['nir_cv'].median():.4f}  max: {stats['nir_cv'].max():.4f}")

    # Flag pixels inside the high-density bbox
    stats["in_hd_bbox"] = (
        stats["lon"].between(HD_LON_MIN, HD_LON_MAX) &
        stats["lat"].between(HD_LAT_MIN, HD_LAT_MAX)
    )
    n_hd = stats["in_hd_bbox"].sum()
    log(f"\n  Pixels in high-density bbox: {n_hd}")
    if n_hd > 0:
        hd = stats[stats["in_hd_bbox"]]
        all_rank_mean = stats["nir_mean"].rank(pct=True)
        hd_pct_mean   = all_rank_mean[stats["in_hd_bbox"]].mean() * 100
        all_rank_cv   = stats["nir_cv"].rank(pct=True)
        hd_pct_cv     = all_rank_cv[stats["in_hd_bbox"]].mean() * 100
        log(f"  HD bbox nir_mean percentile (higher = more NIR): {hd_pct_mean:.1f}th")
        log(f"  HD bbox nir_cv   percentile (lower = more stable): {100 - hd_pct_cv:.1f}th (inverted)")
        log(f"  HD bbox nir_mean range: {hd['nir_mean'].min():.4f} – {hd['nir_mean'].max():.4f}")
        log(f"  HD bbox nir_cv   range: {hd['nir_cv'].min():.4f} – {hd['nir_cv'].max():.4f}")

    return stats


def save_stats(stats: pd.DataFrame, out_dir: Path) -> None:
    out_path = out_dir / "longreach_dry_nir_stats.parquet"
    cols = ["point_id", "lon", "lat", "nir_mean", "nir_std", "nir_cv", "n_years", "in_hd_bbox"]
    stats[cols].to_parquet(out_path, index=False)
    log(f"\nSaved stats: {out_path.relative_to(PROJECT_ROOT)}")


def plot_spatial(stats: pd.DataFrame, col: str, cmap: str, label: str,
                 out_path: Path, title: str,
                 bg_img: np.ndarray | None = None) -> None:
    """Scatter plot of pixel centroids coloured by a summary statistic.

    Zoomed to HD_BBOX. If bg_img is provided it is displayed as the background
    using HD_BBOX as the geographic extent. Scatter markers are semi-transparent
    so the satellite imagery remains visible beneath them.
    """
    lon_min, lat_min, lon_max, lat_max = SURVEY_BBOX
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Figure height preserves metric aspect ratio at the scene latitude
    lat_centre = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320
    fig_w = 7
    fig_h = max(4.0, fig_w * (lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    if bg_img is not None:
        ax.imshow(
            bg_img,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )

    # Marker size: approximate one 10 m S2 pixel in points².
    # fig_w inches at 150 dpi → fig_w*150 device px across lon_span degrees.
    # 72 pt/inch conversion gives points per degree.
    pt_per_deg = fig_w * 72 / lon_span
    marker_pt  = (10 / lon_m_per_deg) * pt_per_deg   # diameter in points
    marker_s   = max(0.6, marker_pt ** 2 / 10)        # scatter s = area in pt²

    log(f"  Plotting {len(stats)} pixels")

    sc = ax.scatter(
        stats["lon"], stats["lat"],
        c=stats[col], cmap=cmap,
        s=marker_s,
        linewidths=0.0,
        alpha=0.55,   # semi-transparent so imagery shows through
        zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label, fontsize=8)

    # Outline the original infestation patch
    ax.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN,
        HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
        label="Infestation patch", zorder=4,
    ))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
              facecolor="black", labelcolor="white", edgecolor="none")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved map: {out_path.relative_to(PROJECT_ROOT)}")


def plot_cv_histogram(stats: pd.DataFrame, out_path: Path) -> None:
    """Histogram of nir_cv across all pixels."""
    median_cv = stats["nir_cv"].median()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(stats["nir_cv"], bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(median_cv, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Median CV = {median_cv:.4f}")

    # Overlay HD bbox pixels
    hd = stats[stats["in_hd_bbox"]]
    if len(hd) > 0:
        ax.scatter(
            hd["nir_cv"],
            [2] * len(hd),    # rug at y=2
            marker="|", color="orange", s=80, linewidths=1.5, zorder=3,
            label=f"HD bbox pixels (n={len(hd)})",
        )

    ax.set_xlabel("NIR CV (dry-season, inter-annual)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of dry-season NIR stability (CV)\n"
                 "Lower CV = more stable year-to-year = more persistent canopy", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved histogram: {out_path.relative_to(PROJECT_ROOT)}")


def log_success_criteria(stats: pd.DataFrame) -> None:
    log("\n--- Success criteria (research/LONGREACH-DRY-NIR.md) ---")

    # 1. HD pixels in top quartile of nir_mean
    q75_mean = stats["nir_mean"].quantile(0.75)
    hd = stats[stats["in_hd_bbox"]]
    n_hd_top = (hd["nir_mean"] >= q75_mean).sum()
    pct = 100 * n_hd_top / len(hd) if len(hd) else 0
    status = "PASS" if pct >= 50 else "FAIL"
    log(f"  [1] HD pixels in top-quartile nir_mean (>= {q75_mean:.4f}): "
        f"{n_hd_top}/{len(hd)} ({pct:.0f}%)  → {status}")

    # 2. HD pixels nir_cv below dataset median
    median_cv = stats["nir_cv"].median()
    n_hd_stable = (hd["nir_cv"] <= median_cv).sum()
    pct2 = 100 * n_hd_stable / len(hd) if len(hd) else 0
    status2 = "PASS" if pct2 >= 50 else "FAIL"
    log(f"  [2] HD pixels with nir_cv ≤ dataset median ({median_cv:.4f}): "
        f"{n_hd_stable}/{len(hd)} ({pct2:.0f}%)  → {status2}")

    # 3. Spatial coherence: Moran's I approximation via local variance
    #    Simple proxy: mean nir_mean of spatial neighbours should correlate with pixel value.
    #    We use the Pearson r between a pixel's nir_mean and the mean of its 8 nearest
    #    neighbours as a rough coherence score.
    from scipy.spatial import cKDTree
    coords = stats[["lon", "lat"]].values
    tree   = cKDTree(coords)
    dists, idxs = tree.query(coords, k=9)   # self + 8 neighbours
    neighbour_means = stats["nir_mean"].values[idxs[:, 1:]].mean(axis=1)
    r = np.corrcoef(stats["nir_mean"].values, neighbour_means)[0, 1]
    status3 = "PASS" if r >= 0.5 else "FAIL"
    log(f"  [3] Spatial coherence (Pearson r with 8-neighbour mean): "
        f"r = {r:.3f}  → {status3}")

    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach dry-season NIR stability ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_dry  = load_and_filter(PARQUET_PATH)
    medians = per_year_medians(df_dry)
    stats   = per_pixel_stats(medians, df_dry)

    save_stats(stats, OUT_DIR)

    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px for bbox {SURVEY_BBOX}")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    log("\nGenerating spatial maps...")
    plot_spatial(
        stats, col="nir_mean", cmap="YlOrRd",
        label="Mean dry-season B08 (reflectance)",
        out_path=OUT_DIR / "longreach_dry_nir_map.png",
        title="Longreach — mean dry-season NIR reflectance (B08), 2020–2025\n(Jun–Oct median per year, then mean across years)",
        bg_img=bg_img,
    )
    plot_spatial(
        stats, col="nir_cv", cmap="RdYlGn_r",
        label="CV (std / mean) of dry-season B08",
        out_path=OUT_DIR / "longreach_dry_nir_cv_map.png",
        title="Longreach — inter-annual NIR stability (CV), 2020–2025\n(lower = more stable = more persistent canopy)",
        bg_img=bg_img,
    )

    log("\nGenerating CV histogram...")
    plot_cv_histogram(stats, OUT_DIR / "longreach_dry_nir_cv_hist.png")

    log_success_criteria(stats)

    log("\nDone.")


if __name__ == "__main__":
    main()
