"""Wet/dry seasonal amplitude analysis over the Longreach Parkinsonia infestation.

Primary metric: window-free NDVI amplitude = annual 90th-percentile NDVI minus annual
10th-percentile NDVI, computed per pixel per year and averaged across years (rec_p).
This is agnostic to the timing of the wet-season peak and dry-season trough — both are
allowed to shift year to year with rainfall.

Secondary (reference) metric: fixed-window recession = median NDVI (Mar–May) minus
median NDVI (Jul–Sep) as previously computed (rec_mean). Logged for comparison.

The hypothesis is *inverted* from a naive wet/dry B08 swing:
  - Parkinsonia: deep-rooted, sustains active chlorophyll into Apr–May (strong
    post-wet peak), then recedes as days shorten → HIGH amplitude
  - Riparian / grassland: lower post-wet NDVI, flatter year-round → LOW amplitude

Combined with dry-season NIR CV from the previous analysis, this gives a 2D
discriminator:
  Parkinsonia  → low nir_cv  + HIGH rec_p   (bottom-right)
  Grassland    → high nir_cv + low  rec_p   (top-left)
  Riparian     → moderate CV + low  rec_p   (top-left, lower CV than grassland)

Outputs:
  outputs/longreach-wet-dry-amp/longreach_amp_stats.parquet
  outputs/longreach-wet-dry-amp/longreach_ndvi_contrast.png   (new)
  outputs/longreach-wet-dry-amp/longreach_rec_map.png
  outputs/longreach-wet-dry-amp/longreach_rec_hist.png
  outputs/longreach-wet-dry-amp/longreach_nir_cv_vs_rec.png
  outputs/longreach-wet-dry-amp/longreach_monthly_profiles.png

See research/LONGREACH-WET-DRY-AMP.md for the analysis plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

from utils.location import get as _get_loc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_PATH  = _get_loc("longreach").parquet_path()
DRY_NIR_STATS = PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
OUT_DIR       = PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp"

# Fixed-window seasons (kept for reference / per-year contrast table)
PEAK_MONTHS   = {3, 4, 5}      # March–May
TROUGH_MONTHS = {7, 8, 9}      # July–September

# Minimum qualifying observations per (pixel, year) for percentile computation
MIN_OBS_PER_YEAR = 10

# Minimum qualifying observations per (pixel, window, year) for fixed-window medians
MIN_OBS = 5

# Rolling window for contrast time series (days)
ROLLING_DAYS = 30

# Quality filter
SCL_PURITY_MIN = 0.5

# Infestation patch bbox
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Full survey area
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

# Riparian proxy: top-10% nir_mean from dry-NIR results
RIPARIAN_PERCENTILE = 0.90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_filter(path: Path) -> pd.DataFrame:
    log(f"Loading parquet: {path}")
    df = pd.read_parquet(path)
    log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log(f"  Pixels: {df['point_id'].nunique()}  |  "
        f"Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    before = len(df)
    df = df[df["scl_purity"] >= SCL_PURITY_MIN].copy()
    dropped = before - len(df)
    log(f"  Quality filter (scl_purity ≥ {SCL_PURITY_MIN}): "
        f"dropped {dropped:,} rows ({100*dropped/before:.1f}%), "
        f"retained {len(df):,}")

    df["ndvi"]  = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    return df


def assign_classes(df: pd.DataFrame, dry_nir: pd.DataFrame) -> pd.DataFrame:
    """Add in_hd_bbox, is_riparian, is_grassland columns to observation-level df."""
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    dn = dry_nir.copy()
    dn["is_riparian"]  = (~dn["in_hd_bbox"]) & (dn["nir_mean"] >= rip_thresh)
    dn["is_grassland"] = (~dn["in_hd_bbox"]) & (~dn["is_riparian"])
    return df.merge(dn[["point_id", "in_hd_bbox", "is_riparian", "is_grassland"]],
                    on="point_id", how="left")


def class_ids(dry_nir: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Return (hd_ids, rip_ids, grass_ids) Series of point_id."""
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    hd_ids    = dry_nir.loc[dry_nir["in_hd_bbox"], "point_id"]
    rip_ids   = dry_nir.loc[
        ~dry_nir["in_hd_bbox"] & (dry_nir["nir_mean"] >= rip_thresh), "point_id"]
    grass_ids = dry_nir.loc[
        ~dry_nir["in_hd_bbox"] & (dry_nir["nir_mean"] < rip_thresh), "point_id"]
    return hd_ids, rip_ids, grass_ids


# ---------------------------------------------------------------------------
# Step A — Daily NDVI contrast time series (window-free diagnostic)
# ---------------------------------------------------------------------------

def ndvi_contrast_time_series(df: pd.DataFrame) -> None:
    """Daily infestation − grassland NDVI contrast with 30-day rolling mean."""
    log("\n--- NDVI contrast time series (window-free) ---")

    daily_inf   = (df[df["in_hd_bbox"]]
                   .groupby("date")["ndvi"].mean().rename("inf"))
    daily_grass = (df[df["is_grassland"]]
                   .groupby("date")["ndvi"].mean().rename("grass"))

    contrast = pd.concat([daily_inf, daily_grass], axis=1).dropna()
    contrast["diff"] = contrast["inf"] - contrast["grass"]

    frac_positive = (contrast["diff"] > 0).mean()
    log(f"  Dates with contrast > 0: {(contrast['diff'] > 0).sum()}/{len(contrast)} "
        f"({100 * frac_positive:.1f}%)")

    contrast["year"] = contrast.index.year
    log("\n  Per-year contrast summary:")
    for yr, grp in contrast.groupby("year"):
        idx_max = grp["diff"].idxmax()
        idx_min = grp["diff"].idxmin()
        frac    = (grp["diff"] > 0).mean()
        log(f"    {yr}: max={grp['diff'].max():.4f} on {idx_max.date()}  "
            f"min={grp['diff'].min():.4f} on {idx_min.date()}  "
            f"frac>0={frac:.2f}")

    contrast_sorted = contrast.sort_index()
    contrast_sorted["rolling"] = (
        contrast_sorted["diff"]
        .rolling(f"{ROLLING_DAYS}D", center=True, min_periods=3)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.scatter(contrast_sorted.index, contrast_sorted["diff"],
               s=8, alpha=0.4, color="steelblue", zorder=2, label="Daily contrast")
    ax.plot(contrast_sorted.index, contrast_sorted["rolling"],
            color="darkorange", linewidth=1.8, zorder=3,
            label=f"{ROLLING_DAYS}-day rolling mean")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Infestation − Grassland NDVI", fontsize=9)
    ax.set_title(
        "NDVI inter-class contrast (Infestation − Grassland)\n"
        f"Daily means, 2020–2025  |  frac > 0: {frac_positive:.0%}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_ndvi_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step B — Fixed-window seasonal medians (reference)
# ---------------------------------------------------------------------------

def seasonal_ndvi_medians(df: pd.DataFrame, months: set[int],
                          season_name: str) -> pd.DataFrame:
    """Return per-(point_id, year) NDVI medians for the given month set."""
    subset = df[df["month"].isin(months)]
    counts = subset.groupby(["point_id", "year"])["ndvi"].count()

    valid = counts[counts >= MIN_OBS].index
    n_excluded = len(counts) - len(valid)
    log(f"  {season_name} (months {sorted(months)}): "
        f"{len(subset):,} rows, {len(counts):,} (pixel×year) groups, "
        f"{n_excluded} excluded (< {MIN_OBS} obs)")

    subset_valid = subset.set_index(["point_id", "year"]).loc[valid].reset_index()
    medians = (
        subset_valid
        .groupby(["point_id", "year"])["ndvi"]
        .median()
        .rename(f"ndvi_{season_name}")
        .reset_index()
    )
    log(f"    Annual medians: {len(medians):,}  |  "
        f"range [{medians[f'ndvi_{season_name}'].min():.4f}, "
        f"{medians[f'ndvi_{season_name}'].max():.4f}]")
    return medians


def compute_fixed_window_recession(peak_med: pd.DataFrame,
                                   trough_med: pd.DataFrame) -> pd.DataFrame:
    """Paired join of peak and trough medians; compute recession per (pixel, year)."""
    paired = peak_med.merge(trough_med, on=["point_id", "year"], how="inner")
    log(f"\nFixed-window paired (pixel, year) rows: {len(paired):,}  "
        f"(of {len(peak_med):,} peak × {len(trough_med):,} trough)")
    paired["recession"] = paired["ndvi_peak"] - paired["ndvi_trough"]
    log(f"  Recession — min: {paired['recession'].min():.4f}  "
        f"median: {paired['recession'].median():.4f}  "
        f"max: {paired['recession'].max():.4f}")
    neg = (paired["recession"] < 0).sum()
    if neg:
        log(f"  Note: {neg} rows with negative recession "
            f"(trough > peak — late wet season or anomalous year)")
    return paired


def log_per_year_contrast(paired: pd.DataFrame, hd_ids, grass_ids) -> None:
    """Log per-year infestation vs grassland fixed-window recession contrast."""
    log("\n--- Per-year fixed-window contrast table ---")
    for year in sorted(paired["year"].unique()):
        yr = paired[paired["year"] == year]
        dataset_med = yr["recession"].median()
        hd_yr    = yr[yr["point_id"].isin(hd_ids)]
        grass_yr = yr[yr["point_id"].isin(grass_ids)]
        if hd_yr.empty or grass_yr.empty:
            continue
        inf_mean   = hd_yr["recession"].mean()
        grass_mean = grass_yr["recession"].mean()
        contrast   = inf_mean - grass_mean
        frac_above = (hd_yr["recession"] > dataset_med).sum() / len(hd_yr)
        log(f"  {year}: infestation={inf_mean:.4f}  grassland={grass_mean:.4f}  "
            f"contrast={contrast:+.4f}  "
            f"inf_above_median={frac_above:.0%}")
    log("-------------------------------------------------------")


def log_peak_trough_decomposition(paired: pd.DataFrame, hd_ids, grass_ids,
                                  rip_ids) -> None:
    """Log per-class mean peak / trough / recession."""
    log("\n--- Peak/trough decomposition by class ---")
    log(f"  {'Class':12s}  {'ndvi_peak':>10}  {'ndvi_trough':>12}  {'rec_mean':>9}")
    for label, ids in [("Infestation", hd_ids), ("Grassland", grass_ids),
                       ("Riparian", rip_ids)]:
        sub = paired[paired["point_id"].isin(ids)]
        log(f"  {label:12s}  "
            f"{sub['ndvi_peak'].mean():>10.4f}  "
            f"{sub['ndvi_trough'].mean():>12.4f}  "
            f"{sub['recession'].mean():>9.4f}")
    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Step C — Window-free annual percentile amplitude
# ---------------------------------------------------------------------------

def annual_percentile_recession(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel annual (p90 − p10) NDVI amplitude, window-free.

    Returns a DataFrame with one row per (point_id, year):
      ndvi_p90, ndvi_p10, rec_p  (= p90 − p10)
    Only years with ≥ MIN_OBS_PER_YEAR qualifying observations are included.
    """
    log("\n--- Window-free annual percentile amplitude ---")

    counts = df.groupby(["point_id", "year"])["ndvi"].count()
    valid  = counts[counts >= MIN_OBS_PER_YEAR].index
    n_excl = len(counts) - len(valid)
    log(f"  (pixel, year) groups: {len(counts):,}  |  "
        f"excluded (< {MIN_OBS_PER_YEAR} obs): {n_excl}")

    df_valid = df.set_index(["point_id", "year"]).loc[valid].reset_index()

    p90 = (df_valid.groupby(["point_id", "year"])["ndvi"]
           .quantile(0.90).rename("ndvi_p90").reset_index())
    p10 = (df_valid.groupby(["point_id", "year"])["ndvi"]
           .quantile(0.10).rename("ndvi_p10").reset_index())

    annual = p90.merge(p10, on=["point_id", "year"])
    annual["rec_p"] = annual["ndvi_p90"] - annual["ndvi_p10"]

    log(f"  Annual (pixel×year) rows: {len(annual):,}")
    log(f"  rec_p — min: {annual['rec_p'].min():.4f}  "
        f"median: {annual['rec_p'].median():.4f}  "
        f"max: {annual['rec_p'].max():.4f}")
    neg = (annual["rec_p"] < 0).sum()
    if neg:
        log(f"  Note: {neg} rows with rec_p < 0 (p10 > p90 — data anomaly)")

    return annual


# ---------------------------------------------------------------------------
# Step D — Per-pixel summary statistics
# ---------------------------------------------------------------------------

def per_pixel_stats(annual_pctile: pd.DataFrame, paired_fw: pd.DataFrame,
                    df: pd.DataFrame, dry_nir: pd.DataFrame) -> pd.DataFrame:
    log("\nAggregating per-pixel statistics across years...")

    # Primary: window-free percentile amplitude
    stats = (
        annual_pctile
        .groupby("point_id")
        .agg(
            rec_p=      ("rec_p",     "mean"),
            rec_p_std=  ("rec_p",     "std"),
            n_years=    ("rec_p",     "count"),
            ndvi_p90_mean=("ndvi_p90", "mean"),
            ndvi_p10_mean=("ndvi_p10", "mean"),
        )
        .reset_index()
    )
    stats["rec_p_cv"] = stats["rec_p_std"] / stats["rec_p"].abs()

    # Reference: fixed-window recession
    fw = (
        paired_fw
        .groupby("point_id")
        .agg(
            rec_mean=        ("recession",   "mean"),
            rec_std=         ("recession",   "std"),
            ndvi_peak_mean=  ("ndvi_peak",   "mean"),
            ndvi_trough_mean=("ndvi_trough", "mean"),
        )
        .reset_index()
    )
    stats = stats.merge(fw, on="point_id", how="left")

    coords = df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    stats = stats.merge(coords, on="point_id", how="left")
    stats = stats.merge(dry_nir[["point_id", "nir_cv", "nir_mean", "in_hd_bbox"]],
                        on="point_id", how="left")

    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    stats["is_riparian"]  = (~stats["in_hd_bbox"]) & (stats["nir_mean"] >= rip_thresh)
    stats["is_grassland"] = (~stats["in_hd_bbox"]) & (~stats["is_riparian"])

    log(f"  Pixels: {len(stats)}")
    log(f"  n_years per pixel — min: {stats['n_years'].min()}  "
        f"median: {stats['n_years'].median():.0f}  max: {stats['n_years'].max()}")
    log(f"  rec_p  — min: {stats['rec_p'].min():.4f}  "
        f"median: {stats['rec_p'].median():.4f}  "
        f"max: {stats['rec_p'].max():.4f}")
    log(f"  rec_mean (fixed-window ref) — min: {stats['rec_mean'].min():.4f}  "
        f"median: {stats['rec_mean'].median():.4f}  "
        f"max: {stats['rec_mean'].max():.4f}")

    log(f"\n  Class counts:")
    log(f"    Infestation (Parkinsonia proxy): {stats['in_hd_bbox'].sum()}")
    log(f"    Riparian (top-{100*(1-RIPARIAN_PERCENTILE):.0f}% nir_mean, "
        f"not HD bbox):  {stats['is_riparian'].sum()}")
    log(f"    Grassland (remainder):          {stats['is_grassland'].sum()}")

    log("\n  Class centroids — primary metric (nir_cv, rec_p):")
    for label, mask in [("Infestation", stats["in_hd_bbox"]),
                        ("Riparian",    stats["is_riparian"]),
                        ("Grassland",   stats["is_grassland"])]:
        sub = stats[mask]
        log(f"    {label:12s}: nir_cv={sub['nir_cv'].mean():.4f}  "
            f"rec_p={sub['rec_p'].mean():.4f}  "
            f"rec_p median={sub['rec_p'].median():.4f}  "
            f"(ref rec_mean={sub['rec_mean'].mean():.4f})")

    return stats


def save_stats(stats: pd.DataFrame) -> None:
    out_path = OUT_DIR / "longreach_amp_stats.parquet"
    cols = ["point_id", "lon", "lat",
            "rec_p", "rec_p_std", "rec_p_cv", "ndvi_p90_mean", "ndvi_p10_mean",
            "rec_mean", "rec_std", "ndvi_peak_mean", "ndvi_trough_mean",
            "n_years", "nir_cv", "nir_mean", "in_hd_bbox", "is_riparian", "is_grassland"]
    stats[cols].to_parquet(out_path, index=False)
    log(f"\nSaved stats: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step E — Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(stats: pd.DataFrame) -> None:
    """Pearson r between rec_p and prior signals."""
    log("\n--- Correlation analysis ---")
    pairs = [
        ("nir_cv",   "Dry-season NIR CV"),
        ("rec_mean", "Fixed-window rec_mean"),
        ("nir_mean", "Dry-season NIR mean"),
    ]
    for col, desc in pairs:
        sub = stats[["rec_p", col]].dropna()
        if len(sub) < 10:
            log(f"  {desc}: insufficient data ({len(sub)} rows)")
            continue
        r, p = pearsonr(sub["rec_p"], sub[col])
        tag = "REDUNDANT (r ≥ 0.7)" if abs(r) >= 0.7 else "independent"
        log(f"  rec_p vs {desc}: r={r:.3f}  p={p:.4f}  → {tag}")
    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_base(stats: pd.DataFrame, col: str, cmap: str, label: str,
                  title: str, bg_img: np.ndarray | None) -> tuple:
    lon_min, lat_min, lon_max, lat_max = SURVEY_BBOX
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_centre = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320
    fig_w = 7
    fig_h = max(4.0, fig_w * (lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    if bg_img is not None:
        ax.imshow(bg_img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", interpolation="bilinear", zorder=0)

    pt_per_deg = fig_w * 72 / lon_span
    marker_pt  = (10 / lon_m_per_deg) * pt_per_deg
    marker_s   = max(0.6, marker_pt ** 2 / 10)

    sc = ax.scatter(stats["lon"], stats["lat"], c=stats[col], cmap=cmap,
                    s=marker_s, linewidths=0, alpha=0.6, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label, fontsize=8)

    ax.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN), HD_LON_MAX - HD_LON_MIN, HD_LAT_MAX - HD_LAT_MIN,
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
    return fig, ax


def plot_recession_map(stats: pd.DataFrame, bg_img: np.ndarray | None) -> None:
    fig, _ = _scatter_base(
        stats, col="rec_p", cmap="RdYlGn",
        label="Mean NDVI amplitude (annual p90 − p10)",
        title="Longreach — NDVI annual amplitude (window-free p90 − p10)\n"
              "Higher = larger seasonal swing = Parkinsonia signature",
        bg_img=bg_img,
    )
    out = OUT_DIR / "longreach_rec_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved recession map: {out.relative_to(PROJECT_ROOT)}")


def plot_recession_histogram(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    colors = {"Infestation": "darkorange", "Riparian": "steelblue", "Grassland": "olivedrab"}
    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]],   colors["Infestation"]),
        ("Riparian",    stats[stats["is_riparian"]],   colors["Riparian"]),
        ("Grassland",   stats[stats["is_grassland"]],  colors["Grassland"]),
    ]

    bins = np.linspace(stats["rec_p"].min() - 0.005,
                       stats["rec_p"].max() + 0.005, 35)
    for label, sub, color in groups:
        ax.hist(sub["rec_p"], bins=bins, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.4,
                label=f"{label} (n={len(sub)}, med={sub['rec_p'].median():.3f})")

    ax.set_xlabel("Mean NDVI annual amplitude (window-free p90 − p10)", fontsize=9)
    ax.set_ylabel("Pixel count", fontsize=9)
    ax.set_title("Distribution of NDVI annual amplitude by land-cover class\n"
                 "Higher = stronger seasonal swing (Parkinsonia signature)", fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(stats["rec_p"].median(), color="black", linestyle=":",
               linewidth=1.2, label="_nolegend_")
    ax.text(stats["rec_p"].median() + 0.001, ax.get_ylim()[1] * 0.92,
            f"dataset median\n{stats['rec_p'].median():.3f}",
            fontsize=7, va="top")

    fig.tight_layout()
    out = OUT_DIR / "longreach_rec_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"Saved recession histogram: {out.relative_to(PROJECT_ROOT)}")


def plot_2d_scatter(stats: pd.DataFrame) -> None:
    """nir_cv × rec_p two-axis discriminator."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    groups = [
        ("Grassland",   stats[stats["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    stats[stats["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", stats[stats["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]

    for label, sub, color, size, alpha in groups:
        ax.scatter(sub["nir_cv"], sub["rec_p"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(sub)})", zorder=3)

    for label, sub, color, *_ in groups:
        cx, cy = sub["nir_cv"].mean(), sub["rec_p"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    med_cv  = stats["nir_cv"].median()
    med_rec = stats["rec_p"].median()
    ax.axvline(med_cv,  color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(med_rec, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(med_cv + 0.001, ax.get_ylim()[0] + 0.001,
            f"CV median\n{med_cv:.3f}", fontsize=7, color="grey", va="bottom")
    ax.text(ax.get_xlim()[0] + 0.001, med_rec + 0.001,
            f"rec_p median {med_rec:.3f}", fontsize=7, color="grey", va="bottom")

    ax.set_xlabel("Dry-season NIR CV (inter-annual stability; lower = more stable)",
                  fontsize=9)
    ax.set_ylabel("Mean NDVI annual amplitude (window-free p90 − p10;\n"
                  "higher = stronger seasonal swing)", fontsize=9)
    ax.set_title("Two-axis NIR/NDVI discriminator\n"
                 "Infestation target: low CV + high amplitude (bottom-right quadrant)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "longreach_nir_cv_vs_rec.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved 2D scatter: {out.relative_to(PROJECT_ROOT)}")


def plot_monthly_profiles(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Mean NDVI monthly profile per class, with ±1 std band and per-year lines."""
    fig = plt.figure(figsize=(14, 5), dpi=150)
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]]["point_id"],   "darkorange"),
        ("Riparian",    stats[stats["is_riparian"]]["point_id"],   "steelblue"),
        ("Grassland",   stats[stats["is_grassland"]]["point_id"],  "olivedrab"),
    ]

    months_ordered = list(range(1, 13))
    month_labels   = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    for i, (label, pids, color) in enumerate(groups):
        ax  = fig.add_subplot(gs[i])
        sub = df[df["point_id"].isin(pids)]

        yr_month = sub.groupby(["year", "month"])["ndvi"].mean().unstack("month")
        for yr in yr_month.index:
            vals = [yr_month.loc[yr, m] if m in yr_month.columns else np.nan
                    for m in months_ordered]
            ax.plot(months_ordered, vals, color=color, alpha=0.25, linewidth=0.8)

        mean_profile = yr_month.mean()
        std_profile  = yr_month.std()
        ym = np.array([mean_profile.get(m, np.nan) for m in months_ordered], dtype=float)
        ys = np.array([std_profile.get(m, np.nan)  for m in months_ordered], dtype=float)
        ax.plot(months_ordered, ym, color=color, linewidth=2, zorder=4)
        ax.fill_between(months_ordered, ym - ys, ym + ys, color=color, alpha=0.2)

        for months_set, shade in [(PEAK_MONTHS, "limegreen"), (TROUGH_MONTHS, "tomato")]:
            for m in months_set:
                ax.axvspan(m - 0.5, m + 0.5, color=shade, alpha=0.08, zorder=0)

        ax.set_xticks(months_ordered)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel("NDVI" if i == 0 else "", fontsize=8)
        ax.set_title(f"{label}\n(n={len(pids)} pixels)", fontsize=9)
        ax.tick_params(labelsize=7)

        if len(pids) > 0:
            rec_med = stats.loc[stats["point_id"].isin(pids), "rec_p"].median()
            ax.text(0.97, 0.04, f"amplitude median: {rec_med:.3f}",
                    transform=ax.transAxes, fontsize=7.5, ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    fig.suptitle(
        "Monthly NDVI profiles by class — Longreach 2020–2025\n"
        "Green shading = reference peak window (Mar–May), red = trough (Jul–Sep)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "longreach_monthly_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved monthly profiles: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Success criteria  (updated to use rec_p as primary)
# ---------------------------------------------------------------------------

def log_success_criteria(stats: pd.DataFrame) -> None:
    log("\n--- Success criteria ---")

    hd      = stats[stats["in_hd_bbox"]]
    non_hd  = stats[~stats["in_hd_bbox"]]
    rip     = stats[stats["is_riparian"]]
    grass   = stats[stats["is_grassland"]]
    med_rec = stats["rec_p"].median()

    # 1. Infestation above dataset median amplitude
    n_above = (hd["rec_p"] > med_rec).sum()
    pct = 100 * n_above / len(hd)
    status = "PASS" if pct >= 60 else "FAIL"
    log(f"  [1] Infestation pixels above dataset median amplitude "
        f"({med_rec:.4f}): {n_above}/{len(hd)} ({pct:.0f}%)  → {status}")

    # 2. IQR separation: infestation vs combined other classes
    hd_q25, hd_q75 = hd["rec_p"].quantile([0.25, 0.75])
    ot_q25, ot_q75 = non_hd["rec_p"].quantile([0.25, 0.75])
    overlap = max(0, min(hd_q75, ot_q75) - max(hd_q25, ot_q25))
    span    = max(hd_q75, ot_q75) - min(hd_q25, ot_q25)
    overlap_frac = overlap / span if span > 0 else 1.0
    status2 = "PASS" if overlap_frac < 0.5 else "FAIL"
    log(f"  [2] IQR separation — infestation [{hd_q25:.4f}, {hd_q75:.4f}] vs "
        f"non-infestation [{ot_q25:.4f}, {ot_q75:.4f}], "
        f"overlap fraction: {overlap_frac:.2f}  → {status2}")

    # 3. Amplitude ordering: infestation > grassland > riparian
    parko_med = hd["rec_p"].median()
    grass_med = grass["rec_p"].median()
    rip_med   = rip["rec_p"].median()
    parko_gt_grass = parko_med > grass_med
    parko_gt_rip   = parko_med > rip_med
    status3 = "PASS" if (parko_gt_grass and parko_gt_rip) else "PARTIAL"
    log(f"  [3] Amplitude ordering (infestation > others):\n"
        f"        infestation={parko_med:.4f}  grassland={grass_med:.4f}  "
        f"riparian={rip_med:.4f}\n"
        f"        infestation > grassland: {parko_gt_grass} | "
        f"infestation > riparian: {parko_gt_rip}  → {status3}")

    # 4. 2D separation: infestation centroid in low-CV / high-amp quadrant
    hd_cv_cent  = hd["nir_cv"].mean()
    ot_cv_cent  = non_hd["nir_cv"].mean()
    hd_rec_cent = hd["rec_p"].mean()
    ot_rec_cent = non_hd["rec_p"].mean()
    low_cv   = hd_cv_cent  < ot_cv_cent
    high_rec = hd_rec_cent > ot_rec_cent
    status4 = "PASS" if (low_cv and high_rec) else "PARTIAL"
    log(f"  [4] Infestation centroid in low-CV/high-amplitude quadrant:\n"
        f"        CV:  infestation={hd_cv_cent:.4f}  others={ot_cv_cent:.4f} → lower: {low_cv}\n"
        f"        amp: infestation={hd_rec_cent:.4f}  others={ot_rec_cent:.4f} → higher: {high_rec}"
        f"  → {status4}")

    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach wet/dry seasonal amplitude (NDVI recession) ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(PARQUET_PATH)

    log("\nLoading dry-NIR stats...")
    dry_nir = pd.read_parquet(DRY_NIR_STATS)
    log(f"  {len(dry_nir)} pixel records loaded")

    df = assign_classes(df, dry_nir)
    hd_ids, rip_ids, grass_ids = class_ids(dry_nir)

    log(f"\nClass counts (pixel-observations):")
    log(f"  Infestation: {df['in_hd_bbox'].sum():,} obs  "
        f"({df[df['in_hd_bbox']]['point_id'].nunique()} pixels)")
    log(f"  Riparian:    {df['is_riparian'].sum():,} obs  "
        f"({df[df['is_riparian']]['point_id'].nunique()} pixels)")
    log(f"  Grassland:   {df['is_grassland'].sum():,} obs  "
        f"({df[df['is_grassland']]['point_id'].nunique()} pixels)")

    # A — Daily NDVI contrast time series
    ndvi_contrast_time_series(df)

    # B — Fixed-window medians (reference)
    log("\nComputing fixed-window seasonal NDVI medians (reference)...")
    peak_med   = seasonal_ndvi_medians(df, PEAK_MONTHS,   "peak")
    trough_med = seasonal_ndvi_medians(df, TROUGH_MONTHS, "trough")
    paired_fw  = compute_fixed_window_recession(peak_med, trough_med)
    log_per_year_contrast(paired_fw, hd_ids, grass_ids)
    log_peak_trough_decomposition(paired_fw, hd_ids, grass_ids, rip_ids)

    # C — Window-free annual percentile amplitude (primary)
    annual_pctile = annual_percentile_recession(df)

    # D — Per-pixel summary
    stats = per_pixel_stats(annual_pctile, paired_fw, df, dry_nir)
    save_stats(stats)

    # E — Correlation analysis
    correlation_analysis(stats)

    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    log("\nGenerating plots...")
    plot_recession_map(stats, bg_img)
    plot_recession_histogram(stats)
    plot_2d_scatter(stats)
    plot_monthly_profiles(df, stats)

    log_success_criteria(stats)
    log("\nDone.")


if __name__ == "__main__":
    main()
