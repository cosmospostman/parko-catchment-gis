"""Red-edge ratio analysis over the Longreach Parkinsonia infestation.

Metric: B07/B05 ratio — proxy for active chlorophyll (higher = more active).
Per-pixel summary statistic: annual 10th-percentile re_ratio, averaged across years.

See research/LONGREACH-RED-EDGE.md for the full analysis plan.
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
AMP_STATS     = PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp" / "longreach_amp_stats.parquet"
OUT_DIR       = PROJECT_ROOT / "outputs" / "longreach-red-edge"

SCL_PURITY_MIN = 0.5

# Minimum qualifying observations per (pixel, year) to include that year's 10th pctile
MIN_OBS_PER_YEAR = 10

# Rolling window for contrast time series (days)
ROLLING_DAYS = 30

HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

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

    df["re_ratio"] = df["B07"] / df["B05"]
    df["re_ndvi"]  = (df["B07"] - df["B05"]) / (df["B07"] + df["B05"])
    df["month"]    = df["date"].dt.month
    df["year"]     = df["date"].dt.year

    log(f"  re_ratio — min: {df['re_ratio'].min():.4f}  "
        f"median: {df['re_ratio'].median():.4f}  "
        f"max: {df['re_ratio'].max():.4f}")
    log(f"  re_ndvi  — min: {df['re_ndvi'].min():.4f}  "
        f"median: {df['re_ndvi'].median():.4f}  "
        f"max: {df['re_ndvi'].max():.4f}")
    return df


def assign_classes(df: pd.DataFrame, dry_nir: pd.DataFrame) -> pd.DataFrame:
    """Add in_hd_bbox, is_riparian, is_grassland columns to pixel-level df."""
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    dry_nir = dry_nir.copy()
    dry_nir["is_riparian"] = (~dry_nir["in_hd_bbox"]) & (dry_nir["nir_mean"] >= rip_thresh)
    dry_nir["is_grassland"] = (~dry_nir["in_hd_bbox"]) & (~dry_nir["is_riparian"])
    class_cols = dry_nir[["point_id", "in_hd_bbox", "is_riparian", "is_grassland"]]
    return df.merge(class_cols, on="point_id", how="left")


# ---------------------------------------------------------------------------
# Step 3 — Inter-class contrast time series
# ---------------------------------------------------------------------------

def contrast_time_series(df: pd.DataFrame) -> None:
    log("\n--- Step 3: Inter-class contrast time series ---")

    inf_pids  = df[df["in_hd_bbox"]]["point_id"].unique()
    grass_pids = df[df["is_grassland"]]["point_id"].unique()

    daily_inf   = df[df["point_id"].isin(inf_pids)].groupby("date")["re_ratio"].mean().rename("inf")
    daily_grass = df[df["point_id"].isin(grass_pids)].groupby("date")["re_ratio"].mean().rename("grass")

    contrast = pd.concat([daily_inf, daily_grass], axis=1).dropna()
    contrast["diff"] = contrast["inf"] - contrast["grass"]

    frac_positive = (contrast["diff"] > 0).mean()
    log(f"  Dates with contrast > 0: {(contrast['diff']>0).sum()}/{len(contrast)} "
        f"({100*frac_positive:.1f}%)")

    # Per-year stats
    contrast["year"] = contrast.index.year
    log("\n  Per-year contrast summary:")
    for yr, grp in contrast.groupby("year"):
        idx_max = grp["diff"].idxmax()
        idx_min = grp["diff"].idxmin()
        frac = (grp["diff"] > 0).mean()
        log(f"    {yr}: max={grp['diff'].max():.4f} on {idx_max.date()}  "
            f"min={grp['diff'].min():.4f} on {idx_min.date()}  "
            f"frac>0={frac:.2f}")

    # Rolling mean
    contrast_sorted = contrast.sort_index()
    contrast_sorted["rolling"] = (
        contrast_sorted["diff"]
        .rolling(f"{ROLLING_DAYS}D", center=True, min_periods=3)
        .mean()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.scatter(contrast_sorted.index, contrast_sorted["diff"],
               s=8, alpha=0.4, color="steelblue", zorder=2, label="Daily contrast")
    ax.plot(contrast_sorted.index, contrast_sorted["rolling"],
            color="darkorange", linewidth=1.8, zorder=3,
            label=f"{ROLLING_DAYS}-day rolling mean")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Infestation − Grassland re_ratio", fontsize=9)
    ax.set_title(
        "Red-edge ratio inter-class contrast (Infestation − Grassland)\n"
        f"B07/B05 daily means, 2020–2025  |  frac > 0: {frac_positive:.0%}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_re_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 4 — Monthly median profile by class
# ---------------------------------------------------------------------------

def monthly_profiles(df: pd.DataFrame) -> None:
    log("\n--- Step 4: Monthly median profiles ---")

    fig = plt.figure(figsize=(14, 5), dpi=150)
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    groups = [
        ("Infestation", df["in_hd_bbox"],    "darkorange"),
        ("Riparian",    df["is_riparian"],    "steelblue"),
        ("Grassland",   df["is_grassland"],   "olivedrab"),
    ]

    months_ordered = list(range(1, 13))
    month_labels   = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    for i, (label, mask, color) in enumerate(groups):
        ax  = fig.add_subplot(gs[i])
        sub = df[mask]
        n_pixels = sub["point_id"].nunique()

        yr_month = sub.groupby(["year", "month"])["re_ratio"].median().unstack("month")
        for yr in yr_month.index:
            vals = [yr_month.loc[yr, m] if m in yr_month.columns else np.nan
                    for m in months_ordered]
            ax.plot(months_ordered, vals, color=color, alpha=0.25, linewidth=0.8)

        mean_p = yr_month.mean()
        std_p  = yr_month.std()
        ym = np.array([mean_p.get(m, np.nan) for m in months_ordered], dtype=float)
        ys = np.array([std_p.get(m, np.nan)  for m in months_ordered], dtype=float)
        ax.plot(months_ordered, ym, color=color, linewidth=2, zorder=4)
        ax.fill_between(months_ordered, ym - ys, ym + ys, color=color, alpha=0.2)

        valid = ~np.isnan(ym)
        if valid.any():
            peak_m  = months_ordered[np.nanargmax(ym)]
            trough_m = months_ordered[np.nanargmin(ym)]
            amp = np.nanmax(ym) - np.nanmin(ym)
            log(f"  {label}: peak month={peak_m}  trough month={trough_m}  range={amp:.4f}")

        ax.set_xticks(months_ordered)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel("re_ratio (B07/B05)" if i == 0 else "", fontsize=8)
        ax.set_title(f"{label}\n(n={n_pixels} pixels)", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Monthly red-edge ratio (B07/B05) profiles by class — Longreach 2020–2025\n"
        "Higher = more active chlorophyll",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "longreach_re_monthly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 5 — Per-pixel annual low percentile
# ---------------------------------------------------------------------------

def per_pixel_re_stats(df: pd.DataFrame, dry_nir: pd.DataFrame,
                       amp_stats: pd.DataFrame) -> pd.DataFrame:
    log("\n--- Step 5: Per-pixel annual 10th-percentile summary ---")

    # Annual 10th percentile per (pixel, year) — require MIN_OBS_PER_YEAR obs
    counts = df.groupby(["point_id", "year"])["re_ratio"].count()
    valid  = counts[counts >= MIN_OBS_PER_YEAR].index
    n_excl = len(counts) - len(valid)
    log(f"  (pixel, year) groups: {len(counts):,}  |  "
        f"excluded (< {MIN_OBS_PER_YEAR} obs): {n_excl}")

    df_valid = df.set_index(["point_id", "year"]).loc[valid].reset_index()
    annual_p10 = (
        df_valid.groupby(["point_id", "year"])["re_ratio"]
        .quantile(0.10)
        .rename("re_p10_yr")
        .reset_index()
    )

    stats = (
        annual_p10.groupby("point_id")["re_p10_yr"]
        .agg(re_p10="mean", re_p10_std="std", n_years="count")
        .reset_index()
    )
    stats["re_p10_cv"] = stats["re_p10_std"] / stats["re_p10"]

    # Also log re_ndvi equivalents for reference
    annual_re_ndvi = (
        df_valid.groupby(["point_id", "year"])["re_ndvi"]
        .quantile(0.10)
        .rename("re_ndvi_p10_yr")
        .reset_index()
    )
    ndvi_stats = (
        annual_re_ndvi.groupby("point_id")["re_ndvi_p10_yr"]
        .mean()
        .rename("re_ndvi_p10")
        .reset_index()
    )
    stats = stats.merge(ndvi_stats, on="point_id", how="left")

    coords = df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    stats = stats.merge(coords, on="point_id", how="left")

    # Class flags
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    stats = stats.merge(
        dry_nir[["point_id", "nir_cv", "nir_mean", "in_hd_bbox"]],
        on="point_id", how="left",
    )
    stats["is_riparian"]  = (~stats["in_hd_bbox"]) & (stats["nir_mean"] >= rip_thresh)
    stats["is_grassland"] = (~stats["in_hd_bbox"]) & (~stats["is_riparian"])

    # Merge rec_mean from amp_stats
    stats = stats.merge(amp_stats[["point_id", "rec_mean"]], on="point_id", how="left")

    log(f"  Pixels with stats: {len(stats)}")
    log(f"  n_years per pixel — min: {stats['n_years'].min()}  "
        f"median: {stats['n_years'].median():.0f}  max: {stats['n_years'].max()}")

    log("\n  Class-level re_p10 distributions:")
    for label, mask in [("Infestation", stats["in_hd_bbox"]),
                        ("Riparian",    stats["is_riparian"]),
                        ("Grassland",   stats["is_grassland"])]:
        sub = stats[mask]
        if len(sub) == 0:
            continue
        q25, q75 = sub["re_p10"].quantile([0.25, 0.75])
        log(f"    {label:12s}: mean={sub['re_p10'].mean():.4f}  "
            f"median={sub['re_p10'].median():.4f}  "
            f"IQR=[{q25:.4f},{q75:.4f}]  "
            f"re_ndvi_p10 mean={sub['re_ndvi_p10'].mean():.4f}")

    return stats


def save_stats(stats: pd.DataFrame) -> None:
    cols = ["point_id", "lon", "lat", "re_p10", "re_p10_std", "re_p10_cv",
            "re_ndvi_p10", "n_years", "nir_cv", "nir_mean", "rec_mean",
            "in_hd_bbox", "is_riparian", "is_grassland"]
    out  = OUT_DIR / "longreach_re_stats.parquet"
    stats[cols].to_parquet(out, index=False)
    log(f"  Saved stats: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plots (steps 6–9)
# ---------------------------------------------------------------------------

def _map_base(stats: pd.DataFrame, col: str, cmap: str, label: str,
              title: str, bg_img) -> tuple:
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


def plot_map(stats: pd.DataFrame, bg_img) -> None:
    log("\n--- Step 6: Spatial map ---")
    fig, _ = _map_base(
        stats, col="re_p10", cmap="RdYlGn",
        label="Mean annual 10th-pctile re_ratio (B07/B05)",
        title="Longreach — red-edge ratio annual p10 (re_p10)\n"
              "Higher = active chlorophyll retained; Parkinsonia signature",
        bg_img=bg_img,
    )
    out = OUT_DIR / "longreach_re_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_histogram(stats: pd.DataFrame) -> None:
    log("\n--- Step 7: Histogram by class ---")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]],  "darkorange"),
        ("Riparian",    stats[stats["is_riparian"]],  "steelblue"),
        ("Grassland",   stats[stats["is_grassland"]], "olivedrab"),
    ]
    bins = np.linspace(stats["re_p10"].min() - 0.005,
                       stats["re_p10"].max() + 0.005, 35)
    for label, sub, color in groups:
        ax.hist(sub["re_p10"], bins=bins, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.4,
                label=f"{label} (n={len(sub)}, med={sub['re_p10'].median():.3f})")

    ax.set_xlabel("re_p10 (mean annual 10th-pctile of B07/B05)", fontsize=9)
    ax.set_ylabel("Pixel count", fontsize=9)
    ax.set_title("Distribution of red-edge annual p10 by land-cover class\n"
                 "Higher = chlorophyll retained at dry-season low", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_re_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_2d_nir_cv_vs_re(stats: pd.DataFrame) -> None:
    log("\n--- Step 8a: 2D scatter nir_cv × re_p10 ---")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    groups = [
        ("Grassland",   stats[stats["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    stats[stats["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", stats[stats["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]
    for label, sub, color, size, alpha in groups:
        ax.scatter(sub["nir_cv"], sub["re_p10"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(sub)})", zorder=3)
    for label, sub, color, *_ in groups:
        cx, cy = sub["nir_cv"].mean(), sub["re_p10"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel("Dry-season NIR CV (lower = more stable)", fontsize=9)
    ax.set_ylabel("re_p10 (mean annual 10th-pctile of B07/B05; higher = active chlorophyll)", fontsize=9)
    ax.set_title("nir_cv × re_p10 — stability vs chlorophyll activity", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_nir_cv_vs_re.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_2d_rec_vs_re(stats: pd.DataFrame) -> None:
    log("\n--- Step 8b: 2D scatter rec_mean × re_p10 ---")
    sub = stats.dropna(subset=["rec_mean"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    groups = [
        ("Grassland",   sub[sub["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    sub[sub["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", sub[sub["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]
    for label, grp, color, size, alpha in groups:
        ax.scatter(grp["rec_mean"], grp["re_p10"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(grp)})", zorder=3)
    for label, grp, color, *_ in groups:
        cx, cy = grp["rec_mean"].mean(), grp["re_p10"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel("rec_mean (NDVI seasonal recession; higher = stronger wet-season flush)", fontsize=9)
    ax.set_ylabel("re_p10 (mean annual 10th-pctile of B07/B05)", fontsize=9)
    ax.set_title("rec_mean × re_p10 — recession vs chlorophyll activity", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_rec_vs_re.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def correlation_analysis(stats: pd.DataFrame) -> None:
    log("\n--- Step 9: Correlation analysis ---")
    pairs = [
        ("nir_cv",   "Dry-season NIR CV"),
        ("rec_mean", "NDVI recession (rec_mean)"),
        ("nir_mean", "Dry-season NIR mean"),
    ]
    for col, desc in pairs:
        sub = stats[["re_p10", col]].dropna()
        if len(sub) < 10:
            log(f"  {desc}: insufficient data ({len(sub)} rows)")
            continue
        r, p = pearsonr(sub["re_p10"], sub[col])
        redundant = "REDUNDANT (r ≥ 0.7)" if abs(r) >= 0.7 else "independent"
        log(f"  re_p10 vs {desc}: r={r:.3f}  p={p:.4f}  → {redundant}")


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def log_success_criteria(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    log("\n--- Success criteria ---")

    inf   = stats[stats["in_hd_bbox"]]
    grass = stats[stats["is_grassland"]]
    rip   = stats[stats["is_riparian"]]

    # 1. Contrast time series: frac > 0 ≥ 0.6
    inf_pids   = df[df["in_hd_bbox"]]["point_id"].unique()
    grass_pids = df[df["is_grassland"]]["point_id"].unique()
    daily_inf   = df[df["point_id"].isin(inf_pids)].groupby("date")["re_ratio"].mean()
    daily_grass = df[df["point_id"].isin(grass_pids)].groupby("date")["re_ratio"].mean()
    contrast    = (daily_inf - daily_grass).dropna()
    frac = (contrast > 0).mean()
    status1 = "PASS" if frac >= 0.6 else "FAIL"
    log(f"  [1] Fraction of dates with infestation re_ratio > grassland: "
        f"{frac:.2f}  → {status1}")

    # 2. Class ordering in re_p10
    parko_med = inf["re_p10"].median()
    grass_med = grass["re_p10"].median()
    status2 = "PASS" if parko_med > grass_med else "FAIL"
    log(f"  [2] re_p10 ordering — infestation={parko_med:.4f}  "
        f"grassland={grass_med:.4f}  → {status2}")

    # 3. IQR separation: infestation vs grassland
    inf_q25, inf_q75   = inf["re_p10"].quantile([0.25, 0.75])
    grs_q25, grs_q75   = grass["re_p10"].quantile([0.25, 0.75])
    overlap = max(0, min(inf_q75, grs_q75) - max(inf_q25, grs_q25))
    span    = max(inf_q75, grs_q75) - min(inf_q25, grs_q25)
    overlap_frac = overlap / span if span > 0 else 1.0
    status3 = "PASS" if overlap_frac < 0.5 else "FAIL"
    log(f"  [3] IQR overlap fraction (infestation vs grassland): "
        f"{overlap_frac:.2f}  → {status3}")

    # 4. Low correlation with rec_mean
    sub = stats[["re_p10", "rec_mean"]].dropna()
    if len(sub) >= 10:
        r, _ = pearsonr(sub["re_p10"], sub["rec_mean"])
        status4 = "PASS" if abs(r) < 0.7 else "FAIL"
        log(f"  [4] Pearson r(re_p10, rec_mean) = {r:.3f}  → {status4}")
    else:
        log("  [4] Insufficient data for correlation check")

    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach red-edge ratio analysis ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df      = load_and_filter(PARQUET_PATH)
    dry_nir = pd.read_parquet(DRY_NIR_STATS)
    amp     = pd.read_parquet(AMP_STATS)

    log(f"\nLoaded dry-NIR stats: {len(dry_nir)} pixels")
    log(f"Loaded amp stats:     {len(amp)} pixels")

    df = assign_classes(df, dry_nir)

    log(f"\nClass counts (pixel-observations):")
    log(f"  Infestation: {df['in_hd_bbox'].sum():,} obs  ({df[df['in_hd_bbox']]['point_id'].nunique()} pixels)")
    log(f"  Riparian:    {df['is_riparian'].sum():,} obs  ({df[df['is_riparian']]['point_id'].nunique()} pixels)")
    log(f"  Grassland:   {df['is_grassland'].sum():,} obs  ({df[df['is_grassland']]['point_id'].nunique()} pixels)")

    contrast_time_series(df)
    monthly_profiles(df)

    stats = per_pixel_re_stats(df, dry_nir, amp)
    save_stats(stats)

    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    plot_map(stats, bg_img)
    plot_histogram(stats)
    plot_2d_nir_cv_vs_re(stats)
    plot_2d_rec_vs_re(stats)
    correlation_analysis(stats)
    log_success_criteria(df, stats)

    log("\nDone.")


if __name__ == "__main__":
    main()
