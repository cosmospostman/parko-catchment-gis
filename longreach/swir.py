"""SWIR moisture index analysis over the Longreach Parkinsonia infestation.

Metric: (B08 - B11) / (B08 + B11) — proxy for canopy water content.
Per-pixel summary statistic: annual 10th-percentile swir_mi, averaged across years.

See research/LONGREACH-SWIR.md for the full analysis plan.
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
RE_STATS      = PROJECT_ROOT / "outputs" / "longreach-red-edge" / "longreach_re_stats.parquet"
OUT_DIR       = PROJECT_ROOT / "outputs" / "longreach-swir"

SCL_PURITY_MIN = 0.5
MIN_OBS_PER_YEAR = 10
ROLLING_DAYS = 30

HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

RIPARIAN_PERCENTILE = 0.90

# Sample pixels for B11 time series diagnostic
N_SAMPLE = 10

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

    df["swir_mi"] = (df["B08"] - df["B11"]) / (df["B08"] + df["B11"])
    df["month"]   = df["date"].dt.month
    df["year"]    = df["date"].dt.year

    log(f"  swir_mi — min: {df['swir_mi'].min():.4f}  "
        f"median: {df['swir_mi'].median():.4f}  "
        f"max: {df['swir_mi'].max():.4f}")
    log(f"  B11     — min: {df['B11'].min():.4f}  "
        f"median: {df['B11'].median():.4f}  "
        f"max: {df['B11'].max():.4f}")
    log(f"  B08     — min: {df['B08'].min():.4f}  "
        f"median: {df['B08'].median():.4f}  "
        f"max: {df['B08'].max():.4f}")
    return df


def assign_classes(df: pd.DataFrame, dry_nir: pd.DataFrame) -> pd.DataFrame:
    """Add in_hd_bbox, is_riparian, is_grassland columns."""
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    dry_nir = dry_nir.copy()
    dry_nir["is_riparian"]  = (~dry_nir["in_hd_bbox"]) & (dry_nir["nir_mean"] >= rip_thresh)
    dry_nir["is_grassland"] = (~dry_nir["in_hd_bbox"]) & (~dry_nir["is_riparian"])
    class_cols = dry_nir[["point_id", "in_hd_bbox", "is_riparian", "is_grassland"]]
    return df.merge(class_cols, on="point_id", how="left")


# ---------------------------------------------------------------------------
# Step 3 — Inter-class contrast time series (swir_mi, B11, B08)
# ---------------------------------------------------------------------------

def contrast_time_series(df: pd.DataFrame) -> None:
    log("\n--- Step 3: Inter-class contrast time series ---")

    inf_pids   = df[df["in_hd_bbox"]]["point_id"].unique()
    grass_pids = df[df["is_grassland"]]["point_id"].unique()

    def daily_means(signal: str) -> pd.DataFrame:
        inf_s   = df[df["point_id"].isin(inf_pids)].groupby("date")[signal].mean().rename("inf")
        grass_s = df[df["point_id"].isin(grass_pids)].groupby("date")[signal].mean().rename("grass")
        c = pd.concat([inf_s, grass_s], axis=1).dropna()
        c["diff"] = c["inf"] - c["grass"]
        return c

    c_swir = daily_means("swir_mi")
    c_b11  = daily_means("B11")
    c_b08  = daily_means("B08")

    frac_positive = (c_swir["diff"] > 0).mean()
    log(f"  swir_mi dates with contrast > 0: {(c_swir['diff']>0).sum()}/{len(c_swir)} "
        f"({100*frac_positive:.1f}%)")

    log("\n  Per-year swir_mi contrast summary:")
    c_swir["year"] = c_swir.index.year
    c_b11["year"]  = c_b11.index.year
    c_b08["year"]  = c_b08.index.year

    for yr, grp in c_swir.groupby("year"):
        idx_max = grp["diff"].idxmax()
        frac    = (grp["diff"] > 0).mean()
        b11_yr  = c_b11[c_b11["year"] == yr]
        b08_yr  = c_b08[c_b08["year"] == yr]

        # Contributor at date of max swir_mi contrast
        if idx_max in b11_yr.index and idx_max in b08_yr.index:
            b11_contrib = b11_yr.loc[idx_max, "diff"]
            b08_contrib = b08_yr.loc[idx_max, "diff"]
            driver = "B11" if abs(b11_contrib) > abs(b08_contrib) else "B08"
        else:
            driver = "n/a"

        log(f"    {yr}: max_swir_mi_contrast={grp['diff'].max():.4f} on {idx_max.date()}  "
            f"frac>0={frac:.2f}  driver_at_peak={driver}")

    # Rolling means
    def rolling(c: pd.DataFrame) -> pd.Series:
        return (c.sort_index()["diff"]
                .rolling(f"{ROLLING_DAYS}D", center=True, min_periods=3)
                .mean())

    roll_swir = rolling(c_swir)
    roll_b11  = rolling(c_b11)
    roll_b08  = rolling(c_b08)

    # 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150, sharex=True)
    panel_data = [
        (axes[0], c_swir,  roll_swir, "swir_mi contrast (Inf − Grass)",
         "steelblue",  "darkorange", f"SWIR moisture index  |  frac > 0: {frac_positive:.0%}"),
        (axes[1], c_b11,   roll_b11,  "B11 contrast (Inf − Grass)",
         "mediumpurple","indigo",     "Raw B11 (SWIR reflectance)"),
        (axes[2], c_b08,   roll_b08,  "B08 contrast (Inf − Grass)",
         "olivedrab",  "darkgreen",  "Raw B08 (NIR reflectance)"),
    ]
    for ax, c, roll, ylabel, scatter_col, line_col, title in panel_data:
        ax.scatter(c.index, c["diff"], s=8, alpha=0.35, color=scatter_col,
                   zorder=2, label="Daily contrast")
        ax.plot(c.sort_index().index, roll, color=line_col, linewidth=1.8,
                zorder=3, label=f"{ROLLING_DAYS}-day rolling mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Date", fontsize=9)
    fig.suptitle("Longreach — SWIR contrast time series with B11/B08 decomposition\n"
                 "Infestation − Grassland daily means, 2020–2025",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "longreach_swir_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")

    return frac_positive


# ---------------------------------------------------------------------------
# Step 4 — Monthly median profiles (swir_mi, B11, B08) × 3 classes (3×3)
# ---------------------------------------------------------------------------

def monthly_profiles(df: pd.DataFrame) -> None:
    log("\n--- Step 4: Monthly median profiles (3×3 panel) ---")

    signals = [
        ("swir_mi", "swir_mi (B08−B11)/(B08+B11)", "RdYlGn"),
        ("B11",     "Raw B11 (SWIR reflectance)",   "Purples"),
        ("B08",     "Raw B08 (NIR reflectance)",     "Greens"),
    ]
    groups = [
        ("Infestation", df["in_hd_bbox"],   "darkorange"),
        ("Riparian",    df["is_riparian"],   "steelblue"),
        ("Grassland",   df["is_grassland"],  "olivedrab"),
    ]

    months_ordered = list(range(1, 13))
    month_labels   = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs  = GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # amplitude log
    log("\n  Seasonal amplitude (max−min of monthly mean) per class per signal:")
    for row, (sig, sig_label, _) in enumerate(signals):
        for col, (cls_label, mask, color) in enumerate(groups):
            ax  = fig.add_subplot(gs[row, col])
            sub = df[mask]
            n_pixels = sub["point_id"].nunique()

            yr_month = sub.groupby(["year", "month"])[sig].median().unstack("month")
            for yr in yr_month.index:
                vals = [yr_month.loc[yr, m] if m in yr_month.columns else np.nan
                        for m in months_ordered]
                ax.plot(months_ordered, vals, color=color, alpha=0.2, linewidth=0.7)

            mean_p = yr_month.mean()
            std_p  = yr_month.std()
            ym = np.array([mean_p.get(m, np.nan) for m in months_ordered], dtype=float)
            ys = np.array([std_p.get(m, np.nan)  for m in months_ordered], dtype=float)
            ax.plot(months_ordered, ym, color=color, linewidth=2, zorder=4)
            ax.fill_between(months_ordered, ym - ys, ym + ys, color=color, alpha=0.2)

            amp = np.nanmax(ym) - np.nanmin(ym)
            log(f"    {sig:8s}  {cls_label:12s}: amplitude={amp:.4f}")

            ax.set_xticks(months_ordered)
            ax.set_xticklabels(month_labels, fontsize=7)
            if row == 2:
                ax.set_xlabel("Month", fontsize=8)
            if col == 0:
                ax.set_ylabel(sig_label, fontsize=8)
            ax.set_title(f"{cls_label} (n={n_pixels})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Longreach — Monthly profiles of swir_mi, B11, B08 by class (2020–2025)\n"
        "Thick line = mean across years; shaded band = ±1 std",
        fontsize=10, y=1.01,
    )
    out = OUT_DIR / "longreach_swir_monthly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 5 — Per-pixel annual low percentile
# ---------------------------------------------------------------------------

def per_pixel_swir_stats(df: pd.DataFrame, dry_nir: pd.DataFrame,
                         amp_stats: pd.DataFrame, re_stats: pd.DataFrame) -> pd.DataFrame:
    log("\n--- Step 5: Per-pixel annual 10th-percentile summary ---")

    counts = df.groupby(["point_id", "year"])["swir_mi"].count()
    valid  = counts[counts >= MIN_OBS_PER_YEAR].index
    n_excl = len(counts) - len(valid)
    log(f"  (pixel, year) groups: {len(counts):,}  |  "
        f"excluded (< {MIN_OBS_PER_YEAR} obs): {n_excl}")

    # obs-count distribution
    obs_counts = counts.values
    log(f"  Obs per (pixel, year) — min: {obs_counts.min()}  "
        f"median: {np.median(obs_counts):.0f}  max: {obs_counts.max()}")
    below_thresh = counts[counts < MIN_OBS_PER_YEAR]
    if len(below_thresh):
        # Which pixels are consistently under threshold
        by_pixel = below_thresh.groupby("point_id").size()
        always_sparse = by_pixel[by_pixel >= 3]  # flagged if ≥3 years sparse
        if len(always_sparse):
            log(f"  WARNING: {len(always_sparse)} pixels sparse in ≥3 years — "
                f"consider lower threshold or exclusion")

    df_valid = df.set_index(["point_id", "year"]).loc[valid].reset_index()

    annual_p10 = (
        df_valid.groupby(["point_id", "year"])["swir_mi"]
        .quantile(0.10)
        .rename("swir_p10_yr")
        .reset_index()
    )
    stats = (
        annual_p10.groupby("point_id")["swir_p10_yr"]
        .agg(swir_p10="mean", swir_p10_std="std", n_years="count")
        .reset_index()
    )
    stats["swir_p10_cv"] = stats["swir_p10_std"] / stats["swir_p10"]

    coords = df[["point_id", "lon", "lat"]].drop_duplicates("point_id")
    stats  = stats.merge(coords, on="point_id", how="left")

    # Class flags + nir_cv, nir_mean
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    stats = stats.merge(
        dry_nir[["point_id", "nir_cv", "nir_mean", "in_hd_bbox"]],
        on="point_id", how="left",
    )
    stats["is_riparian"]  = (~stats["in_hd_bbox"]) & (stats["nir_mean"] >= rip_thresh)
    stats["is_grassland"] = (~stats["in_hd_bbox"]) & (~stats["is_riparian"])

    # rec_mean from amp_stats
    stats = stats.merge(amp_stats[["point_id", "rec_mean"]], on="point_id", how="left")

    # re_p10 from red-edge stats
    stats = stats.merge(re_stats[["point_id", "re_p10"]], on="point_id", how="left")

    log(f"  Pixels with stats: {len(stats)}")
    log(f"  n_years per pixel — min: {stats['n_years'].min()}  "
        f"median: {stats['n_years'].median():.0f}  max: {stats['n_years'].max()}")

    log("\n  Class-level swir_p10 distributions:")
    for label, mask in [("Infestation", stats["in_hd_bbox"]),
                        ("Riparian",    stats["is_riparian"]),
                        ("Grassland",   stats["is_grassland"])]:
        sub = stats[mask]
        if len(sub) == 0:
            continue
        q25, q75 = sub["swir_p10"].quantile([0.25, 0.75])
        log(f"    {label:12s}: mean={sub['swir_p10'].mean():.4f}  "
            f"median={sub['swir_p10'].median():.4f}  "
            f"IQR=[{q25:.4f},{q75:.4f}]  n={len(sub)}")

    return stats


def save_stats(stats: pd.DataFrame) -> None:
    cols = ["point_id", "lon", "lat",
            "swir_p10", "swir_p10_std", "swir_p10_cv", "n_years",
            "nir_cv", "nir_mean", "rec_mean", "re_p10",
            "in_hd_bbox", "is_riparian", "is_grassland"]
    out = OUT_DIR / "longreach_swir_stats.parquet"
    stats[cols].to_parquet(out, index=False)
    log(f"  Saved stats: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 6 — B11 time series diagnostic (sample pixels)
# ---------------------------------------------------------------------------

def b11_timeseries_diagnostic(df: pd.DataFrame, dry_nir: pd.DataFrame) -> None:
    log("\n--- Step 6: B11 time series diagnostic ---")

    # Select 10 highest-nir_mean infestation pixels and 10 lowest-nir_mean grassland pixels
    rip_thresh = dry_nir.loc[~dry_nir["in_hd_bbox"], "nir_mean"].quantile(RIPARIAN_PERCENTILE)
    dry_nir = dry_nir.copy()
    dry_nir["is_grassland"] = (~dry_nir["in_hd_bbox"]) & (dry_nir["nir_mean"] < rip_thresh)

    inf_sample  = (dry_nir[dry_nir["in_hd_bbox"]]
                   .nlargest(N_SAMPLE, "nir_mean")["point_id"].tolist())
    grass_sample = (dry_nir[dry_nir["is_grassland"]]
                    .nsmallest(N_SAMPLE, "nir_mean")["point_id"].tolist())

    log(f"  Infestation sample ({len(inf_sample)}): {inf_sample}")
    log(f"  Grassland sample   ({len(grass_sample)}): {grass_sample}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)

    month_cmap = plt.cm.get_cmap("hsv", 12)

    for ax, pids, title in [
        (axes[0], inf_sample,   f"Infestation — top-{N_SAMPLE} NIR pixels"),
        (axes[1], grass_sample, f"Grassland   — bottom-{N_SAMPLE} NIR pixels"),
    ]:
        sub = df[df["point_id"].isin(pids)]

        # Scatter coloured by month
        sc = ax.scatter(sub["date"], sub["B11"],
                        c=sub["month"], cmap="hsv", vmin=1, vmax=12,
                        s=6, alpha=0.4, zorder=2)

        # Annual 10th-percentile markers per pixel
        for pid in pids:
            pix = sub[sub["point_id"] == pid]
            for yr, grp in pix.groupby("year"):
                if len(grp) < MIN_OBS_PER_YEAR:
                    continue
                p10_val  = grp["B11"].quantile(0.10)
                p10_date = pd.Timestamp(f"{yr}-07-01")  # representative x position
                ax.scatter([p10_date], [p10_val], marker="_", s=120,
                           color="black", linewidths=1.2, zorder=4)

        ax.set_title(title, fontsize=9)
        ax.set_ylabel("B11 reflectance", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.015, pad=0.01)
        cb.set_label("Month", fontsize=7)
        cb.set_ticks(list(range(1, 13)))
        cb.ax.tick_params(labelsize=6)

        # Flag if any pixel-year has unusually high dry-season B11
        for pid in pids:
            pix = sub[sub["point_id"] == pid]
            for yr, grp in pix.groupby("year"):
                b11_range = grp["B11"].max() - grp["B11"].min()
                if b11_range > 2 * sub["B11"].std() * 2:
                    log(f"  WARNING: pixel {pid} year {yr} has B11 range={b11_range:.4f} "
                        f"— potential outlier")

    axes[-1].set_xlabel("Date", fontsize=9)
    fig.suptitle("B11 raw time series — per-acquisition scatter coloured by month\n"
                 "Horizontal bars = annual 10th percentile; diagnostic for outliers & signal attribution",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "longreach_b11_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Steps 7-8 — Spatial map and histogram
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
    log("\n--- Step 7: Spatial map ---")
    fig, _ = _map_base(
        stats, col="swir_p10", cmap="RdYlGn",
        label="Mean annual 10th-pctile swir_mi",
        title="Longreach — SWIR moisture index annual p10 (swir_p10)\n"
              "Higher = canopy water retained; Parkinsonia signature",
        bg_img=bg_img,
    )
    out = OUT_DIR / "longreach_swir_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_histogram(stats: pd.DataFrame) -> None:
    log("\n--- Step 8: Histogram by class ---")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    groups = [
        ("Infestation", stats[stats["in_hd_bbox"]],   "darkorange"),
        ("Riparian",    stats[stats["is_riparian"]],   "steelblue"),
        ("Grassland",   stats[stats["is_grassland"]],  "olivedrab"),
    ]
    bins = np.linspace(stats["swir_p10"].min() - 0.005,
                       stats["swir_p10"].max() + 0.005, 35)
    for label, sub, color in groups:
        ax.hist(sub["swir_p10"], bins=bins, alpha=0.65, color=color,
                edgecolor="white", linewidth=0.4,
                label=f"{label} (n={len(sub)}, med={sub['swir_p10'].median():.3f})")

    ax.set_xlabel("swir_p10 (mean annual 10th-pctile of swir_mi)", fontsize=9)
    ax.set_ylabel("Pixel count", fontsize=9)
    ax.set_title("Distribution of SWIR moisture index annual p10 by land-cover class\n"
                 "Higher = canopy water retained at dry-season low", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_swir_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 9 — 2D projections
# ---------------------------------------------------------------------------

def plot_2d_nir_cv_vs_swir(stats: pd.DataFrame) -> None:
    log("\n--- Step 9a: 2D scatter nir_cv × swir_p10 ---")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    groups = [
        ("Grassland",   stats[stats["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    stats[stats["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", stats[stats["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]
    for label, sub, color, size, alpha in groups:
        ax.scatter(sub["nir_cv"], sub["swir_p10"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(sub)})", zorder=3)
    for label, sub, color, *_ in groups:
        cx, cy = sub["nir_cv"].mean(), sub["swir_p10"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel("Dry-season NIR CV (lower = more stable)", fontsize=9)
    ax.set_ylabel("swir_p10 (higher = canopy water retained)", fontsize=9)
    ax.set_title("nir_cv × swir_p10 — stability vs hydration", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_nir_cv_vs_swir.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_2d_rec_vs_swir(stats: pd.DataFrame) -> None:
    log("\n--- Step 9b: 2D scatter rec_mean × swir_p10 ---")
    sub = stats.dropna(subset=["rec_mean"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    groups = [
        ("Grassland",   sub[sub["is_grassland"]],  "olivedrab",  10, 0.45),
        ("Riparian",    sub[sub["is_riparian"]],    "steelblue",  18, 0.65),
        ("Infestation", sub[sub["in_hd_bbox"]],     "darkorange", 22, 0.80),
    ]
    for label, grp, color, size, alpha in groups:
        ax.scatter(grp["rec_mean"], grp["swir_p10"],
                   c=color, s=size, alpha=alpha, linewidths=0,
                   label=f"{label} (n={len(grp)})", zorder=3)
    for label, grp, color, *_ in groups:
        cx, cy = grp["rec_mean"].mean(), grp["swir_p10"].mean()
        ax.scatter([cx], [cy], c=color, s=80, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(f"{label}\n({cx:.3f}, {cy:.3f})",
                    xy=(cx, cy), xytext=(8, 6), textcoords="offset points",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel("rec_mean (NDVI seasonal recession)", fontsize=9)
    ax.set_ylabel("swir_p10 (higher = canopy water retained)", fontsize=9)
    ax.set_title("rec_mean × swir_p10 — recession vs hydration", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "longreach_rec_vs_swir.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Step 10 — Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(stats: pd.DataFrame) -> None:
    log("\n--- Step 10: Correlation analysis ---")
    pairs = [
        ("nir_cv",   "Dry-season NIR CV"),
        ("rec_mean", "NDVI recession (rec_mean)"),
        ("re_p10",   "Red-edge annual p10 (re_p10)"),
        ("nir_mean", "Dry-season NIR mean"),
    ]
    for col, desc in pairs:
        sub = stats[["swir_p10", col]].dropna()
        if len(sub) < 10:
            log(f"  {desc}: insufficient data ({len(sub)} rows)")
            continue
        r, p = pearsonr(sub["swir_p10"], sub[col])
        redundant = "REDUNDANT (r ≥ 0.7)" if abs(r) >= 0.7 else "independent"
        log(f"  swir_p10 vs {desc}: r={r:.3f}  p={p:.4f}  → {redundant}")

    # Partial correlation of swir_p10 with rec_mean, controlling for nir_mean
    sub = stats[["swir_p10", "rec_mean", "nir_mean"]].dropna()
    if len(sub) >= 10:
        from numpy.linalg import lstsq

        def residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
            coef, *_ = lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)
            return y - np.c_[np.ones(len(X)), X] @ coef

        res_swir = residuals(sub["swir_p10"].values, sub["nir_mean"].values)
        res_rec  = residuals(sub["rec_mean"].values,  sub["nir_mean"].values)
        r_partial, p_partial = pearsonr(res_swir, res_rec)
        log(f"  Partial r(swir_p10, rec_mean | nir_mean) = {r_partial:.3f}  p={p_partial:.4f}")
        if abs(r_partial) < 0.7:
            log("    → swir_p10 carries information beyond shared NIR structure")
        else:
            log("    → swir_p10 and rec_mean collinear after controlling for NIR")


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def log_success_criteria(df: pd.DataFrame, stats: pd.DataFrame,
                          frac_positive: float) -> None:
    log("\n--- Success criteria ---")

    inf   = stats[stats["in_hd_bbox"]]
    grass = stats[stats["is_grassland"]]

    # 1. Fraction of dates with positive contrast ≥ 0.6
    status1 = "PASS" if frac_positive >= 0.6 else "FAIL"
    log(f"  [1] Fraction of dates with infestation swir_mi > grassland: "
        f"{frac_positive:.2f}  → {status1}")

    # 2. Class ordering: infestation swir_p10 > grassland swir_p10
    parko_med = inf["swir_p10"].median()
    grass_med = grass["swir_p10"].median()
    status2 = "PASS" if parko_med > grass_med else "FAIL"
    log(f"  [2] swir_p10 ordering — infestation={parko_med:.4f}  "
        f"grassland={grass_med:.4f}  → {status2}")

    # 3. IQR overlap fraction < 0.5
    inf_q25, inf_q75 = inf["swir_p10"].quantile([0.25, 0.75])
    grs_q25, grs_q75 = grass["swir_p10"].quantile([0.25, 0.75])
    overlap = max(0, min(inf_q75, grs_q75) - max(inf_q25, grs_q25))
    span    = max(inf_q75, grs_q75) - min(inf_q25, grs_q25)
    overlap_frac = overlap / span if span > 0 else 1.0
    status3 = "PASS" if overlap_frac < 0.5 else "FAIL"
    log(f"  [3] IQR overlap fraction (infestation vs grassland): "
        f"{overlap_frac:.2f}  → {status3}")

    # 4. B11 seasonal amplitude for grassland > infestation (checked in monthly_profiles)
    log("  [4] B11 amplitude check — see monthly profile log above (target: grassland amp > infestation amp)")

    # 5. Low correlation with rec_mean (r < 0.7)
    sub = stats[["swir_p10", "rec_mean"]].dropna()
    if len(sub) >= 10:
        r, _ = pearsonr(sub["swir_p10"], sub["rec_mean"])
        status5 = "PASS" if abs(r) < 0.7 else "FAIL"
        log(f"  [5] Pearson r(swir_p10, rec_mean) = {r:.3f}  → {status5}")
    else:
        log("  [5] Insufficient data for correlation check")

    log("-------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach SWIR moisture index analysis ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df      = load_and_filter(PARQUET_PATH)
    dry_nir = pd.read_parquet(DRY_NIR_STATS)
    amp     = pd.read_parquet(AMP_STATS)
    re      = pd.read_parquet(RE_STATS)

    log(f"\nLoaded dry-NIR stats: {len(dry_nir)} pixels")
    log(f"Loaded amp stats:     {len(amp)} pixels")
    log(f"Loaded re stats:      {len(re)} pixels")

    df = assign_classes(df, dry_nir)

    log("\nClass counts (pixel-observations):")
    log(f"  Infestation: {df['in_hd_bbox'].sum():,} obs  "
        f"({df[df['in_hd_bbox']]['point_id'].nunique()} pixels)")
    log(f"  Riparian:    {df['is_riparian'].sum():,} obs  "
        f"({df[df['is_riparian']]['point_id'].nunique()} pixels)")
    log(f"  Grassland:   {df['is_grassland'].sum():,} obs  "
        f"({df[df['is_grassland']]['point_id'].nunique()} pixels)")

    frac_positive = contrast_time_series(df)
    monthly_profiles(df)

    stats = per_pixel_swir_stats(df, dry_nir, amp, re)
    save_stats(stats)

    b11_timeseries_diagnostic(df, dry_nir)

    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    plot_map(stats, bg_img)
    plot_histogram(stats)
    plot_2d_nir_cv_vs_swir(stats)
    plot_2d_rec_vs_swir(stats)
    correlation_analysis(stats)
    log_success_criteria(df, stats, frac_positive)

    log("\nDone.")


if __name__ == "__main__":
    main()
