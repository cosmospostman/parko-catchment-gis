"""Flowering flash detection — Longreach Parkinsonia infestation.

Computes five visible/NIR indices sensitive to a yellow-flower spectral signature
and scans the full 2020–2025 time series for a phenological window where infestation
pixels show an anomalous spike above their own baseline.

The extension population is retained for context but the primary signal is within-
infestation: for each pixel, each observation is expressed as a z-score anomaly
relative to that pixel's own annual median for the same DOY bin. A flowering flash
would appear as a transient positive anomaly shared across many infestation pixels
simultaneously, at any time of year.

Indices computed per observation:
  FI_rg   = (B03 + B04) / B08
  FI_r    = B04 / B08
  FI_by   = (B03 + B04) / (B02 + B08)   ← primary; suppresses bare-soil false positives
  dNDVI   = −(B08 − B04) / (B08 + B04)
  FI_swir = B11 / B08

Outputs:
  outputs/longreach-flowering/fi_doy_profiles.png       (DOY anomaly profiles, all indices)
  outputs/longreach-flowering/fi_by_timeseries.png      (raw FI_by + anomaly time series)
  outputs/longreach-flowering/fi_by_spatial_peak.png    (pixel map at peak anomaly DOY bin)
  outputs/longreach-flowering/flowering_window_by_year.csv

See research/LONGREACH-FLOWERING.md for the full analysis plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

PARQUET_PATH   = PROJECT_ROOT / "data" / "longreach_pixels.parquet"
OUT_DIR        = PROJECT_ROOT / "outputs" / "longreach-flowering"

# Quality filter
SCL_PURITY_MIN = 0.5

# Minimum total observations across the archive to include a pixel
MIN_PIXEL_OBS  = 10

# DOY bin width in days (26 bins × 14 days ≈ full year)
DOY_BIN_DAYS   = 14

# Original infestation patch bbox (used for spatial overlay and success criteria)
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Infestation patch lat boundary: pixels north of this are infestation, south are extension
INFESTATION_LAT_MIN = -22.764033   # southernmost row of the original 374-pixel fetch

# Full survey bbox (infestation + southern extension) — used for WMS and spatial maps
SURVEY_BBOX = [145.423948, -22.767104, 145.424956, -22.761054]

# Indices to compute — (column_name, label_for_plots)
INDICES = [
    ("FI_rg",   "FI_rg  = (B03+B04)/B08"),
    ("FI_r",    "FI_r   = B04/B08"),
    ("FI_by",   "FI_by  = (B03+B04)/(B02+B08)"),
    ("dNDVI",   "dNDVI  = −(B08−B04)/(B08+B04)"),
    ("FI_swir", "FI_swir = B11/B08"),
]
PRIMARY_INDEX = "FI_by"

# Year-by-year peak detection: flag dates where FI_by exceeds this percentile
# of the pixel's own annual distribution
PEAK_PERCENTILE = 75

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
    log(f"  Quality filter (scl_purity ≥ {SCL_PURITY_MIN}): dropped {dropped:,} rows "
        f"({100 * dropped / before:.1f}%), retained {len(df):,}")

    # Drop pixels with too few total observations (edge pixels)
    obs_per_pixel = df.groupby("point_id").size()
    valid_pixels  = obs_per_pixel[obs_per_pixel >= MIN_PIXEL_OBS].index
    n_dropped_px  = df["point_id"].nunique() - len(valid_pixels)
    df = df[df["point_id"].isin(valid_pixels)].copy()
    log(f"  Min-obs filter (≥ {MIN_PIXEL_OBS} obs per pixel): dropped {n_dropped_px} pixels, "
        f"retained {df['point_id'].nunique()} pixels")

    return df


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    log("\nComputing flowering indices...")
    df = df.copy()

    df["FI_rg"]   = (df["B03"] + df["B04"]) / df["B08"]
    df["FI_r"]    = df["B04"] / df["B08"]
    df["FI_by"]   = (df["B03"] + df["B04"]) / (df["B02"] + df["B08"])
    df["dNDVI"]   = -(df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["FI_swir"] = df["B11"] / df["B08"]

    # Replace inf / -inf from near-zero denominators with NaN
    index_cols = [name for name, _ in INDICES]
    df[index_cols] = df[index_cols].replace([np.inf, -np.inf], np.nan)

    for name, label in INDICES:
        valid = df[name].dropna()
        log(f"  {name:8s}  n_valid={len(valid):,}  "
            f"min={valid.min():.4f}  median={valid.median():.4f}  max={valid.max():.4f}")

    # Label pixels as infestation or extension
    coords = df[["point_id", "lat"]].drop_duplicates("point_id")
    infestation_ids = coords[coords["lat"] >= INFESTATION_LAT_MIN]["point_id"]
    df["population"] = np.where(df["point_id"].isin(infestation_ids), "infestation", "extension")
    n_inf = df[df["population"] == "infestation"]["point_id"].nunique()
    n_ext = df[df["population"] == "extension"]["point_id"].nunique()
    log(f"\n  Infestation pixels: {n_inf}  |  Extension pixels: {n_ext}")

    return df


def doy_bin(doy: pd.Series) -> pd.Series:
    """Map day-of-year to bin start DOY (0-indexed bins of DOY_BIN_DAYS days)."""
    return ((doy - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1


def compute_pixel_anomalies(pop_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-pixel z-score anomaly columns to a population subset.

    Baseline: per-pixel per-DOY-bin median across all years.
    Denominator: per-pixel overall std across all dates and seasons.

    Works on any population — call separately for infestation and extension
    so each pixel's baseline is estimated from its own time series.
    """
    pop_df = pop_df.copy()
    pop_df["doy"]     = pop_df["date"].dt.dayofyear
    pop_df["doy_bin"] = doy_bin(pop_df["doy"])
    pop_df["year"]    = pop_df["date"].dt.year

    index_cols = [name for name, _ in INDICES]

    baseline = (
        pop_df.groupby(["point_id", "doy_bin"])[index_cols]
        .median()
        .reset_index()
        .rename(columns={c: f"{c}_base" for c in index_cols})
    )
    pixel_std = (
        pop_df.groupby("point_id")[index_cols]
        .std()
        .reset_index()
        .rename(columns={c: f"{c}_std" for c in index_cols})
    )

    pop_df = pop_df.merge(baseline,   on=["point_id", "doy_bin"], how="left")
    pop_df = pop_df.merge(pixel_std,  on="point_id",              how="left")

    for name in index_cols:
        std_safe = pop_df[f"{name}_std"].where(pop_df[f"{name}_std"] > 1e-6, np.nan)
        pop_df[f"{name}_z"] = (pop_df[name] - pop_df[f"{name}_base"]) / std_safe

    return pop_df


def build_doy_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-pixel anomaly z-scores for both populations; DOY profiles for infestation.

    Returns
    -------
    profiles      : infestation mean z-score per DOY bin (used for plots + criteria)
    inf_anomalies : per-(pixel, date) anomaly scores for infestation pixels
    ext_anomalies : per-(pixel, date) anomaly scores for extension pixels
    """
    log("\nBuilding within-pixel DOY anomaly z-scores (both populations)...")

    inf_anomalies = compute_pixel_anomalies(df[df["population"] == "infestation"])
    ext_anomalies = compute_pixel_anomalies(df[df["population"] == "extension"])

    log(f"  Infestation pixels: {inf_anomalies['point_id'].nunique()}  "
        f"observations: {len(inf_anomalies):,}")
    log(f"  Extension pixels:   {ext_anomalies['point_id'].nunique()}  "
        f"observations: {len(ext_anomalies):,}")

    z_cols    = [f"{name}_z" for name, _ in INDICES]
    primary_z = f"{PRIMARY_INDEX}_z_mean"

    # Mean anomaly across infestation pixels per DOY bin (profiles for plots)
    profiles_long = (
        inf_anomalies.groupby("doy_bin")[z_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    profiles_long.columns = [
        "doy_bin" if c[0] == "doy_bin" else f"{c[0]}_{c[1]}"
        for c in profiles_long.columns
    ]

    log(f"\n  {PRIMARY_INDEX} mean within-infestation z-score anomaly by DOY bin:")
    for _, row in profiles_long.sort_values("doy_bin").iterrows():
        val = row[primary_z]
        if np.isnan(val):
            log(f"    DOY {int(row['doy_bin']):3d}  z=   NaN")
            continue
        bar_pos = "█" * max(0, int(val * 10))
        bar_neg = "▒" * max(0, int(-val * 10))
        log(f"    DOY {int(row['doy_bin']):3d}  z={val:+.3f}  {bar_neg}{bar_pos}")

    peak_bin = int(profiles_long.loc[profiles_long[primary_z].idxmax(), "doy_bin"])
    log(f"\n  Peak mean z-score anomaly at DOY bin {peak_bin} "
        f"(z={profiles_long.loc[profiles_long['doy_bin']==peak_bin, primary_z].values[0]:+.3f})")

    return profiles_long, inf_anomalies, ext_anomalies


def build_timeseries(df: pd.DataFrame, inf_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Scene-mean FI_by per date (both populations) plus mean infestation z-score."""
    log("\nBuilding FI_by scene-mean time series...")

    # Raw FI_by mean per date × population
    ts = (
        df.groupby(["population", "date"])[PRIMARY_INDEX]
        .mean()
        .reset_index()
        .pivot(index="date", columns="population", values=PRIMARY_INDEX)
        .reset_index()
    )
    ts.columns.name = None

    # Mean z-score anomaly across infestation pixels per date
    z_col = f"{PRIMARY_INDEX}_z"
    inf_z = (
        inf_anomalies.groupby("date")[z_col]
        .mean()
        .reset_index()
        .rename(columns={z_col: "inf_z"})
    )
    ts = ts.merge(inf_z, on="date", how="left")

    log(f"  Dates in time series: {len(ts)}")
    log(f"  FI_by mean — infestation: {ts['infestation'].mean():.4f}  "
        f"extension: {ts['extension'].mean():.4f}")
    log(f"  Infestation z-score — mean: {ts['inf_z'].mean():.3f}  "
        f"max: {ts['inf_z'].max():.3f}  "
        f"on {ts.loc[ts['inf_z'].idxmax(), 'date'].date() if ts['inf_z'].notna().any() else 'n/a'}")

    return ts


def build_flowering_windows(inf_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Per-year dates where mean infestation FI_by z-score anomaly exceeds 1.0.

    Uses the within-infestation anomaly time series so the threshold is relative
    to each pixel's own seasonal baseline, not an absolute index value.
    """
    log(f"\nDetecting per-year flowering windows (mean FI_by z-score > 1.0)...")

    z_col = f"{PRIMARY_INDEX}_z"
    inf   = inf_anomalies.copy()
    inf["year"] = inf["date"].dt.year

    rows = []
    for year, grp in inf.groupby("year"):
        # Mean z-score across pixels per acquisition date
        scene_z  = grp.groupby("date")[z_col].mean()
        elevated = scene_z[scene_z >= 1.0]
        if len(elevated) == 0:
            log(f"  {year}: no dates with mean z-score ≥ 1.0  "
                f"(year max z={scene_z.max():.3f})")
            rows.append({"year": year, "n_dates": 0,
                         "doy_start": np.nan, "doy_end": np.nan,
                         "peak_z": scene_z.max()})
            continue
        doys   = elevated.index.dayofyear
        peak_z = elevated.max()
        log(f"  {year}: {len(elevated)} elevated dates, "
            f"DOY {doys.min()}–{doys.max()}  peak z={peak_z:.3f}  "
            f"dates: {', '.join(str(d.date()) for d in elevated.index)}")
        rows.append({"year": year, "n_dates": len(elevated),
                     "doy_start": int(doys.min()), "doy_end": int(doys.max()),
                     "peak_z": peak_z})

    windows = pd.DataFrame(rows)
    return windows


def build_spatial_peak(df: pd.DataFrame, inf_anomalies: pd.DataFrame,
                        ext_anomalies: pd.DataFrame,
                        windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Per-pixel mean FI_by z-score on the single highest-z acquisition date.

    Uses the date with the highest mean z-score across infestation pixels (from the
    flowering windows) as the spatial snapshot.  This avoids the cancellation problem
    of averaging residuals across a whole DOY bin.
    """
    z_col = f"{PRIMARY_INDEX}_z"

    # Find the single date with the highest mean z-score across infestation pixels
    scene_z = inf_anomalies.groupby("date")[z_col].mean()
    peak_date = scene_z.idxmax()
    peak_z_mean = scene_z.max()
    log(f"\nBuilding spatial map for peak date {peak_date.date()} "
        f"(mean z={peak_z_mean:.3f} across infestation pixels)...")

    coords = df[["point_id", "lon", "lat", "population"]].drop_duplicates("point_id")

    # Infestation: z-score anomaly on peak date
    inf_subset = inf_anomalies[inf_anomalies["date"] == peak_date]
    log(f"  Infestation observations on date: {len(inf_subset):,} across {inf_subset['point_id'].nunique()} pixels")

    inf_spatial = (
        inf_subset[["point_id", z_col]]
        .rename(columns={z_col: "fi_peak_z"})
        .merge(coords, on="point_id", how="left")
    )
    log(f"  fi_peak_z range (infestation): "
        f"{inf_spatial['fi_peak_z'].min():.3f} – {inf_spatial['fi_peak_z'].max():.3f}  "
        f"mean={inf_spatial['fi_peak_z'].mean():.3f}")

    # Extension: z-score anomaly on peak date (same method, own baseline)
    ext_subset = ext_anomalies[ext_anomalies["date"] == peak_date]
    log(f"  Extension observations on date: {len(ext_subset):,} across {ext_subset['point_id'].nunique()} pixels")

    z_col = f"{PRIMARY_INDEX}_z"
    ext_spatial = (
        ext_subset[["point_id", z_col]]
        .rename(columns={z_col: "fi_peak_z"})
        .merge(coords, on="point_id", how="left")
    )
    log(f"  fi_peak_z range (extension):   "
        f"{ext_spatial['fi_peak_z'].min():.3f} – {ext_spatial['fi_peak_z'].max():.3f}  "
        f"mean={ext_spatial['fi_peak_z'].mean():.3f}")

    return inf_spatial, ext_spatial, peak_date


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_doy_profiles(doy_profiles: pd.DataFrame, out_path: Path) -> None:
    """5-panel within-infestation z-score anomaly profiles across DOY bins."""
    n = len(INDICES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

    for ax, (name, label) in zip(axes, INDICES):
        mean_col = f"{name}_z_mean"
        std_col  = f"{name}_z_std"

        if mean_col not in doy_profiles.columns:
            continue

        z_mean = doy_profiles[mean_col]
        z_std  = doy_profiles[std_col] if std_col in doy_profiles.columns else None
        x      = doy_profiles["doy_bin"]

        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        if z_std is not None:
            ax.fill_between(x, z_mean - z_std, z_mean + z_std,
                            alpha=0.18, color="steelblue", label="±1 std across pixels")
        ax.plot(x, z_mean, color="steelblue", linewidth=1.8, label="Mean z-score anomaly")
        ax.fill_between(x, 0, z_mean, where=z_mean > 0,
                        alpha=0.25, color="gold", label="Above baseline (candidate flowering)")
        ax.fill_between(x, 0, z_mean, where=z_mean < 0,
                        alpha=0.15, color="tomato", label="Below baseline")

        ax.set_ylabel("z-score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        ax.set_title(label, fontsize=8, loc="left")

    axes[-1].set_xlabel("Day of year", fontsize=8)
    axes[-1].set_xticks(doy_profiles["doy_bin"].values[::2])
    axes[-1].tick_params(axis="x", labelsize=7, rotation=45)

    fig.suptitle(
        "Longreach — within-infestation flowering index anomaly (z-score) by DOY\n"
        "Each panel: mean z-score across infestation pixels per 14-day bin (2020–2025 pooled)\n"
        "Positive = above pixel's own seasonal baseline → candidate flowering flash",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved DOY profiles: {out_path.relative_to(PROJECT_ROOT)}")


def plot_timeseries(ts: pd.DataFrame, out_path: Path) -> None:
    """Raw FI_by scene-mean (both populations) + infestation z-score anomaly."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    if "infestation" in ts.columns:
        ax1.plot(ts["date"], ts["infestation"], color="forestgreen", linewidth=0.9,
                 alpha=0.85, label="Infestation (raw FI_by)")
    if "extension" in ts.columns:
        ax1.plot(ts["date"], ts["extension"], color="peru", linewidth=0.9,
                 alpha=0.85, linestyle="--", label="Extension (raw FI_by)")

    ax1.set_ylabel("FI_by  (B03+B04)/(B02+B08)", fontsize=8)
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.tick_params(labelsize=7)
    ax1.set_title(
        "Longreach — FI_by scene-mean time series 2020–2025\n"
        "Each point = mean across all pixels in that population for one S2 acquisition",
        fontsize=9,
    )

    # Lower panel: within-infestation z-score anomaly
    if "inf_z" in ts.columns:
        ax2.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax2.fill_between(ts["date"], 0, ts["inf_z"],
                         where=ts["inf_z"] > 0, alpha=0.55, color="gold",
                         label="Above pixel baseline (candidate flowering)")
        ax2.fill_between(ts["date"], 0, ts["inf_z"],
                         where=ts["inf_z"] < 0, alpha=0.35, color="steelblue",
                         label="Below pixel baseline")
        ax2.plot(ts["date"], ts["inf_z"], color="darkgoldenrod", linewidth=0.7, alpha=0.7)
        ax2.axhline(1,  color="gold",     linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.axhline(-1, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.7)

    ax2.set_ylabel("Mean z-score\n(infestation pixels)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.7)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax2.set_xlabel("Date", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved time series: {out_path.relative_to(PROJECT_ROOT)}")


def plot_spatial_peak(inf_spatial: pd.DataFrame, ext_spatial: pd.DataFrame,
                      peak_date, out_path: Path,
                      bg_img: np.ndarray | None) -> None:
    """Side-by-side spatial maps: infestation (left) and extension (right).

    Both panels use the same RdYlGn z-score scale anchored to the infestation
    p02–p98 range, so the two populations are directly comparable.
    Each panel shows its own population's pixels only, with the WMS aerial
    as background.
    """
    lon_min, lat_min, lon_max, lat_max = SURVEY_BBOX

    # Infestation occupies north half of survey bbox; extension occupies south half
    inf_lat_min = INFESTATION_LAT_MIN
    inf_lat_max = lat_max
    ext_lat_min = lat_min
    ext_lat_max = INFESTATION_LAT_MIN

    lat_centre    = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320
    lon_span      = lon_max - lon_min

    # Each panel height proportional to its lat span
    inf_lat_span = inf_lat_max - inf_lat_min
    ext_lat_span = ext_lat_max - ext_lat_min
    panel_w      = 5.5
    inf_h        = panel_w * (inf_lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg)
    ext_h        = panel_w * (ext_lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg)

    # Symmetric colour scale anchored to infestation p02–p98
    inf_z = inf_spatial["fi_peak_z"].dropna()
    vlim  = max(abs(inf_z.quantile(0.02)), abs(inf_z.quantile(0.98)), 1.0)

    # Marker size scaled to 10 m pixel at panel width
    pt_per_deg = panel_w * 72 / lon_span
    marker_pt  = (10 / lon_m_per_deg) * pt_per_deg
    marker_s   = max(1.0, marker_pt ** 2 / 8)

    fig, (ax_inf, ax_ext) = plt.subplots(
        1, 2,
        figsize=(panel_w * 2 + 1.5, max(inf_h, ext_h) + 1.2),
        dpi=150,
    )

    for ax, spatial, bbox_lat_min, bbox_lat_max, title in [
        (ax_inf, inf_spatial, inf_lat_min, inf_lat_max,
         f"Infestation (north)\nmean z = {inf_spatial['fi_peak_z'].mean():.2f}"),
        (ax_ext, ext_spatial, ext_lat_min, ext_lat_max,
         f"Extension / grassland (south)\nmean z = {ext_spatial['fi_peak_z'].mean():.2f}"),
    ]:
        if bg_img is not None:
            # Slice the WMS image to this panel's lat range
            full_lat_span = lat_max - lat_min
            img_h = bg_img.shape[0]
            # WMS origin="upper": row 0 = lat_max, row img_h = lat_min
            row_top    = int((lat_max - bbox_lat_max) / full_lat_span * img_h)
            row_bottom = int((lat_max - bbox_lat_min) / full_lat_span * img_h)
            row_top    = max(0, row_top)
            row_bottom = min(img_h, row_bottom)
            panel_bg   = bg_img[row_top:row_bottom, :, :]
            ax.imshow(
                panel_bg,
                extent=[lon_min, lon_max, bbox_lat_min, bbox_lat_max],
                origin="upper", aspect="auto", interpolation="bilinear", zorder=0,
            )

        sc = ax.scatter(
            spatial["lon"], spatial["lat"],
            c=spatial["fi_peak_z"], cmap="RdYlGn",
            s=marker_s, linewidths=0.0, alpha=0.75, zorder=2,
            vmin=-vlim, vmax=vlim,
        )

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(bbox_lat_min, bbox_lat_max)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.set_title(title, fontsize=8)

    # Infestation bbox outline on left panel
    ax_inf.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN,
        HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.0, linestyle="--",
        label="High-density bbox", zorder=3,
    ))
    ax_inf.legend(loc="lower right", fontsize=6, framealpha=0.7,
                  facecolor="black", labelcolor="white", edgecolor="none")

    # Shared colorbar
    cb = fig.colorbar(sc, ax=[ax_inf, ax_ext], fraction=0.02, pad=0.02, aspect=30)
    cb.set_label("FI_by z-score anomaly (above own seasonal baseline)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Longreach — FI_by z-score anomaly on peak date {pd.Timestamp(peak_date).date()}\n"
        f"Uniform scale ±{vlim:.1f} σ — positive (green) = above pixel's own seasonal baseline",
        fontsize=9, y=1.01,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved spatial peak map: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def log_success_criteria(doy_profiles: pd.DataFrame, windows: pd.DataFrame,
                          spatial: pd.DataFrame) -> None:
    log("\n--- Success criteria (research/LONGREACH-FLOWERING.md) ---")

    primary_z = f"{PRIMARY_INDEX}_z_mean"
    peak_row  = doy_profiles.loc[doy_profiles[primary_z].idxmax()]
    peak_bin  = int(peak_row["doy_bin"])
    peak_z    = peak_row[primary_z]

    # 1. Peak mean z-score > 1.0 (one std above pixel's own baseline)
    status1 = "PASS" if peak_z > 1.0 else "FAIL"
    log(f"  [1] Peak mean z-score anomaly > 1.0: "
        f"z = {peak_z:.3f} at DOY bin {peak_bin}  → {status1}")

    # 2. Window recurs in ≥ 3 of 6 years with any elevated date within ±30 DOY of peak bin
    years_with_window = windows[windows["n_dates"] > 0]
    if len(years_with_window) > 0:
        in_range = years_with_window[
            (years_with_window["doy_start"] <= peak_bin + 30) &
            (years_with_window["doy_end"]   >= peak_bin - 30)
        ]
    else:
        in_range = years_with_window
    n_recurring = len(in_range)
    status2 = "PASS" if n_recurring >= 3 else "FAIL"
    log(f"  [2] Window recurs in ≥ 3 years with elevated date within ±30 DOY of peak: "
        f"{n_recurring} qualifying years  → {status2}")

    # 3. Spatial coherence: Pearson r between pixel z-score and 8-neighbour mean
    from scipy.spatial import cKDTree
    coords = spatial[["lon", "lat"]].values
    tree   = cKDTree(coords)
    dists, idxs = tree.query(coords, k=9)   # self + 8 neighbours
    neighbour_means = spatial["fi_peak_z"].values[idxs[:, 1:]].mean(axis=1)
    r = np.corrcoef(spatial["fi_peak_z"].values, neighbour_means)[0, 1]
    status3 = "PASS" if r >= 0.5 else "FAIL"
    log(f"  [3] Spatial coherence (Pearson r with 8-neighbour mean): "
        f"r = {r:.3f}  → {status3}")

    log("----------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Longreach flowering flash detection ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(PARQUET_PATH)
    df = add_indices(df)

    doy_profiles, inf_anomalies, ext_anomalies = build_doy_profiles(df)
    ts                          = build_timeseries(df, inf_anomalies)
    windows                     = build_flowering_windows(inf_anomalies)
    inf_spatial, ext_spatial, peak_date = build_spatial_peak(df, inf_anomalies, ext_anomalies, windows)

    # Save tabular outputs
    windows_path = OUT_DIR / "flowering_window_by_year.csv"
    windows.to_csv(windows_path, index=False)
    log(f"\nSaved flowering windows: {windows_path.relative_to(PROJECT_ROOT)}")

    # Fetch WMS background
    log("\nFetching Queensland Globe WMS background tile...")
    try:
        bg_img = fetch_wms_image(SURVEY_BBOX, width_px=2048)
        log(f"  Background tile: {bg_img.shape[1]}×{bg_img.shape[0]} px for bbox {SURVEY_BBOX}")
    except Exception as exc:
        log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without background")
        bg_img = None

    log("\nGenerating plots...")
    plot_doy_profiles(doy_profiles, OUT_DIR / "fi_doy_profiles.png")
    plot_timeseries(ts,             OUT_DIR / "fi_by_timeseries.png")
    plot_spatial_peak(inf_spatial, ext_spatial, peak_date, OUT_DIR / "fi_by_spatial_peak.png", bg_img)

    log_success_criteria(doy_profiles, windows, inf_spatial)

    log("\nDone.")


if __name__ == "__main__":
    main()
