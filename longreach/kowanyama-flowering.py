"""kowanyama-flowering.py — flowering flash detection at Kowanyama.

Applies the FI_by = (B03 + B04) / (B02 + B08) flowering index to the full
Kowanyama S2 time series. The signal of interest is a transient anomaly where
a pixel's FI_by spikes above its own seasonal baseline — the within-pixel
z-score approach from longreach/flowering.py.

Key differences from the Longreach analysis:
- No labelled infestation population: contrast-gating is not used.
- The 35 anchor pixels (confirmed presence, within 100 m of -15.457794,
  141.535690) serve as the positive reference.
- All other pixels serve as background.
- We ask: do anchor pixels have higher fi_p90 than the landscape median?

The dataset is large (145M rows). To keep memory manageable the script loads
only the four required bands (B02, B03, B04, B08) plus metadata, computes
FI_by and DOY-bin baselines in polars, then converts to pandas only for the
small per-pixel summary table (~250k rows).

Produces:
  outputs/kowanyama/
    kowanyama_fi_p90_map.png          — spatial map of fi_p90 on WMS
    kowanyama_fi_doy_profile.png      — mean FI_by z-score by DOY bin,
                                        anchor vs full background
    kowanyama_fi_anchor_timeseries.png — raw FI_by + z-score for anchor
                                        pixel centroid over time
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "kowanyama"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCL_PURITY_MIN  = 0.5
DOY_BIN_DAYS    = 14          # 14-day bins (~26 per year)
HAZE_B02_MAX    = 0.010       # scene-mean B02 anomaly above DOY baseline → hazy
MIN_PIXEL_OBS   = 10          # drop pixels with fewer total observations

ANCHOR_LAT      = -15.457794
ANCHOR_LON      = 141.535690
ANCHOR_RADIUS_M = 100.0

LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity", "B02", "B03", "B04", "B08"]


def log(msg: str) -> None:
    print(msg, flush=True)


def haversine_m(lat1, lon1, lats, lons):
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlam = np.radians(lons - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# 1. Load and quality-filter (polars)
# ---------------------------------------------------------------------------

log("Loading Kowanyama parquet (B02/B03/B04/B08 only)...")
df = pl.read_parquet(
    PROJECT_ROOT / "data" / "kowanyama_pixels.parquet",
    columns=LOAD_COLS,
)
log(f"  {len(df):,} rows  |  {df['point_id'].n_unique():,} pixels")

before = len(df)
df = df.filter(pl.col("scl_purity") >= SCL_PURITY_MIN).drop("scl_purity")
log(f"  Quality filter: dropped {before - len(df):,} rows, retained {len(df):,}")

# Derive FI_by and date metadata
df = df.with_columns([
    pl.col("date").dt.year().cast(pl.Int16).alias("year"),
    pl.col("date").dt.ordinal_day().alias("doy"),
    (
        (pl.col("B03") + pl.col("B04")) / (pl.col("B02") + pl.col("B08"))
    ).alias("fi_by"),
]).drop(["B02", "B03", "B04", "B08"])

# DOY bin (14-day bins, same as longreach/flowering.py)
df = df.with_columns(
    (((pl.col("doy") - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1).cast(pl.Int16).alias("doy_bin")
)

# Replace inf/-inf from near-zero denominators
df = df.with_columns(
    pl.when(pl.col("fi_by").is_infinite()).then(None).otherwise(pl.col("fi_by")).alias("fi_by")
)

log(f"  fi_by: median={df['fi_by'].median():.4f}  "
    f"p01={df['fi_by'].quantile(0.01):.4f}  "
    f"p99={df['fi_by'].quantile(0.99):.4f}")

# Extract coords lookup (before dropping lon/lat)
coords_pl = df.select(["point_id", "lon", "lat"]).unique("point_id")
df = df.drop(["lon", "lat"])

# ---------------------------------------------------------------------------
# 2. Haze filter — drop dates with elevated scene-mean B02
#    (B02 was dropped; approximate via fi_by scene mean being anomalously high
#    due to haze raising B03+B04. Use scene-mean fi_by deviation from DOY baseline.)
#
# We re-load B02 briefly just for the haze mask, then discard it.
# ---------------------------------------------------------------------------

log("\nBuilding haze mask...")
b02_df = pl.read_parquet(
    PROJECT_ROOT / "data" / "kowanyama_pixels.parquet",
    columns=["date", "scl_purity", "B02"],
).filter(pl.col("scl_purity") >= SCL_PURITY_MIN).drop("scl_purity")

b02_df = b02_df.with_columns([
    pl.col("B02").cast(pl.Float32),
    pl.col("date").dt.ordinal_day().alias("doy"),
])
b02_df = b02_df.with_columns(
    (((pl.col("doy") - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1).cast(pl.Int16).alias("doy_bin")
)

# Scene-mean B02 per date
scene_b02 = b02_df.group_by("date").agg(pl.col("B02").mean().alias("b02_scene"))
scene_b02 = scene_b02.with_columns(
    pl.col("date").dt.ordinal_day().alias("doy")
).with_columns(
    (((pl.col("doy") - 1) // DOY_BIN_DAYS) * DOY_BIN_DAYS + 1).cast(pl.Int16).alias("doy_bin")
)

# DOY-bin baseline for scene-mean B02
b02_baseline = scene_b02.group_by("doy_bin").agg(
    pl.col("b02_scene").median().alias("b02_base")
)
scene_b02 = scene_b02.join(b02_baseline, on="doy_bin").with_columns(
    (pl.col("b02_scene") - pl.col("b02_base")).alias("b02_anom")
)

clean_dates_pl = scene_b02.filter(pl.col("b02_anom") <= HAZE_B02_MAX).select("date")
hazy_count = len(scene_b02) - len(clean_dates_pl)
log(f"  Total acquisition dates: {len(scene_b02)}")
log(f"  Hazy dates removed:      {hazy_count}")
log(f"  Clean dates retained:    {len(clean_dates_pl)}")
del b02_df, scene_b02, b02_baseline

clean_dates_set = set(clean_dates_pl["date"].to_list())
df = df.filter(pl.col("date").is_in(clean_dates_set))
log(f"  Rows after haze filter: {len(df):,}")

# ---------------------------------------------------------------------------
# 3. Min-obs filter
# ---------------------------------------------------------------------------

obs_per_pixel = df.group_by("point_id").agg(pl.len().alias("n_obs"))
valid_pixels  = obs_per_pixel.filter(pl.col("n_obs") >= MIN_PIXEL_OBS).select("point_id")
before_px = df["point_id"].n_unique()
df = df.join(valid_pixels, on="point_id", how="inner")
log(f"\nMin-obs filter (>= {MIN_PIXEL_OBS}): dropped {before_px - df['point_id'].n_unique()} pixels, "
    f"retained {df['point_id'].n_unique():,}")

# ---------------------------------------------------------------------------
# 4. Per-pixel z-score anomaly
#
# Baseline: per-(pixel, doy_bin) median fi_by across all years.
# Std:      per-pixel overall std across all dates.
# z = (fi_by - baseline) / std
# ---------------------------------------------------------------------------

log("\nComputing per-pixel DOY-bin baselines...")

baseline = (
    df.group_by(["point_id", "doy_bin"])
    .agg(pl.col("fi_by").median().alias("fi_by_base"))
)
pixel_std = (
    df.group_by("point_id")
    .agg(pl.col("fi_by").std().alias("fi_by_std"))
)

df = (
    df
    .join(baseline,   on=["point_id", "doy_bin"], how="left")
    .join(pixel_std,  on="point_id",              how="left")
    .with_columns(
        pl.when(pl.col("fi_by_std") > 1e-6)
        .then((pl.col("fi_by") - pl.col("fi_by_base")) / pl.col("fi_by_std"))
        .otherwise(None)
        .alias("fi_by_z")
    )
)

log(f"  fi_by_z: median={df['fi_by_z'].median():.4f}  "
    f"p90={df['fi_by_z'].quantile(0.90):.4f}  "
    f"p99={df['fi_by_z'].quantile(0.99):.4f}")

# ---------------------------------------------------------------------------
# 5. Per-pixel fi_p90 summary
# ---------------------------------------------------------------------------

log("\nAggregating fi_p90 per pixel...")

fi_p90_pl = (
    df.group_by("point_id")
    .agg([
        pl.col("fi_by_z").quantile(0.90).alias("fi_p90"),
        pl.col("fi_by_z").median().alias("fi_z_median"),
        pl.col("fi_by").mean().alias("fi_by_mean"),
        pl.len().alias("n_obs"),
    ])
    .join(coords_pl, on="point_id", how="left")
)

fi_p90 = fi_p90_pl.to_pandas()
log(f"  {len(fi_p90):,} pixels")
log(f"  fi_p90: p10={fi_p90['fi_p90'].quantile(0.10):.3f}  "
    f"median={fi_p90['fi_p90'].median():.3f}  "
    f"p90={fi_p90['fi_p90'].quantile(0.90):.3f}")

# Anchor pixels
fi_p90["dist_anchor_m"] = haversine_m(
    ANCHOR_LAT, ANCHOR_LON, fi_p90["lat"].values, fi_p90["lon"].values
)
anchor = fi_p90[fi_p90["dist_anchor_m"] <= ANCHOR_RADIUS_M].copy()
background = fi_p90[fi_p90["dist_anchor_m"] > ANCHOR_RADIUS_M].copy()

log(f"\nAnchor pixels (n={len(anchor)}):")
log(f"  fi_p90: min={anchor['fi_p90'].min():.3f}  "
    f"median={anchor['fi_p90'].median():.3f}  "
    f"max={anchor['fi_p90'].max():.3f}")
log(f"Background (n={len(background):,}):")
log(f"  fi_p90: p10={background['fi_p90'].quantile(0.10):.3f}  "
    f"median={background['fi_p90'].median():.3f}  "
    f"p90={background['fi_p90'].quantile(0.90):.3f}")

# ---------------------------------------------------------------------------
# 6. DOY profile: mean fi_by_z by DOY bin, anchor vs background
# ---------------------------------------------------------------------------

log("\nBuilding DOY profiles...")

# Convert to pandas for profile aggregation (small after groupby)
doy_df = df.select(["point_id", "doy_bin", "fi_by_z"]).to_pandas()
doy_df["is_anchor"] = doy_df["point_id"].isin(set(anchor["point_id"]))

profile_anchor = (
    doy_df[doy_df["is_anchor"]]
    .groupby("doy_bin")["fi_by_z"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
profile_bg = (
    doy_df[~doy_df["is_anchor"]]
    .groupby("doy_bin")["fi_by_z"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(
    "FI_by z-score anomaly by DOY — anchor pixels vs background\n"
    "Kowanyama Mitchell River delta  |  FI_by = (B03+B04)/(B02+B08)",
    fontsize=10,
)

# Background with ±1 std band
ax.fill_between(
    profile_bg["doy_bin"],
    profile_bg["mean"] - profile_bg["std"],
    profile_bg["mean"] + profile_bg["std"],
    alpha=0.15, color="#3498db", label="_nolegend_",
)
ax.plot(profile_bg["doy_bin"], profile_bg["mean"],
        color="#3498db", linewidth=1.5, label="Background mean (±1 std)")

# Anchor with ±1 std band
ax.fill_between(
    profile_anchor["doy_bin"],
    profile_anchor["mean"] - profile_anchor["std"],
    profile_anchor["mean"] + profile_anchor["std"],
    alpha=0.25, color="#e74c3c", label="_nolegend_",
)
ax.plot(profile_anchor["doy_bin"], profile_anchor["mean"],
        color="#e74c3c", linewidth=2.0, label=f"Anchor mean (n={len(anchor)} px, ±1 std)")

ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Day of year (DOY bin start)", fontsize=9)
ax.set_ylabel("FI_by z-score anomaly", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_facecolor("#1a1a1a")
ax.legend(fontsize=8, framealpha=0.7, facecolor="#2a2a2a",
          labelcolor="white", edgecolor="none")

# Month labels on x axis
month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticks(month_doys)
ax.set_xticklabels(month_names, fontsize=7)

plt.tight_layout()
out_doy = OUT_DIR / "kowanyama_fi_doy_profile.png"
fig.savefig(out_doy, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_doy.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 7. Anchor pixel time series: raw FI_by + z-score for the nearest anchor pixel
# ---------------------------------------------------------------------------

log("\nBuilding anchor pixel time series...")

# Use the anchor pixel closest to the anchor coordinate
nearest_id = anchor.loc[anchor["dist_anchor_m"].idxmin(), "point_id"]
log(f"  Nearest anchor pixel: {nearest_id}  "
    f"(dist={anchor.loc[anchor['dist_anchor_m'].idxmin(), 'dist_anchor_m']:.1f} m)")

ts_df = (
    df.filter(pl.col("point_id") == nearest_id)
    .select(["date", "doy_bin", "fi_by", "fi_by_base", "fi_by_z"])
    .sort("date")
    .to_pandas()
)
ts_df["date"] = pd.to_datetime(ts_df["date"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(
    f"Anchor pixel {nearest_id} — FI_by time series\n"
    f"Kowanyama  |  lon={anchor.loc[anchor['dist_anchor_m'].idxmin(), 'lon']:.5f}  "
    f"lat={anchor.loc[anchor['dist_anchor_m'].idxmin(), 'lat']:.5f}",
    fontsize=10,
)

ax1.plot(ts_df["date"], ts_df["fi_by"], color="#2ecc71", linewidth=0.6,
         alpha=0.7, label="FI_by (observed)")
ax1.plot(ts_df["date"], ts_df["fi_by_base"], color="white", linewidth=1.0,
         linestyle="--", alpha=0.6, label="DOY-bin baseline")
ax1.set_ylabel("FI_by = (B03+B04)/(B02+B08)", fontsize=9)
ax1.tick_params(labelsize=8)
ax1.set_facecolor("#1a1a1a")
ax1.legend(fontsize=8, framealpha=0.7, facecolor="#2a2a2a",
           labelcolor="white", edgecolor="none")

ax2.bar(ts_df["date"], ts_df["fi_by_z"], width=5,
        color=np.where(ts_df["fi_by_z"] > 0, "#e74c3c", "#3498db"),
        alpha=0.8, label="fi_by_z")
ax2.axhline(0,   color="white",  linewidth=0.8, linestyle="--", alpha=0.5)
ax2.axhline(1.5, color="#e67e22", linewidth=0.8, linestyle=":", alpha=0.7,
            label="z = 1.5")
ax2.set_ylabel("fi_by z-score anomaly", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_facecolor("#1a1a1a")
ax2.legend(fontsize=8, framealpha=0.7, facecolor="#2a2a2a",
           labelcolor="white", edgecolor="none")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

plt.tight_layout()
out_ts = OUT_DIR / "kowanyama_fi_anchor_timeseries.png"
fig.savefig(out_ts, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_ts.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Spatial fi_p90 map on WMS
# ---------------------------------------------------------------------------

log("\nFetching WMS tile...")
margin  = 0.001
lon_min = fi_p90["lon"].min() - margin
lon_max = fi_p90["lon"].max() + margin
lat_min = fi_p90["lat"].min() - margin
lat_max = fi_p90["lat"].max() + margin

try:
    img = fetch_wms_image([lon_min, lat_min, lon_max, lat_max], width_px=4096)
    log(f"  {img.shape[1]} x {img.shape[0]} px")
except Exception as exc:
    log(f"  WARNING: WMS fetch failed ({exc})")
    img = None

DPI = 200
fig_w = img.shape[1] / DPI if img is not None else 20
fig_h = img.shape[0] / DPI if img is not None else 10

lat_centre    = (lat_min + lat_max) / 2
lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
lon_span_deg  = lon_max - lon_min
pt_per_deg    = fig_w * 72 / lon_span_deg
marker_pt     = (10 / lon_m_per_deg) * pt_per_deg
marker_s      = max(0.5, marker_pt ** 2 / 8)

# Symmetric colormap centred at 0
vmax = max(abs(fi_p90["fi_p90"].quantile(0.02)),
           abs(fi_p90["fi_p90"].quantile(0.98)))
norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.suptitle(
    "FI_by flowering index — fi_p90 (90th percentile z-score)\n"
    f"Kowanyama  |  {len(fi_p90):,} pixels  |  "
    "warm = anomalously high flowering flash",
    fontsize=10,
)

if img is not None:
    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="auto", zorder=0)
else:
    ax.set_facecolor("#1a1a1a")

sc = ax.scatter(
    fi_p90["lon"], fi_p90["lat"],
    c=fi_p90["fi_p90"], cmap=plt.cm.RdYlGn, norm=norm,
    s=marker_s, linewidths=0, alpha=0.6, zorder=2,
)
plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02,
             label="fi_p90 (FI_by z-score 90th pct)")

if len(anchor):
    ax.scatter(anchor["lon"], anchor["lat"],
               marker="D", s=marker_s * 4, color="#f1c40f",
               edgecolors="white", linewidths=0.6, zorder=5,
               label="Anchor: presence")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Latitude", fontsize=8)
ax.tick_params(labelsize=7)

plt.tight_layout()
out_map = OUT_DIR / "kowanyama_fi_p90_map.png"
fig.savefig(out_map, dpi=DPI, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_map.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------

log("\n=== Summary ===")
log(f"Pixels: {len(fi_p90):,}  |  Anchor: {len(anchor)}")
log(f"Anchor fi_p90:     median={anchor['fi_p90'].median():.3f}  "
    f"range [{anchor['fi_p90'].min():.3f}, {anchor['fi_p90'].max():.3f}]")
log(f"Background fi_p90: p25={background['fi_p90'].quantile(0.25):.3f}  "
    f"median={background['fi_p90'].median():.3f}  "
    f"p75={background['fi_p90'].quantile(0.75):.3f}")

anchor_pct_rank = (background["fi_p90"] < anchor["fi_p90"].median()).mean() * 100
log(f"Anchor median fi_p90 is at the {anchor_pct_rank:.0f}th percentile of background")

log("\nDone.")
