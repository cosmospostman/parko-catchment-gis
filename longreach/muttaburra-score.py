"""Score every pixel in the Muttaburra bbox with the Longreach-trained pipeline.

Reads data/muttaburra_pixels.parquet, extracts the three primary features
(nir_cv, rec_p, re_p10) using the same logic as kowanyama-score.py, then
applies a logistic regression trained on the Longreach infestation and
grassland end-member pixels to score every pixel.

Site: 2 km × 2 km centred on ALA centroid −22.538000, 144.558000
Bbox: 144.548274, −22.546983, 144.567726, −22.529017
ALA records in bbox: 17 (GPS-precision, 1 m uncertainty)

Dry season definition: Jun–Oct (months 6–10), same as Longreach — this is
semi-arid Queensland, not monsoonal.

Produces:
  outputs/muttaburra/
    muttaburra_features.parquet             — per-pixel feature table with prob_lr
    muttaburra_prob_wms.png                 — probability heatmap overlaid on WMS imagery
    muttaburra_prob_heatmap.png             — probability heatmap on dark background
    muttaburra_top_decile.png               — top-decile pixels on WMS imagery
    muttaburra_bottom_decile.png            — bottom-decile pixels on WMS imagery
    muttaburra_feature_distributions.png    — nir_cv / rec_p / re_p10 vs Longreach training
    muttaburra_classifier_space.png         — 2D nir_cv vs rec_p with decision boundary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "muttaburra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCL_PURITY_MIN = 0.5
DRY_MONTHS     = [6, 7, 8, 9, 10]   # semi-arid Qld, same as Longreach
MIN_OBS_DRY    = 5
MIN_OBS_ANNUAL = 10

FEATURES = ["nir_cv", "rec_p", "re_p10"]

# ALA GPS-precision anchor: centroid of the 17 in-bbox records
ANCHOR_PRESENCE = (-22.538000, 144.558000)   # (lat, lon)
ANCHOR_RADIUS_M = 20.0    # ~2 pixel widths — selects nearest pixel(s) to centroid


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Load and quality-filter
# ---------------------------------------------------------------------------

log("Loading Muttaburra parquet ...")
LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity", "B04", "B05", "B07", "B08"]
df = pl.read_parquet(
    PROJECT_ROOT / "data" / "muttaburra_pixels.parquet",
    columns=LOAD_COLS,
)
log(f"  {len(df):,} rows  |  {df['point_id'].n_unique():,} pixels")

before = len(df)
df = df.filter(pl.col("scl_purity") >= SCL_PURITY_MIN)
log(f"  Quality filter: dropped {before - len(df):,} rows, retained {len(df):,}")

df = (
    df
    .drop("scl_purity")
    .with_columns([
        pl.col("date").dt.year().cast(pl.Int16).alias("year"),
        pl.col("date").dt.month().cast(pl.Int8).alias("month"),
        ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
        (pl.col("B07") / pl.col("B05")).alias("re_ratio"),
    ])
    .drop(["date", "B04", "B05"])
)

coords = df.select(["point_id", "lon", "lat"]).unique("point_id")
df = df.drop(["lon", "lat"])

# ---------------------------------------------------------------------------
# 2. Feature extraction (Polars — parallelised groupby)
# ---------------------------------------------------------------------------

log("\nExtracting features ...")

# --- nir_cv ---

dry = df.filter(pl.col("month").is_in(DRY_MONTHS)).select(["point_id", "year", "B08"])
df  = df.drop("B08")

yr_nir = (
    dry
    .group_by(["point_id", "year"])
    .agg([
        pl.col("B08").median().alias("nir_yr"),
        pl.col("B08").count().alias("n_dry"),
    ])
    .filter(pl.col("n_dry") >= MIN_OBS_DRY)
)
del dry

nir_stats = (
    yr_nir
    .group_by("point_id")
    .agg([
        pl.col("nir_yr").mean().alias("nir_mean"),
        pl.col("nir_yr").std().alias("nir_std"),
        pl.col("nir_yr").count().alias("n_dry_years"),
    ])
    .with_columns(
        (pl.col("nir_std") / pl.col("nir_mean")).alias("nir_cv")
    )
)
del yr_nir
log(f"  nir_cv: {len(nir_stats):,} pixels with >= {MIN_OBS_DRY} dry obs/year")

# --- rec_p ---

ann = df.select(["point_id", "year", "ndvi"])
df  = df.drop("ndvi")

ndvi_pcts = (
    ann
    .group_by(["point_id", "year"])
    .agg([
        pl.col("ndvi").quantile(0.90).alias("p90"),
        pl.col("ndvi").quantile(0.10).alias("p10"),
        pl.col("ndvi").count().alias("n_ann"),
    ])
    .filter(pl.col("n_ann") >= MIN_OBS_ANNUAL)
    .with_columns(
        (pl.col("p90") - pl.col("p10")).alias("rec_p_yr")
    )
)
del ann

rec_stats = (
    ndvi_pcts
    .group_by("point_id")
    .agg([
        pl.col("rec_p_yr").mean().alias("rec_p"),
        pl.col("rec_p_yr").count().alias("n_amp_years"),
    ])
)
del ndvi_pcts
log(f"  rec_p:  {len(rec_stats):,} pixels with >= {MIN_OBS_ANNUAL} obs/year")

# --- re_p10 ---

re = df.select(["point_id", "year", "re_ratio"])
del df

re_p10_yr = (
    re
    .group_by(["point_id", "year"])
    .agg([
        pl.col("re_ratio").quantile(0.10).alias("re_p10_yr"),
        pl.col("re_ratio").count().alias("n_re"),
    ])
    .filter(pl.col("n_re") >= MIN_OBS_ANNUAL)
)
del re

re_stats = (
    re_p10_yr
    .group_by("point_id")
    .agg(pl.col("re_p10_yr").mean().alias("re_p10"))
)
del re_p10_yr
log(f"  re_p10: {len(re_stats):,} pixels")

# --- Merge and convert to pandas ---

feat_pl = (
    coords
    .join(nir_stats.select(["point_id", "nir_cv", "nir_mean", "n_dry_years"]), on="point_id", how="inner")
    .join(rec_stats.select(["point_id", "rec_p", "n_amp_years"]),              on="point_id", how="inner")
    .join(re_stats.select(["point_id", "re_p10"]),                             on="point_id", how="inner")
)
feat = feat_pl.to_pandas()
log(f"\n  Feature table: {len(feat):,} pixels with all three features")
log(f"  n_dry_years — median: {feat['n_dry_years'].median():.1f}  "
    f"min: {feat['n_dry_years'].min()}  max: {feat['n_dry_years'].max()}")
log(f"  n_amp_years — median: {feat['n_amp_years'].median():.1f}  "
    f"min: {feat['n_amp_years'].min()}  max: {feat['n_amp_years'].max()}")

# ---------------------------------------------------------------------------
# 3. Train logistic regression on Longreach end-member pixels
# ---------------------------------------------------------------------------

log("\nBuilding Longreach training set ...")

orig_amp = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp" / "longreach_amp_stats.parquet"
)
orig_nir = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
)
orig_re  = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-red-edge" / "longreach_re_stats.parquet"
)

train_df = orig_amp[["point_id", "rec_p", "in_hd_bbox", "is_riparian", "is_grassland"]].copy()
train_df = train_df.merge(orig_nir[["point_id", "nir_cv"]], on="point_id")
train_df = train_df.merge(orig_re[["point_id",  "re_p10"]], on="point_id")

train = train_df[train_df["in_hd_bbox"] | train_df["is_grassland"]].copy()
y     = train["in_hd_bbox"].astype(int)
log(f"  Training pixels — Infestation: {y.sum()}  |  Grassland: {(y == 0).sum()}")

scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES].values)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y)
log(f"  Training accuracy (Longreach end-members): {lr.score(X_train, y):.3f}")

X_mut = scaler.transform(feat[FEATURES].values)
feat["prob_lr"] = lr.predict_proba(X_mut)[:, 1]

log(f"\nMuttaburra probability distribution:")
log(f"  median: {feat['prob_lr'].median():.3f}  "
    f"p10: {feat['prob_lr'].quantile(0.10):.3f}  "
    f"p90: {feat['prob_lr'].quantile(0.90):.3f}  "
    f"p99: {feat['prob_lr'].quantile(0.99):.3f}")

# ---------------------------------------------------------------------------
# 4. Save feature table
# ---------------------------------------------------------------------------

out_parquet = OUT_DIR / "muttaburra_features.parquet"
feat.to_parquet(out_parquet, index=False)
log(f"\nSaved feature table: {out_parquet.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 5. Anchor pixel report
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lats, lons):
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlam = np.radians(lons - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

feat["dist_anchor_m"] = haversine_m(
    ANCHOR_PRESENCE[0], ANCHOR_PRESENCE[1],
    feat["lat"].values, feat["lon"].values,
)
anchor_px = feat[feat["dist_anchor_m"] <= ANCHOR_RADIUS_M].copy()

log(f"\nAnchor pixels (presence, within {ANCHOR_RADIUS_M:.0f} m of {ANCHOR_PRESENCE}):")
log(f"  {len(anchor_px)} pixels found")
if len(anchor_px):
    for _, row in anchor_px.iterrows():
        log(
            f"  {row['point_id']}  lon={row['lon']:.6f}  lat={row['lat']:.6f}  "
            f"dist={row['dist_anchor_m']:.1f} m  "
            f"nir_cv={row['nir_cv']:.4f}  rec_p={row['rec_p']:.4f}  "
            f"re_p10={row['re_p10']:.4f}  prob_lr={row['prob_lr']:.3f}"
        )

# ---------------------------------------------------------------------------
# 6. Fetch WMS imagery
# ---------------------------------------------------------------------------

margin  = 0.001
lon_min = feat["lon"].min() - margin
lon_max = feat["lon"].max() + margin
lat_min = feat["lat"].min() - margin
lat_max = feat["lat"].max() + margin
bbox    = [lon_min, lat_min, lon_max, lat_max]

log(f"\nFetching WMS tile for bbox {[round(x, 5) for x in bbox]} ...")
try:
    img = fetch_wms_image(bbox, width_px=4096)
    log(f"  WMS tile: {img.shape[1]} x {img.shape[0]} px")
except Exception as exc:
    log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without imagery")
    img = None

# ---------------------------------------------------------------------------
# 7. Probability heatmaps
# ---------------------------------------------------------------------------

log("\nGenerating probability maps ...")

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=1)

lon_span_deg  = lon_max - lon_min
lat_span_deg  = lat_max - lat_min
lat_centre    = (lat_min + lat_max) / 2
lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))

DPI = 200
if img is not None:
    fig_w = img.shape[1] / DPI
    fig_h = img.shape[0] / DPI
else:
    fig_w = 14
    fig_h = max(4.0, fig_w * (lat_span_deg * 111320) / (lon_span_deg * lon_m_per_deg))

pt_per_deg = fig_w * 72 / lon_span_deg
marker_pt  = (10 / lon_m_per_deg) * pt_per_deg
marker_s   = max(0.5, marker_pt ** 2 / 8)

SUPTITLE = (
    "Parkinsonia probability — Muttaburra (Site 4)\n"
    f"{len(feat):,} pixels  |  Longreach-trained LR on (nir_cv, rec_p, re_p10)"
)

for title_suffix, show_img, alpha, out_name in [
    ("probability overlay on WMS imagery", True,  0.55, "muttaburra_prob_wms.png"),
    ("probability score only",             False, 1.0,  "muttaburra_prob_heatmap.png"),
]:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(SUPTITLE, fontsize=12)

    if show_img and img is not None:
        ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#1a1a1a")

    sc = ax.scatter(
        feat["lon"], feat["lat"],
        c=feat["prob_lr"], cmap=cmap, norm=norm,
        s=marker_s, linewidths=0, alpha=alpha, zorder=2,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Parkinsonia probability", fontsize=8)

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title_suffix, fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Top / bottom decile maps
# ---------------------------------------------------------------------------

log("Generating decile maps ...")

p90        = feat["prob_lr"].quantile(0.90)
p10        = feat["prob_lr"].quantile(0.10)
top_decile = feat[feat["prob_lr"] >= p90]
bot_decile = feat[feat["prob_lr"] <= p10]

for subset, colour, title, out_name in [
    (top_decile, "#e74c3c",
     f"Top decile — prob >= {p90:.2f}  (n={len(top_decile):,})\nMuttaburra (Longreach-trained model)",
     "muttaburra_top_decile.png"),
    (bot_decile, "#3498db",
     f"Bottom decile — prob <= {p10:.2f}  (n={len(bot_decile):,})\nMuttaburra (Longreach-trained model)",
     "muttaburra_bottom_decile.png"),
]:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=11)

    if img is not None:
        ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#1a1a1a")

    ax.scatter(subset["lon"], subset["lat"],
               c=colour, s=marker_s * 1.2, linewidths=0, alpha=0.7, zorder=2)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 9. Feature distribution comparison vs Longreach training populations
# ---------------------------------------------------------------------------

log("Generating feature distribution comparison ...")

lr_inf   = train_df[train_df["in_hd_bbox"]]
lr_grass = train_df[train_df["is_grassland"]]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Feature distributions: Muttaburra (all pixels) vs Longreach training populations",
    fontsize=11,
)
for ax, feat_col, xlabel in [
    (axes[0], "nir_cv", "nir_cv (dry-season NIR CV)"),
    (axes[1], "rec_p",  "rec_p (annual NDVI amplitude)"),
    (axes[2], "re_p10", "re_p10 (annual B07/B05 p10)"),
]:
    bins = np.linspace(
        min(feat[feat_col].quantile(0.01), lr_inf[feat_col].quantile(0.01)),
        max(feat[feat_col].quantile(0.99), lr_grass[feat_col].quantile(0.99)),
        40,
    )
    ax.hist(lr_inf[feat_col],   bins=bins, alpha=0.5, color="#e74c3c", label="LR infestation", density=True)
    ax.hist(lr_grass[feat_col], bins=bins, alpha=0.5, color="#3498db", label="LR grassland",   density=True)
    ax.hist(feat[feat_col],     bins=bins, alpha=0.4, color="#2ecc71", label="Muttaburra all",  density=True)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8); ax.legend(fontsize=7)

plt.tight_layout()
out_dist = OUT_DIR / "muttaburra_feature_distributions.png"
fig.savefig(out_dist, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_dist.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 10. 2D classifier space: nir_cv vs rec_p
# ---------------------------------------------------------------------------

log("Generating 2D classifier space plot ...")

w   = lr.coef_[0]
b   = lr.intercept_[0]
mu  = scaler.mean_
sig = scaler.scale_

nir_cv_grid = np.linspace(
    min(feat["nir_cv"].quantile(0.005), lr_inf["nir_cv"].quantile(0.005)),
    max(feat["nir_cv"].quantile(0.995), lr_grass["nir_cv"].quantile(0.995)),
    300,
)
boundary_rec_p = mu[1] - (sig[1] / w[1]) * (w[0] * (nir_cv_grid - mu[0]) / sig[0] + b)

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle(
    "2D classifier space: nir_cv vs rec_p\nMuttaburra pixels coloured by Parkinsonia probability",
    fontsize=11,
)

sc = ax.scatter(
    feat["nir_cv"], feat["rec_p"],
    c=feat["prob_lr"], cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(vmin=0, vmax=1),
    s=4, linewidths=0, alpha=0.5, zorder=2,
)
plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Parkinsonia probability")

ax.scatter(lr_inf["nir_cv"].mean(), lr_inf["rec_p"].mean(),
           marker="*", s=220, color="#e74c3c", zorder=5,
           label=f"LR infestation centroid (n={len(lr_inf)})")
ax.scatter(lr_grass["nir_cv"].mean(), lr_grass["rec_p"].mean(),
           marker="*", s=220, color="#3498db", zorder=5,
           label=f"LR grassland centroid (n={len(lr_grass)})")

for df_pop, colour in [(lr_inf, "#e74c3c"), (lr_grass, "#3498db")]:
    x0, x1 = df_pop["nir_cv"].quantile([0.25, 0.75])
    y0, y1 = df_pop["rec_p"].quantile([0.25, 0.75])
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                 fill=False, edgecolor=colour, linewidth=1.2, linestyle="--", zorder=4))

ax.plot(nir_cv_grid, boundary_rec_p,
        color="white", linewidth=1.5, linestyle="-", zorder=6,
        label="Decision boundary (prob = 0.5, re_p10 @ LR mean)")

if False:  # anchor diamonds disabled
    pass

ax.set_xlabel("nir_cv  (dry-season inter-annual NIR CV)", fontsize=9)
ax.set_ylabel("rec_p  (annual NDVI amplitude)", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_facecolor("#1a1a1a")
ax.legend(fontsize=8, loc="upper right", framealpha=0.7,
          facecolor="#2a2a2a", labelcolor="white", edgecolor="none")

plt.tight_layout()
out_scatter = OUT_DIR / "muttaburra_classifier_space.png"
fig.savefig(out_scatter, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_scatter.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------

log("\n=== Summary ===")
log(f"Total pixels scored:  {len(feat):,}")
log(f"prob_lr — median: {feat['prob_lr'].median():.3f}  "
    f"p90: {feat['prob_lr'].quantile(0.90):.3f}  "
    f"p99: {feat['prob_lr'].quantile(0.99):.3f}")
log(f"Top-decile threshold (prob >= {p90:.3f}): {len(top_decile):,} pixels")
log(f"Bottom-decile threshold (prob <= {p10:.3f}): {len(bot_decile):,} pixels")
log(f"Anchor pixels (within {ANCHOR_RADIUS_M:.0f} m): {len(anchor_px)}")
if len(anchor_px):
    log(f"  Anchor prob_lr — mean: {anchor_px['prob_lr'].mean():.3f}  "
        f"min: {anchor_px['prob_lr'].min():.3f}  max: {anchor_px['prob_lr'].max():.3f}")
log("\nDone.")
