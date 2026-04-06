"""Score every pixel in the expanded Longreach bbox with the Stage 2 pipeline.

Reads longreach_expansion_pixels.parquet (40,996 pixels covering the full
~2.7 km × 1.5 km bbox), extracts the three primary features (nir_cv, rec_p,
re_p10) using the same logic as the Stage 1/2 analysis scripts, then applies
a logistic regression trained on the known infestation and grassland end-member
pixels from longreach_pixels.parquet to score every pixel.

Produces:
  outputs/longreach-expansion-map/
    longreach_expansion_features.parquet   — per-pixel feature table
    longreach_expansion_prob_map.png       — probability heatmap vs WMS imagery
    longreach_expansion_top_decile.png     — top-decile pixels on WMS imagery
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

from utils.location import get as _get_loc

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-expansion-map"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config — identical thresholds to the Stage 1/2 scripts
# ---------------------------------------------------------------------------

SCL_PURITY_MIN   = 0.5
DRY_MONTHS       = {6, 7, 8, 9, 10}
MIN_OBS_DRY      = 5    # (pixel, year) min obs for nir_cv
MIN_OBS_ANNUAL   = 10   # (pixel, year) min obs for rec_p and re_p10

# Original infestation patch bbox (used to assign training labels)
HD_LON_MIN, HD_LON_MAX = 145.423948, 145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033, -22.761054

# Southern extension bbox (grassland training population in Stage 1)
# Pixels south of the infestation patch and not flagged as riparian proxy
EXT_LAT_MAX = HD_LAT_MIN   # everything south of the infestation patch

FEATURES = ["nir_cv", "rec_p", "re_p10"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Load and quality-filter the expansion parquet
# ---------------------------------------------------------------------------

log("Loading expansion parquet...")
exp = pd.read_parquet(PROJECT_ROOT / "data" / "pixels" / "longreach-expansion" / "longreach-expansion.parquet")
log(f"  {len(exp):,} rows  |  {exp['point_id'].nunique():,} pixels")

before = len(exp)
exp = exp[exp["scl_purity"] >= SCL_PURITY_MIN].copy()
log(f"  Quality filter: dropped {before - len(exp):,} rows, retained {len(exp):,}")

exp["year"]  = exp["date"].dt.year
exp["month"] = exp["date"].dt.month
exp["ndvi"]  = (exp["B08"] - exp["B04"]) / (exp["B08"] + exp["B04"])
exp["re_ratio"] = exp["B07"] / exp["B05"]

# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

log("\nExtracting features...")

# --- nir_cv: dry-season inter-annual NIR coefficient of variation ---

dry = exp[exp["month"].isin(DRY_MONTHS)].copy()
dry_counts = dry.groupby(["point_id", "year"])["B08"].count()
valid_dry = dry_counts[dry_counts >= MIN_OBS_DRY].index
dry_valid = dry.set_index(["point_id", "year"]).loc[valid_dry].reset_index()

yr_nir = (
    dry_valid.groupby(["point_id", "year"])["B08"]
    .median()
    .rename("nir_yr")
    .reset_index()
)
nir_stats = (
    yr_nir.groupby("point_id")["nir_yr"]
    .agg(nir_mean="mean", nir_std="std", n_dry_years="count")
    .reset_index()
)
nir_stats["nir_cv"] = nir_stats["nir_std"] / nir_stats["nir_mean"]
log(f"  nir_cv: {len(nir_stats):,} pixels with ≥ {MIN_OBS_DRY} dry obs/year")

# --- rec_p: annual NDVI amplitude (p90 − p10), mean across years ---

ann_counts = exp.groupby(["point_id", "year"])["ndvi"].count()
valid_ann = ann_counts[ann_counts >= MIN_OBS_ANNUAL].index
ann_valid = exp.set_index(["point_id", "year"]).loc[valid_ann].reset_index()

ndvi_p90 = (
    ann_valid.groupby(["point_id", "year"])["ndvi"]
    .quantile(0.90).rename("p90").reset_index()
)
ndvi_p10 = (
    ann_valid.groupby(["point_id", "year"])["ndvi"]
    .quantile(0.10).rename("p10").reset_index()
)
amp = ndvi_p90.merge(ndvi_p10, on=["point_id", "year"])
amp["rec_p_yr"] = amp["p90"] - amp["p10"]

rec_stats = (
    amp.groupby("point_id")["rec_p_yr"]
    .agg(rec_p="mean", n_amp_years="count")
    .reset_index()
)
log(f"  rec_p:  {len(rec_stats):,} pixels with ≥ {MIN_OBS_ANNUAL} obs/year")

# --- re_p10: annual B07/B05 10th-percentile, mean across years ---

re_counts = exp.groupby(["point_id", "year"])["re_ratio"].count()
valid_re = re_counts[re_counts >= MIN_OBS_ANNUAL].index
re_valid = exp.set_index(["point_id", "year"]).loc[valid_re].reset_index()

re_p10_yr = (
    re_valid.groupby(["point_id", "year"])["re_ratio"]
    .quantile(0.10).rename("re_p10_yr").reset_index()
)
re_stats = (
    re_p10_yr.groupby("point_id")["re_p10_yr"]
    .mean().rename("re_p10").reset_index()
)
log(f"  re_p10: {len(re_stats):,} pixels")

# --- Merge all features with pixel coordinates ---

coords = exp[["point_id", "lon", "lat"]].drop_duplicates("point_id")

feat = (
    coords
    .merge(nir_stats[["point_id", "nir_cv", "nir_mean"]], on="point_id", how="inner")
    .merge(rec_stats[["point_id", "rec_p"]],               on="point_id", how="inner")
    .merge(re_stats[["point_id",  "re_p10"]],              on="point_id", how="inner")
)
log(f"\n  Feature table: {len(feat):,} pixels with all three features")

# Flag infestation bbox pixels
feat["in_hd_bbox"] = (
    feat["lon"].between(HD_LON_MIN, HD_LON_MAX) &
    feat["lat"].between(HD_LAT_MIN, HD_LAT_MAX)
)
log(f"  Infestation bbox pixels in expansion: {feat['in_hd_bbox'].sum()}")

# ---------------------------------------------------------------------------
# 3. Train logistic regression on the original Stage 1/2 end-member pixels
#
# The training set is the 748-pixel longreach_pixels.parquet population:
#   - infestation = in_hd_bbox pixels (362)
#   - grassland   = the southern extension non-riparian pixels (347)
# Using the expansion's own pixel features for training would grossly imbalance
# the class ratio (~362 vs ~36,000) and collapse the classifier.
# ---------------------------------------------------------------------------

log("\nBuilding training set from original end-member pixels...")

# Load the pre-computed feature stats for the original 748-pixel population.
# These parquets are written by dry-season-nir.py, wet-dry-amp.py, red-edge.py.
orig_nir = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
)
orig_amp = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp" / "longreach_amp_stats.parquet"
)
orig_re = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-red-edge" / "longreach_re_stats.parquet"
)

train_df = orig_amp[["point_id", "rec_p", "is_riparian", "is_grassland"]].copy()
train_df = train_df.merge(orig_nir[["point_id", "nir_cv"]], on="point_id")
train_df = train_df.merge(orig_re[["point_id",  "re_p10"]], on="point_id")
train_df["in_hd_bbox"] = ~train_df["is_grassland"] & ~train_df["is_riparian"]

# Annotate expansion feature table with class flags derived from expansion data
# (riparian proxy = top-10% nir_mean of non-infestation pixels in expansion)
ext_pixels = feat[~feat["in_hd_bbox"]]
rip_thresh = ext_pixels["nir_mean"].quantile(0.90)
feat["is_riparian"]  = (~feat["in_hd_bbox"]) & (feat["nir_mean"] >= rip_thresh)
feat["is_grassland"] = (~feat["in_hd_bbox"]) & (~feat["is_riparian"])

n_inf   = feat["in_hd_bbox"].sum()
n_grass = feat["is_grassland"].sum()
n_rip   = feat["is_riparian"].sum()
log(f"  Expansion class flags — Infestation: {n_inf}  |  Grassland: {n_grass}  |  Riparian proxy: {n_rip}")

# Train only on infestation vs grassland end-members (exclude riparian proxy, as in Stage 2)
train = train_df[train_df["in_hd_bbox"] | train_df["is_grassland"]].copy()
y = train["in_hd_bbox"].astype(int)
log(f"  Training pixels — Infestation: {y.sum()}  |  Grassland: {(y == 0).sum()}")

scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES].values)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y)

X_all = scaler.transform(feat[FEATURES].values)
feat["prob_lr"] = lr.predict_proba(X_all)[:, 1]

log(f"\n  Training accuracy: {lr.score(X_train, y):.3f}")
log(f"  Infestation median prob: {feat.loc[feat['in_hd_bbox'],   'prob_lr'].median():.3f}")
log(f"  Grassland   median prob: {feat.loc[feat['is_grassland'],  'prob_lr'].median():.3f}")
log(f"  Riparian    median prob: {feat.loc[feat['is_riparian'],   'prob_lr'].median():.3f}")

# ---------------------------------------------------------------------------
# 4. Save feature table
# ---------------------------------------------------------------------------

out_parquet = OUT_DIR / "longreach_expansion_features.parquet"
feat.to_parquet(out_parquet, index=False)
log(f"\nSaved feature table: {out_parquet.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 5. Fetch WMS imagery for the full expansion bbox
# ---------------------------------------------------------------------------

margin = 0.0002
lon_min = feat["lon"].min() - margin
lon_max = feat["lon"].max() + margin
lat_min = feat["lat"].min() - margin
lat_max = feat["lat"].max() + margin
bbox = [lon_min, lat_min, lon_max, lat_max]

log(f"\nFetching WMS tile for bbox {[round(x,5) for x in bbox]} ...")
try:
    img = fetch_wms_image(bbox, width_px=4096)
    log(f"  WMS tile: {img.shape[1]} × {img.shape[0]} px")
except Exception as exc:
    log(f"  WARNING: WMS fetch failed ({exc}) — maps will render without imagery")
    img = None

# ---------------------------------------------------------------------------
# 6. Probability heatmap + WMS imagery
# ---------------------------------------------------------------------------

log("\nGenerating probability map...")

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=1)

# Approximate marker size: one 10m S2 pixel in points²
lat_centre = (lat_min + lat_max) / 2
lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
lon_span_deg = lon_max - lon_min
lat_span_deg = lat_max - lat_min

DPI = 200
# Size each panel to match the WMS tile resolution: panel width = WMS width / DPI inches
if img is not None:
    fig_w = img.shape[1] / DPI        # one panel width in inches
    fig_h = img.shape[0] / DPI        # panel height in inches
else:
    fig_w = 20
    fig_h = max(4.0, fig_w * (lat_span_deg * 111320) / (lon_span_deg * lon_m_per_deg))

fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h))
fig.suptitle(
    "Parkinsonia probability — full Longreach expansion bbox\n"
    f"{len(feat):,} pixels  |  logistic regression on (nir_cv, rec_p, re_p10)",
    fontsize=12,
)

pt_per_deg = fig_w * 72 / lon_span_deg
marker_pt = (10 / lon_m_per_deg) * pt_per_deg
marker_s = max(0.5, marker_pt ** 2 / 8)

for ax, title_suffix, show_img, alpha in [
    (axes[0], "probability overlay on WMS imagery", True,  0.55),
    (axes[1], "probability score only",             False, 1.0),
]:
    if show_img and img is not None:
        ax.imshow(
            img, extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper", aspect="auto", zorder=0,
        )
    else:
        ax.set_facecolor("#1a1a1a")

    sc = ax.scatter(
        feat["lon"], feat["lat"],
        c=feat["prob_lr"], cmap=cmap, norm=norm,
        s=marker_s, linewidths=0,
        alpha=alpha,
        zorder=2,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Parkinsonia probability", fontsize=8)

    # Mark the infestation training bbox
    ax.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN, HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
        label="Training: infestation patch", zorder=4,
    ))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
              facecolor="black", labelcolor="white", edgecolor="none")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title_suffix, fontsize=9)

plt.tight_layout()
out_map = OUT_DIR / "longreach_expansion_prob_map.png"
fig.savefig(out_map, dpi=DPI, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_map.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 7. Top-decile pixels on WMS imagery
# ---------------------------------------------------------------------------

log("Generating top-decile map...")

p90 = feat["prob_lr"].quantile(0.90)
p10 = feat["prob_lr"].quantile(0.10)
top_decile = feat[feat["prob_lr"] >= p90]
bot_decile  = feat[feat["prob_lr"] <= p10]

fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h))
fig.suptitle(
    "Top vs bottom decile Parkinsonia probability\nLongreach expansion bbox",
    fontsize=12,
)

for ax, subset, colour, title in [
    (axes[0], top_decile, "#e74c3c",
     f"Top decile (prob ≥ {p90:.2f}, n={len(top_decile):,})"),
    (axes[1], bot_decile, "#3498db",
     f"Bottom decile (prob ≤ {p10:.2f}, n={len(bot_decile):,})"),
]:
    if img is not None:
        ax.imshow(
            img, extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper", aspect="auto", zorder=0,
        )
    else:
        ax.set_facecolor("#1a1a1a")

    ax.scatter(
        subset["lon"], subset["lat"],
        c=colour, s=marker_s * 1.2, linewidths=0, alpha=0.7, zorder=2,
    )
    ax.add_patch(mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN, HD_LAT_MAX - HD_LAT_MIN,
        fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
        label="Training: infestation patch", zorder=4,
    ))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
              facecolor="black", labelcolor="white", edgecolor="none")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)

plt.tight_layout()
out_decile = OUT_DIR / "longreach_expansion_top_decile.png"
fig.savefig(out_decile, dpi=DPI, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_decile.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 8. Summary stats
# ---------------------------------------------------------------------------

log("\n=== Summary ===")
log(f"Total pixels scored: {len(feat):,}")
log(f"Expansion bbox: lon {lon_min:.5f}–{lon_max:.5f}, lat {lat_min:.5f}–{lat_max:.5f}")
log(f"Top-decile threshold (prob ≥ {p90:.3f}): {len(top_decile):,} pixels")
log(f"  Of which in training infestation bbox: {top_decile['in_hd_bbox'].sum()}")
log(f"  Of which outside training bbox: {(~top_decile['in_hd_bbox']).sum()}")

prob_outside = feat.loc[~feat["in_hd_bbox"], "prob_lr"]
log(f"\nPixels outside training bbox (n={len(prob_outside):,}):")
log(f"  prob_lr — median: {prob_outside.median():.3f}  "
    f"p90: {prob_outside.quantile(0.90):.3f}  "
    f"p99: {prob_outside.quantile(0.99):.3f}")

log("\nDone.")
