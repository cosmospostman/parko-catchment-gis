"""Sanity check: rank pixels by Parkinsonia probability and compare against
Queensland Globe 20cm satellite imagery.

Computes a simple linear probability score from the three Stage 1 features
(nir_cv, rec_p, re_p10), ranks all 748 pixels, and produces a side-by-side
figure: WMS imagery on the left, score heatmap on the right, with the pixel
grid overlaid on both. Decile bands are marked so individual pixel scores can
be visually verified against crown structure in the imagery.

See research/LONGREACH-STAGE2.md — sanity check task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "utils" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-feature-space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and merge features
# ---------------------------------------------------------------------------

amp_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp" / "longreach_amp_stats.parquet"
)
nir_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
)
re_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-red-edge" / "longreach_re_stats.parquet"
)

df = amp_df[["point_id", "lon", "lat", "rec_p", "is_riparian", "is_grassland"]].copy()
df = df.merge(nir_df[["point_id", "nir_cv"]], on="point_id")
df = df.merge(re_df[["point_id", "re_p10"]], on="point_id")

# Parse grid row/col from point_id (format: px_CCCC_RRRR)
df[["col", "row"]] = df.point_id.str.extract(r"px_(\d+)_(\d+)").astype(int)

def assign_class(row):
    if row.is_riparian:
        return "Riparian"
    if row.is_grassland:
        return "Grassland"
    return "Infestation"

df["class"] = df.apply(assign_class, axis=1)

# ---------------------------------------------------------------------------
# Compute probability score
# Two versions:
#   1. Logistic regression trained on infestation vs grassland end-members
#   2. Simple rank-based score (percentile of nir_cv inverted + rec_p + re_p10)
#      — no model, purely descriptive
# ---------------------------------------------------------------------------

FEATURES = ["nir_cv", "rec_p", "re_p10"]

# Logistic regression on end-member pixels only (exclude riparian proxy)
train = df[df["class"].isin(["Infestation", "Grassland"])].copy()
y = (train["class"] == "Infestation").astype(int)
X_train = train[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

X_all_scaled = scaler.transform(df[FEATURES].values)
df["prob_lr"] = lr.predict_proba(X_all_scaled)[:, 1]

# Raw rank score (no model): invert nir_cv (lower = more Parkinsonia), add rec_p and re_p10
df["nir_cv_inv"] = 1 - (df.nir_cv - df.nir_cv.min()) / (df.nir_cv.max() - df.nir_cv.min())
df["rec_p_norm"] = (df.rec_p - df.rec_p.min()) / (df.rec_p.max() - df.rec_p.min())
df["re_p10_norm"] = (df.re_p10 - df.re_p10.min()) / (df.re_p10.max() - df.re_p10.min())
df["score_raw"] = (df.nir_cv_inv + df.rec_p_norm + df.re_p10_norm) / 3.0

df["rank"] = rankdata(-df.prob_lr, method="min")  # rank 1 = highest probability

print(f"Pixels: {len(df)}")
print("\nTop 20 by Parkinsonia probability:")
top20 = df.nsmallest(20, "rank")[["point_id", "col", "row", "lon", "lat", "class", "prob_lr", "rank"]]
print(top20.to_string(index=False))

print("\nBottom 20 by Parkinsonia probability:")
bot20 = df.nlargest(20, "rank")[["point_id", "col", "row", "lon", "lat", "class", "prob_lr", "rank"]]
print(bot20.to_string(index=False))

print("\nClass mean probability:")
print(df.groupby("class")["prob_lr"].agg(["mean", "median", "std"]).round(3))

# ---------------------------------------------------------------------------
# Fetch WMS imagery for the full pixel extent
# ---------------------------------------------------------------------------

# Add a small margin (half an S2 pixel = ~5m) around the pixel centroids
MARGIN_DEG = 0.00005   # ~5 m

lon_min = df.lon.min() - MARGIN_DEG
lon_max = df.lon.max() + MARGIN_DEG
lat_min = df.lat.min() - MARGIN_DEG
lat_max = df.lat.max() + MARGIN_DEG

bbox = [lon_min, lat_min, lon_max, lat_max]
print(f"\nFetching WMS tile for bbox {bbox} ...")
img = fetch_wms_image(bbox, width_px=512)
print(f"WMS tile: {img.shape[1]} × {img.shape[0]} px")

# ---------------------------------------------------------------------------
# Map pixel (lon, lat) → image pixel coordinates
# ---------------------------------------------------------------------------

img_h, img_w = img.shape[:2]

def lonlat_to_imgxy(lon, lat):
    """Convert WGS84 lon/lat to image pixel coordinates (x right, y down)."""
    x = (lon - lon_min) / (lon_max - lon_min) * img_w
    y = (lat_max - lat) / (lat_max - lat_min) * img_h
    return x, y

df["img_x"], df["img_y"] = zip(*df.apply(
    lambda r: lonlat_to_imgxy(r.lon, r.lat), axis=1
))

# Approximate S2 pixel half-width in image pixels (~10 m)
lon_span_m = (lon_max - lon_min) * 111320 * np.cos(np.radians(df.lat.mean()))
px_per_m = img_w / lon_span_m
s2_half_px = 5.0 * px_per_m   # 5 m half-width → 10 m pixel

# ---------------------------------------------------------------------------
# Figure 1: WMS imagery + probability heatmap, side by side
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 18))
fig.suptitle(
    "Parkinsonia probability vs Queensland Globe 20cm imagery\n"
    "Longreach infestation + southern extension (748 S2 pixels, 2020–2025)",
    fontsize=12
)

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=1)

# Left panel: WMS imagery with probability-coloured squares overlaid
ax_img = axes[0]
ax_img.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="auto")
ax_img.set_title("Queensland Globe 20cm + probability overlay", fontsize=10)

for _, row in df.iterrows():
    half = 0.000045   # ~5 m in degrees lon (approximate S2 half-pixel)
    half_lat = 0.000045
    rect = mpatches.Rectangle(
        (row.lon - half, row.lat - half_lat),
        width=half * 2, height=half_lat * 2,
        linewidth=0, facecolor=cmap(norm(row.prob_lr)), alpha=0.55
    )
    ax_img.add_patch(rect)

ax_img.set_xlabel("Longitude", fontsize=9)
ax_img.set_ylabel("Latitude", fontsize=9)
ax_img.tick_params(labelsize=8)

# Right panel: probability heatmap only (no imagery) — easier to read the score gradient
ax_score = axes[1]
ax_score.set_facecolor("#222222")
ax_score.set_title("Parkinsonia probability score (logistic regression)", fontsize=10)

sc = ax_score.scatter(
    df.lon, df.lat,
    c=df.prob_lr, cmap=cmap, norm=norm,
    s=60, marker="s", linewidths=0
)

# Annotate decile boundaries
decile_thresholds = np.percentile(df.prob_lr, [10, 20, 30, 40, 50, 60, 70, 80, 90])
cb = plt.colorbar(sc, ax=ax_score, fraction=0.03, pad=0.04)
cb.set_label("Probability (Parkinsonia)", fontsize=9)
for thresh in decile_thresholds:
    cb.ax.axhline(thresh, color="white", linewidth=0.8, linestyle="--")

ax_score.set_xlabel("Longitude", fontsize=9)
ax_score.set_ylabel("Latitude", fontsize=9)
ax_score.tick_params(labelsize=8)

# Mark infestation / extension boundary (lat of the first infestation pixel)
inf_lat_min = df.loc[df["class"] == "Infestation", "lat"].min()
for ax in axes:
    ax.axhline(inf_lat_min, color="white", linewidth=1.0, linestyle=":",
               alpha=0.8, label="Infestation boundary")

plt.tight_layout()
out_path = OUT_DIR / "longreach_prob_vs_imagery.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Figure 2: top / bottom decile pixels highlighted on imagery
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 18))
fig.suptitle(
    "Top decile (highest Parkinsonia probability) vs bottom decile\n"
    "overlaid on Queensland Globe 20cm imagery",
    fontsize=12
)

p10 = df.prob_lr.quantile(0.10)
p90 = df.prob_lr.quantile(0.90)

top_decile = df[df.prob_lr >= p90]
bot_decile  = df[df.prob_lr <= p10]

for ax, subset, colour, title in [
    (axes[0], top_decile, "#e74c3c",
     f"Top decile (prob ≥ {p90:.2f}, n={len(top_decile)})"),
    (axes[1], bot_decile, "#3498db",
     f"Bottom decile (prob ≤ {p10:.2f}, n={len(bot_decile)})"),
]:
    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="auto")
    half = 0.000045
    for _, row in subset.iterrows():
        rect = mpatches.Rectangle(
            (row.lon - half, row.lat - half),
            width=half * 2, height=half * 2,
            linewidth=0.5, edgecolor="white",
            facecolor=colour, alpha=0.65
        )
        ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.axhline(inf_lat_min, color="white", linewidth=1.0, linestyle=":",
               alpha=0.8)

plt.tight_layout()
out_path2 = OUT_DIR / "longreach_prob_deciles_imagery.png"
fig.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path2}")

# ---------------------------------------------------------------------------
# Print ranked summary table
# ---------------------------------------------------------------------------

print("\n=== Full pixel ranking (sample: every 50th) ===")
df_sorted = df.sort_values("rank")
cols = ["rank", "point_id", "col", "row", "class", "prob_lr", "score_raw", "nir_cv", "rec_p", "re_p10"]
print(df_sorted[cols].iloc[::50].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Save full ranked table
ranked_path = OUT_DIR / "longreach_pixel_ranking.csv"
df_sorted[cols].to_csv(ranked_path, index=False, float_format="%.4f")
print(f"\nFull ranking saved to: {ranked_path}")
