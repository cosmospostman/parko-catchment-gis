"""3D feature space evaluation for Longreach Parkinsonia signals.

Joins the three per-pixel summary statistics from Stage 1:
  - nir_cv   (dry-season inter-annual NIR stability)
  - rec_p    (window-free annual NDVI amplitude, p90 − p10)
  - re_p10   (annual red-edge low percentile, B07/B05)

Produces:
  - Pairwise 2D scatters (3 panels) with class ellipses
  - 3D scatter (two viewing angles)
  - Class centroid and separation table
  - Pairwise Pearson r table
  - Inter-class Mahalanobis distance table
  - Histogram grid (one row per feature, one column per class)

See research/LONGREACH-STAGE2.md for context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy.spatial.distance import mahalanobis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-feature-space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours / class labels (consistent with prior analyses)
# ---------------------------------------------------------------------------
CLASS_COLOURS = {
    "Infestation": "#2ca02c",
    "Grassland":   "#ff7f0e",
    "Riparian":    "#1f77b4",
}

# ---------------------------------------------------------------------------
# Load and merge the three parquets
# ---------------------------------------------------------------------------

nir_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "longreach_dry_nir_stats.parquet"
)
amp_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-wet-dry-amp" / "longreach_amp_stats.parquet"
)
re_df = pd.read_parquet(
    PROJECT_ROOT / "outputs" / "longreach-red-edge" / "longreach_re_stats.parquet"
)

# amp_df is the richest — has class flags and rec_p
df = amp_df[["point_id", "lon", "lat", "rec_p", "rec_mean", "is_riparian", "is_grassland"]].copy()
df = df.merge(nir_df[["point_id", "nir_cv"]], on="point_id")
df = df.merge(re_df[["point_id",  "re_p10"]], on="point_id")

# Derive class label
def assign_class(row):
    if row.is_riparian:
        return "Riparian"
    if row.is_grassland:
        return "Grassland"
    return "Infestation"

df["class"] = df.apply(assign_class, axis=1)

FEATURES = ["nir_cv", "rec_p", "re_p10"]
FEATURE_LABELS = {
    "nir_cv":  "NIR CV (dry-season stability)",
    "rec_p":   "NDVI amplitude (rec_p)",
    "re_p10":  "Red-edge p10 (re_p10)",
}
CLASSES = ["Infestation", "Grassland", "Riparian"]

print(f"Pixels loaded: {len(df)}")
for cls in CLASSES:
    print(f"  {cls}: {(df['class'] == cls).sum()}")

# ---------------------------------------------------------------------------
# Helper: confidence ellipse (1-sigma) for a 2D scatter
# ---------------------------------------------------------------------------

def confidence_ellipse_patch(x, y, ax, n_std=1.5, **kwargs):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                   angle=theta, **kwargs)

# ---------------------------------------------------------------------------
# 1. Pairwise 2D scatter grid (3 panels)
# ---------------------------------------------------------------------------

pairs = [("nir_cv", "rec_p"), ("nir_cv", "re_p10"), ("rec_p", "re_p10")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Longreach — pairwise 2D feature projections", fontsize=13, y=1.01)

for ax, (fx, fy) in zip(axes, pairs):
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        col = CLASS_COLOURS[cls]
        ax.scatter(sub[fx], sub[fy], s=8, alpha=0.4, color=col, label=cls)
        # Confidence ellipse
        ep = confidence_ellipse_patch(
            sub[fx].values, sub[fy].values, ax,
            n_std=1.5, edgecolor=col, facecolor="none", linewidth=1.5, zorder=3
        )
        ax.add_patch(ep)
        # Centroid marker
        ax.scatter(sub[fx].mean(), sub[fy].mean(), s=80, color=col,
                   marker="D", edgecolors="black", linewidth=0.8, zorder=4)

    ax.set_xlabel(FEATURE_LABELS[fx], fontsize=10)
    ax.set_ylabel(FEATURE_LABELS[fy], fontsize=10)
    ax.tick_params(labelsize=9)

patches = [mpatches.Patch(color=CLASS_COLOURS[c], label=c) for c in CLASSES]
axes[0].legend(handles=patches, fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "longreach_pairwise_2d.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: longreach_pairwise_2d.png")

# ---------------------------------------------------------------------------
# 2. 3D scatter — two viewing angles
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Longreach — 3D feature space (nir_cv × rec_p × re_p10)", fontsize=13)

for i, (elev, azim) in enumerate([(25, -60), (10, 20)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        ax.scatter(sub["nir_cv"], sub["rec_p"], sub["re_p10"],
                   s=8, alpha=0.4, color=CLASS_COLOURS[cls], label=cls)
        # Centroid
        ax.scatter(sub["nir_cv"].mean(), sub["rec_p"].mean(), sub["re_p10"].mean(),
                   s=100, color=CLASS_COLOURS[cls], marker="D",
                   edgecolors="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("nir_cv", fontsize=9, labelpad=6)
    ax.set_ylabel("rec_p", fontsize=9, labelpad=6)
    ax.set_zlabel("re_p10", fontsize=9, labelpad=6)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)

handles = [mpatches.Patch(color=CLASS_COLOURS[c], label=c) for c in CLASSES]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
fig.savefig(OUT_DIR / "longreach_3d_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: longreach_3d_scatter.png")

# ---------------------------------------------------------------------------
# 3. Histogram grid (3 features × 3 classes)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False)
fig.suptitle("Longreach — feature distributions by class", fontsize=13)

for row, feat in enumerate(FEATURES):
    for col, cls in enumerate(CLASSES):
        ax = axes[row, col]
        vals = df.loc[df["class"] == cls, feat]
        ax.hist(vals, bins=30, color=CLASS_COLOURS[cls], alpha=0.75, edgecolor="none")
        ax.axvline(vals.median(), color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{cls}" if row == 0 else "", fontsize=10)
        ax.set_xlabel(FEATURE_LABELS[feat] if col == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.97, 0.95, f"med={vals.median():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)

# Row labels
for row, feat in enumerate(FEATURES):
    axes[row, 0].set_ylabel(FEATURE_LABELS[feat], fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "longreach_feature_histograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: longreach_feature_histograms.png")

# ---------------------------------------------------------------------------
# 4. Numeric summaries
# ---------------------------------------------------------------------------

print("\n=== Class centroids ===")
centroid_rows = []
for cls in CLASSES:
    sub = df[df["class"] == cls]
    row = {"class": cls, "n": len(sub)}
    for f in FEATURES:
        row[f"{f}_mean"] = sub[f].mean()
        row[f"{f}_median"] = sub[f].median()
        row[f"{f}_iqr_lo"] = sub[f].quantile(0.25)
        row[f"{f}_iqr_hi"] = sub[f].quantile(0.75)
    centroid_rows.append(row)
centroid_df = pd.DataFrame(centroid_rows)
print(centroid_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== IQR overlap fractions (pairwise classes) ===")
def iqr_overlap(a, b):
    lo = max(a.quantile(0.25), b.quantile(0.25))
    hi = min(a.quantile(0.75), b.quantile(0.75))
    return max(0.0, hi - lo)

class_pairs = [("Infestation", "Grassland"), ("Infestation", "Riparian"), ("Grassland", "Riparian")]
for feat in FEATURES:
    for c1, c2 in class_pairs:
        v1 = df.loc[df["class"] == c1, feat]
        v2 = df.loc[df["class"] == c2, feat]
        overlap = iqr_overlap(v1, v2)
        print(f"  {feat:12s}  {c1} vs {c2}: IQR overlap = {overlap:.4f}")

print("\n=== Pairwise Pearson r (all pixels) ===")
for i, f1 in enumerate(FEATURES):
    for f2 in FEATURES[i+1:]:
        r, p = pearsonr(df[f1], df[f2])
        print(f"  {f1} × {f2}: r = {r:.3f}  (p = {p:.2e})")

print("\n=== Inter-class Mahalanobis distances (3D feature space) ===")
X = df[FEATURES].values
# Pooled covariance
cov_pooled = np.cov(X.T)
cov_inv = np.linalg.inv(cov_pooled)

for c1, c2 in class_pairs:
    mu1 = df.loc[df["class"] == c1, FEATURES].mean().values
    mu2 = df.loc[df["class"] == c2, FEATURES].mean().values
    dist = mahalanobis(mu1, mu2, cov_inv)
    print(f"  {c1} vs {c2}: Mahalanobis d = {dist:.2f}")

print("\n=== 2D subspace Mahalanobis distances ===")
for fx, fy in pairs:
    cov2 = np.cov(df[[fx, fy]].values.T)
    cov2_inv = np.linalg.inv(cov2)
    for c1, c2 in class_pairs:
        mu1 = df.loc[df["class"] == c1, [fx, fy]].mean().values
        mu2 = df.loc[df["class"] == c2, [fx, fy]].mean().values
        dist = mahalanobis(mu1, mu2, cov2_inv)
        print(f"  ({fx}, {fy})  {c1} vs {c2}: d = {dist:.2f}")

print(f"\nOutputs written to {OUT_DIR}")
