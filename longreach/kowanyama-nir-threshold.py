"""kowanyama-nir-threshold.py — nir_cv threshold scan for Kowanyama.

Loads the pre-computed feature table (outputs/kowanyama/kowanyama_features.parquet)
and applies a series of hard thresholds on nir_cv alone. For each threshold,
pixels with nir_cv <= T are flagged as candidate Parkinsonia and plotted on the
WMS imagery.

The anchor pixel range (0.13–0.21, median 0.17) motivates the scan range.

Produces:
  outputs/kowanyama/
    kowanyama_nir_threshold_T.png   — one map per threshold value T
    kowanyama_nir_threshold_all.png — four-panel comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

ANCHOR_PRESENCE = (-15.457794, 141.535690)  # (lat, lon)
THRESHOLDS = [0.13, 0.17, 0.20, 0.22]


def log(msg: str) -> None:
    print(msg, flush=True)


def haversine_m(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlam = np.radians(lons - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# 1. Load feature table
# ---------------------------------------------------------------------------

feat_path = OUT_DIR / "kowanyama_features.parquet"
log(f"Loading feature table: {feat_path.relative_to(PROJECT_ROOT)}")
feat = pd.read_parquet(feat_path)
log(f"  {len(feat):,} pixels")

log(f"\nnir_cv distribution:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    log(f"  p{p:02d}: {feat['nir_cv'].quantile(p/100):.4f}")

# Anchor pixels
feat["dist_anchor_m"] = haversine_m(
    ANCHOR_PRESENCE[0], ANCHOR_PRESENCE[1],
    feat["lat"].values, feat["lon"].values,
)
anchor_px = feat[feat["dist_anchor_m"] <= 100.0]
log(f"\nAnchor pixels: {len(anchor_px)}  "
    f"nir_cv range [{anchor_px['nir_cv'].min():.4f}, {anchor_px['nir_cv'].max():.4f}]  "
    f"median {anchor_px['nir_cv'].median():.4f}")

# ---------------------------------------------------------------------------
# 2. Fetch WMS imagery (once)
# ---------------------------------------------------------------------------

margin  = 0.001
lon_min = feat["lon"].min() - margin
lon_max = feat["lon"].max() + margin
lat_min = feat["lat"].min() - margin
lat_max = feat["lat"].max() + margin

log(f"\nFetching WMS tile...")
try:
    img = fetch_wms_image([lon_min, lat_min, lon_max, lat_max], width_px=4096)
    log(f"  {img.shape[1]} x {img.shape[0]} px")
except Exception as exc:
    log(f"  WARNING: WMS fetch failed ({exc})")
    img = None

DPI = 200
if img is not None:
    fig_w = img.shape[1] / DPI
    fig_h = img.shape[0] / DPI
else:
    lat_centre    = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lon_span_deg  = lon_max - lon_min
    lat_span_deg  = lat_max - lat_min
    fig_w = 20
    fig_h = max(4.0, fig_w * (lat_span_deg * 111320) / (lon_span_deg * lon_m_per_deg))

lat_centre    = (lat_min + lat_max) / 2
lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
lon_span_deg  = lon_max - lon_min
pt_per_deg    = fig_w * 72 / lon_span_deg
marker_pt     = (10 / lon_m_per_deg) * pt_per_deg
marker_s      = max(0.5, marker_pt ** 2 / 8)

# ---------------------------------------------------------------------------
# 3. Per-threshold maps
# ---------------------------------------------------------------------------

log("\nThreshold scan:")
for T in THRESHOLDS:
    hits    = feat[feat["nir_cv"] <= T]
    nonhits = feat[feat["nir_cv"] > T]
    pct     = 100 * len(hits) / len(feat)
    log(f"  nir_cv <= {T:.2f}: {len(hits):,} pixels ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"nir_cv ≤ {T:.2f}  —  {len(hits):,} pixels ({pct:.1f}% of bbox)\n"
        f"Kowanyama Mitchell River delta  |  Longreach anchor: nir_cv 0.13–0.21",
        fontsize=10,
    )

    if img is not None:
        ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#1a1a1a")

    # Non-hits: dim grey
    ax.scatter(nonhits["lon"], nonhits["lat"],
               c="#555555", s=marker_s, linewidths=0, alpha=0.25, zorder=1)
    # Hits: orange
    ax.scatter(hits["lon"], hits["lat"],
               c="#e67e22", s=marker_s * 1.5, linewidths=0, alpha=0.8, zorder=2,
               label=f"nir_cv ≤ {T:.2f}")
    # Anchor
    if len(anchor_px):
        ax.scatter(anchor_px["lon"], anchor_px["lat"],
                   marker="D", s=marker_s * 4, color="#f1c40f",
                   edgecolors="white", linewidths=0.6, zorder=5,
                   label="Anchor: presence")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / f"kowanyama_nir_threshold_{str(T).replace('.', '')}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: {out.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 4. Four-panel comparison
# ---------------------------------------------------------------------------

log("\nGenerating four-panel comparison...")

fig, axes = plt.subplots(2, 2, figsize=(fig_w * 1.05, fig_h * 2.1))
fig.suptitle(
    "nir_cv threshold scan — Kowanyama  |  orange = candidate Parkinsonia  |  diamond = anchor presence",
    fontsize=10,
)

for ax, T in zip(axes.flat, THRESHOLDS):
    hits    = feat[feat["nir_cv"] <= T]
    nonhits = feat[feat["nir_cv"] > T]
    pct     = 100 * len(hits) / len(feat)

    if img is not None:
        ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#1a1a1a")

    ax.scatter(nonhits["lon"], nonhits["lat"],
               c="#555555", s=marker_s * 0.5, linewidths=0, alpha=0.2, zorder=1)
    ax.scatter(hits["lon"], hits["lat"],
               c="#e67e22", s=marker_s, linewidths=0, alpha=0.8, zorder=2)
    if len(anchor_px):
        ax.scatter(anchor_px["lon"], anchor_px["lat"],
                   marker="D", s=marker_s * 3, color="#f1c40f",
                   edgecolors="white", linewidths=0.5, zorder=5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"nir_cv ≤ {T:.2f}  ({len(hits):,} px, {pct:.1f}%)", fontsize=9)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6)

plt.tight_layout()
out_all = OUT_DIR / "kowanyama_nir_threshold_all.png"
fig.savefig(out_all, dpi=DPI, bbox_inches="tight")
plt.close(fig)
log(f"Saved: {out_all.relative_to(PROJECT_ROOT)}")

log("\nDone.")
