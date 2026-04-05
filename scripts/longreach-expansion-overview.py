"""Final Queensland Globe view of the Longreach expansion bbox.

Fetches a high-resolution WMS tile of the expansion region and overlays:
  - The expansion bbox (rounded coordinates)
  - The existing confirmed-infestation strip for reference
  - 0.005° grid (~500 m at this latitude)

Output: outputs/longreach-overview/longreach_expansion_final.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bboxes
# ---------------------------------------------------------------------------

# Expansion bbox (rounded, from LONGREACH-EXPANSION.md)
EXP = dict(lon_min=145.421, lat_min=-22.771, lon_max=145.448, lat_max=-22.758)

# Existing confirmed-infestation strip
INF = dict(lon_min=145.4213, lat_min=-22.7671, lon_max=145.4287, lat_max=-22.7597)

# Fetch tile: expansion bbox + small margin
MARGIN = 0.003   # ~300 m
FETCH_BBOX = [
    EXP["lon_min"] - MARGIN,
    EXP["lat_min"] - MARGIN,
    EXP["lon_max"] + MARGIN,
    EXP["lat_max"] + MARGIN,
]

print(f"Fetching WMS tile {FETCH_BBOX} ...")
img = fetch_wms_image(FETCH_BBOX, width_px=2048)
lon_min, lat_min, lon_max, lat_max = FETCH_BBOX
print(f"Tile: {img.shape[1]} × {img.shape[0]} px")

lat_c = (EXP["lat_min"] + EXP["lat_max"]) / 2
exp_w = (EXP["lon_max"] - EXP["lon_min"]) * 111_320 * math.cos(math.radians(lat_c))
exp_h = (EXP["lat_max"] - EXP["lat_min"]) * 111_320
n_px  = int(exp_w / 10) * int(exp_h / 10)
gb    = n_px * 387 * 120 / 1e9

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

aspect = (lat_max - lat_min) / (lon_max - lon_min)
fig, ax = plt.subplots(figsize=(14, 14 * aspect), dpi=150)
fig.suptitle(
    "Longreach expansion — eastern riparian zone\n"
    f"bbox lon [{EXP['lon_min']}, {EXP['lon_max']}]  lat [{EXP['lat_min']}, {EXP['lat_max']}]\n"
    f"{exp_w/1000:.2f} km × {exp_h/1000:.2f} km  |  ~{n_px:,} S2 pixels  |  ~{gb:.1f} GB",
    fontsize=10,
)

ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
          origin="upper", aspect="auto", interpolation="bilinear")

# Expansion bbox
exp_rect = mpatches.Rectangle(
    (EXP["lon_min"], EXP["lat_min"]),
    EXP["lon_max"] - EXP["lon_min"],
    EXP["lat_max"] - EXP["lat_min"],
    linewidth=2.5, edgecolor="#e74c3c", facecolor="#e74c3c", alpha=0.10, zorder=4,
)
ax.add_patch(exp_rect)
exp_border = mpatches.Rectangle(
    (EXP["lon_min"], EXP["lat_min"]),
    EXP["lon_max"] - EXP["lon_min"],
    EXP["lat_max"] - EXP["lat_min"],
    linewidth=2.5, edgecolor="#e74c3c", facecolor="none", zorder=5,
)
ax.add_patch(exp_border)

# Existing strip
inf_rect = mpatches.Rectangle(
    (INF["lon_min"], INF["lat_min"]),
    INF["lon_max"] - INF["lon_min"],
    INF["lat_max"] - INF["lat_min"],
    linewidth=2.0, edgecolor="white", facecolor="none", linestyle=":", zorder=5,
)
ax.add_patch(inf_rect)

# Zone annotations
zones = [
    (145.4215, -22.7645, "Native\nriparian\nwoodland", "left"),
    (145.4340, -22.7645, "Floodplain\ngilgai\nmosaic",  "center"),
    (145.4430, -22.7645, "Scattered\nParkinsonia\non clay",  "center"),
]
for lon, lat, label, ha in zones:
    ax.text(lon, lat, label,
            color="white", fontsize=7.5, ha=ha, va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.60, edgecolor="none"),
            zorder=6)

# 0.005° grid (~550 m)
for lon in np.arange(math.ceil(lon_min * 200) / 200, lon_max, 0.005):
    ax.axvline(lon, color="white", linewidth=0.45, alpha=0.40, linestyle="--")
    ax.text(lon, lat_max - (lat_max - lat_min) * 0.015, f"{lon:.3f}",
            ha="center", va="top", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.45, edgecolor="none"))
for lat in np.arange(math.ceil(lat_min * 200) / 200, lat_max, 0.005):
    ax.axhline(lat, color="white", linewidth=0.45, alpha=0.40, linestyle="--")
    ax.text(lon_min + (lon_max - lon_min) * 0.01, lat, f"{lat:.3f}",
            ha="left", va="center", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.45, edgecolor="none"))

legend_patches = [
    mpatches.Patch(edgecolor="#e74c3c", facecolor="#e74c3c", alpha=0.4,
                   label=f"Expansion bbox  [{EXP['lon_min']}, {EXP['lon_max']}] × [{EXP['lat_min']}, {EXP['lat_max']}]"),
    mpatches.Patch(edgecolor="white", facecolor="none", linestyle=":",
                   label="Existing strip (confirmed infestation, 748 px)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          framealpha=0.8, facecolor="black", labelcolor="white", edgecolor="none")

ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

plt.tight_layout()
out_path = OUT_DIR / "longreach_expansion_final.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
