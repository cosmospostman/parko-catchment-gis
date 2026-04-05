"""Fetch a zoomed WMS tile of the eastern riparian zone (~2 km east of the
confirmed infestation) to assess what vegetation is actually there.

Output: outputs/longreach-overview/longreach_east_zoom.png
"""

from __future__ import annotations

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

# Zoom into the eastern zone: from the confirmed infestation lon_max (~145.429)
# to ~2.5 km further east, same N-S extent as candidate B
ZOOM_BBOX = [145.428, -22.782, 145.460, -22.744]

print(f"Fetching eastern zoom tile {ZOOM_BBOX} ...")
img = fetch_wms_image(ZOOM_BBOX, width_px=2048)
print(f"Tile: {img.shape[1]} × {img.shape[0]} px")

lon_min, lat_min, lon_max, lat_max = ZOOM_BBOX

# Approximate ground dimensions
import math
lon_m = (lon_max - lon_min) * 111_320 * math.cos(math.radians((lat_min + lat_max) / 2))
lat_m = (lat_max - lat_min) * 111_320
print(f"Coverage: {lon_m:.0f} m × {lat_m:.0f} m")

aspect = (lat_max - lat_min) / (lon_max - lon_min)
fig, ax = plt.subplots(figsize=(12, 12 * aspect), dpi=150)
fig.suptitle(
    "Eastern zone zoom — Longreach\n"
    f"lon [{lon_min:.4f}–{lon_max:.4f}], lat [{lat_min:.4f}–{lat_max:.4f}]  "
    f"({lon_m:.0f} m × {lat_m:.0f} m)",
    fontsize=10,
)

ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
          origin="upper", aspect="auto", interpolation="bilinear")

# Mark confirmed infestation eastern edge for reference
ax.axvline(145.4287, color="white", linewidth=1.5, linestyle=":",
           label="Infestation eastern edge (145.4287)")

# 0.005° grid (~500 m)
for lon in np.arange(math.ceil(lon_min * 200) / 200, lon_max, 0.005):
    ax.axvline(lon, color="white", linewidth=0.4, alpha=0.4, linestyle="--")
    ax.text(lon, lat_max - (lat_max - lat_min) * 0.012, f"{lon:.4f}",
            ha="center", va="top", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))
for lat in np.arange(math.ceil(lat_min * 200) / 200, lat_max, 0.005):
    ax.axhline(lat, color="white", linewidth=0.4, alpha=0.4, linestyle="--")
    ax.text(lon_min + (lon_max - lon_min) * 0.01, lat, f"{lat:.4f}",
            ha="left", va="center", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))

ax.legend(loc="lower right", fontsize=8, framealpha=0.8,
          facecolor="black", labelcolor="white", edgecolor="none")
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)

plt.tight_layout()
out_path = OUT_DIR / "longreach_east_zoom.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
