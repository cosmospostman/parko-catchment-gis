"""Re-render the saved 6 km overview tile with only the two revised candidate bboxes:

  B  — original 2×4 km rectangle (centred on infestation)
  B2 — revised 2×4 km rectangle shifted ~0.003° east to include dense eastern riparian

Output: outputs/longreach-overview/longreach_revised_bboxes.png
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

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-overview"

# ---------------------------------------------------------------------------
# Fetch (or reload from cache) the raw 6 km WMS tile
# Raw tile cached separately so we don't re-fetch every run.
# ---------------------------------------------------------------------------

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

# The tile covers bbox C from the previous script
lon_min = 145.40065424036143
lat_min = -22.79035133524973
lon_max = 145.44936375963854
lat_max = -22.736452664750267

raw_tile_path = OUT_DIR / "longreach_6km_raw.npy"
if raw_tile_path.exists():
    img = np.load(raw_tile_path)
    print(f"Loaded cached raw tile: {img.shape}")
else:
    ov_list = [lon_min, lat_min, lon_max, lat_max]
    print(f"Fetching WMS tile {ov_list} ...")
    img = fetch_wms_image(ov_list, width_px=2048)
    np.save(raw_tile_path, img)
    print(f"Cached: {raw_tile_path}")

CENTRE_LAT = -22.763402
CENTRE_LON = 145.425009

INF_BBOX = dict(lon_min=145.4213, lat_min=-22.7671, lon_max=145.4287, lat_max=-22.7597)

# ---------------------------------------------------------------------------
# Two candidate bboxes
# ---------------------------------------------------------------------------

# B — original 2×4 km, centred on the confirmed infestation
B = dict(
    lon_min=145.4153, lat_min=-22.7814,
    lon_max=145.4348, lat_max=-22.7454,
    label="B — 2×4 km (original, centred)",
    colour="#f39c12",   # orange
)

# B2 — same 2×4 km footprint as B, shifted ~0.003° east so the eastern
# riparian corridor sits closer to centre rather than at the right edge.
# lon shift: +0.003° ≈ +307 m at this latitude
B2 = dict(
    lon_min=145.4183, lat_min=-22.7814,
    lon_max=145.4378, lat_max=-22.7454,
    label="B2 — 2×4 km (shifted east, riparian-centred)",
    colour="#e74c3c",   # red
)

candidates = [B, B2]

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

aspect = (lat_max - lat_min) / (lon_max - lon_min)
fig_w = 14
fig_h = fig_w * aspect

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
fig.suptitle(
    "Longreach — revised candidate expansion bboxes\n"
    "B (orange) = centred; B2 (red) = shifted east to capture riparian corridor",
    fontsize=11,
)

ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
          origin="upper", aspect="auto", interpolation="bilinear")

legend_patches = []

# Current strip
rect0 = mpatches.Rectangle(
    (INF_BBOX["lon_min"], INF_BBOX["lat_min"]),
    INF_BBOX["lon_max"] - INF_BBOX["lon_min"],
    INF_BBOX["lat_max"] - INF_BBOX["lat_min"],
    linewidth=1.8, edgecolor="white", facecolor="none", linestyle=":", zorder=4,
)
ax.add_patch(rect0)
legend_patches.append(
    mpatches.Patch(edgecolor="white", facecolor="none", linestyle=":",
                   label="Current strip (confirmed infestation)")
)

for bb in candidates:
    w = bb["lon_max"] - bb["lon_min"]
    h = bb["lat_max"] - bb["lat_min"]
    rect = mpatches.Rectangle(
        (bb["lon_min"], bb["lat_min"]), w, h,
        linewidth=2.2, edgecolor=bb["colour"], facecolor=bb["colour"],
        alpha=0.12, zorder=4,
    )
    ax.add_patch(rect)
    # Solid border on top
    rect_border = mpatches.Rectangle(
        (bb["lon_min"], bb["lat_min"]), w, h,
        linewidth=2.2, edgecolor=bb["colour"], facecolor="none", zorder=5,
    )
    ax.add_patch(rect_border)
    # Label inside top-left
    ax.text(
        bb["lon_min"] + 0.0004, bb["lat_max"] - 0.0004,
        bb["label"],
        color=bb["colour"], fontsize=8, va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65, edgecolor="none"),
        zorder=6,
    )
    legend_patches.append(
        mpatches.Patch(edgecolor=bb["colour"], facecolor=bb["colour"], alpha=0.4,
                       label=bb["label"])
    )

# Overlap shading — show where B and B2 overlap
ovlp_lon_min = max(B["lon_min"], B2["lon_min"])
ovlp_lat_min = max(B["lat_min"], B2["lat_min"])
ovlp_lon_max = min(B["lon_max"], B2["lon_max"])
ovlp_lat_max = min(B["lat_max"], B2["lat_max"])
if ovlp_lon_max > ovlp_lon_min and ovlp_lat_max > ovlp_lat_min:
    ovlp_rect = mpatches.Rectangle(
        (ovlp_lon_min, ovlp_lat_min),
        ovlp_lon_max - ovlp_lon_min,
        ovlp_lat_max - ovlp_lat_min,
        linewidth=0, facecolor="yellow", alpha=0.12, zorder=3,
    )
    ax.add_patch(ovlp_rect)
    ax.text(
        (ovlp_lon_min + ovlp_lon_max) / 2,
        (ovlp_lat_min + ovlp_lat_max) / 2,
        "overlap",
        color="yellow", fontsize=7, ha="center", va="center", alpha=0.8,
        zorder=6,
    )

# Site centre
ax.scatter([CENTRE_LON], [CENTRE_LAT], s=80, color="white", marker="+",
           linewidths=2.0, zorder=7)

# 0.01° grid
import math as _m
for lon in np.arange(_m.ceil(lon_min * 100) / 100, lon_max, 0.01):
    ax.axvline(lon, color="white", linewidth=0.4, alpha=0.35, linestyle="--")
    ax.text(lon, lat_max - (lat_max - lat_min) * 0.012, f"{lon:.3f}",
            ha="center", va="top", fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4, edgecolor="none"))
for lat in np.arange(_m.ceil(lat_min * 100) / 100, lat_max, 0.01):
    ax.axhline(lat, color="white", linewidth=0.4, alpha=0.35, linestyle="--")
    ax.text(lon_min + (lon_max - lon_min) * 0.008, lat, f"{lat:.3f}",
            ha="left", va="center", fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4, edgecolor="none"))

ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          framealpha=0.8, facecolor="black", labelcolor="white", edgecolor="none")
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

plt.tight_layout()
out_path = OUT_DIR / "longreach_revised_bboxes.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# Print summary
print("\n=== Bbox coordinates ===")
for bb in candidates:
    import math
    lon_span = bb["lon_max"] - bb["lon_min"]
    lat_span = bb["lat_max"] - bb["lat_min"]
    lon_m = lon_span * 111_320 * math.cos(math.radians(CENTRE_LAT))
    lat_m = lat_span * 111_320
    n_px = int(lon_m / 10) * int(lat_m / 10)
    gb = n_px * 387 * 120 / 1e9
    print(f"\n  {bb['label']}")
    print(f"    lon: [{bb['lon_min']:.4f}, {bb['lon_max']:.4f}]   ({lon_m:.0f} m)")
    print(f"    lat: [{bb['lat_min']:.4f}, {bb['lat_max']:.4f}]   ({lat_m:.0f} m)")
    print(f"    ~{n_px:,} pixels  ~{gb:.1f} GB")
