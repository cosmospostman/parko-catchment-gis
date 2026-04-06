"""Fetch a 6 km overview WMS tile around the Longreach infestation site
and annotate candidate expansion bboxes.

Site centre: -22.763402, 145.425009

Candidate bboxes produced (all centred on the confirmed infestation patch):

  A — 1 km × 3 km strip   along watercourse corridor (cheap: ~31 K pixels, ~1 GB)
  B — 2 km × 4 km targeted rectangle (medium: ~80 K pixels, ~3 GB)
  C — 5 km × 6 km regional context    (large: ~300 K pixels, ~10 GB)

The 6 km overview tile itself is saved for visual review so you can judge
which corridor/bbox best captures the riparian and grassland diversity needed.

Output: outputs/longreach-overview/longreach_6km_overview.png
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
    "qglobe_plot", PROJECT_ROOT / "utils" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Site anchor
# ---------------------------------------------------------------------------

CENTRE_LAT = -22.763402
CENTRE_LON = 145.425009

# Current confirmed infestation bbox
INF_BBOX = dict(lon_min=145.4213, lat_min=-22.7671, lon_max=145.4287, lat_max=-22.7597)

# ---------------------------------------------------------------------------
# Helper: convert metres offsets from centre to lon/lat bbox
# ---------------------------------------------------------------------------

def metres_to_deg(metres_lon: float, metres_lat: float, lat: float) -> tuple[float, float]:
    """Return (dlon, dlat) for the given metre offsets at the given latitude."""
    dlon = metres_lon / (111_320 * math.cos(math.radians(lat)))
    dlat = metres_lat / 111_320
    return dlon, dlat

def make_bbox(half_lon_m: float, half_lat_m: float) -> dict:
    dlon, dlat = metres_to_deg(half_lon_m, half_lat_m, CENTRE_LAT)
    return dict(
        lon_min=CENTRE_LON - dlon,
        lat_min=CENTRE_LAT - dlat,
        lon_max=CENTRE_LON + dlon,
        lat_max=CENTRE_LAT + dlat,
    )

# ---------------------------------------------------------------------------
# Candidate bboxes
# ---------------------------------------------------------------------------

candidates = {
    "A — 1×3 km strip": make_bbox(half_lon_m=500,  half_lat_m=1500),
    "B — 2×4 km rect":  make_bbox(half_lon_m=1000, half_lat_m=2000),
    "C — 5×6 km region": make_bbox(half_lon_m=2500, half_lat_m=3000),
}

# Watercourse-following narrow strip (N-S corridor, wider N-S than E-W)
# The infestation sits on a N-S drainage line, so a narrow E-W / long N-S strip
# captures more of the riparian corridor per pixel fetched.
candidates["D — narrow 0.5×6 km corridor"] = make_bbox(half_lon_m=250, half_lat_m=3000)

COLOURS = {
    "A — 1×3 km strip":          "#e74c3c",  # red
    "B — 2×4 km rect":           "#f39c12",  # orange
    "C — 5×6 km region":         "#2ecc71",  # green
    "D — narrow 0.5×6 km corridor": "#3498db",  # blue
}

# ---------------------------------------------------------------------------
# Print pixel / data volume estimates
# ---------------------------------------------------------------------------

print("=== Candidate bbox summary ===")
print(f"{'Name':<30}  {'lon range':>10}  {'lat range':>10}  {'~px':>10}  {'~GB':>6}")
for name, bb in candidates.items():
    lon_span = bb["lon_max"] - bb["lon_min"]
    lat_span = bb["lat_max"] - bb["lat_min"]
    lon_m = lon_span * 111_320 * math.cos(math.radians(CENTRE_LAT))
    lat_m = lat_span * 111_320
    n_px = int(lon_m / 10) * int(lat_m / 10)
    # ~387 obs/pixel, 11 float32 bands + overhead ≈ 120 bytes/row
    gb = n_px * 387 * 120 / 1e9
    print(f"  {name:<28}  {lon_span:>10.4f}  {lat_span:>10.4f}  {n_px:>10,}  {gb:>6.1f}")

# ---------------------------------------------------------------------------
# Fetch 6 km overview tile (= candidate C extent)
# ---------------------------------------------------------------------------

overview_bbox = candidates["C — 5×6 km region"]
ov_list = [overview_bbox["lon_min"], overview_bbox["lat_min"],
           overview_bbox["lon_max"], overview_bbox["lat_max"]]

print(f"\nFetching 6 km overview WMS tile for bbox {ov_list} ...")
img = fetch_wms_image(ov_list, width_px=2048)
print(f"Tile: {img.shape[1]} × {img.shape[0]} px")

# ---------------------------------------------------------------------------
# Render overview with candidate bboxes overlaid
# ---------------------------------------------------------------------------

lon_min, lat_min, lon_max, lat_max = ov_list

aspect = (lat_max - lat_min) / (lon_max - lon_min)
fig_w = 14
fig_h = fig_w * aspect
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
fig.suptitle(
    "Longreach — 6 km regional overview with candidate expansion bboxes\n"
    f"Centre: {CENTRE_LAT}, {CENTRE_LON}",
    fontsize=11
)

ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
          origin="upper", aspect="auto", interpolation="bilinear")

# Draw candidate boxes
legend_patches = []
for name, bb in candidates.items():
    colour = COLOURS[name]
    w = bb["lon_max"] - bb["lon_min"]
    h = bb["lat_max"] - bb["lat_min"]
    rect = mpatches.Rectangle(
        (bb["lon_min"], bb["lat_min"]), w, h,
        linewidth=1.8, edgecolor=colour, facecolor="none", zorder=4
    )
    ax.add_patch(rect)
    # Label at top-left corner
    ax.text(
        bb["lon_min"], bb["lat_max"],
        name,
        color=colour, fontsize=7, va="bottom", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
        zorder=5,
    )
    legend_patches.append(mpatches.Patch(edgecolor=colour, facecolor="none", label=name))

# Draw current infestation bbox
rect0 = mpatches.Rectangle(
    (INF_BBOX["lon_min"], INF_BBOX["lat_min"]),
    INF_BBOX["lon_max"] - INF_BBOX["lon_min"],
    INF_BBOX["lat_max"] - INF_BBOX["lat_min"],
    linewidth=2.0, edgecolor="white", facecolor="none", zorder=4, linestyle=":"
)
ax.add_patch(rect0)
legend_patches.append(
    mpatches.Patch(edgecolor="white", facecolor="none", linestyle=":", label="Current strip (confirmed infestation)")
)

# Site centre marker
ax.scatter([CENTRE_LON], [CENTRE_LAT], s=60, color="white", marker="+",
           linewidths=1.5, zorder=6)

# Lon/lat grid — 0.01° spacing (~1.1 km at this lat)
import math as _m
for lon in np.arange(_m.ceil(lon_min * 100) / 100, lon_max, 0.01):
    ax.axvline(lon, color="white", linewidth=0.4, alpha=0.35, linestyle="--")
    ax.text(lon, lat_max - (lat_max - lat_min) * 0.01, f"{lon:.3f}",
            ha="center", va="top", fontsize=5, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4, edgecolor="none"))
for lat in np.arange(_m.ceil(lat_min * 100) / 100, lat_max, 0.01):
    ax.axhline(lat, color="white", linewidth=0.4, alpha=0.35, linestyle="--")
    ax.text(lon_min + (lon_max - lon_min) * 0.005, lat, f"{lat:.3f}",
            ha="left", va="center", fontsize=5, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4, edgecolor="none"))

ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          framealpha=0.75, facecolor="black", labelcolor="white", edgecolor="none")
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

plt.tight_layout()
out_path = OUT_DIR / "longreach_6km_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Print bbox coordinates for reference
# ---------------------------------------------------------------------------

print("\n=== Candidate bbox coordinates ===")
for name, bb in candidates.items():
    print(f"\n  {name}")
    print(f"    lon: [{bb['lon_min']:.4f}, {bb['lon_max']:.4f}]")
    print(f"    lat: [{bb['lat_min']:.4f}, {bb['lat_max']:.4f}]")
