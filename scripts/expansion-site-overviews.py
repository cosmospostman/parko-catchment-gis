"""Queensland Globe overviews for the expansion site candidates.

Produces one PNG per site (outputs/expansion-site-overviews/site_<N>_*.png):
  - WMS tile fetched at the proposed fetch bbox from SITE-EXPANSION.md
  - ALA occurrence points overlaid (red dots, no label clutter at this scale)
  - Fetch bbox rectangle highlighted
  - Labelled lon/lat grid
  - Title with site name and bbox

Sites are loaded from data/locations/*.yaml via utils.location.
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
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

OUT_DIR = PROJECT_ROOT / "outputs" / "expansion-site-overviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALA_PATH = PROJECT_ROOT / "outputs" / "australia_occurrences" / "ala_australia_occurrences.gpkg"

# ---------------------------------------------------------------------------
# Site definitions — loaded from data/locations/*.yaml
# ---------------------------------------------------------------------------

from utils.location import all_locations  # noqa: E402

# Exclude longreach (training site) and kowanyama (separate environment).
_EXCLUDE = {"longreach", "kowanyama"}
SITES = sorted(
    [loc for loc in all_locations() if loc.id not in _EXCLUDE],
    key=lambda l: l.id,
)

# ---------------------------------------------------------------------------
# Load ALA points
# ---------------------------------------------------------------------------

print("Loading ALA occurrences ...")
ala = gpd.read_file(ALA_PATH)
ala_lons = ala["decimalLongitude"].values
ala_lats = ala["decimalLatitude"].values
print(f"  {len(ala):,} total records")

# ---------------------------------------------------------------------------
# Render each site
# ---------------------------------------------------------------------------

def _nice_grid_spacing(span: float, target_lines: int = 6) -> float:
    raw = span / target_lines
    mag = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        step = factor * mag
        if span / step <= target_lines:
            return step
    return 10 * mag


for sid, loc in enumerate(SITES, 1):
    lon_min, lat_min, lon_max, lat_max = loc.bbox
    name = loc.name

    print(f"\nSite {sid} — {name}")
    print(f"  Fetching WMS tile {loc.bbox} ...")
    img = fetch_wms_image(loc.bbox, width_px=2048)
    print(f"  Tile: {img.shape[1]} × {img.shape[0]} px")

    # ALA points inside this bbox
    mask = (
        (ala_lons >= lon_min) & (ala_lons <= lon_max) &
        (ala_lats >= lat_min) & (ala_lats <= lat_max)
    )
    pts_lon = ala_lons[mask]
    pts_lat = ala_lats[mask]
    n_pts = mask.sum()
    print(f"  ALA points in bbox: {n_pts}")

    # Figure sizing: preserve geographic aspect ratio
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    aspect = lat_span / lon_span
    fig_w = 14
    fig_h = max(5.0, fig_w * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    fig.suptitle(
        f"Site {sid} — {name}\n"
        f"ALA records in bbox: {n_pts}\n"
        f"fetch bbox  lon [{lon_min}, {lon_max}]  lat [{lat_min}, {lat_max}]",
        fontsize=10,
    )

    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="auto", interpolation="bilinear")

    # Fetch bbox outline
    bbox_rect = mpatches.Rectangle(
        (lon_min, lat_min), lon_span, lat_span,
        linewidth=2.0, edgecolor="#f39c12", facecolor="#f39c12", alpha=0.08, zorder=3,
    )
    ax.add_patch(bbox_rect)
    bbox_border = mpatches.Rectangle(
        (lon_min, lat_min), lon_span, lat_span,
        linewidth=2.0, edgecolor="#f39c12", facecolor="none", zorder=4,
    )
    ax.add_patch(bbox_border)

    # ALA points
    if n_pts > 0:
        ax.scatter(pts_lon, pts_lat, s=14, color="#e74c3c",
                   edgecolors="white", linewidths=0.5, zorder=5,
                   label=f"ALA occurrences ({n_pts})")

    # ALA centroid marker
    if loc.centroid:
        clat, clon = loc.centroid
        ax.scatter([clon], [clat], s=80, color="white", marker="+",
                   linewidths=2.0, zorder=6)
        ax.scatter([clon], [clat], s=80, color="#e74c3c", marker="+",
                   linewidths=1.0, zorder=7,
                   label=f"ALA centroid ({clat}, {clon})")
    else:
        clat, clon = None, None

    # Lon/lat grid
    lon_step = _nice_grid_spacing(lon_span)
    lat_step = _nice_grid_spacing(lat_span)
    lon_start = math.ceil(lon_min / lon_step) * lon_step
    lat_start = math.ceil(lat_min / lat_step) * lat_step
    grid_kw = dict(color="white", linewidth=0.5, alpha=0.45, linestyle="--")
    for x in np.arange(lon_start, lon_max, lon_step):
        ax.axvline(x, **grid_kw)
        ax.text(x, lat_max - lat_span * 0.012, f"{x:.3f}",
                ha="center", va="top", fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))
    for y in np.arange(lat_start, lat_max, lat_step):
        ax.axhline(y, **grid_kw)
        ax.text(lon_min + lon_span * 0.008, y, f"{y:.3f}",
                ha="left", va="center", fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))

    legend_handles = [
        mpatches.Patch(edgecolor="#f39c12", facecolor="#f39c12", alpha=0.3,
                       label=f"Fetch bbox"),
    ]
    if n_pts > 0:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                       markeredgecolor="white", markersize=7,
                       label=f"ALA occurrences in bbox ({n_pts})")
        )
    if clat is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="+", color="#e74c3c", markersize=10,
                       linewidth=0, label=f"ALA centroid ({clat:.3f}, {clon:.3f})")
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.8, facecolor="black", labelcolor="white", edgecolor="none")

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    out_path = OUT_DIR / f"site_{sid}_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print(f"\nAll done — {len(SITES)} overviews in {OUT_DIR}")
