"""Queensland Globe overviews for the 5 expansion site candidates.

Produces one PNG per site (outputs/expansion-site-overviews/site_<N>_*.png):
  - WMS tile fetched at the proposed fetch bbox from SITE-EXPANSION.md
  - ALA occurrence points overlaid (red dots, no label clutter at this scale)
  - Fetch bbox rectangle highlighted
  - Labelled lon/lat grid
  - Title with site name, ALA record count, and coordinate uncertainty

Sites 1–5 from SITE-EXPANSION.md (Galilee Basin East):
  1  Barcaldine corridor      bbox 144.88, -21.55, 145.07, -21.30
  2  Aramac Road cluster      bbox 145.16, -22.17, 145.24, -22.07
  3  Jericho area cluster     bbox 144.57, -22.53, 144.80, -22.36
  4  Barcaldine South cluster bbox 144.48, -22.58, 144.59, -22.50
  5  Longreach South cluster  bbox 145.41, -22.79, 145.45, -22.75
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
# Site definitions (from SITE-EXPANSION.md)
# ---------------------------------------------------------------------------

SITES = [
    dict(
        id=1,
        name="Barcaldine corridor",
        ala_records=472,
        coord_uncertainty_m=100,
        centroid=(-21.438, 145.004),
        bbox=[144.88, -21.55, 145.07, -21.30],
    ),
    dict(
        id=2,
        name="Aramac Road cluster",
        ala_records=109,
        coord_uncertainty_m=100,
        centroid=(-22.104, 145.205),
        bbox=[145.16, -22.17, 145.24, -22.07],
    ),
    dict(
        id=3,
        name="Jericho area cluster",
        ala_records=78,
        coord_uncertainty_m=1,
        centroid=(-22.452, 144.693),
        bbox=[144.57, -22.53, 144.80, -22.36],
    ),
    dict(
        id=4,
        name="Muttaburra",
        ala_records=37,
        coord_uncertainty_m=1,
        centroid=(-22.538, 144.558),
        bbox=[144.548274, -22.546983, 144.567726, -22.529017],
    ),
    dict(
        id=5,
        name="Longreach South cluster",
        ala_records=6,
        coord_uncertainty_m=13,
        centroid=(-22.773, 145.428),
        bbox=[145.41, -22.79, 145.45, -22.75],
    ),
]

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


for site in SITES:
    lon_min, lat_min, lon_max, lat_max = site["bbox"]
    sid = site["id"]
    name = site["name"]

    print(f"\nSite {sid} — {name}")
    print(f"  Fetching WMS tile {site['bbox']} ...")
    img = fetch_wms_image(site["bbox"], width_px=2048)
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

    unc_str = (
        f"{site['coord_uncertainty_m']} m" if site["coord_uncertainty_m"] < 10
        else f"~{site['coord_uncertainty_m']} m"
    )
    fig.suptitle(
        f"Site {sid} — {name}  (Priority {sid})\n"
        f"ALA records in bbox: {n_pts} / {site['ala_records']} cluster total  |  "
        f"coord uncertainty: {unc_str}\n"
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
    clat, clon = site["centroid"]
    ax.scatter([clon], [clat], s=80, color="white", marker="+",
               linewidths=2.0, zorder=6)
    ax.scatter([clon], [clat], s=80, color="#e74c3c", marker="+",
               linewidths=1.0, zorder=7,
               label=f"ALA centroid ({clat}, {clon})")

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
