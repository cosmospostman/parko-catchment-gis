"""Render docs/diagrams/54LWH_cog_blocks/54LWH_cog_blocks.png

Left panel:  COG internal 11×11 block grid for S2 tile 54LWH, labelled (col, row).
Right panel: Same grid with strips coloured by the pixels they contain within the
             Mitchell River catchment boundary.  Each strip corresponds to one
             absolute COG block row and is stored as 54LWH_strip_NN.parquet.

Run from the repo root:
    .venv/bin/python docs/diagrams/54LWH_cog_blocks/render.py
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from pyproj import Transformer
import shapely.geometry as sg
from shapely.ops import transform as shp_transform

# ---------------------------------------------------------------------------
# S2 tile 54LWH parameters
# ---------------------------------------------------------------------------
EPSG     = "EPSG:32754"
ORIGIN_X = 499980.0
Y_TOP    = 8300020.0
BLOCK_PX = 1024
BLOCK_M  = BLOCK_PX * 10.0          # 10 m/px → 10 240 m/block
N_BLOCKS = math.ceil(10980 / BLOCK_PX)  # 11

tile_x0 = ORIGIN_X
tile_x1 = ORIGIN_X + N_BLOCKS * BLOCK_M
tile_y0 = Y_TOP - N_BLOCKS * BLOCK_M
tile_y1 = Y_TOP

# ---------------------------------------------------------------------------
# Mitchell River catchment — load, project to UTM 54S, clip to tile
# ---------------------------------------------------------------------------
with open("data/catchments/mitchell_river.geojson") as f:
    gj = json.load(f)
catch_wgs = sg.shape(gj["features"][0]["geometry"])
to_utm = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
catch_utm = shp_transform(to_utm.transform, catch_wgs)
catch_clipped = catch_utm.intersection(sg.box(tile_x0, tile_y0, tile_x1, tile_y1))

# ---------------------------------------------------------------------------
# Strip layout: absolute COG block rows that overlap the catchment bbox
# ---------------------------------------------------------------------------
cb = catch_clipped.bounds  # (minx, miny, maxx, maxy)
strips = [
    (row, Y_TOP - (row + 1) * BLOCK_M, Y_TOP - row * BLOCK_M)
    for row in range(N_BLOCKS)
    if Y_TOP - row * BLOCK_M > cb[1] and Y_TOP - (row + 1) * BLOCK_M < cb[3]
]


def poly_to_patch(geom, **kwargs):
    """Convert a Shapely (Multi)Polygon to a matplotlib PathPatch."""
    def _ring(coords):
        v = list(coords)
        return v, [MPath.MOVETO] + [MPath.LINETO] * (len(v) - 2) + [MPath.CLOSEPOLY]
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    verts, codes = [], []
    for p in polys:
        v, c = _ring(p.exterior.coords);   verts += v; codes += c
        for interior in p.interiors:
            v, c = _ring(interior.coords); verts += v; codes += c
    return PathPatch(MPath(verts, codes), **kwargs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
cmap = plt.colormaps["tab10"]

for ax_idx, ax in enumerate(axes):
    # COG block grid — identical on both panels
    for col in range(N_BLOCKS):
        for row in range(N_BLOCKS):
            bx = ORIGIN_X + col * BLOCK_M
            by = Y_TOP - (row + 1) * BLOCK_M
            ax.add_patch(mpatches.Rectangle(
                (bx, by), BLOCK_M, BLOCK_M,
                linewidth=0.4, edgecolor="#666", facecolor="#e8eef4", zorder=1,
            ))
            if ax_idx == 0:
                ax.text(bx + BLOCK_M / 2, by + BLOCK_M / 2, f"({col},{row})",
                        ha="center", va="center", fontsize=4.5, color="#555", zorder=2)

    ax.set_xlim(tile_x0, tile_x1)
    ax.set_ylim(tile_y0, tile_y1)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, UTM 54S)", fontsize=8)
    ax.set_ylabel("Northing (m, UTM 54S)", fontsize=8)
    ax.tick_params(labelsize=7)

    if ax_idx == 1:
        # Each strip: fill only the intersection with the catchment
        for i, (block_row, y_bot, y_top_blk) in enumerate(strips):
            color = cmap(i % 10)
            strip_catch = catch_clipped.intersection(sg.box(tile_x0, y_bot, tile_x1, y_top_blk))
            if strip_catch.is_empty:
                continue
            ax.add_patch(poly_to_patch(strip_catch, linewidth=0, facecolor=color, alpha=0.55, zorder=2))
            ax.text(tile_x1 - 1000, (y_bot + y_top_blk) / 2, f"54LWH_strip_{block_row:02d}",
                    ha="right", va="center", fontsize=7, color=color, fontweight="bold", zorder=5)

        # Catchment boundary
        ax.add_patch(poly_to_patch(catch_clipped,
                                   linewidth=1.5, edgecolor="black", facecolor="none", zorder=4))
        ax.set_title(
            "54LWH strips — pixels within Mitchell catchment\n"
            "Labels = absolute COG block row · zero over-fetch on HTTP range reads",
            fontsize=9,
        )
    else:
        ax.set_title(
            "54LWH · COG internal block grid\n11×11 blocks · 1024 px/block = 10.24 km/block",
            fontsize=9,
        )

plt.tight_layout()
out = "docs/diagrams/54LWH_cog_blocks/54LWH_cog_blocks.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
