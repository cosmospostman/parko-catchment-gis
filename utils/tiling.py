"""utils/tiling.py — Spatial tiling helpers for memory-bounded raster processing.

Tiles are computed in projected space (EPSG:7855) then converted back to
EPSG:4326 for passing to stackstac (bounds_latlon) and odc-stac (bbox).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_tile_bboxes(
    full_bbox: list[float],   # [minx, miny, maxx, maxy] in degrees (EPSG:4326)
    resolution_m: int,        # e.g. 10
    tile_size_px: int,        # e.g. 256
) -> list[list[float]]:
    """Return a list of [minx, miny, maxx, maxy] tile bboxes covering full_bbox.

    Tiles are computed in projected space (EPSG:7855) then converted back to
    EPSG:4326.  Tiles at the edges are clipped to the full_bbox extent so every
    returned bbox lies within full_bbox.
    """
    from pyproj import Transformer

    to_proj = Transformer.from_crs("EPSG:4326", "EPSG:7855", always_xy=True)
    to_geo  = Transformer.from_crs("EPSG:7855",  "EPSG:4326", always_xy=True)

    minx_geo, miny_geo, maxx_geo, maxy_geo = full_bbox

    # Project corners to EPSG:7855
    (proj_minx, proj_maxx), (proj_miny, proj_maxy) = to_proj.transform(
        [minx_geo, maxx_geo], [miny_geo, maxy_geo]
    )

    tile_m = resolution_m * tile_size_px  # tile side length in metres

    # Build grid rows/cols in projected space
    import math
    n_cols = math.ceil((proj_maxx - proj_minx) / tile_m)
    n_rows = math.ceil((proj_maxy - proj_miny) / tile_m)

    # Guard: at least one tile even if extent is smaller than one tile
    n_cols = max(n_cols, 1)
    n_rows = max(n_rows, 1)

    # Pre-compute all projected grid-line positions and back-project them in a
    # single batch call.  Because each shared edge is represented by exactly one
    # value in the batch, adjacent tiles receive the *same* geographic coordinate
    # for that edge, eliminating floating-point overlap between neighbours.
    col_edges_proj = [proj_minx + i * tile_m for i in range(n_cols)] + [proj_maxx]
    row_edges_proj = [proj_miny + i * tile_m for i in range(n_rows)] + [proj_maxy]

    # Back-project along the left column (constant x = proj_minx representative)
    # and the bottom row to get y-edges and x-edges respectively.
    # We only need the scalar back-projection of each edge at a reference point
    # on the opposite axis (the midpoint), which gives a good geographic
    # approximation for the grid lines.
    mid_y_proj = (proj_miny + proj_maxy) / 2
    mid_x_proj = (proj_minx + proj_maxx) / 2

    col_edges_geo_x, _ = to_geo.transform(col_edges_proj,
                                           [mid_y_proj] * len(col_edges_proj))
    _, row_edges_geo_y  = to_geo.transform([mid_x_proj] * len(row_edges_proj),
                                            row_edges_proj)

    bboxes = []
    for row in range(n_rows):
        for col in range(n_cols):
            g_minx = col_edges_geo_x[col]
            g_maxx = col_edges_geo_x[col + 1]
            g_miny = row_edges_geo_y[row]
            g_maxy = row_edges_geo_y[row + 1]

            # Clip to full_bbox (guard against floating-point overshoot at edges)
            g_minx = max(g_minx, minx_geo)
            g_miny = max(g_miny, miny_geo)
            g_maxx = min(g_maxx, maxx_geo)
            g_maxy = min(g_maxy, maxy_geo)

            bboxes.append([g_minx, g_miny, g_maxx, g_maxy])

    logger.debug(
        "make_tile_bboxes: %d tiles (%d cols × %d rows) at %d m × %d px",
        len(bboxes), n_cols, n_rows, resolution_m, tile_size_px,
    )
    return bboxes


def merge_tile_rasters(
    tile_paths: list[Path],
    out_path: Path,
    nodata: float,
    crs: str,
) -> None:
    """Mosaic tile GeoTIFFs into a single COG using rioxarray.merge_arrays."""
    import numpy as np
    import rioxarray  # noqa: F401
    import xarray as xr
    from rioxarray.merge import merge_arrays

    arrays = []
    for p in tile_paths:
        if p is None:
            continue
        da = xr.open_dataarray(str(p))
        da = da.rio.write_crs(crs)
        arrays.append(da)

    if not arrays:
        raise ValueError("merge_tile_rasters: no valid tile paths provided")

    merged = merge_arrays(arrays, nodata=nodata)
    merged = merged.rio.write_crs(crs)
    merged = merged.rio.write_nodata(nodata)

    merged.rio.to_raster(
        str(out_path),
        driver="GTiff",
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        dtype="float32",
    )
    logger.info("merge_tile_rasters: written %s (%d tiles)", out_path, len(arrays))
