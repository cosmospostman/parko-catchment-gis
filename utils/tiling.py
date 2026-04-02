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

    bboxes = []
    for row in range(n_rows):
        for col in range(n_cols):
            t_minx = proj_minx + col * tile_m
            t_miny = proj_miny + row * tile_m
            t_maxx = min(t_minx + tile_m, proj_maxx)
            t_maxy = min(t_miny + tile_m, proj_maxy)

            # Back-project all four corners and take the geographic envelope.
            corners_x, corners_y = to_geo.transform(
                [t_minx, t_maxx, t_minx, t_maxx],
                [t_miny, t_miny, t_maxy, t_maxy],
            )
            g_minx = min(corners_x)
            g_maxx = max(corners_x)
            g_miny = min(corners_y)
            g_maxy = max(corners_y)

            # Clamp every tile to full_bbox.  Interior tiles may bulge slightly
            # beyond the full extent due to CRS nonlinearity; clamping keeps them
            # within bounds.  Outermost edges are snapped exactly so the union of
            # all tiles equals full_bbox precisely (no gap at the boundary).
            g_minx = minx_geo if col == 0         else max(g_minx, minx_geo)
            g_maxx = maxx_geo if col == n_cols - 1 else min(g_maxx, maxx_geo)
            g_miny = miny_geo if row == 0         else max(g_miny, miny_geo)
            g_maxy = maxy_geo if row == n_rows - 1 else min(g_maxy, maxy_geo)

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
    """Mosaic tile GeoTIFFs into a single COG using gdalbuildvrt + gdal_translate.

    Uses GDAL command-line tools so pixel data is never fully loaded into RAM —
    gdalbuildvrt writes an XML index and gdal_translate streams tiles to the output.
    """
    import subprocess
    import tempfile

    valid = [str(p) for p in tile_paths if p is not None]
    if not valid:
        raise ValueError("merge_tile_rasters: no valid tile paths provided")

    nodata_str = str(nodata) if nodata == nodata else "nan"  # handle float nan

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as vrt_file:
        vrt_path = vrt_file.name

    try:
        r1 = subprocess.run(
            ["gdalbuildvrt", "-vrtnodata", nodata_str, vrt_path] + valid,
            capture_output=True,
        )
        if r1.returncode != 0:
            raise subprocess.CalledProcessError(
                r1.returncode, r1.args, r1.stdout, r1.stderr
            )
        r2 = subprocess.run(
            [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "COMPRESS=DEFLATE",
                "-co", "TILED=YES",
                "-co", "BLOCKXSIZE=512",
                "-co", "BLOCKYSIZE=512",
                "-co", "BIGTIFF=IF_SAFER",
                "-ot", "Float32",
                "-a_nodata", nodata_str,
                vrt_path, str(out_path),
            ],
            capture_output=True,
        )
        if r2.returncode != 0:
            logger.error("gdal_translate stderr:\n%s", r2.stderr.decode(errors="replace"))
            raise subprocess.CalledProcessError(
                r2.returncode, r2.args, r2.stdout, r2.stderr
            )
    finally:
        Path(vrt_path).unlink(missing_ok=True)

    logger.info("merge_tile_rasters: written %s (%d tiles)", out_path, len(valid))
