"""utils/s2_tiles.py — S2 MGRS tile grid, fetch-and-cache.

The full global S2 tile grid (~56k tiles) is fetched once from a public mirror,
clipped to Australia, and cached at data/cache/s2_tiles_au.gpkg.  Subsequent
calls load from the cache — no network access needed.

Public source: Sentinel-2 MGRS tiling grid published by ESA/Sinergise,
mirrored at https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "s2_tiles_au.gpkg"

# Australia bounding box (generous margin)
_AU_BBOX = (110.0, -45.0, 155.0, -10.0)

# GeoJSON source — global S2 tiling grid (~25 MB download, one-time)
_GRID_URL = (
    "https://raw.githubusercontent.com/justinelliotmeyers/"
    "Sentinel-2-Shapefile-Index/master/Sentinel_2_tiling_grid.geojson"
)


def get_au_tile_grid() -> "geopandas.GeoDataFrame":
    """Return S2 MGRS tile grid clipped to Australia as a GeoDataFrame.

    Fetches and caches on first call; subsequent calls load from disk.
    Columns: Name (tile ID e.g. '54LWH'), geometry (Polygon, EPSG:4326).
    """
    import geopandas as gpd

    if _CACHE_PATH.exists():
        return gpd.read_file(_CACHE_PATH)

    logger.info("S2 tile grid not cached — fetching from %s", _GRID_URL)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        grid = gpd.read_file(_GRID_URL)
    except Exception as exc:
        logger.error("Failed to fetch S2 tile grid: %s", exc)
        raise RuntimeError(
            f"Could not fetch S2 tile grid from {_GRID_URL}. "
            "Check your internet connection or manually place the file at "
            f"{_CACHE_PATH}."
        ) from exc

    west, south, east, north = _AU_BBOX
    from shapely.geometry import box
    au_box = box(west, south, east, north)
    au_grid = grid[grid.geometry.intersects(au_box)].copy()
    au_grid = au_grid.reset_index(drop=True)

    au_grid.to_file(_CACHE_PATH, driver="GPKG")
    logger.info(
        "Cached %d AU S2 tiles to %s", len(au_grid), _CACHE_PATH
    )
    return au_grid


def bbox_to_tile_ids(bbox: tuple[float, float, float, float]) -> list[str]:
    """Return S2 MGRS tile IDs that intersect the given bbox.

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.

    Returns
    -------
    list[str]
        Tile IDs e.g. ['54LWH', '54LWJ'].  Empty if no tiles intersect.
    """
    from shapely.geometry import box

    grid = get_au_tile_grid()
    query_box = box(*bbox)

    # Column may be 'Name' or 'name' depending on source version
    name_col = "Name" if "Name" in grid.columns else "name"
    hits = grid[grid.geometry.intersects(query_box)]
    return sorted(hits[name_col].tolist())
