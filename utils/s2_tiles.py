"""utils/s2_tiles.py — S2 MGRS tile grid, fetch-and-cache.

The full global S2 tile grid (~56k tiles) is fetched once from a public mirror,
clipped to Australia, and cached at data/cache/s2_tiles_au.gpkg.  Subsequent
calls load from the cache — no network access needed.

Public source: Sentinel-2 MGRS tiling grid published by ESA/Sinergise.
Fetched from Zenodo (DOI 10.5281/zenodo.10998972) with CDN fallback.
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
# Tried in order; first success wins.
_GRID_URLS = [
    # Zenodo — persistent DOI, most reliable
    "https://zenodo.org/records/10998972/files/sentinel2_tiling_grid_wgs84.geojson?download=1",
    # CDN mirror via bencevans/sentinel-2-grid npm package
    "https://unpkg.com/sentinel-2-grid/data/grid.json",
]


def get_au_tile_grid() -> "geopandas.GeoDataFrame":
    """Return S2 MGRS tile grid clipped to Australia as a GeoDataFrame.

    Fetches and caches on first call; subsequent calls load from disk.
    Columns: Name (tile ID e.g. '54LWH'), geometry (Polygon, EPSG:4326).
    """
    import geopandas as gpd

    if _CACHE_PATH.exists():
        return gpd.read_file(_CACHE_PATH)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    grid = None
    last_exc: Exception | None = None
    for url in _GRID_URLS:
        logger.info("S2 tile grid not cached — fetching from %s", url)
        try:
            grid = gpd.read_file(url)
            break
        except Exception as exc:
            logger.warning("Failed to fetch S2 tile grid from %s: %s", url, exc)
            last_exc = exc

    if grid is None:
        raise RuntimeError(
            f"Could not fetch S2 tile grid from any source: {_GRID_URLS}. "
            "Check your internet connection or manually place the file at "
            f"{_CACHE_PATH}."
        ) from last_exc

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
