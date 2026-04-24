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


def geometry_to_tile_ids(
    geometry,
    bbox: tuple[float, float, float, float],
) -> list[str]:
    """Return S2 MGRS tile IDs needed to cover all catchment pixels.

    A tile can be skipped if every part of the catchment that falls within it
    also falls within the overlap zone of at least one neighbouring tile — because
    those pixels will be read from the neighbour instead.

    The algorithm iterates to convergence: removing a tile may expose another tile
    as now-entirely-covered, so we repeat until the set stabilises.

    Parameters
    ----------
    geometry:
        Shapely geometry (polygon or multipolygon) of the catchment.
    bbox:
        (lon_min, lat_min, lon_max, lat_max) envelope of *geometry*.

    Returns
    -------
    list[str]
        Sorted tile IDs that will actually produce observations.
    """
    from shapely.geometry import box

    grid = get_au_tile_grid()
    name_col = "Name" if "Name" in grid.columns else "name"

    # Candidates: tiles whose footprint intersects the catchment polygon itself
    # (not just the bbox envelope).
    candidates = grid[grid.geometry.intersects(geometry)]
    tile_geoms: dict[str, object] = {
        row[name_col]: row.geometry for _, row in candidates.iterrows()
    }

    if len(tile_geoms) <= 1:
        return sorted(tile_geoms)

    # Precompute: catchment portion inside each tile.
    catchment_in: dict[str, object] = {
        tid: tg.intersection(geometry) for tid, tg in tile_geoms.items()
    }
    # Precompute: pairwise overlap between tiles (symmetric, skip empties).
    tile_ids = sorted(tile_geoms)
    overlaps: dict[str, list[str]] = {tid: [] for tid in tile_ids}
    for i, a in enumerate(tile_ids):
        for b in tile_ids[i + 1:]:
            ov = tile_geoms[a].intersection(tile_geoms[b])
            if not ov.is_empty:
                overlaps[a].append(b)
                overlaps[b].append(a)

    # Iteratively drop tiles whose entire catchment intersection is covered by
    # the union of overlap zones with tiles that are still in the kept set.
    kept = set(tile_ids)
    changed = True
    while changed:
        changed = False
        for tid in sorted(kept):   # deterministic order
            ci = catchment_in[tid]
            if ci.is_empty:
                kept.discard(tid)
                changed = True
                continue
            # Union of overlap zones between this tile and each kept neighbour.
            covered = None
            for nbr in overlaps[tid]:
                if nbr not in kept:
                    continue
                ov = tile_geoms[tid].intersection(tile_geoms[nbr])
                covered = ov if covered is None else covered.union(ov)
            if covered is None:
                continue
            # If the catchment inside this tile is entirely within the covered
            # zone, this tile adds nothing that a neighbour won't provide.
            unique = ci.difference(covered)
            if unique.is_empty:
                kept.discard(tid)
                changed = True
                logger.debug("geometry_to_tile_ids: dropping %s (entirely in overlap zone)", tid)

    skipped = [t for t in tile_ids if t not in kept]
    if skipped:
        logger.info(
            "geometry_to_tile_ids: skipping %d tile(s) whose catchment pixels "
            "are covered by neighbours: %s",
            len(skipped), skipped,
        )
    return sorted(kept)
