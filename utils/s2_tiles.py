"""utils/s2_tiles.py — S2 MGRS tile grid, fetch-and-cache.

The full global S2 tile grid (~56k tiles) is fetched once from a public mirror,
clipped to Australia, and cached at data/cache/s2_tiles_au.parquet.  Subsequent
calls load from the cache — no network access needed.

Public source: Sentinel-2 MGRS tiling grid published by ESA/Sinergise.
Fetched from Zenodo (DOI 10.5281/zenodo.10998972) with CDN fallback.

Each record has two fields:
    name     : str     — MGRS tile ID e.g. '54LWH'
    geometry : object  — shapely Polygon (EPSG:4326)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import shapely.wkb
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "s2_tiles_au.parquet"

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

# Internal representation: list of (name, shapely_geometry)
_TileRow = tuple[str, object]


def _load_parquet(path: Path) -> list[_TileRow]:
    import polars as pl
    df = pl.read_parquet(path)
    return [
        (row["name"], shapely.wkb.loads(bytes(row["geometry"])))
        for row in df.iter_rows(named=True)
    ]


def _save_parquet(rows: list[_TileRow], path: Path) -> None:
    import polars as pl
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "name": [r[0] for r in rows],
        "geometry": [shapely.wkb.dumps(r[1]) for r in rows],
    })
    df.write_parquet(path)


def _fetch_and_clip() -> list[_TileRow]:
    """Download the global tile grid, clip to Australia, return rows."""
    import urllib.request

    au_box = box(*_AU_BBOX)
    last_exc: Exception | None = None

    for url in _GRID_URLS:
        logger.info("S2 tile grid not cached — fetching from %s", url)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            break
        except Exception as exc:
            logger.warning("Failed to fetch S2 tile grid from %s: %s", url, exc)
            last_exc = exc
    else:
        raise RuntimeError(
            f"Could not fetch S2 tile grid from any source: {_GRID_URLS}. "
            "Check your internet connection or manually place the file at "
            f"{_CACHE_PATH}."
        ) from last_exc

    fc = json.loads(raw)
    features = fc.get("features", fc) if isinstance(fc, dict) else fc

    rows: list[_TileRow] = []
    for feat in features:
        props = feat.get("properties") or {}
        name = props.get("Name") or props.get("name") or props.get("NAME", "")
        geom = shape(feat["geometry"])
        if geom.intersects(au_box):
            rows.append((name, geom))

    return rows


def get_au_tile_grid() -> list[_TileRow]:
    """Return S2 MGRS tile grid clipped to Australia.

    Returns a list of (name, geometry) tuples where name is the MGRS tile ID
    (e.g. '54LWH') and geometry is a Shapely Polygon in EPSG:4326.

    Fetches and caches on first call; subsequent calls load from disk.
    """
    if _CACHE_PATH.exists():
        return _load_parquet(_CACHE_PATH)

    rows = _fetch_and_clip()
    _save_parquet(rows, _CACHE_PATH)
    logger.info("Cached %d AU S2 tiles to %s", len(rows), _CACHE_PATH)
    return rows


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
    query_box = box(*bbox)
    grid = get_au_tile_grid()
    return sorted(name for name, geom in grid if geom.intersects(query_box))


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
    grid = get_au_tile_grid()

    # Candidates: tiles whose footprint intersects the catchment polygon itself
    tile_geoms: dict[str, object] = {
        name: geom for name, geom in grid if geom.intersects(geometry)
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
        for tid in sorted(kept):
            ci = catchment_in[tid]
            if ci.is_empty:
                kept.discard(tid)
                changed = True
                continue
            covered = None
            for nbr in overlaps[tid]:
                if nbr not in kept:
                    continue
                ov = tile_geoms[tid].intersection(tile_geoms[nbr])
                covered = ov if covered is None else covered.union(ov)
            if covered is None:
                continue
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
