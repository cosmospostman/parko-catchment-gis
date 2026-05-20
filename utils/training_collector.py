"""utils/training_collector.py — Collect training pixels organised by S2 tile.

Fetches Sentinel-2 observations for a set of TrainingRegions, writing one
parquet per S2 tile to data/training/tiles/{tile_id}.parquet, and maintaining
a sidecar index at data/training/index.parquet that maps region_id → tile_ids.

Each region is fetched independently using its own small bbox, so only the
pixels that fall within the actual training bbox are fetched from the COG.
Per-region parquets are concatenated into the tile parquet.

Use via cli/location.py:
    python cli/location.py training fetch --all
    python cli/location.py training fetch --regions lake_mueller_presence barcoorah_presence
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import pickle
import re as _re

from utils.regions import TrainingRegion, load_regions, select_regions
from utils.location import tile_chips_path
from utils.parquet_utils import append_s1_to_tile_parquet
from utils.pixel_collector import collect
from utils.s2_tiles import bbox_to_tile_ids
from utils.stac import search_sentinel2
from utils.s1_collector import collect_s1, _DEFAULT_CACHE_DIR as _S1_CHIP_CACHE_DIR

logger = logging.getLogger(__name__)


_TRAINING_DIR = PROJECT_ROOT / "data" / "training"
_TILES_DIR    = _TRAINING_DIR / "tiles"
_INDEX_PATH   = _TRAINING_DIR / "index.parquet"


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _load_index() -> pl.DataFrame:
    """Load the region→tile sidecar index, or return an empty frame."""
    if _INDEX_PATH.exists():
        return pl.read_parquet(_INDEX_PATH)
    return pl.DataFrame({"region_id": pl.Series([], dtype=pl.Utf8),
                          "tile_id":   pl.Series([], dtype=pl.Utf8)})


def _save_index(df: pl.DataFrame) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_INDEX_PATH)


def _update_index(region_id: str, tile_ids: list[str]) -> None:
    """Add or refresh the index entries for one region."""
    df = _load_index()
    df = df.filter(pl.col("region_id") != region_id)  # remove stale entries
    if not tile_ids:
        raise ValueError(
            f"_update_index: region {region_id!r} maps to zero tiles — "
            "check that bbox_to_tile_ids() returns at least one tile."
        )
    new_rows = pl.DataFrame({"region_id": [region_id] * len(tile_ids), "tile_id": tile_ids})
    df = pl.concat([df, new_rows])
    _save_index(df)


def tile_ids_for_regions(region_ids: list[str]) -> list[str]:
    """Return the deduplicated tile IDs that cover the given region IDs.

    Consults the sidecar index — regions must have been collected first.
    """
    df = _load_index()
    hits = df.filter(pl.col("region_id").is_in(region_ids))
    if hits.is_empty():
        raise RuntimeError(
            f"No tile index entries found for regions {region_ids}. "
            "Run 'ensure-training-pixels' first."
        )
    return sorted(hits["tile_id"].unique().to_list())


def tile_parquet_path(tile_id: str) -> Path:
    return _TILES_DIR / f"{tile_id}.parquet"


def _region_parquet_path(region_id: str) -> Path:
    return _TILES_DIR / "regions" / f"{region_id}.parquet"


_STAC_ENDPOINT  = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION  = "sentinel-2-l2a"
_GRANULE_RE     = _re.compile(r"_\d+_L2A$")


def _best_tile_for_region(region: "TrainingRegion", tile_ids: list[str]) -> str:
    """Return the tile whose footprint contains the region centroid, or the first tile."""
    from shapely.geometry import Point
    from utils.s2_tiles import get_au_tile_grid
    cx = (region.bbox[0] + region.bbox[2]) / 2
    cy = (region.bbox[1] + region.bbox[3]) / 2
    centroid = Point(cx, cy)
    grid = get_au_tile_grid()
    tile_map = dict(grid)
    for tile_id in tile_ids:
        geom = tile_map.get(tile_id)
        if geom is not None and geom.contains(centroid):
            return tile_id
    return tile_ids[0]


def _union_bbox(regions: list[TrainingRegion]) -> list[float]:
    lon_min = min(r.bbox[0] for r in regions)
    lat_min = min(r.bbox[1] for r in regions)
    lon_max = max(r.bbox[2] for r in regions)
    lat_max = max(r.bbox[3] for r in regions)
    return [lon_min, lat_min, lon_max, lat_max]


def _fetch_tile_items(
    tile_id: str,
    tile_regions: list[TrainingRegion],
    cloud_max: int,
) -> list:
    """Search STAC once for all regions on a tile; return deduplicated items.

    Result is cached under data/training/stac/ so re-runs are instant.
    """
    start, end = _tile_date_window(tile_regions)
    union = _union_bbox(tile_regions)
    stac_key = hashlib.md5(
        f"{union}|{start}|{end}|{cloud_max}".encode()
    ).hexdigest()
    stac_dir = _TRAINING_DIR / "stac"
    stac_dir.mkdir(parents=True, exist_ok=True)
    stac_cache = stac_dir / f"{tile_id}_{stac_key}.pkl"

    if stac_cache.exists():
        with stac_cache.open("rb") as fh:
            items = pickle.load(fh)
        logger.info(
            "Tile %s: %d STAC items loaded from cache (%s)",
            tile_id, len(items), stac_cache.name,
        )
        return items

    logger.info(
        "Tile %s: STAC search  %s → %s  cloud < %d%%  bbox=%s",
        tile_id, start, end, cloud_max, union,
    )
    raw = search_sentinel2(
        bbox=union,
        start=start,
        end=end,
        cloud_cover_max=cloud_max,
        endpoint=_STAC_ENDPOINT,
        collection=_S2_COLLECTION,
    )
    if not raw:
        raise RuntimeError(f"No STAC items found for tile {tile_id} — check bboxes and date range")

    seen: set[str] = set()
    items = []
    for item in raw:
        key = _GRANULE_RE.sub("", item.id)
        if key not in seen:
            seen.add(key)
            items.append(item)

    logger.info(
        "Tile %s: %d items → %d deduplicated",
        tile_id, len(raw), len(items),
    )
    with stac_cache.open("wb") as fh:
        pickle.dump(items, fh)
    return items


def _tile_date_window(tile_regions: list[TrainingRegion]) -> tuple[str, str]:
    """Derive fetch window from the explicit years lists across all regions on this tile."""
    all_years = [yr for r in tile_regions for yr in r.years]
    if not all_years:
        raise ValueError(
            f"Regions {[r.id for r in tile_regions]} have empty years lists"
        )
    return f"{min(all_years)}-01-01", f"{max(all_years)}-12-31"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()


def _collect_one_region(
    region: TrainingRegion,
    tile_id: str,
    tile_items: list,
    start: str,
    end: str,
    cloud_max: int,
    apply_nbar: bool,
    max_concurrent: int,
) -> str | None:
    """Fetch S2 and S1 concurrently for one region, write region parquet.

    S2 and S1 COG fetches are launched in parallel threads since they hit
    independent endpoints (AWS us-west-2 for S2, Azure West Europe for S1)
    and share no mutable state.  S1 only needs the pixel grid, which is
    derived from the bbox without waiting for S2 data.

    Returns the actual S2 tile_id the data was sourced from (which may differ
    from the ``tile_id`` argument when the region straddles a tile boundary and
    no items matched the primary tile), or None if no data was written.
    """
    from utils.pixel_collector import make_pixel_grid

    out_path    = _region_parquet_path(region.id)
    collect_dir = _TILES_DIR / "regions" / f"{region.id}.collect"
    # Use a per-region chip cache, not the shared per-tile one. Parallel workers
    # on the same tile would otherwise race — each writing patches for its own
    # small bbox, then the stale-check deleting the other worker's patches.
    cache_dir   = tile_chips_path(tile_id) / region.id
    bbox = list(region.bbox)

    logger.info("Region %s: S2+S1 fetch start  bbox=%s  %s→%s", region.id, bbox, start, end)

    # Derive the pixel grid immediately — needed by both S2 (point_id_prefix)
    # and S1 (points list), and costs nothing to compute.
    points = make_pixel_grid(bbox_wgs84=bbox, point_id_prefix=region.id)

    s2_tile_paths: list[Path] = []
    s2_exc: BaseException | None = None
    s1_exc: BaseException | None = None

    def _fetch_s2() -> None:
        nonlocal s2_tile_paths, s2_exc
        try:
            # Filter to items from this tile only. tile_items comes from a STAC
            # search over the union bbox, which may include adjacent tiles when a
            # region straddles a tile boundary. Mixing items from different MGRS
            # tiles into one cache_dir breaks CachedNpzChipStore: each tile has
            # its own CRS/transform, so point projections from one tile are
            # invalid for patches fetched for another.
            this_tile_items = [it for it in tile_items if tile_id in it.id]
            tile_paths = collect(
                bbox_wgs84=bbox,
                start=start,
                end=end,
                out_dir=collect_dir,
                cloud_max=cloud_max,
                cache_dir=cache_dir,
                apply_nbar=apply_nbar,
                max_concurrent=max_concurrent,
                items=this_tile_items,
                point_id_prefix=region.id,
            )
            if not tile_paths:
                tile_paths = sorted(collect_dir.glob("*.parquet"))
                if tile_paths:
                    logger.info(
                        "Region %s: collect() returned no paths but found %d existing "
                        "parquet(s) in collect dir — using those",
                        region.id, len(tile_paths),
                    )
            s2_tile_paths = tile_paths
        except Exception as exc:
            s2_exc = exc

    def _fetch_s1() -> None:
        nonlocal s1_exc
        try:
            # S1 fetch runs inside append_s1_to_tile_parquet after S2 writes
            # out_path, so we only pre-warm the S1 STAC cache here.  The actual
            # band reads happen after S2 completes (they need the parquet schema).
            # Warming the cache now means the STAC search round-trip to Planetary
            # Computer overlaps with the S2 COG fetch rather than following it.
            from utils.s1_collector import _resolve_s1_items
            _resolve_s1_items(bbox, start, end, _S1_CHIP_CACHE_DIR)
        except Exception as exc:
            s1_exc = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        s2_future = pool.submit(_fetch_s2)
        s1_future = pool.submit(_fetch_s1)
        s2_future.result()  # propagates exception if _fetch_s2 raised
        s1_future.result()  # propagates exception if _fetch_s1 raised

    if s2_exc:
        raise s2_exc
    if s1_exc:
        logger.warning("Region %s: S1 STAC pre-warm failed: %s", region.id, s1_exc)

    if not s2_tile_paths:
        logger.warning("Region %s: no S2 data — skipping", region.id)
        return None

    # Detect the actual S2 tile from the first shard filename (e.g. "54KZC.parquet").
    # This may differ from tile_id when the region straddles a boundary and STAC
    # had no items for the primary tile — the fallback uses a neighbouring tile's
    # cached shards.  We index under the actual tile so _rebuild_tile_parquet
    # finds the region parquet.
    actual_tile_id = s2_tile_paths[0].stem
    if actual_tile_id != tile_id:
        logger.warning(
            "Region %s: S2 data came from tile %s, not primary tile %s — "
            "indexing under %s",
            region.id, actual_tile_id, tile_id, actual_tile_id,
        )

    # Merge S2 shards into region parquet, then append S1 (STAC already cached).
    writer = pq.ParquetWriter(out_path, pq.ParquetFile(s2_tile_paths[0]).schema_arrow)
    for tp in s2_tile_paths:
        pf = pq.ParquetFile(tp)
        for rg_idx in range(pf.metadata.num_row_groups):
            writer.write_table(pf.read_row_group(rg_idx))
    writer.close()

    append_s1_to_tile_parquet(
        tile_path=out_path,
        bbox_wgs84=bbox,
        start=start,
        end=end,
        collect_s1_fn=collect_s1,
        s1_cache_dir=_S1_CHIP_CACHE_DIR,
    )

    logger.info("Region %s: done", region.id)
    return actual_tile_id


def _rebuild_tile_parquet(tile_id: str) -> None:
    """Concatenate all collected region parquets for tile_id into the tile parquet."""
    with _index_lock:
        all_indexed = _load_index()
        all_indexed_ids = set(
            all_indexed.filter(pl.col("tile_id") == tile_id)["region_id"].to_list()
        )

    region_paths = [
        _region_parquet_path(rid)
        for rid in sorted(all_indexed_ids)
        if _region_parquet_path(rid).exists()
    ]
    tile_path = tile_parquet_path(tile_id)
    logger.info(
        "Building tile %s from %d region parquets → %s",
        tile_id, len(region_paths), tile_path.name,
    )
    writer = None
    total_rows = 0
    for rp in region_paths:
        pf = pq.ParquetFile(rp)
        for rg_idx in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg_idx)
            if writer is None:
                writer = pq.ParquetWriter(tile_path, tbl.schema)
            writer.write_table(tbl)
            total_rows += len(tbl)
    if writer is not None:
        writer.close()
    logger.info("Tile %s: %d rows", tile_id, total_rows)


def ensure_training_pixels(
    regions: list[TrainingRegion],
    cloud_max: int = 80,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    max_region_workers: int = 4,
) -> None:
    """Ensure pixel parquets exist for all tiles covered by the given regions.

    For each region:
      1. Fetches S2 COGs and pre-warms the S1 STAC cache concurrently.
      2. Writes data/training/tiles/regions/{region_id}.parquet (S2 + S1).
      3. After all regions on a tile finish, rebuilds the tile parquet and
         updates the sidecar index.

    Regions on different tiles are fetched in parallel (up to max_region_workers
    at a time).  Within each region, S2 and S1 STAC searches overlap.

    Skips regions whose parquet already exists (idempotent).
    """
    # Group regions by tile.  For each region, record which tile should own it
    # (the one containing its centroid) so multi-tile regions aren't submitted
    # under the wrong tile (e.g. a sliver tile whose patches are too small).
    region_primary_tile: dict[str, str] = {}
    tile_to_regions: dict[str, list[TrainingRegion]] = {}
    for region in regions:
        tile_ids = bbox_to_tile_ids(region.bbox_tuple)
        region_primary_tile[region.id] = _best_tile_for_region(region, tile_ids)
        for tile_id in tile_ids:
            tile_to_regions.setdefault(tile_id, []).append(region)

    if not tile_to_regions:
        logger.error("No S2 tiles found for the given regions — check bboxes")
        return

    logger.info(
        "%d regions → %d S2 tiles: %s",
        len(regions), len(tile_to_regions), sorted(tile_to_regions),
    )

    _TILES_DIR.mkdir(parents=True, exist_ok=True)
    (_TILES_DIR / "regions").mkdir(parents=True, exist_ok=True)

    # Track which tiles have new regions so we know what to rebuild afterwards.
    tiles_with_new: set[str] = set()
    tiles_with_new_lock = threading.Lock()
    submitted_region_ids: set[str] = set()

    # One lock per tile guards the shared STAC cache file so concurrent workers
    # on the same tile don't race to write it.
    tile_stac_locks: dict[str, threading.Lock] = {
        tile_id: threading.Lock() for tile_id in tile_to_regions
    }

    def _do_region(tile_id, region, tile_regions) -> None:
        start, end = _tile_date_window(tile_regions)
        # STAC search is per-tile and cached; the lock ensures only one worker
        # per tile does the live search — the rest load from the pickle.
        with tile_stac_locks[tile_id]:
            tile_items = _fetch_tile_items(tile_id, tile_regions, cloud_max)
        actual_tile = _collect_one_region(
            region=region,
            tile_id=tile_id,
            tile_items=tile_items,
            start=start,
            end=end,
            cloud_max=cloud_max,
            apply_nbar=apply_nbar,
            max_concurrent=max_concurrent,
        )
        if actual_tile is not None:
            with tiles_with_new_lock:
                tiles_with_new.add(actual_tile)
            # Update the index as soon as this region is done so a crash
            # mid-run doesn't lose track of already-completed regions.
            with _index_lock:
                _update_index(region.id, [actual_tile])

    with ThreadPoolExecutor(max_workers=max_region_workers) as pool:
        futures: dict = {}
        for tile_id, tile_regions in sorted(tile_to_regions.items()):
            pending = [r for r in tile_regions if not _region_parquet_path(r.id).exists()]
            for r in tile_regions:
                if r not in pending:
                    logger.info("Region %s already collected — skipping", r.id)
            for region in pending:
                if region.id in submitted_region_ids:
                    # Region spans multiple tiles — already submitted under its
                    # primary tile (centroid-containing). Skip other tiles.
                    continue
                if region_primary_tile[region.id] != tile_id:
                    # This tile is not the primary for this region — it will be
                    # submitted when we reach the primary tile in sorted order.
                    continue
                submitted_region_ids.add(region.id)
                futures[pool.submit(_do_region, tile_id, region, tile_regions)] = region.id

        if not futures:
            logger.info("All regions already collected.")
        else:
            logger.info(
                "Fetching %d region(s) with up to %d parallel workers",
                len(futures), max_region_workers,
            )

        for future in as_completed(futures):
            region_id = futures[future]
            exc = future.exception()
            if exc:
                logger.error("Region %s failed: %s", region_id, exc, exc_info=exc)

    # Rebuild tile parquets for any tile that got new regions, and update the
    # index for already-complete regions that weren't in the work list.
    for tile_id, tile_regions in sorted(tile_to_regions.items()):
        tile_missing = not tile_parquet_path(tile_id).exists()
        if tile_id in tiles_with_new or tile_missing:
            _rebuild_tile_parquet(tile_id)
        # Ensure already-complete regions are indexed (idempotent).
        for region in tile_regions:
            if region.id not in submitted_region_ids:
                with _index_lock:
                    _update_index(region.id, [tile_id])

    logger.info("Done. Index: %s", _INDEX_PATH)
