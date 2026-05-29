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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import pickle
import re as _re

from utils.location import tile_chips_path  # noqa: F401 — re-exported for monkeypatching
from utils.regions import TrainingRegion, load_regions, select_regions
from utils.s2_tiles import bbox_to_tile_ids
from utils.stac import search_sentinel2

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
        mgrs_tile=tile_id,
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
    cloud_max: int,
    apply_nbar: bool,
    max_concurrent: int,
    n_sort_workers: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> str | None:
    """Fetch S2 and S1 for one region via FetchSpec, write region parquet.

    Returns the actual S2 tile_id the data was sourced from, or None if no data
    was written.
    """
    from utils.fetch_spec import FetchSpec, fetch_spec

    out_path  = _region_parquet_path(region.id)
    cache_dir = tile_chips_path(tile_id) / region.id
    bbox      = list(region.bbox)

    this_tile_items = [it for it in tile_items if tile_id in it.id]

    all_years = sorted(set(region.years))
    spec = FetchSpec(
        id=region.id,
        bbox=bbox,
        years=all_years,
        point_id_prefix=region.id,
        label=region.label,
        out_dir=_TILES_DIR / "regions" / region.id,
        cache_dir=cache_dir,
    )

    logger.info("Region %s: fetch start  bbox=%s  years=%s", region.id, bbox, all_years)

    year_results = fetch_spec(
        spec,
        cloud_max=cloud_max,
        apply_nbar=apply_nbar,
        max_concurrent=max_concurrent,
        items=this_tile_items,
        n_s1_workers=n_sort_workers or 4,
    )

    merged_paths: list[Path] = [p for paths in year_results.values() for p in paths]

    if not merged_paths:
        logger.warning("Region %s: no data — skipping", region.id)
        return None

    s2_paths = sorted(
        p for yr in all_years
        for p in (spec.out_dir / str(yr)).glob("*.s2.parquet")
        if (spec.out_dir / str(yr)).is_dir()
    )
    actual_tile_id = s2_paths[0].stem.replace(".s2", "") if s2_paths else tile_id

    if actual_tile_id != tile_id:
        logger.warning(
            "Region %s: S2 data from tile %s, not primary tile %s — indexing under %s",
            region.id, actual_tile_id, tile_id, actual_tile_id,
        )

    schema = _superset_schema(merged_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema)
    for mp in merged_paths:
        pf = pq.ParquetFile(mp)
        for rg_idx in range(pf.metadata.num_row_groups):
            tbl = _cast_to_schema(_normalise_source(pf.read_row_group(rg_idx)), schema)
            writer.write_table(tbl)
    writer.close()

    logger.info("Region %s: done (%d years)", region.id, len(merged_paths))
    return actual_tile_id


def _normalise_source(tbl: pa.Table) -> pa.Table:
    if "source" in tbl.schema.names:
        src_idx = tbl.schema.get_field_index("source")
        if tbl.schema.field("source").type != pa.string():
            tbl = tbl.set_column(src_idx, "source", tbl.column("source").cast(pa.string()))
    return tbl


def _superset_schema(paths: list[Path]) -> pa.Schema:
    """Return the union schema across all parquet files (metadata-only reads)."""
    schema: pa.Schema | None = None
    for p in paths:
        s = pq.read_schema(p)
        # Normalise source field type before merging.
        if "source" in s.names:
            idx = s.get_field_index("source")
            if s.field("source").type != pa.string():
                s = s.set(idx, pa.field("source", pa.string()))
        schema = s if schema is None else pa.unify_schemas([schema, s])
    return schema


def _cast_to_schema(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
    """Add any missing columns (as nulls) so tbl conforms to schema."""
    for i, field in enumerate(schema):
        if field.name not in tbl.schema.names:
            tbl = tbl.append_column(field, pa.nulls(len(tbl), type=field.type))
    return tbl.cast(schema)


def _write_region_rows(writer_ref: list, tile_path: Path, rp: Path, schema: pa.Schema) -> int:
    """Append all row-groups from a region parquet into writer_ref[0], returning row count."""
    pf = pq.ParquetFile(rp)
    rows = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = _cast_to_schema(_normalise_source(pf.read_row_group(rg_idx)), schema)
        if writer_ref[0] is None:
            writer_ref[0] = pq.ParquetWriter(tile_path, schema)
        writer_ref[0].write_table(tbl)
        rows += len(tbl)
    return rows


def _rebuild_tile_parquet(tile_id: str, new_region_ids: set[str] | None = None) -> None:
    """Rebuild (or incrementally update) the merged tile parquet for tile_id.

    If new_region_ids is given and the tile parquet already exists, performs an
    incremental update: streams the existing tile, dropping rows whose point_id
    prefix matches a changed region, then appends the fresh region parquets.
    This avoids re-reading the unchanged majority of regions on large tiles.

    Falls back to a full rebuild when the tile parquet is missing or
    new_region_ids is not supplied.
    """
    with _index_lock:
        all_indexed = _load_index()
        all_indexed_ids = set(
            all_indexed.filter(pl.col("tile_id") == tile_id)["region_id"].to_list()
        )

    tile_path = tile_parquet_path(tile_id)
    new_paths = [
        _region_parquet_path(rid)
        for rid in sorted(new_region_ids or [])
        if _region_parquet_path(rid).exists()
    ] if new_region_ids else []

    # --- Incremental path: tile exists and we know which regions changed ---
    if new_region_ids and tile_path.exists() and new_paths:
        tmp_path = tile_path.with_suffix(".tmp.parquet")
        logger.info(
            "Updating tile %s: replacing %d region(s) incrementally",
            tile_id, len(new_paths),
        )
        # Derive superset schema from existing tile + new region parquets.
        schema = _superset_schema([tile_path] + new_paths)
        writer_ref: list = [None]
        total_rows = 0
        prefixes = tuple(f"{rid}_" for rid in new_region_ids)
        pf = pq.ParquetFile(tile_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            tbl = _cast_to_schema(_normalise_source(pf.read_row_group(rg_idx)), schema)
            pid_col = tbl.column("point_id")
            drop = pc.starts_with(pid_col, prefixes[0])
            for p in prefixes[1:]:
                drop = pc.or_(drop, pc.starts_with(pid_col, p))
            tbl = tbl.filter(pc.invert(drop))
            if len(tbl):
                if writer_ref[0] is None:
                    writer_ref[0] = pq.ParquetWriter(tmp_path, schema)
                writer_ref[0].write_table(tbl)
                total_rows += len(tbl)
        for rp in new_paths:
            total_rows += _write_region_rows(writer_ref, tmp_path, rp, schema)
        if writer_ref[0] is not None:
            writer_ref[0].close()
            tmp_path.replace(tile_path)
        logger.info("Tile %s: %d rows (incremental)", tile_id, total_rows)
        return

    # --- Full rebuild ---
    region_paths = [
        _region_parquet_path(rid)
        for rid in sorted(all_indexed_ids)
        if _region_parquet_path(rid).exists()
    ]
    logger.info(
        "Building tile %s from %d region parquets → %s",
        tile_id, len(region_paths), tile_path.name,
    )
    schema = _superset_schema(region_paths)
    writer_ref = [None]
    total_rows = 0
    for rp in region_paths:
        total_rows += _write_region_rows(writer_ref, tile_path, rp, schema)
    if writer_ref[0] is not None:
        writer_ref[0].close()
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

    import math, os as _os
    # Divide CPU cores evenly across concurrent region workers so parallel
    # sort pools don't over-subscribe the machine.
    _n_sort_workers = max(1, math.ceil((_os.cpu_count() or 4) / max_region_workers))

    def _do_region(tile_id, region, tile_regions) -> None:
        # STAC search is per-tile and cached; the lock ensures only one worker
        # per tile does the live search — the rest load from the pickle.
        with tile_stac_locks[tile_id]:
            tile_items = _fetch_tile_items(tile_id, tile_regions, cloud_max)
        actual_tile = _collect_one_region(
            region=region,
            tile_id=tile_id,
            tile_items=tile_items,
            cloud_max=cloud_max,
            apply_nbar=apply_nbar,
            max_concurrent=max_concurrent,
            n_sort_workers=_n_sort_workers,
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
            new_ids = {r.id for r in tile_regions if r.id in submitted_region_ids}
            _rebuild_tile_parquet(tile_id, new_region_ids=new_ids if not tile_missing else None)
        # Ensure already-complete regions are indexed (idempotent).
        for region in tile_regions:
            if region.id not in submitted_region_ids:
                with _index_lock:
                    _update_index(region.id, [tile_id])

    logger.info("Done. Index: %s", _INDEX_PATH)
