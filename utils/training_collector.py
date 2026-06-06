"""utils/training_collector.py — Collect training pixels organised by S2 tile.

Fetches Sentinel-2 + Sentinel-1 observations for a set of TrainingRegions by
reusing the main tile fetch pipeline (fetch_tile_local / run_tile_pipeline_v2).

For each S2 tile that has training regions:
  1. Build a MultiPolygon of the union of all region bboxes on the tile.
  2. Call fetch_tile_local(), which uses compute_chunks() to select only the
     COG-block-aligned chunks that intersect at least one region bbox — no empty
     space between scattered regions is fetched.
  3. Chunks are written to the chunkstore: {chunkstore}/{year}/{tile_id}/...
  4. Extract each training region from the chunkstore via ChunkIndex.query_bbox(),
     filtering rows to the region's own year list.
  5. Write per-region parquets to data/training/tiles/regions/{region_id}.parquet.
  6. Rebuild per-tile parquets by concatenating region parquets.

Use via cli/location.py:
    python cli/location.py training fetch --all
    python cli/location.py training fetch --regions lake_mueller_presence barcoorah_presence
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from utils.regions import TrainingRegion, load_regions, select_regions
from utils.s2_tiles import bbox_to_tile_ids
from utils.location import tile_chips_path  # noqa: F401 — re-exported for monkeypatching in tests

logger = logging.getLogger(__name__)


_TRAINING_DIR = PROJECT_ROOT / "data" / "training"
_TILES_DIR    = _TRAINING_DIR / "tiles"
_INDEX_PATH   = _TRAINING_DIR / "index.parquet"


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _load_index() -> pl.DataFrame:
    if _INDEX_PATH.exists():
        return pl.read_parquet(_INDEX_PATH)
    return pl.DataFrame({"region_id": pl.Series([], dtype=pl.Utf8),
                          "tile_id":   pl.Series([], dtype=pl.Utf8)})


def _save_index(df: pl.DataFrame) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_INDEX_PATH)


_index_lock = threading.Lock()


def _update_index(region_id: str, tile_ids: list[str]) -> None:
    df = _load_index()
    df = df.filter(pl.col("region_id") != region_id)
    if not tile_ids:
        raise ValueError(
            f"_update_index: region {region_id!r} maps to zero tiles — "
            "check that bbox_to_tile_ids() returns at least one tile."
        )
    new_rows = pl.DataFrame({"region_id": [region_id] * len(tile_ids), "tile_id": tile_ids})
    df = pl.concat([df, new_rows])
    _save_index(df)


def tile_ids_for_regions(region_ids: list[str]) -> list[str]:
    """Return deduplicated tile IDs covering the given region IDs (index must exist)."""
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


# ---------------------------------------------------------------------------
# Chunkstore location
# ---------------------------------------------------------------------------

def _chunkstore_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    import os
    return Path(os.environ.get("CHUNKSTORE_DIR", "/mnt/external/chunkstore"))


# ---------------------------------------------------------------------------
# Tile grouping helpers
# ---------------------------------------------------------------------------

def _best_tile_for_region(region: TrainingRegion, tile_ids: list[str]) -> str:
    """Return the tile whose footprint contains the region centroid, or the first."""
    from shapely.geometry import Point
    from utils.s2_tiles import get_au_tile_grid
    cx = (region.bbox[0] + region.bbox[2]) / 2
    cy = (region.bbox[1] + region.bbox[3]) / 2
    centroid = Point(cx, cy)
    grid = dict(get_au_tile_grid())
    for tile_id in tile_ids:
        geom = grid.get(tile_id)
        if geom is not None and geom.contains(centroid):
            return tile_id
    return tile_ids[0]


# ---------------------------------------------------------------------------
# Small bbox/date utilities (used by tests and helpers)
# ---------------------------------------------------------------------------

def _tile_date_window(regions: list) -> tuple[str, str]:
    """Return (start, end) ISO date strings spanning all years across regions."""
    all_years = [yr for r in regions for yr in r.years]
    if not all_years:
        raise ValueError("regions have no years")
    return f"{min(all_years)}-01-01", f"{max(all_years)}-12-31"


def _union_bbox(regions: list) -> list[float]:
    """Return [minx, miny, maxx, maxy] bbox covering all region bboxes."""
    if not regions:
        raise ValueError("regions is empty")
    return [
        min(r.bbox[0] for r in regions),
        min(r.bbox[1] for r in regions),
        max(r.bbox[2] for r in regions),
        max(r.bbox[3] for r in regions),
    ]


# ---------------------------------------------------------------------------
# Startup summary table
# ---------------------------------------------------------------------------

def _chunk_summary_table(
    tile_to_regions: dict[str, list[TrainingRegion]],
    chunkstore: Path,
) -> None:
    """Log a Year × Tile table showing collected vs. to-fetch chunk counts."""
    import json
    from shapely.geometry import box
    from shapely.ops import unary_union

    all_years: list[int] = sorted({yr for rs in tile_to_regions.values() for r in rs for yr in r.years})
    tile_ids: list[str] = sorted(tile_to_regions)

    # Expected chunks per tile (requires cached _grid.json + compute_chunks).
    tile_expected: dict[str, int] = {}
    for tile_id, tile_regions in tile_to_regions.items():
        grid_path = chunkstore / tile_id / "_grid.json"
        if not grid_path.exists():
            tile_expected[tile_id] = -1
            continue
        try:
            from proxy._pipeline import compute_chunks
            g = json.loads(grid_path.read_text())
            multi_geom = unary_union([box(*r.bbox) for r in tile_regions])
            chunks, _ = compute_chunks(
                list(multi_geom.bounds),
                g["block_height"], g["block_width"],
                multi_geom,
                cog_utm_crs=g["crs"], cog_y_top=g["y_top"], cog_x_left=g["x_left"],
            )
            tile_expected[tile_id] = len(chunks)
        except Exception:
            tile_expected[tile_id] = -1

    # Existing chunks per (year, tile).
    present: dict[tuple[int, str], int] = {}
    for tile_id in tile_ids:
        for yr in all_years:
            tile_dir = chunkstore / str(yr) / tile_id
            present[(yr, tile_id)] = len(list(tile_dir.glob(f"{tile_id}_r??_c??.parquet"))) if tile_dir.exists() else 0

    tile_w = max(len(t) for t in tile_ids)
    num_w  = 3
    gap    = "    "
    header = f"  {'Year':<6}  {'Tile':<{tile_w}}  {'In store':>{num_w + 7}}  {gap}  {'To fetch':>{num_w + 7}}"
    sep    = "  " + "-" * (len(header) - 2)
    logger.info(header)
    logger.info(sep)
    for yr in all_years:
        for i, tile_id in enumerate(tile_ids):
            yr_label     = str(yr) if i == 0 else ""
            n_present    = present[(yr, tile_id)]
            expected     = tile_expected[tile_id]
            n_fetch      = "?" if expected < 0 else str(max(0, expected - n_present))
            in_store_str = f"{n_present} chunks"
            to_fetch_str = f"{n_fetch} chunks"
            logger.info(
                f"  {yr_label:<6}  {tile_id:<{tile_w}}  {in_store_str:>{num_w + 7}}  {gap}  {to_fetch_str:>{num_w + 7}}"
            )


# ---------------------------------------------------------------------------
# Fetch: write chunks to chunkstore
# ---------------------------------------------------------------------------

def _fetch_tile_chunks(
    tile_id: str,
    tile_regions: list[TrainingRegion],
    chunkstore: Path,
    cloud_max: int,
    apply_nbar: bool,
    max_concurrent: int,
    work_dir: Path | None = None,
) -> None:
    """Fetch only the chunkstore chunks that intersect the training regions on this tile."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    from utils.tile_pipeline import fetch_tile_local

    multi_geom = unary_union([box(*r.bbox) for r in tile_regions])
    all_years = sorted({yr for r in tile_regions for yr in r.years})

    logger.info(
        "Tile %s: fetching %d chunk(s) for %d region(s), years %d–%d",
        tile_id, len(tile_regions), len(tile_regions), min(all_years), max(all_years),
    )

    fetch_tile_local(
        tile_id=tile_id,
        years=all_years,
        polygon_geometry=multi_geom,
        out_dir=chunkstore,
        cloud_max=cloud_max,
        apply_nbar=apply_nbar,
        max_concurrent=max_concurrent,
        work_dir=work_dir,
    )


# ---------------------------------------------------------------------------
# Extract: pull region rows from chunkstore chunks
# ---------------------------------------------------------------------------

def _extract_region(
    region: TrainingRegion,
    tile_id: str,
    chunkstore: Path,
) -> str | None:
    """Query the chunkstore for one region and write its region parquet.

    Returns the tile_id on success, None if no data found.
    """
    from utils.pixel_reader import ChunkIndex

    out_path = _region_parquet_path(region.id)
    lon_min, lat_min, lon_max, lat_max = region.bbox
    region_years = set(region.years)

    parts: list[pa.Table] = []
    for year in sorted(region_years):
        tile_dir = chunkstore / str(year) / tile_id
        if not tile_dir.exists():
            logger.debug("Region %s: no chunkstore dir for %d/%s", region.id, year, tile_id)
            continue
        idx = ChunkIndex(root=chunkstore, year=year, tile_id=tile_id)
        tbl = idx.query_bbox(lon_min, lat_min, lon_max, lat_max)
        if tbl.num_rows > 0:
            parts.append(tbl)

    if not parts:
        logger.warning("Region %s: no rows found in chunkstore", region.id)
        return None

    combined = pa.concat_tables(parts, promote_options="permissive")

    if "source" in combined.schema.names:
        src_idx = combined.schema.get_field_index("source")
        filled = pc.if_else(pc.is_null(combined.column("source")), "S2", combined.column("source"))
        combined = combined.set_column(src_idx, "source", filled)

    # Dedup multi-orbit S2 rows: same pixel+date from two passes → keep highest
    # scl_purity; ties broken by item_id (orbit _0_ sorts before _1_).
    n_before = len(combined)
    df = pl.from_arrow(combined)
    s2_mask = pl.col("source") == "S2"
    s2 = (
        df.filter(s2_mask)
        .sort(["point_id", "date", "scl_purity", "item_id"], descending=[False, False, True, False])
        .unique(subset=["point_id", "date"], keep="first")
    )
    s1 = df.filter(~s2_mask)
    df = pl.concat([s2, s1], how="diagonal")
    combined = df.to_arrow().cast(combined.schema)
    n_dropped = n_before - len(combined)
    if n_dropped:
        logger.info("Region %s: dropped %d multi-orbit duplicate S2 rows", region.id, n_dropped)

    # Drop S2 rows with any null band value (detector gaps, missing COG tiles).
    _BAND_COLS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
    present_bands = [b for b in _BAND_COLS if b in combined.schema.names]
    if present_bands:
        df2 = pl.from_arrow(combined)
        s2_rows = df2.filter(pl.col("source") == "S2")
        null_mask = pl.any_horizontal([pl.col(b).is_null() for b in present_bands])
        n_null = s2_rows.filter(null_mask).height
        if n_null:
            s2_rows = s2_rows.filter(~null_mask)
            s1_rows = df2.filter(pl.col("source") != "S2")
            df2 = pl.concat([s2_rows, s1_rows], how="diagonal")
            combined = df2.to_arrow().cast(combined.schema)
            logger.info("Region %s: dropped %d S2 rows with null band values", region.id, n_null)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, out_path)
    logger.info("Region %s: %d rows → %s", region.id, combined.num_rows, out_path.name)
    return tile_id


# ---------------------------------------------------------------------------
# Tile parquet rebuild (unchanged logic, same as before)
# ---------------------------------------------------------------------------

# Canonical types for fields that have drifted across schema generations.
# string→large_string: old extractions used plain utf8; newer use large_utf8.
# uint16 bands: earliest burdekin-era files stored bands as float after
# index computation; the current pipeline writes raw uint16.
_CANONICAL_TYPES: dict[str, pa.DataType] = {
    "point_id":   pa.large_utf8(),
    "item_id":    pa.large_utf8(),
    "tile_id":    pa.large_utf8(),
    "source":     pa.large_utf8(),
    "orbit":      pa.large_utf8(),
    **{b: pa.uint16() for b in ("B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12")},
    "scl_purity": pa.int8(),
    "aot":        pa.uint8(),
    "view_zenith": pa.uint8(),
    "sun_zenith":  pa.uint8(),
}


def _normalise_schema(s: pa.Schema) -> pa.Schema:
    """Apply canonical type overrides to a schema."""
    for name, canonical in _CANONICAL_TYPES.items():
        if name in s.names:
            idx = s.get_field_index(name)
            if s.field(name).type != canonical:
                s = s.set(idx, pa.field(name, canonical))
    return s


def _normalise_source(tbl: pa.Table) -> pa.Table:
    """Cast every column that has a canonical type to that type."""
    for name, canonical in _CANONICAL_TYPES.items():
        if name in tbl.schema.names:
            idx = tbl.schema.get_field_index(name)
            if tbl.schema.field(name).type != canonical:
                tbl = tbl.set_column(idx, name, tbl.column(name).cast(canonical))
    return tbl


def _superset_schema(paths: list[Path]) -> pa.Schema:
    """Union schema across files, enforcing canonical types and promoting
    string→large_string where unify_schemas would otherwise refuse."""
    schema: pa.Schema | None = None
    for p in paths:
        s = _normalise_schema(pq.read_schema(p))
        if schema is None:
            schema = s
        else:
            schema = pa.unify_schemas([schema, s], promote_options="permissive")
    return schema


def _cast_to_schema(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
    for i, field in enumerate(schema):
        if field.name not in tbl.schema.names:
            tbl = tbl.append_column(field, pa.nulls(len(tbl), type=field.type))
    return tbl.cast(schema)


def _write_region_rows(writer_ref: list, tile_path: Path, rp: Path, schema: pa.Schema) -> int:
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
    """Rebuild (or incrementally update) the merged tile parquet for tile_id."""
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

    if new_region_ids and tile_path.exists() and new_paths:
        tmp_path = tile_path.with_suffix(".tmp.parquet")
        logger.info(
            "Updating tile %s: replacing %d region(s) incrementally",
            tile_id, len(new_paths),
        )
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ensure_training_pixels(
    regions: list[TrainingRegion],
    cloud_max: int = 80,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    max_region_workers: int = 4,
    chunkstore_dir: Path | None = None,
    work_dir: Path | None = None,
) -> None:
    """Ensure pixel parquets exist for all tiles covered by the given regions.

    For each S2 tile:
      1. Fetch only the chunks that intersect the training region bboxes on the
         tile, writing to the chunkstore (idempotent — existing chunks skipped).
      2. Extract each pending region from the chunkstore via ChunkIndex.query_bbox().
      3. Rebuild the tile parquet and update the sidecar index.

    Skips regions whose parquet already exists (idempotent).
    """
    chunkstore = _chunkstore_dir(chunkstore_dir)

    # Group regions by primary tile (centroid rule for multi-tile regions).
    region_primary_tile: dict[str, str] = {}
    tile_to_regions: dict[str, list[TrainingRegion]] = {}
    for region in regions:
        tile_ids = bbox_to_tile_ids(region.bbox_tuple)
        primary = _best_tile_for_region(region, tile_ids)
        region_primary_tile[region.id] = primary
        tile_to_regions.setdefault(primary, []).append(region)

    if not tile_to_regions:
        logger.error("No S2 tiles found for the given regions — check bboxes")
        return

    logger.info(
        "%d regions → %d S2 tiles: %s",
        len(regions), len(tile_to_regions), sorted(tile_to_regions),
    )
    _chunk_summary_table(tile_to_regions, chunkstore)

    _TILES_DIR.mkdir(parents=True, exist_ok=True)
    (_TILES_DIR / "regions").mkdir(parents=True, exist_ok=True)

    tiles_new_regions: dict[str, set[str]] = {}
    tiles_lock = threading.Lock()

    for tile_id, tile_regions in sorted(tile_to_regions.items()):
        pending = [r for r in tile_regions if not _region_parquet_path(r.id).exists()]
        for r in tile_regions:
            if r not in pending:
                logger.info("Region %s already collected — skipping", r.id)

        if not pending:
            with _index_lock:
                for r in tile_regions:
                    _update_index(r.id, [tile_id])
            continue

        # Step 1: fetch chunks (union geometry, all regions on tile).
        _fetch_tile_chunks(
            tile_id=tile_id,
            tile_regions=tile_regions,
            chunkstore=chunkstore,
            cloud_max=cloud_max,
            apply_nbar=apply_nbar,
            max_concurrent=max_concurrent,
            work_dir=work_dir,
        )

        # Step 2: extract pending regions in parallel.
        def _do_extract(region: TrainingRegion, _tile_id: str = tile_id) -> str | None:
            actual_tile = _extract_region(region, _tile_id, chunkstore)
            if actual_tile is not None:
                with _index_lock:
                    _update_index(region.id, [actual_tile])
                with tiles_lock:
                    tiles_new_regions.setdefault(actual_tile, set()).add(region.id)
            return actual_tile

        with ThreadPoolExecutor(max_workers=max_region_workers) as pool:
            futures = {pool.submit(_do_extract, r): r.id for r in pending}
            for fut in as_completed(futures):
                rid = futures[fut]
                exc = fut.exception()
                if exc:
                    logger.error("Region %s failed: %s", rid, exc, exc_info=exc)

        with _index_lock:
            for r in tile_regions:
                if r not in pending:
                    _update_index(r.id, [tile_id])

    # Step 3: rebuild tile parquets for tiles that received new regions.
    for tile_id in sorted(tile_to_regions):
        tile_missing = not tile_parquet_path(tile_id).exists()
        new_ids = tiles_new_regions.get(tile_id)
        if new_ids or tile_missing:
            _rebuild_tile_parquet(
                tile_id,
                new_region_ids=new_ids if not tile_missing else None,
            )

    logger.info("Done. Index: %s", _INDEX_PATH)
