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
from pathlib import Path

import pandas as pd
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

def _load_index() -> pd.DataFrame:
    """Load the region→tile sidecar index, or return an empty frame."""
    if _INDEX_PATH.exists():
        return pd.read_parquet(_INDEX_PATH)
    return pd.DataFrame(columns=["region_id", "tile_id"])


def _save_index(df: pd.DataFrame) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_INDEX_PATH, index=False)


def _update_index(region_id: str, tile_ids: list[str]) -> None:
    """Add or refresh the index entries for one region."""
    df = _load_index()
    df = df[df["region_id"] != region_id]  # remove stale entries
    if not tile_ids:
        raise ValueError(
            f"_update_index: region {region_id!r} maps to zero tiles — "
            "check that bbox_to_tile_ids() returns at least one tile."
        )
    new_rows = pd.DataFrame({"region_id": region_id, "tile_id": tile_ids})
    df = pd.concat([df, new_rows], ignore_index=True)
    _save_index(df)


def tile_ids_for_regions(region_ids: list[str]) -> list[str]:
    """Return the deduplicated tile IDs that cover the given region IDs.

    Consults the sidecar index — regions must have been collected first.
    """
    df = _load_index()
    hits = df[df["region_id"].isin(region_ids)]
    if hits.empty:
        raise RuntimeError(
            f"No tile index entries found for regions {region_ids}. "
            "Run 'ensure-training-pixels' first."
        )
    return sorted(hits["tile_id"].unique().tolist())


def tile_parquet_path(tile_id: str) -> Path:
    return _TILES_DIR / f"{tile_id}.parquet"


def _region_parquet_path(region_id: str) -> Path:
    return _TILES_DIR / "regions" / f"{region_id}.parquet"


_STAC_ENDPOINT  = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION  = "sentinel-2-l2a"
_GRANULE_RE     = _re.compile(r"_\d+_L2A$")


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
    """Derive fetch window as [min(year)-5, max(year)] across regions."""
    years = [r.year for r in tile_regions if r.year is not None]
    if not years:
        raise ValueError(
            f"Regions {[r.id for r in tile_regions]} have no year set — "
            "all training regions must specify a year"
        )
    return f"{min(years) - 5}-01-01", f"{max(years)}-12-31"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def ensure_training_pixels(
    regions: list[TrainingRegion],
    cloud_max: int = 80,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
) -> None:
    """Ensure pixel parquets exist for all tiles covered by the given regions.

    For each region:
      1. Calls pixel_collector.collect() with the region's own bbox →
         data/training/tiles/regions/{region_id}.parquet
      2. Concatenates all per-region parquets for a tile into
         data/training/tiles/{tile_id}.parquet
      3. Updates the sidecar index.

    Skips regions whose parquet already exists (idempotent).
    Rebuilds tile parquets whenever any constituent region parquet is new.
    """
    # Group regions by tile
    tile_to_regions: dict[str, list[TrainingRegion]] = {}
    for region in regions:
        for tile_id in bbox_to_tile_ids(region.bbox_tuple):
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

    for tile_id, tile_regions in sorted(tile_to_regions.items()):
        new_regions = []

        tile_path = tile_parquet_path(tile_id)
        tile_missing = not tile_path.exists()
        pending = [r for r in tile_regions if not _region_parquet_path(r.id).exists()]
        for r in tile_regions:
            if r not in pending:
                logger.info("Region %s already collected — skipping", r.id)

        if pending or tile_missing:
            tile_items = _fetch_tile_items(tile_id, tile_regions, cloud_max) if pending else None
            start, end = _tile_date_window(tile_regions)

        for region in pending:
            out_path = _region_parquet_path(region.id)
            collect_dir = _TILES_DIR / "regions" / f"{region.id}.collect"
            logger.info(
                "Collecting region %s: bbox=%s  window=%s→%s",
                region.id, region.bbox, start, end,
            )
            cache_dir = tile_chips_path(tile_id)
            tile_paths = collect(
                bbox_wgs84=list(region.bbox),
                start=start,
                end=end,
                out_dir=collect_dir,
                cloud_max=cloud_max,
                cache_dir=cache_dir,
                apply_nbar=apply_nbar,
                max_concurrent=max_concurrent,
                items=tile_items,
                point_id_prefix=region.id,
            )
            # collect() returns [] when all shards were already marked done but no
            # sorted intermediate exists (happens when a previous run completed the
            # shard and wrote the tile parquet but crashed before writing the region
            # parquet).  Recover by scanning the collect dir for existing outputs.
            if not tile_paths:
                tile_paths = sorted(collect_dir.glob("*.parquet"))
                if tile_paths:
                    logger.info(
                        "Region %s: collect() returned no paths but found %d existing "
                        "parquet(s) in collect dir — using those",
                        region.id, len(tile_paths),
                    )
            # Merge per-tile S2 parquets into a single region parquet,
            # then fetch and append S1 rows using the same pixel grid.
            if tile_paths:
                writer = pq.ParquetWriter(out_path, pq.ParquetFile(tile_paths[0]).schema_arrow)
                for tp in tile_paths:
                    pf = pq.ParquetFile(tp)
                    for rg_idx in range(pf.metadata.num_row_groups):
                        writer.write_table(pf.read_row_group(rg_idx))
                writer.close()
                append_s1_to_tile_parquet(
                    tile_path=out_path,
                    bbox_wgs84=list(region.bbox),
                    start=start,
                    end=end,
                    collect_s1_fn=collect_s1,
                    s1_cache_dir=_S1_CHIP_CACHE_DIR,
                )
            new_regions.append(region.id)

        # Rebuild tile parquet if any region is new or the tile parquet was missing.
        # Include ALL regions indexed for this tile (not just the ones being fetched
        # now) so that a partial fetch never overwrites previously collected data.
        if new_regions or tile_missing:
            all_indexed = _load_index()
            all_indexed_ids = set(
                all_indexed.loc[all_indexed["tile_id"] == tile_id, "region_id"]
            )
            region_paths = [
                _region_parquet_path(rid)
                for rid in sorted(all_indexed_ids)
                if _region_parquet_path(rid).exists()
            ]
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

        for region in tile_regions:
            _update_index(region.id, [tile_id])

    logger.info("Done. Index: %s", _INDEX_PATH)
