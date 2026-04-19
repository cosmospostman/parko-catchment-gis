"""utils/training_collector.py — Collect training pixels organised by S2 tile.

Fetches Sentinel-2 observations for a set of TrainingRegions, writing one
parquet per S2 tile to data/training/tiles/{tile_id}.parquet, and maintaining
a sidecar index at data/training/index.parquet that maps region_id → tile_ids.

This keeps the storage layer tile-centric (efficient for fetch and I/O) while
the training API remains bbox-centric (each TrainingRegion is just a labeled bbox).

Usage (CLI)
-----------
    python -m utils.training_collector ensure \\
        --regions longreach_presence longreach_absence \\
        --start 2020-01-01 --end 2025-12-31

    python -m utils.training_collector ensure --all \\
        --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.regions import TrainingRegion, load_regions, select_regions
from utils.s2_tiles import bbox_to_tile_ids

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


# ---------------------------------------------------------------------------
# Core: ensure pixels exist for one tile + region list
# ---------------------------------------------------------------------------

def _bbox_for_tile_regions(
    tile_id: str,
    regions: list[TrainingRegion],
) -> list[float]:
    """Return the union bbox of all regions that intersect this tile."""
    lon_mins, lat_mins, lon_maxs, lat_maxs = [], [], [], []
    for r in regions:
        if tile_id in bbox_to_tile_ids(r.bbox_tuple):
            lon_min, lat_min, lon_max, lat_max = r.bbox
            lon_mins.append(lon_min)
            lat_mins.append(lat_min)
            lon_maxs.append(lon_max)
            lat_maxs.append(lat_max)
    return [min(lon_mins), min(lat_mins), max(lon_maxs), max(lat_maxs)]


def ensure_training_pixels(
    regions: list[TrainingRegion],
    start: str,
    end: str,
    cloud_max: int = 80,
    stride: int = 1,
    apply_nbar: bool = True,
) -> None:
    """Ensure pixel parquets exist for all tiles covered by the given regions.

    For each S2 tile that intersects at least one region bbox:
      1. Resolves the union fetch bbox (all regions on that tile).
      2. Calls pixel_collector.collect() → data/training/tiles/{tile_id}.parquet
      3. Updates the sidecar index.

    Skips tiles whose parquet already exists (idempotent).
    """
    from utils.pixel_collector import collect

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

    for tile_id, tile_regions in sorted(tile_to_regions.items()):
        out_path = tile_parquet_path(tile_id)
        if out_path.exists():
            logger.info("Tile %s already collected (%s) — skipping", tile_id, out_path.name)
            _update_index(tile_id, [r.id for r in tile_regions])
            continue

        fetch_bbox = _bbox_for_tile_regions(tile_id, tile_regions)
        logger.info(
            "Collecting tile %s: bbox=%s  regions=%s",
            tile_id, fetch_bbox, [r.id for r in tile_regions],
        )

        cache_dir = _TRAINING_DIR / "chips" / tile_id
        collect(
            bbox_wgs84=fetch_bbox,
            start=start,
            end=end,
            out_path=out_path,
            cloud_max=cloud_max,
            cache_dir=cache_dir,
            stride=stride,
            apply_nbar=apply_nbar,
        )

        # Update index after each successful tile
        for region in tile_regions:
            _update_index(region.id, [tile_id])

    logger.info("Done. Index: %s", _INDEX_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ensure training pixel parquets exist for selected regions."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ensure_p = sub.add_parser("ensure", help="Fetch pixels for training regions")
    grp = ensure_p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--regions", nargs="+", metavar="ID",
        help="Region IDs to ensure (from training_regions.yaml)",
    )
    grp.add_argument(
        "--all", action="store_true",
        help="Ensure pixels for all regions in training_regions.yaml",
    )
    ensure_p.add_argument("--start",     required=True, help="Start date YYYY-MM-DD")
    ensure_p.add_argument("--end",       required=True, help="End date YYYY-MM-DD")
    ensure_p.add_argument("--cloud-max", type=int, default=80)
    ensure_p.add_argument("--stride",    type=int, default=1)
    ensure_p.add_argument("--no-nbar",   action="store_true")

    args = parser.parse_args()

    if args.cmd == "ensure":
        if args.all:
            regions = load_regions()
        else:
            regions = select_regions(args.regions)

        ensure_training_pixels(
            regions=regions,
            start=args.start,
            end=args.end,
            cloud_max=args.cloud_max,
            stride=args.stride,
            apply_nbar=not args.no_nbar,
        )
