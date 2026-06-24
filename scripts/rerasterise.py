"""Re-rasterise already-scored tiles to PMTiles — no GPU, no rescoring.

The scores parquet on disk is authoritative; this only regenerates the PMTiles
placement (which had a bug where chunked tiles were placed by per-chunk-relative
xi/yi on one origin, misplacing whole regions). It reads <tile>.scores.parquet
and rebuilds <tile>.pmtiles via rasterize_tile_to_pmtiles, which places pixels by
their real lon/lat using coords gathered from ALL of the tile's chunk parquets.

Usage:
    python scripts/rerasterise.py --location mitchell --checkpoint tam-v10 \
        --years 2025 --tiles 54LWH 54LWJ 55KCB \
        --pixel-dir /mnt/gis-archive/chunkstore

    # all scored tiles for the year (omit --tiles):
    python scripts/rerasterise.py --location mitchell --checkpoint tam-v10 --years 2025
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.raster import rasterize_tile_to_pmtiles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rerasterise")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-rasterise scored tiles to PMTiles")
    ap.add_argument("--location", required=True, help="Location id (e.g. mitchell)")
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint name under outputs/scores/<loc>/<name> (e.g. tam-v10)")
    ap.add_argument("--years", type=int, nargs="+", required=True, metavar="YEAR")
    ap.add_argument("--tiles", nargs="*", default=None, metavar="TILE_ID",
                    help="MGRS tile ids; default = all scored tiles for the year")
    ap.add_argument("--pixel-dir", default="/mnt/gis-archive/chunkstore",
                    help="Chunkstore base dir: <year>/<tile>/<tile>_r##_c##.parquet")
    args = ap.parse_args()

    chunkstore = Path(args.pixel_dir)
    for year in args.years:
        scores_dir = PROJECT_ROOT / "outputs/scores" / args.location / args.checkpoint / str(year)
        pmtiles_dir = PROJECT_ROOT / "outputs/scores" / args.location
        if not scores_dir.is_dir():
            logger.error("No scores dir for year %d: %s", year, scores_dir)
            continue

        tiles = args.tiles or sorted(
            p.name[: -len(".scores.parquet")] for p in scores_dir.glob("*.scores.parquet")
        )
        if not tiles:
            logger.error("No scored tiles under %s", scores_dir)
            continue

        for tile_id in tiles:
            scores_path = scores_dir / f"{tile_id}.scores.parquet"
            if not scores_path.exists():
                logger.error("MISSING scores for %s: %s", tile_id, scores_path)
                continue
            coords_paths = sorted(
                (chunkstore / str(year) / tile_id).glob(f"{tile_id}_r*_c*.parquet")
            )
            if not coords_paths:
                logger.error("No chunk coords for %s under %s",
                             tile_id, chunkstore / str(year) / tile_id)
                continue
            out = pmtiles_dir / f"{tile_id}.pmtiles"
            logger.info("Re-rasterising %s (%d) — %d coord chunks → %s",
                        tile_id, year, len(coords_paths), out)
            t0 = time.perf_counter()
            rasterize_tile_to_pmtiles(scores_path, coords_paths, tile_id, out)
            logger.info("Done %s in %.1fs → %s (%.0f MB)",
                        tile_id, time.perf_counter() - t0, out.name, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
