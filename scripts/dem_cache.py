"""
scripts/dem_cache.py — Download Copernicus GLO-30 DEM tiles for the catchment
and store them in the EBS tile cache.

Tiles are stored individually (not pre-mosaicked) so that merge_and_reproject_dem()
in utils/dem.py can read them directly at analysis time.  The script is idempotent:
already-downloaded tiles are skipped.

EBS layout after this script:
    /mnt/ebs/dem/copernicus-dem-30m/
        Copernicus_DSM_COG_10_S15_00_E141_00_DEM.tif
        Copernicus_DSM_COG_10_S15_00_E142_00_DEM.tif
        ...  (~20 tiles, ~200 MB total for the Mitchell catchment)

Usage:
    python scripts/dem_cache.py [--tile-dir /mnt/ebs/dem/copernicus-dem-30m]

Add to EBS-SETUP.md as a one-time setup step (tiles are static, no refresh needed).
"""
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache Copernicus GLO-30 DEM tiles for the Mitchell catchment"
    )
    parser.add_argument(
        "--tile-dir",
        default="/mnt/ebs/dem/copernicus-dem-30m",
        help="Directory to store individual DEM tiles (default: /mnt/ebs/dem/copernicus-dem-30m)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    tile_dir = Path(args.tile_dir)

    # Ensure project root is on the path when the script is run from any directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import config
    import geopandas as gpd
    from utils.dem import download_dem_tiles

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)
    logger.info("Catchment bbox (WGS84): %s", bbox)

    tile_paths = download_dem_tiles(bbox, tile_dir)

    if not tile_paths:
        print("ERROR: no DEM tiles downloaded", file=sys.stderr)
        sys.exit(1)

    print(f"DEM tiles cached: {len(tile_paths)} tiles in {tile_dir}")
    for p in sorted(tile_paths):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
