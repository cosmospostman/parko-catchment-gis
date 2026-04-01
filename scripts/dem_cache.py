"""
scripts/dem_cache.py — Download Copernicus GLO-30 DEM tiles covering the
catchment and mosaic them to a single GeoTIFF on the local EBS volume.

The output file is written once and reused on subsequent runs (idempotent).
sarsen's terrain_correction() is then pointed at this local path instead of
fetching tiles on every scene.

Usage:
    python scripts/dem_cache.py --out /mnt/ebs/dem/mitchell_dem.tif

Tile naming convention (Copernicus GLO-30 on S3):
    s3://copernicus-dem-30m/Copernicus_DSM_COG_10_<NS><lat>_00_<EW><lon>_00_DEM/
        Copernicus_DSM_COG_10_<NS><lat>_00_<EW><lon>_00_DEM.tif
e.g. S14 / E143 →
    Copernicus_DSM_COG_10_S14_00_E143_00_DEM/Copernicus_DSM_COG_10_S14_00_E143_00_DEM.tif
"""
import argparse
import logging
import math
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _tile_url(lat: int, lon: int) -> str:
    """Return the public HTTPS URL for the COP-DEM GLO-30 tile at (lat, lon).

    lat/lon are the integer degree of the tile's south-west corner.
    Copernicus tile names use the *northern* edge for latitude.
    """
    # Tile name latitude = north edge of tile = lat + 1 (for southern hemisphere)
    # Actually COP-DEM uses the SW corner lat directly for the label, with N/S prefix.
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    abs_lat = abs(lat)
    abs_lon = abs(lon)
    stem = (
        f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"
    )
    return (
        f"https://copernicus-dem-30m.s3.amazonaws.com/{stem}/{stem}.tif"
    )


def _tiles_for_bbox(minx: float, miny: float, maxx: float, maxy: float):
    """Yield (lat, lon) integer SW-corner pairs covering the bbox."""
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            yield lat, lon


def download_tiles(bbox, dest_dir: Path) -> list[Path]:
    """Download all COP-DEM tiles covering bbox to dest_dir. Returns local paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bbox
    tiles = list(_tiles_for_bbox(minx, miny, maxx, maxy))
    logger.info("COP-DEM: %d tiles to fetch for bbox %s", len(tiles), bbox)

    paths = []
    for lat, lon in tiles:
        url = _tile_url(lat, lon)
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        filename = f"cop_dem_{ns}{abs(lat):02d}_{ew}{abs(lon):03d}.tif"
        local = dest_dir / filename

        if local.exists():
            logger.info("  skip (cached): %s", filename)
            paths.append(local)
            continue

        logger.info("  downloading: %s", url)
        try:
            urllib.request.urlretrieve(url, local)
            logger.info("  saved: %s (%.1f MB)", filename, local.stat().st_size / 1e6)
            paths.append(local)
        except Exception as exc:
            # Some edge tiles don't exist (ocean) — skip gracefully
            logger.warning("  skipped (not found): %s — %s", filename, exc)

    return paths


def mosaic(tile_paths: list[Path], out_path: Path) -> None:
    """Merge tile GeoTIFFs into a single mosaic using rasterio."""
    import rasterio
    from rasterio.merge import merge

    logger.info("Mosaicking %d tiles → %s", len(tile_paths), out_path)
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaic_arr.shape[1],
        width=mosaic_arr.shape[2],
        transform=mosaic_transform,
        compress="deflate",
        bigtiff="YES",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic_arr)
    for ds in datasets:
        ds.close()
    logger.info("Written: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache COP-DEM GLO-30 tiles for the catchment")
    parser.add_argument("--out", required=True, help="Output mosaic GeoTIFF path")
    parser.add_argument("--tile-dir", help="Directory to store individual tiles (default: <out>/../dem_tiles)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        logger.info("DEM mosaic already exists: %s — skipping", out_path)
        print(f"DEM already cached: {out_path}")
        return

    import config
    import geopandas as gpd

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)  # [minx, miny, maxx, maxy]

    tile_dir = Path(args.tile_dir) if args.tile_dir else out_path.parent / "dem_tiles"
    tile_paths = download_tiles(bbox, tile_dir)

    if not tile_paths:
        print("ERROR: no DEM tiles downloaded", file=sys.stderr)
        sys.exit(1)

    mosaic(tile_paths, out_path)
    print(f"DEM cached: {out_path}")


if __name__ == "__main__":
    main()
