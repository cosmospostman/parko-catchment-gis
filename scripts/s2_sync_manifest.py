"""
scripts/s2_sync_manifest.py — STAC search → manifest of S3 keys to sync.

Searches for Sentinel-2 scenes covering the composite window (May–Oct, which
is a superset of the flowering window Aug–Oct) and writes one S3 URI per line.

Usage:
    python scripts/s2_sync_manifest.py YEAR [--out manifest.txt]
"""
import argparse
import sys
from pathlib import Path

# Required bands for stage 01 (red, nir, scl) + stage 03 (green, rededge1, rededge2, nir, scl)
REQUIRED_BANDS = ["red", "nir", "scl", "green", "rededge1", "rededge2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write S3 sync manifest for Sentinel-2 scenes")
    parser.add_argument("year", type=int, help="Processing year")
    parser.add_argument("--out", default="manifest.txt", help="Output manifest file path")
    args = parser.parse_args()

    import config
    from utils.io import configure_logging
    from utils.stac import search_sentinel2
    import geopandas as gpd
    import logging

    configure_logging()
    logger = logging.getLogger(__name__)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

    # Use the wider composite window — flowering is a subset
    composite_start = f"{args.year}-{config.COMPOSITE_START}"
    composite_end   = f"{args.year}-{config.COMPOSITE_END}"

    logger.info("Searching Sentinel-2 items: %s → %s", composite_start, composite_end)
    items = search_sentinel2(
        bbox=bbox,
        start=composite_start,
        end=composite_end,
        cloud_cover_max=config.CLOUD_COVER_MAX,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S2_COLLECTION,
    )

    if not items:
        print(f"ERROR: No Sentinel-2 items found for {args.year}", file=sys.stderr)
        sys.exit(1)

    logger.info("Found %d items; extracting asset hrefs", len(items))

    uris: list[str] = []
    for item in items:
        for band, asset in item.assets.items():
            if band not in REQUIRED_BANDS:
                continue
            href = asset.href
            if href.startswith("s3://"):
                uris.append(href)
            # Some STAC endpoints return HTTPS COG URLs — convert to s3:// if sentinel-cogs host
            elif "sentinel-cogs.s3" in href or "sentinel-cogs.s3.us-west-2.amazonaws.com" in href:
                from urllib.parse import urlparse
                parsed = urlparse(href)
                path = parsed.path.lstrip("/")
                uris.append(f"s3://sentinel-cogs/{path}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for uri in uris:
        if uri not in seen:
            seen.add(uri)
            deduped.append(uri)

    out_path = Path(args.out)
    out_path.write_text("\n".join(deduped) + "\n")
    logger.info("Wrote %d URIs to %s", len(deduped), out_path)
    print(f"Manifest written: {out_path} ({len(deduped)} files)")


if __name__ == "__main__":
    main()
