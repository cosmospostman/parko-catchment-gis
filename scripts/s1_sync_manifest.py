"""
scripts/s1_sync_manifest.py — STAC search → manifest of S3 keys to sync for Sentinel-1.

Searches for Sentinel-1 GRD scenes covering:
  - the flood season (Jan–May of YEAR), and
  - the dry-season reference window (Oct–Nov of YEAR-1)

Writes one S3 URI per line (vv and vh assets only).

Usage:
    python scripts/s1_sync_manifest.py YEAR [--out manifest_s1.txt]
"""
import argparse
import sys
from pathlib import Path

REQUIRED_ASSETS = ["vv", "vh", "safe-manifest",
                   "schema-calibration-vv", "schema-calibration-vh", "schema-noise-vv", "schema-noise-vh"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write S3 sync manifest for Sentinel-1 scenes")
    parser.add_argument("year", type=int, help="Processing year")
    parser.add_argument("--out", default="manifest_s1.txt", help="Output manifest file path")
    args = parser.parse_args()

    import config
    from utils.io import configure_logging
    from utils.stac import search_sentinel1
    import geopandas as gpd
    import logging

    configure_logging()
    logger = logging.getLogger(__name__)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

    flood_start = f"{args.year}-{config.FLOOD_SEASON_START}"
    flood_end   = f"{args.year}-{config.FLOOD_SEASON_END}"

    logger.info("Searching Sentinel-1 items: %s → %s", flood_start, flood_end)
    items = search_sentinel1(
        bbox=bbox,
        start=flood_start,
        end=flood_end,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S1_COLLECTION,
    )

    if not items:
        print(f"ERROR: No Sentinel-1 items found for {args.year}", file=sys.stderr)
        sys.exit(1)

    # Dry-season reference: Oct–Nov of the analysis year
    dry_start = f"{args.year}-10-01"
    dry_end   = f"{args.year}-11-30"
    logger.info("Searching Sentinel-1 dry-season reference items: %s → %s", dry_start, dry_end)
    dry_items = search_sentinel1(
        bbox=bbox,
        start=dry_start,
        end=dry_end,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S1_COLLECTION,
    )
    if not dry_items:
        logger.warning("No dry-season S1 items found for %s–%s; reference mask will be skipped", dry_start, dry_end)
    else:
        logger.info("Found %d dry-season items", len(dry_items))
        items = items + dry_items

    logger.info("Found %d items; extracting asset hrefs", len(items))

    uris: list[str] = []
    for item in items:
        for band, asset in item.assets.items():
            if band not in REQUIRED_ASSETS:
                continue
            href = asset.href
            if href.startswith("s3://"):
                uris.append(href)

        # Annotation XMLs — use STAC asset keys schema-product-vv/vh which
        # point directly to the per-subswath annotation files on this collection.
        for ann_key in ("schema-product-vv", "schema-product-vh"):
            ann_asset = item.assets.get(ann_key)
            if ann_asset and ann_asset.href.startswith("s3://"):
                uris.append(ann_asset.href)

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
