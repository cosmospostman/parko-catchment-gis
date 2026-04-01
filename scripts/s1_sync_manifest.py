"""
scripts/s1_sync_manifest.py — STAC search → manifest of S3 keys to sync for Sentinel-1.

Searches for Sentinel-1 GRD scenes covering the flood season (Jan–May) and writes
one S3 URI per line (vv and vh assets only).

Usage:
    python scripts/s1_sync_manifest.py YEAR [--out manifest_s1.txt]
"""
import argparse
import sys
from pathlib import Path

REQUIRED_ASSETS = ["vv", "vh", "safe-manifest", "schema-product-vv", "schema-product-vh",
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

    logger.info("Found %d items; extracting asset hrefs", len(items))

    uris: list[str] = []
    for item in items:
        for band, asset in item.assets.items():
            if band not in REQUIRED_ASSETS:
                continue
            href = asset.href
            if href.startswith("s3://"):
                uris.append(href)

        # Main annotation XMLs required by sarsen — not exposed as STAC assets.
        # Filename is derived from the scene ID (all lowercase) with fixed slice
        # numbers: VV=001, VH=002 for IW GRD dual-pol scenes.
        vv_asset = item.assets.get("vv")
        if vv_asset and vv_asset.href.startswith("s3://"):
            scene_root = vv_asset.href.rsplit("/measurement/", 1)[0]
            # S1A_IW_GRDH_1SDV_20250527T195239_20250527T195304_059384_075EF7
            # → s1a-iw-grd-{pol}-20250527t195239-20250527t195304-059384-075ef7-{slice}.xml
            parts = item.id.lower().split("_")
            # parts: [s1a, iw, grdh, 1sdv, start, stop, orbit, datatake]
            sat, _, _, _, start, stop, orbit, datatake = parts[:8]
            base = f"{sat}-iw-grd-{{pol}}-{start}-{stop}-{orbit}-{datatake}"
            uris.append(f"{scene_root}/annotation/{base.format(pol='vv')}-001.xml")
            uris.append(f"{scene_root}/annotation/{base.format(pol='vh')}-002.xml")

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
