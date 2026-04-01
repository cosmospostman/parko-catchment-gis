"""
Step 04 — Wet-season flood extent mapping (Sentinel-1 SAR).

Produces: flood_extent_{year}.gpkg  (GeoPackage, EPSG:7855 geometries)

Approach
--------
1. Build a dry-season reference mask from Oct–Nov of the prior year.
   Pixels with persistently low VV backscatter in the dry season are sodic
   scalds / smooth gully floors — not water — and are excluded from flood
   classification.
2. For each wet-season S1 scene, apply a 3×3 median speckle filter then
   classify water using per-scene Otsu thresholding on VV with a VH guard.
3. Accumulate per-pixel flood frequency; threshold at FLOOD_MIN_FREQUENCY.
"""
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import dask
import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import shape
from shapely.ops import unary_union

# Script-level constants
FLOOD_UNION_SIMPLIFY_TOLERANCE = 100        # metres — coarser than pixel size (50 m) is fine
DASK_CHUNK_SPATIAL = 2048
S1_MAX_WORKERS = 2                          # concurrent S1 scene downloads
S1_RESOLUTION = 50                          # metres — flood mapping doesn't need 10 m
FLOOD_MIN_FREQUENCY = 0.10                  # pixel must be water in ≥10% of scenes

logger = logging.getLogger(__name__)

PIPELINE_RUN = os.environ.get("PIPELINE_RUN") == "1"
LOCAL_S1_ROOT = os.environ.get("LOCAL_S1_ROOT", "")


def _sigma_to_db(arr: xr.DataArray) -> xr.DataArray:
    """Convert linear sigma-naught to decibels."""
    return 10 * np.log10(arr + 1e-12)


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs
    from utils.stac import search_sentinel1
    from utils.sar import flood_mask_from_scene, build_dry_season_reference_mask
    from utils.quicklook import save_quicklook

    configure_logging()

    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)
    bbox_wgs84 = list(catchment.to_crs("EPSG:4326").total_bounds)

    flood_start = f"{config.YEAR}-{config.FLOOD_SEASON_START}"
    flood_end   = f"{config.YEAR}-{config.FLOOD_SEASON_END}"

    # --- Dry-season reference mask (Oct–Nov of prior year) -------------------
    dry_start = f"{config.YEAR}-10-01"
    dry_end   = f"{config.YEAR}-11-30"
    logger.info("Searching dry-season S1 reference items: %s → %s", dry_start, dry_end)
    dry_items = search_sentinel1(
        bbox=bbox_wgs84,
        start=dry_start,
        end=dry_end,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S1_COLLECTION,
    )
    if LOCAL_S1_ROOT and dry_items:
        from utils.stac import rewrite_hrefs_to_local
        dry_items = rewrite_hrefs_to_local(dry_items, LOCAL_S1_ROOT)

    reference_mask = None
    if dry_items:
        logger.info("Building dry-season reference mask from %d scenes", len(dry_items))
        reference_mask = build_dry_season_reference_mask(
            dry_items,
            bbox=bbox_wgs84,
            resolution=S1_RESOLUTION,
        )
    else:
        logger.warning("No dry-season S1 items found; sodic-scald masking will be skipped")

    # --- Flood-season scenes --------------------------------------------------
    logger.info("Searching Sentinel-1 items: %s → %s", flood_start, flood_end)
    items = search_sentinel1(
        bbox=bbox_wgs84,
        start=flood_start,
        end=flood_end,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S1_COLLECTION,
    )
    if not items:
        raise RuntimeError(f"No Sentinel-1 items found for flood season {config.YEAR}")

    if LOCAL_S1_ROOT:
        from utils.stac import rewrite_hrefs_to_local
        logger.info("Rewriting S1 asset hrefs to local cache: %s", LOCAL_S1_ROOT)
        items = rewrite_hrefs_to_local(items, LOCAL_S1_ROOT)

    n_workers = S1_MAX_WORKERS if PIPELINE_RUN else 1
    logger.info("Processing %d S1 scenes (workers=%d)", len(items), n_workers)

    def _process_scene(item):
        return flood_mask_from_scene(
            item,
            bbox=bbox_wgs84,
            resolution=S1_RESOLUTION,
            reference_mask=reference_mask,
        )

    flood_count = None   # uint16 count of scenes flagging each pixel as water
    obs_count   = None   # uint16 count of scenes observing each pixel (footprint)
    combined_coords = None
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_scene, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                scene = future.result()
                if scene is None:
                    continue
                logger.info("S1 scene processed (%d/%d, %.1f%%): %s", completed, len(items), 100 * completed / len(items), item.id)
                water    = scene["water"].values.view(np.uint8)
                observed = scene["observed"].values.view(np.uint8)
                if flood_count is None:
                    combined_coords = scene["water"]
                    flood_count = water.astype(np.uint16)
                    obs_count   = observed.astype(np.uint16)
                else:
                    if water.shape == flood_count.shape:
                        flood_count += water
                        obs_count   += observed
                    else:
                        logger.debug("Shape mismatch for %s — skipping", item.id)
            except Exception as exc:
                msg = str(exc)
                if "no spatial overlap" in msg or "no valid pixels" in msg or "zero-size" in msg:
                    logger.debug("Skipped S1 scene %s (no overlap)", item.id)
                else:
                    logger.warning("Failed to process S1 scene %s: %s", item.id, exc)

    if flood_count is None:
        raise RuntimeError("No valid S1 scenes processed")

    # Flood frequency = flood_count / obs_count per pixel (ignore unobserved pixels)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(obs_count > 0, flood_count / obs_count.astype(np.float32), 0.0)
    logger.info("Flood frequency threshold: %.0f%%  obs_count median: %d",
                FLOOD_MIN_FREQUENCY * 100, int(np.median(obs_count[obs_count > 0])))
    for pct in [50, 75, 90, 95, 99]:
        logger.info("  freq p%d = %.1f%%", pct, np.percentile(freq[obs_count > 0], pct) * 100)
    combined = freq >= FLOOD_MIN_FREQUENCY
    del flood_count, obs_count, freq

    # Vectorise flood mask to polygons
    logger.info("Vectorising flood extent...")
    import rasterio.features
    import rasterio.transform
    from scipy.ndimage import binary_closing

    # Build affine transform from the reference DataArray coordinates
    import affine
    x = combined_coords.coords["x"].values
    y = combined_coords.coords["y"].values
    res_x = float(x[1] - x[0]) if len(x) > 1 else config.TARGET_RESOLUTION
    res_y = float(y[1] - y[0]) if len(y) > 1 else -config.TARGET_RESOLUTION
    transform = affine.Affine(res_x, 0, float(x[0]), 0, res_y, float(y[0]))

    # Morphological closing merges nearby blobs at raster stage, drastically
    # reducing polygon count before vectorisation and unary_union.
    CLOSING_RADIUS_PX = 3  # 3 px × 50 m = 150 m closing radius
    struct = np.ones((CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool)
    data = binary_closing(combined, structure=struct).astype(np.uint8)
    logger.info("Morphological closing done; flood pixels: %d", data.sum())

    shapes = list(
        rasterio.features.shapes(data, mask=data, transform=transform)
    )
    logger.info("Vectorised to %d shapes before union", len(shapes))
    if not shapes:
        logger.warning("No flood pixels found — writing empty GeoDataFrame")
        gdf = gpd.GeoDataFrame(geometry=[], crs=config.TARGET_CRS)
    else:
        geoms = [shape(s) for s, v in shapes if v == 1]
        merged = unary_union(geoms)
        if hasattr(merged, "geoms"):
            geom_list = list(merged.geoms)
        else:
            geom_list = [merged]
        # Simplify
        geom_list = [g.simplify(FLOOD_UNION_SIMPLIFY_TOLERANCE) for g in geom_list]
        gdf = gpd.GeoDataFrame(geometry=geom_list, crs=config.TARGET_CRS)

    # Clip to catchment
    gdf = gpd.clip(gdf, catchment)

    out_path = config.flood_extent_path(config.YEAR)
    gdf.to_file(str(out_path), driver="GPKG")
    logger.info("Written: %s  (%d features)", out_path, len(gdf))

    ql_path = out_path.with_name(out_path.stem + "_quicklook.png")
    x = combined_coords.coords["x"].values
    y = combined_coords.coords["y"].values
    flood_da = xr.DataArray(
        combined.astype(np.float32),
        dims=["y", "x"],
        coords={"x": x, "y": y},
    )
    save_quicklook(
        flood_da,
        ql_path,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        title=f"Flood Extent {config.YEAR}",
    )


if __name__ == "__main__":
    main()
