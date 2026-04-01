"""
Step 04 — Wet-season flood extent mapping (Sentinel-1 SAR).

Produces: flood_extent_{year}.gpkg  (GeoPackage, EPSG:7844 geometries)
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
S1_POLARISATIONS = ["VV", "VH"]
VV_OPEN_WATER_THRESHOLD_DB = -14.0          # dB — below this = open water
FLOOD_UNION_SIMPLIFY_TOLERANCE = 100        # metres — coarser than pixel size (50 m) is fine
DASK_CHUNK_SPATIAL = 2048
S1_MAX_WORKERS = 4                          # concurrent S1 scene downloads
S1_RESOLUTION = 50                          # metres — flood mapping doesn't need 10 m
FLOOD_MIN_FREQUENCY = 0.25                  # pixel must be water in ≥25% of scenes

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
    from utils.sar import flood_mask_from_scene
    from utils.quicklook import save_quicklook

    configure_logging()

    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)
    bbox_wgs84 = list(catchment.to_crs("EPSG:4326").total_bounds)

    flood_start = f"{config.YEAR}-{config.FLOOD_SEASON_START}"
    flood_end   = f"{config.YEAR}-{config.FLOOD_SEASON_END}"

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
            threshold_db=VV_OPEN_WATER_THRESHOLD_DB,
        )

    flood_count = None   # uint16 count of scenes flagging each pixel as water
    scene_count = 0      # number of scenes successfully contributing
    combined_coords = None
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_scene, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                mask = future.result()
                if mask is None:
                    continue
                logger.info("S1 scene processed (%d/%d, %.1f%%): %s", completed, len(items), 100 * completed / len(items), item.id)
                if flood_count is None:
                    combined_coords = mask
                    flood_count = mask.values.astype(np.uint16)
                else:
                    m = mask.values
                    if m.shape == flood_count.shape:
                        flood_count += m.astype(np.uint16)
                    else:
                        logger.debug("Shape mismatch for %s — skipping", item.id)
                scene_count += 1
            except Exception as exc:
                msg = str(exc)
                if "no spatial overlap" in msg or "no valid pixels" in msg or "zero-size" in msg:
                    logger.debug("Skipped S1 scene %s (no overlap)", item.id)
                else:
                    logger.warning("Failed to process S1 scene %s: %s", item.id, exc)

    if flood_count is None:
        raise RuntimeError("No valid S1 scenes processed")

    # Pixels must be flooded in at least FLOOD_MIN_FREQUENCY fraction of scenes
    min_scenes = max(1, int(scene_count * FLOOD_MIN_FREQUENCY))
    logger.info("Flood frequency threshold: %d/%d scenes (%.0f%%)",
                min_scenes, scene_count, FLOOD_MIN_FREQUENCY * 100)
    for pct in [50, 75, 90, 95, 99]:
        logger.info("  flood_count p%d = %d scenes", pct, np.percentile(flood_count, pct))
    combined = flood_count >= min_scenes
    del flood_count

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
    save_quicklook(
        combined_coords.squeeze(drop=True).astype(np.float32),
        ql_path,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        title=f"Flood Extent {config.YEAR}",
    )


if __name__ == "__main__":
    main()
