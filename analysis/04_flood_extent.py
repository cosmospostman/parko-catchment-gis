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
FLOOD_UNION_SIMPLIFY_TOLERANCE = 20         # metres
DASK_CHUNK_SPATIAL = 2048
S1_MAX_WORKERS = 8                          # concurrent S1 scene downloads

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
    from utils.sar import preprocess_s1_scene
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
        ds = preprocess_s1_scene(item, bbox=bbox_wgs84, resolution=config.TARGET_RESOLUTION)
        if "VV" not in ds:
            logger.warning("VV band missing in item %s — skipping", item.id)
            return None
        vv_db = _sigma_to_db(ds["VV"])
        return (vv_db < VV_OPEN_WATER_THRESHOLD_DB).compute()

    flood_masks = []
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_scene, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                mask = future.result()
                if mask is not None:
                    flood_masks.append(mask)
                    logger.info("S1 scene processed (%d/%d, %.1f%%): %s", completed, len(items), 100 * completed / len(items), item.id)
            except Exception as exc:
                logger.warning("Failed to process S1 scene %s: %s", item.id, exc)

    if not flood_masks:
        raise RuntimeError("No valid S1 scenes processed")

    # Union of all flood masks across dates
    combined = flood_masks[0].copy()
    for mask in flood_masks[1:]:
        try:
            combined = combined | mask.reindex_like(combined, method="nearest")
        except Exception:
            combined = combined | mask

    # Vectorise flood mask to polygons
    logger.info("Vectorising flood extent...")
    import rasterio.features
    import rasterio.transform

    data = combined.values.astype(np.uint8)
    # Build an affine transform from the DataArray coordinates
    try:
        transform = combined.rio.transform()
    except Exception:
        # Fallback: construct from coordinate arrays
        import affine
        x = combined.coords.get("x", combined.coords.get("longitude"))
        y = combined.coords.get("y", combined.coords.get("latitude"))
        res_x = float(x[1] - x[0]) if len(x) > 1 else config.TARGET_RESOLUTION
        res_y = float(y[1] - y[0]) if len(y) > 1 else -config.TARGET_RESOLUTION
        transform = affine.Affine(res_x, 0, float(x[0]), 0, res_y, float(y[0]))

    shapes = list(
        rasterio.features.shapes(data, mask=data, transform=transform)
    )
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
        combined.squeeze(drop=True).astype(np.float32),
        ql_path,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        title=f"Flood Extent {config.YEAR}",
    )


if __name__ == "__main__":
    main()
