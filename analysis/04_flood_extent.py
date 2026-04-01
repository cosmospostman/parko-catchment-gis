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
import multiprocessing
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use forkserver so worker processes start clean without inheriting the
# parent's memory (which can be 10+ GB after the dry-season phase).
multiprocessing.set_start_method("forkserver", force=True)

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import shape
from shapely.ops import unary_union

# Script-level constants
FLOOD_UNION_SIMPLIFY_TOLERANCE = 100        # metres — coarser than pixel size (50 m) is fine
DASK_CHUNK_SPATIAL = 2048
S1_DRY_WORKERS   = int(os.environ.get("DRY_WORKERS",   4))   # override with --dry-workers
S1_FLOOD_WORKERS = int(os.environ.get("FETCH_WORKERS", 4))  # override with --fetch-workers
S1_RESOLUTION = 50                          # metres — flood mapping doesn't need 10 m
FLOOD_MIN_FREQUENCY = 0.33                  # pixel must be water in ≥33% of scenes

logger = logging.getLogger(__name__)

PIPELINE_RUN = os.environ.get("PIPELINE_RUN") == "1"
LOCAL_S1_ROOT = os.environ.get("LOCAL_S1_ROOT", "")

# Module-level cache so each worker process loads the mask at most once.
_worker_reference_mask: np.ndarray | None = None
_worker_mask_path: str = ""


def _process_scene_worker(item, bbox, resolution, mask_path: str):
    """Top-level worker function (picklable) for ProcessPoolExecutor."""
    global _worker_reference_mask, _worker_mask_path
    from utils.sar import flood_mask_from_scene
    from utils.io import configure_logging
    configure_logging()
    if mask_path and mask_path != _worker_mask_path:
        _worker_reference_mask = np.load(mask_path)
        _worker_mask_path = mask_path
    return flood_mask_from_scene(item, bbox=bbox, resolution=resolution,
                                 reference_mask=_worker_reference_mask)


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
    dry_workers   = S1_DRY_WORKERS   if PIPELINE_RUN else 1
    flood_workers = S1_FLOOD_WORKERS if PIPELINE_RUN else 1

    # Dry-season reference mask is disabled: the DN²/1e6 normalisation produces
    # uncalibrated backscatter values that vary too much between scenes to set a
    # meaningful absolute threshold.  At -16 dB the mask flags ~76% of the
    # catchment (normal vegetation + bare soil), wiping out real flood signal.
    # The water-fraction sanity guard and frequency threshold handle false
    # positives without needing a reference mask.
    reference_mask = None
    dry_mask_cache = config.dry_season_mask_path(config.YEAR)  # kept for path reference only

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

    logger.info("Processing %d S1 scenes (workers=%d)", len(items), flood_workers)

    # Pass mask path rather than the array — avoids pickling 67 MB per worker.
    # Workers load it once on first use via _process_scene_worker.
    flood_count = None   # uint16 count of scenes flagging each pixel as water
    obs_count   = None   # uint16 count of scenes observing each pixel (footprint)
    combined_coords = None
    completed = 0
    mask_path_str = str(dry_mask_cache) if dry_mask_cache.exists() else ""

    # Submit at most (flood_workers + 1) futures at a time so the executor queue
    # never holds more than one prefetched scene per worker.  Submitting all 109
    # futures upfront causes the pool to begin reading the next scene's raster
    # data before the current result has been consumed, doubling peak memory.
    item_iter = iter(items)
    in_flight: dict = {}

    def _submit_next() -> None:
        item = next(item_iter, None)
        if item is not None:
            f = executor.submit(_process_scene_worker, item, bbox_wgs84, S1_RESOLUTION, mask_path_str)
            in_flight[f] = item

    with ProcessPoolExecutor(max_workers=flood_workers) as executor:
        # Seed the pool — one job per worker, no prefetch queue
        for _ in range(flood_workers):
            _submit_next()

        while in_flight:
            future = next(as_completed(in_flight))
            item = in_flight.pop(future)
            completed += 1
            try:
                scene = future.result()
                if scene is None:
                    logger.info("Flood scene skipped — no valid pixels [%d/%d]: %s",
                                completed, len(items), item.id)
                else:
                    water    = scene["water"].values.view(np.uint8)
                    observed = scene["observed"].values.view(np.uint8)
                    if flood_count is None:
                        combined_coords = scene["water"]
                    del scene
                    if flood_count is None:
                        flood_count = water.astype(np.uint16)
                        obs_count   = observed.astype(np.uint16)
                    elif water.shape == flood_count.shape:
                        flood_count += water
                        obs_count   += observed
                    else:
                        logger.info("Flood scene skipped — shape mismatch [%d/%d]: %s",
                                    completed, len(items), item.id)
                    if flood_count is not None:
                        logger.info("Flood scene accumulated [%d/%d, %.0f%%]: %s  "
                                    "(water=%d obs=%d)",
                                    completed, len(items), 100 * completed / len(items),
                                    item.id, water.sum(), observed.sum())
            except Exception as exc:
                msg = str(exc)
                if "no spatial overlap" in msg or "no valid pixels" in msg or "zero-size" in msg:
                    logger.info("Flood scene skipped — no overlap [%d/%d]: %s",
                                completed, len(items), item.id)
                else:
                    logger.warning("Flood scene failed [%d/%d]: %s: %s",
                                   completed, len(items), item.id, exc)
            # Always submit next — even after skips/failures — to keep the pool full.
            _submit_next()

    if flood_count is None:
        raise RuntimeError("No valid S1 scenes processed")

    # Minimum observations required to classify a pixel — pixels seen by fewer
    # scenes are coverage-edge artefacts (sub-swath boundaries, orbit gaps) and
    # produce false positives because Otsu is calibrated on the full-scene
    # histogram, not on the edge incidence-angle zone.
    # With 13 passes per orbit × 4 orbits = ~52 theoretical observations,
    # requiring at least 4 ensures every valid pixel was seen by at least one
    # full orbit's worth of passes.
    MIN_OBS = 4

    import affine
    import rasterio
    import rasterio.features
    import rasterio.transform
    from scipy.ndimage import binary_closing

    # Build affine transform from the reference DataArray coordinates (used
    # for both the obs_count raster write and the vectorisation below).
    x = combined_coords.coords["x"].values
    y = combined_coords.coords["y"].values
    res_x = float(x[1] - x[0]) if len(x) > 1 else config.TARGET_RESOLUTION
    res_y = float(y[1] - y[0]) if len(y) > 1 else -config.TARGET_RESOLUTION
    transform = affine.Affine(res_x, 0, float(x[0]), 0, res_y, float(y[0]))

    # Persist obs_count as a GeoTIFF for diagnostics and quality masking
    obs_path = config.flood_obs_count_path(config.YEAR)
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(obs_path), "w", driver="GTiff", count=1, dtype="uint16",
        width=obs_count.shape[1], height=obs_count.shape[0],
        crs=config.TARGET_CRS, transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(obs_count, 1)
    logger.info("Written obs_count raster: %s", obs_path)

    # Flood frequency = flood_count / obs_count per pixel (ignore unobserved pixels)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(obs_count > 0, flood_count / obs_count.astype(np.float32), 0.0)
    sufficient_obs = obs_count >= MIN_OBS
    logger.info("Flood frequency threshold: %.0f%%  obs_count median: %d  min_obs filter: %d",
                FLOOD_MIN_FREQUENCY * 100, int(np.median(obs_count[obs_count > 0])), MIN_OBS)
    logger.info("Pixels excluded by min_obs filter: %d (%.1f%%)",
                (~sufficient_obs & (obs_count > 0)).sum(),
                100 * (~sufficient_obs & (obs_count > 0)).sum() / max((obs_count > 0).sum(), 1))
    freq_valid = freq[sufficient_obs]
    if freq_valid.size > 0:
        for pct in [50, 75, 90, 95, 99]:
            logger.info("  freq p%d = %.1f%%", pct, np.percentile(freq_valid, pct) * 100)
    del freq_valid
    combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient_obs
    del flood_count, obs_count, freq, sufficient_obs

    # Vectorise flood mask to polygons
    logger.info("Vectorising flood extent...")

    # Morphological closing merges nearby blobs at raster stage, drastically
    # reducing polygon count before vectorisation and unary_union.
    CLOSING_RADIUS_PX = 3  # 3 px × 50 m = 150 m closing radius
    # Minimum contiguous patch size to retain — removes isolated speckle pixels
    # that survive Otsu + VH guard.  4 px × (50 m)² = 1 ha minimum.
    MIN_PATCH_PX = 4
    struct = np.ones((CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool)
    data = binary_closing(combined, structure=struct).astype(np.uint8)
    logger.info("Morphological closing done; flood pixels: %d", data.sum())

    from scipy.ndimage import label
    labelled, n_features = label(data)
    patch_sizes = np.bincount(labelled.ravel())  # index 0 = background
    small_labels = np.where(patch_sizes < MIN_PATCH_PX)[0]
    small_labels = small_labels[small_labels > 0]  # exclude background
    if small_labels.size:
        data[np.isin(labelled, small_labels)] = 0
    del labelled, patch_sizes, small_labels
    logger.info("After min-patch filter (%d px): flood pixels: %d", MIN_PATCH_PX, data.sum())

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
