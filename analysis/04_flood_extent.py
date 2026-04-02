"""
Step 04 — HAND-based flood connectivity mapping.

Produces:
  flood_extent_{year}.gpkg   — flood connectivity polygons (GeoPackage, EPSG:7855)
  hand_{year}.tif            — HAND raster for diagnostics

Approach
--------
1. Download / load Copernicus GLO-30 DEM tiles for the catchment bbox.
2. Merge tiles, fill voids, reproject to EPSG:7855 at 30 m.
3. Optionally burn the GA TOPO 250K drainage network into the DEM to
   improve flow-path stability on the flat megafan.
4. Compute D8 flow accumulation.
5. Compute HAND (Height Above Nearest Drainage) raster.
6. Apply HAND threshold (HAND_FLOOD_THRESHOLD_M) → binary flood connectivity mask.
7. Morphological closing (150 m radius) to merge adjacent floodplain blobs.
8. Remove patches smaller than MIN_PATCH_PX pixels.
9. Vectorise → union → simplify (100 m tolerance) → clip to catchment.

Why not Sentinel-1?
    C-band GRD (5.5 cm) cannot penetrate the dense grassland / sedgeland
    canopy that dominates the Mitchell floodplain.  Validation shows a
    single unimodal backscatter distribution centred at −16 to −17 dB
    regardless of season — Otsu thresholding splits within the land class,
    not between water and land.  HAND derives flood connectivity from
    terrain geometry and is physically independent of canopy cover.

Note on HAND accuracy:
    HAND has been validated on Amazon megafan floodplains (Rennó et al. 2008;
    Nobre et al. 2011) and is used operationally by the NOAA National Water
    Model.  Independent validation for Gulf of Carpentaria floodplains is
    not yet published.  Results should be treated as a geomorphic flood
    probability layer, not a confirmed inundation extent.
"""
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import shape
from shapely.ops import unary_union

FLOOD_UNION_SIMPLIFY_TOLERANCE = 100   # metres
DEM_RESOLUTION = 30                    # metres (COP-DEM native)
CLOSING_RADIUS_PX = 5                  # 5 px × 30 m = 150 m closing radius
MIN_PATCH_PX = 11                      # ~1 ha at 30 m (10_000 / 900 ≈ 11 px)

logger = logging.getLogger(__name__)

LOCAL_DEM_ROOT = os.environ.get("LOCAL_DEM_ROOT", "")
DRAINAGE_GPKG = os.environ.get("DRAINAGE_GPKG", "")  # optional stream burn


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs
    from utils.dem import (
        download_dem_tiles,
        merge_and_reproject_dem,
        burn_drainage_network,
        compute_flow_accumulation,
        compute_hand,
        flood_connectivity_mask,
    )
    from utils.quicklook import save_quicklook

    configure_logging()
    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)
    bbox_wgs84 = list(catchment.to_crs("EPSG:4326").total_bounds)

    # ------------------------------------------------------------------
    # 1. DEM tiles
    # ------------------------------------------------------------------
    if LOCAL_DEM_ROOT:
        dem_dir = Path(LOCAL_DEM_ROOT)
        tile_paths = sorted(dem_dir.glob("*.tif"))
        logger.info("Using local DEM tiles from %s (%d tiles)", dem_dir, len(tile_paths))
    else:
        dem_dir = Path(
            os.environ.get("DEM_CACHE_DIR", "/mnt/ebs/dem/copernicus-dem-30m")
        )
        tile_paths = download_dem_tiles(bbox_wgs84, dem_dir)

    if not tile_paths:
        raise RuntimeError("No DEM tiles available — set LOCAL_DEM_ROOT or ensure network access")

    # ------------------------------------------------------------------
    # 2. Merge, void-fill, reproject
    # ------------------------------------------------------------------
    dem = merge_and_reproject_dem(
        tile_paths,
        catchment_geom=catchment,
        target_crs=config.TARGET_CRS,
        resolution=DEM_RESOLUTION,
    )

    # ------------------------------------------------------------------
    # 3. Optional stream burn
    # ------------------------------------------------------------------
    if DRAINAGE_GPKG and Path(DRAINAGE_GPKG).exists():
        logger.info("Burning drainage network: %s", DRAINAGE_GPKG)
        dem = burn_drainage_network(
            dem,
            Path(DRAINAGE_GPKG),
            burn_depth_m=config.DEM_BURN_DEPTH_M,
        )
    else:
        logger.info(
            "No drainage network provided (set DRAINAGE_GPKG to enable stream burn). "
            "Proceeding with raw DEM flow routing."
        )

    # ------------------------------------------------------------------
    # 4. Flow accumulation
    # ------------------------------------------------------------------
    logger.info("Computing D8 flow accumulation...")
    flow_accum = compute_flow_accumulation(dem)

    min_upstream_px = int(
        (config.HAND_MIN_UPSTREAM_KM2 * 1e6) / (DEM_RESOLUTION ** 2)
    )
    logger.info(
        "Stream threshold: flow_accum >= %d px (= %.1f km²)",
        min_upstream_px, config.HAND_MIN_UPSTREAM_KM2,
    )

    # ------------------------------------------------------------------
    # 5. HAND raster
    # ------------------------------------------------------------------
    logger.info("Computing HAND raster...")
    hand = compute_hand(dem, flow_accum, min_upstream_px=min_upstream_px)

    # Save HAND diagnostic raster
    hand_path = config.hand_raster_path(config.YEAR)
    hand_path.parent.mkdir(parents=True, exist_ok=True)
    import rasterio
    x = hand.coords["x"].values
    y = hand.coords["y"].values
    res_x = float(x[1] - x[0]) if len(x) > 1 else DEM_RESOLUTION
    res_y = float(y[1] - y[0]) if len(y) > 1 else -DEM_RESOLUTION
    import affine
    transform = affine.Affine(res_x, 0, float(x[0]), 0, res_y, float(y[0]))
    with rasterio.open(
        str(hand_path), "w",
        driver="GTiff", count=1, dtype="float32",
        width=hand.shape[1], height=hand.shape[0],
        crs=config.TARGET_CRS, transform=transform,
        compress="deflate", nodata=-9999.0,
    ) as dst:
        arr = hand.values.copy()
        arr[~np.isfinite(arr)] = -9999.0
        dst.write(arr, 1)
    logger.info("Written HAND raster: %s", hand_path)

    # Reproject HAND to TARGET_RESOLUTION (10 m) so it shares the same grid as
    # the NDVI anomaly and flowering index rasters.  Doing this once here avoids
    # a reproject_match call on every Stage 05 run and keeps all three inputs
    # grid-consistent for the verify-input check.
    # Use gdalwarp via subprocess to stream tiles and avoid loading the full
    # reprojected array into memory (would exceed RAM on 32 GB instances).
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=hand_path.parent) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "gdalwarp",
                "-t_srs", config.TARGET_CRS,
                "-tr", str(config.TARGET_RESOLUTION), str(config.TARGET_RESOLUTION),
                "-r", "bilinear",
                "-srcnodata", "-9999",
                "-dstnodata", "-9999",
                "-ot", "Float32",
                "-co", "COMPRESS=DEFLATE",
                "-co", "TILED=YES",
                "-overwrite",
                str(hand_path),
                tmp_path,
            ],
            check=True,
        )
        Path(tmp_path).replace(hand_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    with rasterio.open(str(hand_path)) as ds:
        logger.info(
            "Reprojected HAND raster to %d m: shape=(%d, %d)",
            config.TARGET_RESOLUTION, ds.height, ds.width,
        )

    # Log HAND percentile diagnostics
    valid_hand = hand.values[np.isfinite(hand.values)]
    if valid_hand.size > 0:
        for pct in [10, 25, 50, 75, 90, 99]:
            logger.info("  HAND p%02d = %.1f m", pct, np.percentile(valid_hand, pct))

    # ------------------------------------------------------------------
    # 6. Flood connectivity mask
    # ------------------------------------------------------------------
    logger.info(
        "Applying HAND threshold %.1f m → flood connectivity mask",
        config.HAND_FLOOD_THRESHOLD_M,
    )
    connected = flood_connectivity_mask(hand, threshold_m=config.HAND_FLOOD_THRESHOLD_M)
    coverage_pct = 100.0 * connected.values.sum() / max(np.isfinite(hand.values).sum(), 1)
    logger.info("Flood connectivity coverage: %.1f%%", coverage_pct)
    if coverage_pct < 5:
        logger.warning(
            "Coverage %.1f%% is very low — possible DEM or flow-routing error",
            coverage_pct,
        )
    elif coverage_pct > 40:
        logger.warning(
            "Coverage %.1f%% is very high — possible DEM or flow-routing error",
            coverage_pct,
        )

    # ------------------------------------------------------------------
    # 7–8. Morphological closing + min-patch filter
    # ------------------------------------------------------------------
    import rasterio.features
    from scipy.ndimage import binary_closing, label

    combined = connected.values.astype(bool)
    struct = np.ones(
        (CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool
    )
    data = binary_closing(combined, structure=struct).astype(np.uint8)
    logger.info("After morphological closing: flood pixels = %d", data.sum())

    labelled, n_features = label(data)
    patch_sizes = np.bincount(labelled.ravel())
    small_labels = np.where(patch_sizes < MIN_PATCH_PX)[0]
    small_labels = small_labels[small_labels > 0]
    if small_labels.size:
        data[np.isin(labelled, small_labels)] = 0
    del labelled, patch_sizes, small_labels
    logger.info("After min-patch filter (%d px): flood pixels = %d", MIN_PATCH_PX, data.sum())

    # ------------------------------------------------------------------
    # 9. Vectorise → union → simplify → clip
    # ------------------------------------------------------------------
    logger.info("Vectorising flood extent...")
    shapes = list(rasterio.features.shapes(data, mask=data, transform=transform))
    logger.info("Vectorised to %d shapes before union", len(shapes))

    if not shapes:
        logger.warning("No flood-connected pixels after filtering — writing empty layer")
        gdf = gpd.GeoDataFrame(geometry=[], crs=config.TARGET_CRS)
    else:
        geoms = [shape(s) for s, v in shapes if v == 1]
        merged = unary_union(geoms)
        geom_list = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        geom_list = [g.simplify(FLOOD_UNION_SIMPLIFY_TOLERANCE) for g in geom_list]
        gdf = gpd.GeoDataFrame(geometry=geom_list, crs=config.TARGET_CRS)

    gdf = gpd.clip(gdf, catchment)

    out_path = config.flood_extent_path(config.YEAR)
    gdf.to_file(str(out_path), driver="GPKG")
    logger.info("Written: %s  (%d features)", out_path, len(gdf))

    # Quicklook — rasterise the clipped GeoDataFrame so the image matches the .gpkg
    ql_path = out_path.with_name(out_path.stem + "_quicklook.png")
    if len(gdf) > 0:
        ql_data = rasterio.features.rasterize(
            gdf.geometry,
            out_shape=data.shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        ).astype(np.float32)
    else:
        ql_data = np.zeros(data.shape, dtype=np.float32)
    flood_da = xr.DataArray(
        ql_data,
        dims=["y", "x"],
        coords={"x": x, "y": y},
    )
    save_quicklook(
        flood_da,
        ql_path,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        title=f"HAND Flood Connectivity {config.YEAR} (threshold={config.HAND_FLOOD_THRESHOLD_M} m)",
    )


if __name__ == "__main__":
    main()
