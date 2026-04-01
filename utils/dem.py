"""utils/dem.py — DEM loading, flow routing, and HAND computation for Step 4.

Flow routing and HAND computation are delegated to pysheds, which implements
priority-flood D8 routing in vectorised NumPy/Cython.  On a 16-core machine
with 32 GB RAM the full Mitchell catchment DEM (~10k × 13k px at 30 m) runs
in a few minutes rather than hours.

HAND reference: Rennó et al. (2008), Remote Sensing of Environment.
pysheds: https://github.com/mdbartos/pysheds
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DEM tile download
# ---------------------------------------------------------------------------

def download_dem_tiles(bbox_wgs84: list, out_dir: Path) -> list[Path]:
    """Download Copernicus GLO-30 tiles covering bbox_wgs84 to out_dir.

    bbox_wgs84: [minx, miny, maxx, maxy] in EPSG:4326.
    Returns list of local tile paths (already-cached tiles are included
    without re-downloading).
    """
    import math
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bbox_wgs84

    paths: list[Path] = []
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            stem = (
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00"
                f"_{ew}{abs(lon):03d}_00_DEM"
            )
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{stem}/{stem}.tif"
            local = out_dir / f"{stem}.tif"

            if local.exists():
                logger.debug("DEM tile cached: %s", local.name)
                paths.append(local)
                continue

            logger.info("Downloading DEM tile: %s", url)
            try:
                urllib.request.urlretrieve(url, local)
                logger.info("  saved %.1f MB: %s", local.stat().st_size / 1e6, local.name)
                paths.append(local)
            except Exception as exc:
                # Edge tiles may not exist (ocean) — skip gracefully
                logger.warning("  skipped (not found): %s — %s", local.name, exc)

    return paths


# ---------------------------------------------------------------------------
# Merge, void-fill, reproject
# ---------------------------------------------------------------------------

def merge_and_reproject_dem(
    tile_paths: list[Path],
    catchment_geom,
    target_crs: str,
    resolution: int,
) -> xr.DataArray:
    """Merge COP-DEM tiles, fill voids, reproject to target_crs at resolution metres.

    catchment_geom: a GeoDataFrame or None.
    Returns a DataArray with (y, x) dims in target_crs.
    """
    import os
    import tempfile

    import rasterio
    import rasterio.enums
    from rasterio.merge import merge
    import rioxarray  # noqa: F401

    logger.info("Merging %d DEM tiles", len(tile_paths))
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = merge(datasets)
    src_crs = datasets[0].crs
    src_nodata = datasets[0].nodata
    for ds in datasets:
        ds.close()

    elev = mosaic_arr[0].astype(np.float32)
    nodata_val = float(src_nodata) if src_nodata is not None else -9999.0

    # Void fill: nearest-valid-neighbour via scipy distance transform
    void_mask = (elev == nodata_val) | ~np.isfinite(elev)
    logger.info("DEM void fraction before fill: %.3f%%", void_mask.mean() * 100)
    if void_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(
            void_mask, return_distances=True, return_indices=True
        )
        elev[void_mask] = elev[tuple(nearest_idx[:, void_mask])]

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff", count=1, dtype="float32",
            crs=src_crs, transform=mosaic_transform,
            width=elev.shape[1], height=elev.shape[0],
            nodata=-9999.0,
        ) as dst:
            dst.write(elev, 1)

        da = xr.open_dataarray(tmp_path, engine="rasterio").squeeze("band", drop=True)
        da = da.rio.write_nodata(-9999.0, encoded=True)
        da = da.rio.write_crs(src_crs)

        da_proj = da.rio.reproject(
            target_crs,
            resolution=(resolution, resolution),
            resampling=rasterio.enums.Resampling.bilinear,
            nodata=-9999.0,
        )

        if catchment_geom is not None:
            import geopandas as gpd
            if hasattr(catchment_geom, "total_bounds"):
                geom_clip = catchment_geom.to_crs(target_crs)
            else:
                geom_clip = gpd.GeoSeries([catchment_geom], crs="EPSG:4326").to_crs(target_crs)
            da_proj = da_proj.rio.clip(
                geom_clip.geometry, crs=target_crs, drop=True, all_touched=True
            )

        da_proj = da_proj.where(da_proj != -9999.0)
        da_proj.name = "elevation"
        logger.info(
            "DEM reprojected: shape=%s  CRS=%s  res=%dm",
            da_proj.shape, target_crs, resolution,
        )
        return da_proj.load()

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Stream burn
# ---------------------------------------------------------------------------

def burn_drainage_network(
    dem: xr.DataArray,
    drainage_gpkg: Path,
    burn_depth_m: float = 10.0,
) -> xr.DataArray:
    """Lower DEM values along the cartographic drainage network by burn_depth_m.

    Rasterises the vector drainage network onto the DEM grid and subtracts
    burn_depth_m from channel pixels.  Ensures DEM-derived flow paths follow
    the known channel network rather than spurious DEM depressions.

    Returns a new DataArray with the same coordinates as dem.
    """
    import geopandas as gpd
    import rasterio.features
    import rasterio.transform

    logger.info("Burning drainage network from %s (depth=%.1f m)", drainage_gpkg, burn_depth_m)
    drains = gpd.read_file(drainage_gpkg)
    crs = dem.rio.crs if dem.rio.crs is not None else "EPSG:7855"
    drains = drains.to_crs(crs)

    x = dem.coords["x"].values
    y = dem.coords["y"].values
    res_x = float(x[1] - x[0]) if len(x) > 1 else 30.0
    res_y = float(y[1] - y[0]) if len(y) > 1 else -30.0
    transform = rasterio.transform.from_origin(
        float(x[0]) - abs(res_x) / 2,
        float(y[0]) + abs(res_y) / 2,
        abs(res_x),
        abs(res_y),
    )

    channel_mask = rasterio.features.rasterize(
        ((geom, 1) for geom in drains.geometry if geom is not None),
        out_shape=dem.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    dem_burned = dem.copy(data=dem.values.copy())
    dem_burned.values[channel_mask] -= burn_depth_m
    logger.info("Burned %d channel pixels", channel_mask.sum())
    return dem_burned


# ---------------------------------------------------------------------------
# Flow accumulation and HAND via pysheds
# ---------------------------------------------------------------------------

def _dem_to_pysheds_grid(dem: xr.DataArray):
    """Write the DEM to a temp GeoTIFF and load it into a pysheds Grid.

    Returns (grid, dem_raster) ready for flow routing.
    pysheds requires a GeoTIFF on disk or an ndarray with affine metadata;
    the GeoTIFF path is the most robust interface across pysheds versions.
    """
    import os
    import tempfile

    import rasterio
    import rasterio.transform
    from pysheds.grid import Grid

    x = dem.coords["x"].values
    y = dem.coords["y"].values
    res_x = float(x[1] - x[0]) if len(x) > 1 else 30.0
    res_y = float(y[1] - y[0]) if len(y) > 1 else -30.0

    # pysheds expects the affine origin at the top-left corner of the top-left pixel
    transform = rasterio.transform.from_origin(
        float(x[0]) - abs(res_x) / 2,
        float(y[0]) + abs(res_y) / 2,
        abs(res_x),
        abs(res_y),
    )

    crs_str = str(dem.rio.crs) if dem.rio.crs is not None else "EPSG:7855"

    arr = dem.values.astype(np.float32)
    # Replace NaN with a nodata sentinel pysheds can recognise
    NODATA = -9999.0
    arr = np.where(np.isfinite(arr), arr, NODATA)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    with rasterio.open(
        tmp_path, "w",
        driver="GTiff", count=1, dtype="float32",
        crs=crs_str, transform=transform,
        width=arr.shape[1], height=arr.shape[0],
        nodata=NODATA,
    ) as dst:
        dst.write(arr, 1)

    grid = Grid.from_raster(tmp_path)
    dem_raster = grid.read_raster(tmp_path)
    os.unlink(tmp_path)
    return grid, dem_raster


def compute_flow_accumulation(dem: xr.DataArray) -> xr.DataArray:
    """D8 flow accumulation using pysheds priority-flood routing.

    Returns upstream contributing area in pixels.  NaN where DEM is void.
    """
    from pysheds.grid import Grid

    logger.info("Computing flow accumulation via pysheds (D8)...")
    grid, dem_raster = _dem_to_pysheds_grid(dem)

    # Fill pits and depressions so all flow reaches an outlet
    pit_filled = grid.fill_pits(dem_raster)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    fdir = grid.flowdir(inflated)
    accum = grid.accumulation(fdir)

    result = np.array(accum).astype(np.float32)

    # Mask void pixels
    nodata_mask = ~np.isfinite(dem.values)
    result[nodata_mask] = np.nan

    logger.info(
        "Flow accumulation done: max=%.0f px  stream pixels (≥1111 px = 1 km²): %d",
        np.nanmax(result),
        int((result >= 1111).sum()),
    )
    return xr.DataArray(result, dims=dem.dims, coords=dem.coords, name="flow_accumulation")


def compute_hand(
    dem: xr.DataArray,
    flow_accumulation: xr.DataArray,
    min_upstream_px: int,
) -> xr.DataArray:
    """Compute Height Above Nearest Drainage (HAND) using pysheds.

    For each pixel, HAND = elevation(pixel) − elevation(nearest stream pixel
    in the D8 flow network), where stream pixels are those with
    flow_accumulation >= min_upstream_px.

    Returns a DataArray of HAND values in metres, NaN where DEM is void.
    """
    from pysheds.grid import Grid

    logger.info(
        "Computing HAND via pysheds (min_upstream_px=%d = %.1f km²)...",
        min_upstream_px, min_upstream_px * (30 ** 2) / 1e6,
    )
    grid, dem_raster = _dem_to_pysheds_grid(dem)

    pit_filled = grid.fill_pits(dem_raster)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    fdir = grid.flowdir(inflated)
    accum = grid.accumulation(fdir)

    # Build stream mask from accumulation threshold – must be a pysheds Raster
    from pysheds.raster import Raster as PyshedsRaster
    stream_mask = PyshedsRaster(
        (np.array(accum) >= min_upstream_px),
        viewfinder=accum.viewfinder,
    )

    hand_arr = grid.compute_hand(fdir, dem_raster, stream_mask)
    result = np.array(hand_arr).astype(np.float32)

    # Clamp negatives (numerical noise in flat areas) and mask voids
    result = np.where(np.isfinite(result), np.maximum(result, 0.0), np.nan)
    result[~np.isfinite(dem.values)] = np.nan

    valid = result[np.isfinite(result)]
    if valid.size > 0:
        logger.info(
            "HAND done: median=%.1f m  p90=%.1f m  p99=%.1f m",
            np.percentile(valid, 50),
            np.percentile(valid, 90),
            np.percentile(valid, 99),
        )

    return xr.DataArray(result, dims=dem.dims, coords=dem.coords, name="HAND")


# ---------------------------------------------------------------------------
# Flood connectivity mask
# ---------------------------------------------------------------------------

def flood_connectivity_mask(
    hand: xr.DataArray,
    threshold_m: float,
) -> xr.DataArray:
    """Return boolean mask: True where HAND <= threshold_m.

    NaN pixels (DEM voids) are False.
    """
    mask = (hand <= threshold_m) & np.isfinite(hand)
    return xr.DataArray(
        mask.values,
        dims=hand.dims,
        coords=hand.coords,
        name="flood_connected",
    )
