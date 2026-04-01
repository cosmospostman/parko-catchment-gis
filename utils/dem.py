"""utils/dem.py — DEM loading, flow routing, and HAND computation for Step 4.

Replaces utils/sar.py for flood connectivity work.  All functions accept and
return xarray DataArrays with spatial coordinates (x, y) in a projected CRS.

HAND reference: Rennó et al. (2008), Remote Sensing of Environment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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

    catchment_geom: a shapely geometry or GeoDataFrame used to clip the output.
    Returns a DataArray with (y, x) dims and spatial_ref coordinate.
    """
    import rasterio
    import rasterio.crs
    import rasterio.warp
    from rasterio.merge import merge
    import rioxarray  # noqa: F401  — activates the .rio accessor

    logger.info("Merging %d DEM tiles", len(tile_paths))
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic_arr, mosaic_transform = merge(datasets)
    src_crs = datasets[0].crs
    src_nodata = datasets[0].nodata
    for ds in datasets:
        ds.close()

    # mosaic_arr shape: (1, H, W)
    elev = mosaic_arr[0].astype(np.float32)
    nodata_val = float(src_nodata) if src_nodata is not None else -9999.0

    # Simple void fill: replace NoData with mean of valid neighbours via
    # scipy distance transform (nearest-valid interpolation).
    void_mask = (elev == nodata_val) | ~np.isfinite(elev)
    void_fraction = void_mask.mean()
    logger.info("DEM void fraction before fill: %.3f%%", void_fraction * 100)
    if void_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(void_mask, return_distances=True, return_indices=True)
        elev[void_mask] = elev[tuple(nearest_idx[:, void_mask])]

    # Build a temporary in-memory rasterio dataset so rioxarray can reproject.
    import io as _io
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            count=1,
            dtype="float32",
            crs=src_crs,
            transform=mosaic_transform,
            width=elev.shape[1],
            height=elev.shape[0],
            nodata=-9999.0,
        ) as dst:
            dst.write(elev, 1)

        da = xr.open_dataarray(tmp_path, engine="rasterio").squeeze("band", drop=True)
        da = da.rio.write_nodata(-9999.0, encoded=True)
        da = da.rio.write_crs(src_crs)

        target_res = (resolution, resolution)
        da_proj = da.rio.reproject(
            target_crs,
            resolution=target_res,
            resampling=rasterio.enums.Resampling.bilinear,
            nodata=-9999.0,
        )

        # Clip to catchment bbox
        if catchment_geom is not None:
            import geopandas as gpd
            if hasattr(catchment_geom, "total_bounds"):
                # GeoDataFrame
                geom_clip = catchment_geom.to_crs(target_crs)
            else:
                geom_clip = gpd.GeoSeries([catchment_geom], crs="EPSG:4326").to_crs(target_crs)
            da_proj = da_proj.rio.clip(geom_clip.geometry, crs=target_crs, drop=True, all_touched=True)

        # Replace encoded nodata with NaN for computation
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
# Flow accumulation (D8)
# ---------------------------------------------------------------------------

def compute_flow_accumulation(dem: xr.DataArray) -> xr.DataArray:
    """D8 flow routing.  Returns upstream contributing area in pixels.

    Uses a simple priority-flood / flow-routing approach on the DEM array.
    NaN pixels in the DEM are treated as barriers (no contribution).

    Note: for large DEMs this is memory-intensive. The implementation uses
    a pure-NumPy approach sufficient for catchment-scale analysis.
    """
    elev = np.where(np.isfinite(dem.values), dem.values, np.inf).astype(np.float64)
    H, W = elev.shape

    # D8 neighbour offsets: (row_delta, col_delta) for 8 directions
    D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    # For each pixel find the steepest downslope neighbour.
    # flow_dir[r, c] = flat index of downslope pixel, or -1 if local minimum/nodata.
    flow_dir = np.full(H * W, -1, dtype=np.int64)
    flat_elev = elev.ravel()

    for r in range(H):
        for c in range(W):
            if not np.isfinite(elev[r, c]):
                continue
            min_slope = 0.0
            best = -1
            for dr, dc in D8_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    dz = elev[r, c] - elev[nr, nc]
                    dist = (2.0 ** 0.5) if (dr != 0 and dc != 0) else 1.0
                    slope = dz / dist
                    if slope > min_slope:
                        min_slope = slope
                        best = nr * W + nc
            flow_dir[r * W + c] = best

    # Accumulate: process in order of decreasing elevation (upstream first)
    order = np.argsort(flat_elev)[::-1]  # highest → lowest
    accum = np.ones(H * W, dtype=np.float64)
    for idx in order:
        down = flow_dir[idx]
        if down >= 0 and np.isfinite(flat_elev[idx]):
            accum[down] += accum[idx]

    result = accum.reshape(H, W)
    # Mask pixels that were NaN in the DEM
    result[~np.isfinite(elev)] = np.nan

    return xr.DataArray(
        result.astype(np.float32),
        dims=dem.dims,
        coords=dem.coords,
        name="flow_accumulation",
    )


# ---------------------------------------------------------------------------
# HAND computation
# ---------------------------------------------------------------------------

def compute_hand(
    dem: xr.DataArray,
    flow_accumulation: xr.DataArray,
    min_upstream_px: int,
) -> xr.DataArray:
    """Compute Height Above Nearest Drainage (HAND).

    For each pixel, HAND = elevation(pixel) − elevation(nearest stream pixel
    in the D8 flow network), where stream pixels are those with
    flow_accumulation >= min_upstream_px.

    Returns a DataArray of HAND values in metres, NaN where DEM is void.

    Algorithm: trace each non-stream pixel downstream until it reaches a
    stream pixel; HAND = current elevation − stream elevation.
    """
    elev = dem.values.astype(np.float64)
    accum = flow_accumulation.values.astype(np.float64)
    H, W = elev.shape

    is_stream = (accum >= min_upstream_px) & np.isfinite(elev)

    # Rebuild flow direction (same logic as compute_flow_accumulation)
    D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    flow_dir = np.full(H * W, -1, dtype=np.int64)
    elev_inf = np.where(np.isfinite(elev), elev, np.inf)

    for r in range(H):
        for c in range(W):
            if not np.isfinite(elev[r, c]):
                continue
            min_slope = 0.0
            best = -1
            for dr, dc in D8_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    dz = elev_inf[r, c] - elev_inf[nr, nc]
                    dist = (2.0 ** 0.5) if (dr != 0 and dc != 0) else 1.0
                    slope = dz / dist
                    if slope > min_slope:
                        min_slope = slope
                        best = nr * W + nc
            flow_dir[r * W + c] = best

    flat_elev = elev.ravel()
    flat_stream = is_stream.ravel()

    # For each pixel, walk downstream until we hit a stream pixel or a sink.
    # Cache the nearest-stream elevation for visited pixels.
    hand_flat = np.full(H * W, np.nan, dtype=np.float64)

    for start in range(H * W):
        if not np.isfinite(flat_elev[start]):
            continue
        if flat_stream[start]:
            hand_flat[start] = 0.0
            continue
        # Walk downstream; collect path for back-filling
        path = [start]
        cur = start
        found_elev = np.nan
        visited = set()
        while True:
            nxt = flow_dir[cur]
            if nxt < 0 or nxt in visited:
                break
            visited.add(cur)
            if flat_stream[nxt]:
                found_elev = flat_elev[nxt]
                break
            if not np.isnan(hand_flat[nxt]):
                # Already resolved — use cached nearest-stream elevation
                found_elev = flat_elev[nxt] - hand_flat[nxt]
                if np.isnan(found_elev):
                    found_elev = np.nan
                else:
                    # Recover stream elevation from cached HAND
                    found_elev = flat_elev[nxt] - hand_flat[nxt]
                break
            path.append(nxt)
            cur = nxt

        if np.isfinite(found_elev):
            for p in path:
                hand_flat[p] = flat_elev[p] - found_elev

    hand = hand_flat.reshape(H, W)
    # Clamp to non-negative (numerical artefacts in flat areas)
    hand = np.where(np.isfinite(hand), np.maximum(hand, 0.0), np.nan)

    return xr.DataArray(
        hand.astype(np.float32),
        dims=dem.dims,
        coords=dem.coords,
        name="HAND",
    )


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
