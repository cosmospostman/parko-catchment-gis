"""utils/sar.py — SAR preprocessing wrapper.

Isolated so tests can mock preprocess_s1_scene() without importing sarsen.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def preprocess_s1_scene(
    item: Any,
    bbox: list,
    resolution: int = 10,
) -> xr.Dataset:
    """Preprocess a Sentinel-1 GRD scene to sigma-naught (linear scale).

    Reads GCPs from the annotation XML, warps the raw measurement TIFF to
    EPSG:7855 at the requested resolution, and clips to bbox.
    No terrain correction — suitable for flat to moderately hilly terrain.
    """
    return _preprocess_gcp_warp(item, bbox, resolution)


def _otsu_threshold(values: np.ndarray, n_bins: int = 512) -> float:
    """Return Otsu's optimal threshold for a 1-D array of positive values."""
    counts, edges = np.histogram(values, bins=n_bins)
    bin_centres = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    sum_total = (counts * bin_centres).sum()
    sum_b, weight_b = 0.0, 0.0
    best_var, best_t = 0.0, bin_centres[0]
    for i in range(n_bins):
        weight_b += counts[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += counts[i] * bin_centres[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = bin_centres[i]
    return float(best_t)


def flood_mask_from_scene(
    item: Any,
    bbox: list,
    resolution: int,
    threshold_db: float,
) -> xr.DataArray | None:
    """Return a boolean flood mask DataArray for a single S1 scene.

    Warps VV band to EPSG:7855, converts to dB, applies threshold.
    Large intermediate arrays are freed before returning — only the boolean
    mask (1 bit per pixel) is kept in memory.
    """
    import rasterio
    import rasterio.control
    import rasterio.crs
    import rasterio.warp
    import rasterio.transform
    import rasterio.io
    import rasterio.windows

    safe_root = Path(_safe_root_from_item(item))
    anno_path = safe_root / "annotation" / "iw-vv.xml"
    meas_asset = item.assets.get("vv")

    if not anno_path.exists() or meas_asset is None:
        raise ValueError(f"Missing VV data for {item.id}")

    gcps = _read_gcps_from_annotation(anno_path)
    if not gcps:
        raise ValueError(f"No GCPs in annotation for {item.id}")

    target_crs = rasterio.crs.CRS.from_epsg(7855)
    src_crs = rasterio.crs.CRS.from_epsg(4326)
    minx, miny, maxx, maxy = bbox

    dst_bounds = rasterio.warp.transform_bounds("EPSG:4326", target_crs, minx, miny, maxx, maxy)
    dst_width  = max(1, int((dst_bounds[2] - dst_bounds[0]) / resolution))
    dst_height = max(1, int((dst_bounds[3] - dst_bounds[1]) / resolution))
    dst_transform = rasterio.transform.from_bounds(*dst_bounds, dst_width, dst_height)

    # Use GCPs to find the source pixel window covering the bbox — avoids reading
    # the full ~16k×26k swath (~1.6 GB float32) when only a subset is needed.
    gcp_lons = np.array([g.x for g in gcps])
    gcp_lats = np.array([g.y for g in gcps])
    gcp_cols = np.array([g.col for g in gcps])
    gcp_rows = np.array([g.row for g in gcps])

    margin = 0.5  # degrees
    mask_near = (
        (gcp_lons >= minx - margin) & (gcp_lons <= maxx + margin) &
        (gcp_lats >= miny - margin) & (gcp_lats <= maxy + margin)
    )

    with rasterio.open(meas_asset.href) as src:
        full_width, full_height = src.width, src.height

    if mask_near.any():
        col_min = max(0, int(gcp_cols[mask_near].min()) - 100)
        col_max = min(full_width,  int(gcp_cols[mask_near].max()) + 100)
        row_min = max(0, int(gcp_rows[mask_near].min()) - 100)
        row_max = min(full_height, int(gcp_rows[mask_near].max()) + 100)
    else:
        col_min, row_min = 0, 0
        col_max, row_max = full_width, full_height

    window = rasterio.windows.Window(
        col_off=col_min, row_off=row_min,
        width=col_max - col_min, height=row_max - row_min,
    )
    with rasterio.open(meas_asset.href) as src:
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        src_data = src.read(1, window=window).astype(np.float32)
        win_col_off = int(window.col_off)
        win_row_off = int(window.row_off)

    win_height, win_width = src_data.shape
    logger.info("Windowed read %s VV: %dx%d px (%.1f MB)",
                item.id, win_width, win_height, win_width * win_height * 4 / 1e6)

    # Adjust GCP coordinates to be relative to the window
    windowed_gcps = [
        rasterio.control.GroundControlPoint(
            row=g.row - win_row_off, col=g.col - win_col_off,
            x=g.x, y=g.y, z=g.z,
        )
        for g in gcps
    ]

    dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(driver="GTiff", count=1, dtype="float32",
                          width=win_width, height=win_height) as mem_ds:
            mem_ds.write(src_data, 1)
            mem_ds.gcps = (windowed_gcps, src_crs)
            del src_data
            rasterio.warp.reproject(
                source=rasterio.band(mem_ds, 1),
                destination=dst_data,
                dst_crs=target_crs,
                dst_transform=dst_transform,
                resampling=rasterio.warp.Resampling.bilinear,
                src_nodata=0,
                dst_nodata=np.nan,
                num_threads=1,
            )

    logger.info("Warped %s VV: %dx%d px (%.0f MB)",
                item.id, dst_width, dst_height, dst_width * dst_height * 4 / 1e6)

    # Otsu threshold on valid (non-nan) DN values — finds the natural land/water
    # break without requiring calibrated sigma-naught.
    valid = dst_data[np.isfinite(dst_data) & (dst_data > 0)]
    if valid.size == 0:
        del dst_data
        return None
    observed = np.isfinite(dst_data) & (dst_data > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_db = 10 * np.log10((dst_data ** 2) / 1e6 + 1e-12)
    water = observed & (vv_db < threshold_db)
    del dst_data, vv_db
    logger.info("Water pixels %s: %d / %d observed (%.1f%%)",
                item.id, water.sum(), observed.sum(), 100 * water.sum() / max(observed.sum(), 1))

    x_coords = np.linspace(dst_bounds[0], dst_bounds[2], dst_width)
    y_coords = np.linspace(dst_bounds[3], dst_bounds[1], dst_height)
    # Return two bool arrays as a Dataset: water and observed footprint
    return xr.Dataset({
        "water":    xr.DataArray(water,    dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
        "observed": xr.DataArray(observed, dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
    })


def _safe_root_from_item(item: Any) -> str:
    vv_asset = item.assets.get("vv")
    if vv_asset:
        return str(Path(vv_asset.href).parent.parent)
    raise ValueError(f"Cannot determine SAFE root for item {item.id}")


def _read_gcps_from_annotation(annotation_path: Path):
    """Parse GCPs from a Sentinel-1 annotation XML.

    Returns a list of rasterio.control.GroundControlPoint.
    """
    import xml.etree.ElementTree as ET
    import rasterio.control

    tree = ET.parse(str(annotation_path))
    gcps = []
    for ggp in tree.findall(".//geolocationGridPoint"):
        col = float(ggp.find("pixel").text)
        row = float(ggp.find("line").text)
        lon = float(ggp.find("longitude").text)
        lat = float(ggp.find("latitude").text)
        z   = float(ggp.find("height").text)
        gcps.append(rasterio.control.GroundControlPoint(row=row, col=col, x=lon, y=lat, z=z))
    return gcps


def _preprocess_gcp_warp(item: Any, bbox: list, resolution: int) -> xr.Dataset:
    """Warp a Sentinel-1 GRD scene to EPSG:7855 using GCPs from the annotation XML."""
    import rasterio
    import rasterio.control
    import rasterio.crs
    import rasterio.warp
    import rasterio.transform

    safe_root = Path(_safe_root_from_item(item))
    anno_dir = safe_root / "annotation"

    # Find annotation XMLs — S3 layout uses iw-vv.xml / iw-vh.xml
    anno_map = {
        "VV": anno_dir / "iw-vv.xml",
        "VH": anno_dir / "iw-vh.xml",
    }
    meas_map = {
        "VV": item.assets.get("vv"),
        "VH": item.assets.get("vh"),
    }

    target_crs = rasterio.crs.CRS.from_epsg(7855)
    minx, miny, maxx, maxy = bbox  # WGS84

    # Compute output bounds in EPSG:7855
    from rasterio.warp import transform_bounds
    dst_bounds = transform_bounds("EPSG:4326", target_crs, minx, miny, maxx, maxy)
    dst_width  = max(1, int((dst_bounds[2] - dst_bounds[0]) / resolution))
    dst_height = max(1, int((dst_bounds[3] - dst_bounds[1]) / resolution))
    dst_transform = rasterio.transform.from_bounds(*dst_bounds, dst_width, dst_height)

    bands = {}
    for pol in ("VV", "VH"):
        anno_path = anno_map[pol]
        meas_asset = meas_map[pol]
        if not anno_path.exists() or meas_asset is None:
            logger.debug("Missing %s data for %s — skipping polarisation", pol, item.id)
            continue

        gcps = _read_gcps_from_annotation(anno_path)
        if not gcps:
            logger.debug("No GCPs in annotation for %s %s", item.id, pol)
            continue

        src_crs = rasterio.crs.CRS.from_epsg(4326)

        # Use GCPs to find the source pixel window covering the bbox,
        # so we only read the relevant subset of the ~16k×26k array.
        gcp_lons = np.array([g.x for g in gcps])
        gcp_lats = np.array([g.y for g in gcps])
        gcp_cols = np.array([g.col for g in gcps])
        gcp_rows = np.array([g.row for g in gcps])

        # Find GCPs inside (or near) the bbox with a small margin
        margin = 0.5  # degrees
        mask_near = (
            (gcp_lons >= minx - margin) & (gcp_lons <= maxx + margin) &
            (gcp_lats >= miny - margin) & (gcp_lats <= maxy + margin)
        )
        if mask_near.any():
            col_min = max(0, int(gcp_cols[mask_near].min()) - 100)
            col_max = int(gcp_cols[mask_near].max()) + 100
            row_min = max(0, int(gcp_rows[mask_near].min()) - 100)
            row_max = int(gcp_rows[mask_near].max()) + 100
        else:
            # bbox may not overlap this scene — fall back to full read and let
            # the warp produce an empty result
            col_min, row_min = 0, 0
            with rasterio.open(meas_asset.href) as src:
                col_max, row_max = src.width, src.height

        import rasterio.windows
        import rasterio.io
        window = rasterio.windows.Window(
            col_off=col_min, row_off=row_min,
            width=col_max - col_min, height=row_max - row_min,
        )
        with rasterio.open(meas_asset.href) as src:
            # Clamp window to actual dataset bounds
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            src_data = src.read(1, window=window).astype(np.float32)
            win_col_off = int(window.col_off)
            win_row_off = int(window.row_off)

        # Adjust GCP pixel/line coordinates to be relative to the window
        windowed_gcps = [
            rasterio.control.GroundControlPoint(
                row=g.row - win_row_off, col=g.col - win_col_off,
                x=g.x, y=g.y, z=g.z,
            )
            for g in gcps
        ]
        win_height, win_width = src_data.shape
        logger.info("Windowed read %s %s: %dx%d px (%.1f MB)",
                    item.id, pol, win_width, win_height,
                    win_width * win_height * 4 / 1e6)

        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(driver="GTiff", count=1, dtype="float32",
                              width=win_width, height=win_height) as mem_ds:
                mem_ds.write(src_data, 1)
                mem_ds.gcps = (windowed_gcps, src_crs)
                rasterio.warp.reproject(
                    source=rasterio.band(mem_ds, 1),
                    destination=dst_data,
                    dst_crs=target_crs,
                    dst_transform=dst_transform,
                    resampling=rasterio.warp.Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=np.nan,
                    num_threads=1,
                )

        # Convert DN to sigma-naught linear scale (S1 GRD: sigma0 = (DN^2) / cal_factor)
        # Without calibration LUT use DN^2 as a proxy — sufficient for flood thresholding
        with np.errstate(invalid="ignore"):
            sigma = (dst_data ** 2) / 1e6  # normalise to roughly linear scale

        x_coords = np.linspace(dst_bounds[0], dst_bounds[2], dst_width)
        y_coords = np.linspace(dst_bounds[3], dst_bounds[1], dst_height)
        bands[pol] = xr.DataArray(sigma, dims=["y", "x"],
                                  coords={"x": x_coords, "y": y_coords})

    if not bands:
        raise ValueError(f"No valid polarisations for {item.id}")

    ds = xr.Dataset(bands)
    logger.info("GCP-warped S1 scene %s: shape=%s", item.id, ds["VV"].shape if "VV" in ds else "VV missing")
    return ds
