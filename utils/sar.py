"""utils/sar.py — SAR preprocessing wrapper.

Isolated so tests can mock preprocess_s1_scene() without importing sarsen.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _focal_median(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a (2*radius+1) × (2*radius+1) median speckle filter, ignoring NaNs."""
    from scipy.ndimage import generic_filter
    size = 2 * radius + 1
    return generic_filter(arr, lambda x: np.nanmedian(x), size=size, mode="reflect").astype(np.float32)


def flood_mask_from_scene(
    item: Any,
    bbox: list,
    resolution: int,
    reference_mask: np.ndarray | None = None,
) -> xr.Dataset | None:
    """Return a flood/observed Dataset for a single S1 scene.

    Warps VV and VH to EPSG:7855, applies a 3×3 median speckle filter, then
    classifies water using per-scene Otsu thresholding on VV with a VH guard
    (pixel must fall below the Otsu threshold in *both* bands).  This reduces
    false positives from wind-roughened open water (VH stays high) and smooth
    dry scalds (VH mimics VV less reliably than true water).

    If reference_mask is provided (True = persistent low-backscatter non-water),
    those pixels are excluded from the water classification.
    """
    ds = _preprocess_gcp_warp(item, bbox, resolution)

    if "VV" not in ds:
        return None

    vv_lin = ds["VV"].values  # linear sigma-naught proxy
    observed = np.isfinite(vv_lin) & (vv_lin > 0)

    if observed.sum() == 0:
        return None

    # Speckle filter (3×3 median) on linear values before dB conversion
    vv_filt = _focal_median(np.where(observed, vv_lin, np.nan), radius=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        vv_db = 10 * np.log10(vv_filt + 1e-12)

    # Per-scene Otsu threshold on VV dB values within the observed footprint
    vv_valid = vv_db[observed]
    if vv_valid.size < 100:
        return None
    otsu_vv = _otsu_threshold(vv_valid)
    logger.info("Otsu VV threshold for %s: %.1f dB", item.id, otsu_vv)

    water = observed & (vv_db < otsu_vv)

    # VH guard — require VH also below its Otsu threshold
    if "VH" in ds:
        vh_lin = ds["VH"].values
        vh_filt = _focal_median(np.where(observed, vh_lin, np.nan), radius=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vh_db = 10 * np.log10(vh_filt + 1e-12)
        vh_valid = vh_db[observed]
        if vh_valid.size >= 100:
            otsu_vh = _otsu_threshold(vh_valid)
            logger.info("Otsu VH threshold for %s: %.1f dB", item.id, otsu_vh)
            water = water & (vh_db < otsu_vh)
        del vh_lin, vh_filt, vh_db

    # Exclude pixels that are persistently low-backscatter in the dry season
    if reference_mask is not None and reference_mask.shape == water.shape:
        water = water & ~reference_mask

    del vv_lin, vv_filt, vv_db

    logger.info("Water pixels %s: %d / %d observed (%.1f%%)",
                item.id, water.sum(), observed.sum(), 100 * water.sum() / max(observed.sum(), 1))

    x_coords = ds["VV"].coords["x"].values
    y_coords = ds["VV"].coords["y"].values
    return xr.Dataset({
        "water":    xr.DataArray(water,    dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
        "observed": xr.DataArray(observed, dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
    })


def build_dry_season_reference_mask(
    items: list,
    bbox: list,
    resolution: int,
    low_backscatter_threshold_db: float = -16.0,
    max_workers: int = 1,
) -> np.ndarray | None:
    """Build a boolean mask of persistently low-backscatter pixels from dry-season scenes.

    Takes the per-pixel median VV backscatter across all provided scenes.  Pixels
    whose median falls below low_backscatter_threshold_db are flagged as
    non-water low-backscatter surfaces (e.g. sodic scalds, smooth gully floors)
    and should be excluded from flood classification.

    Returns a 2-D bool array (True = exclude), or None if no scenes could be
    processed.  The array is in the same grid as the flood-season outputs
    (EPSG:7855 at the requested resolution over bbox).
    """
    total = len(items)

    def _process_dry(item):
        ds = _preprocess_gcp_warp(item, bbox, resolution, polarisations=("VV",))
        if "VV" not in ds:
            return None
        vv_lin = ds["VV"].values
        observed = np.isfinite(vv_lin) & (vv_lin > 0)
        if observed.sum() == 0:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            vv_db = np.where(observed, 10 * np.log10(vv_lin + 1e-12), np.nan)
        return vv_db, observed.sum()

    vv_stack = []
    ref_shape = None
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_dry, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                result = future.result()
                if result is None:
                    logger.debug("Dry-season scene %s: no valid pixels — skipped (%d/%d)",
                                 item.id, completed, total)
                    continue
                vv_db, n_valid = result
                if ref_shape is None:
                    ref_shape = vv_db.shape
                if vv_db.shape == ref_shape:
                    vv_stack.append(vv_db)
                    logger.info("Dry-season reference: added %s (%d valid px) [%d/%d]",
                                item.id, n_valid, completed, total)
                else:
                    logger.debug("Dry-season scene %s: shape mismatch — skipped (%d/%d)",
                                 item.id, completed, total)
            except Exception as exc:
                logger.warning("Dry-season scene %s failed (%d/%d): %s",
                               item.id, completed, total, exc)

    if not vv_stack:
        logger.warning("No dry-season scenes processed — reference mask unavailable")
        return None

    logger.info("Computing reference mask median from %d scenes...", len(vv_stack))
    median_vv = np.nanmedian(np.stack(vv_stack, axis=0), axis=0)
    mask = (median_vv < low_backscatter_threshold_db) & np.isfinite(median_vv)
    logger.info("Dry-season reference mask: %d / %d pixels flagged (%.1f%%)",
                mask.sum(), mask.size, 100 * mask.sum() / mask.size)
    return mask


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


def _preprocess_gcp_warp(
    item: Any,
    bbox: list,
    resolution: int,
    polarisations: tuple = ("VV", "VH"),
) -> xr.Dataset:
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
    for pol in polarisations:
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

        # Find GCPs strictly inside the bbox first; if none, widen to a small margin.
        # Using a tight window first minimises the read size for large scenes that
        # only partially overlap the catchment.
        mask_strict = (
            (gcp_lons >= minx) & (gcp_lons <= maxx) &
            (gcp_lats >= miny) & (gcp_lats <= maxy)
        )
        margin = 0.5  # degrees — fallback search radius
        mask_near = mask_strict | (
            (gcp_lons >= minx - margin) & (gcp_lons <= maxx + margin) &
            (gcp_lats >= miny - margin) & (gcp_lats <= maxy + margin)
        )
        if mask_near.any():
            # Pad by 200 px to avoid clipping edge pixels after GCP reprojection
            col_min = max(0, int(gcp_cols[mask_near].min()) - 200)
            col_max = int(gcp_cols[mask_near].max()) + 200
            row_min = max(0, int(gcp_rows[mask_near].min()) - 200)
            row_max = int(gcp_rows[mask_near].max()) + 200
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
