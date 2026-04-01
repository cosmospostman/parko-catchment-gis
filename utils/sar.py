"""utils/sar.py — SAR preprocessing wrapper.

Isolated so tests can mock preprocess_s1_scene() without importing sarsen.
"""

import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _focal_mean_inplace(arr: np.ndarray, nan_mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a (2*radius+1) × (2*radius+1) mean speckle filter.

    Operates on `arr` in-place (caller must not need the original values).
    NaN pixels (indicated by nan_mask) are filled with the array mean before
    filtering so they don't contaminate neighbours, then restored after.
    Uses uniform_filter (box mean) for O(n) memory overhead.
    """
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1
    if nan_mask.any():
        arr[nan_mask] = float(np.nanmean(arr))
    uniform_filter(arr, size=size, mode="reflect", output=arr)
    arr[nan_mask] = np.nan
    return arr


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
    ds = _preprocess_gcp_warp(item, bbox, resolution, polarisations=("VV",))

    if "VV" not in ds:
        return None

    vv_lin = ds["VV"].values  # linear sigma-naught proxy
    x_coords = ds["VV"].coords["x"].values
    y_coords = ds["VV"].coords["y"].values
    del ds  # release the xr.Dataset before allocating filter/dB arrays

    observed = np.isfinite(vv_lin) & (vv_lin > 0)

    if observed.sum() == 0:
        return None

    # Speckle filter then dB conversion, both in-place on vv_lin to avoid
    # allocating a separate filtered copy.
    vv_nan = ~observed  # unobserved pixels treated as NaN for the filter
    vv_lin[vv_nan] = np.nan
    _focal_mean_inplace(vv_lin, vv_nan, radius=1)  # vv_lin now holds filtered values
    del vv_nan
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(vv_lin + 1e-12, out=vv_lin)
        vv_lin *= 10  # vv_lin now holds dB values
    vv_db = vv_lin
    del vv_lin  # alias only — vv_db IS vv_lin

    vv_valid = vv_db[observed]
    if vv_valid.size < 100:
        return None
    otsu_vv = _otsu_threshold(vv_valid)
    logger.info("Otsu VV threshold for %s: %.1f dB", item.id, otsu_vv)

    water = observed & (vv_db < otsu_vv)
    del vv_db

    # Sanity check: if Otsu classified an implausibly large fraction of the
    # scene as water the histogram was likely unimodal (dry scene) and Otsu
    # split within the land distribution.  Discard the scene in that case.
    # 40% is generous — the Mitchell megafan at peak flood is ~15–20% water.
    water_fraction = water.sum() / max(observed.sum(), 1)
    if water_fraction > 0.40:
        logger.info("Scene %s discarded — water fraction %.1f%% exceeds sanity limit",
                    item.id, 100 * water_fraction)
        return None

    # VH is not used for thresholding — in this dataset the DN²/1e6
    # normalisation compresses the VH dynamic range so that water and land
    # have nearly identical dB values, making any VH threshold unreliable.
    # Water classification relies solely on VV Otsu + the water-fraction guard.
    if vh_lin is not None:
        del vh_lin

    # Exclude pixels that are persistently low-backscatter in the dry season
    if reference_mask is not None and reference_mask.shape == water.shape:
        water = water & ~reference_mask

    logger.info("Water pixels %s: %d / %d observed (%.1f%%)",
                item.id, water.sum(), observed.sum(), 100 * water.sum() / max(observed.sum(), 1))

    return xr.Dataset({
        "water":    xr.DataArray(water,    dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
        "observed": xr.DataArray(observed, dims=["y", "x"], coords={"x": x_coords, "y": y_coords}),
    })


def _process_dry_worker(item, bbox, resolution):
    """Top-level worker function (picklable) for ProcessPoolExecutor."""
    from utils.io import configure_logging
    configure_logging()
    ds = _preprocess_gcp_warp(item, bbox, resolution, polarisations=("VV",))
    if "VV" not in ds:
        return None
    vv_lin = ds["VV"].values
    observed = np.isfinite(vv_lin) & (vv_lin > 0)
    if observed.sum() == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_db = np.where(observed, 10 * np.log10(vv_lin + 1e-12), np.nan)
    return vv_db, int(observed.sum())


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
    ref_shape = None
    completed = 0
    # scene results buffered as (vv_db_float16, n_valid) until shape is known
    pending: list[tuple[np.ndarray, int]] = []
    # index into the mmap once shape is known
    mmap_file = None
    mmap_arr = None   # shape (n_scenes, H, W), float16, memory-mapped
    mmap_idx = 0

    def _write_to_mmap(vv_db: np.ndarray, n_valid: int, item_id: str) -> None:
        nonlocal mmap_arr, mmap_idx
        if mmap_arr is None:
            return
        if vv_db.shape != ref_shape:
            logger.debug("Dry-season scene %s: shape mismatch — skipped", item_id)
            return
        mmap_arr[mmap_idx] = vv_db.astype(np.float16)
        mmap_idx += 1
        logger.info("Dry-season reference: added %s (%d valid px) [%d/%d]",
                    item_id, n_valid, completed, total)

    item_iter = iter(items)
    in_flight: dict = {}

    def _dry_submit_next() -> None:
        item = next(item_iter, None)
        if item is not None:
            f = executor.submit(_process_dry_worker, item, bbox, resolution)
            in_flight[f] = item

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            _dry_submit_next()

        while in_flight:
            future = next(as_completed(in_flight))
            item = in_flight.pop(future)
            completed += 1
            try:
                result = future.result()
                if result is None:
                    logger.debug("Dry-season scene %s: no valid pixels — skipped (%d/%d)",
                                 item.id, completed, total)
                else:
                    vv_db, n_valid = result

                    if ref_shape is None:
                        # First good scene — now we know the grid shape.
                        # Allocate a memory-mapped array for all scenes (worst case = total).
                        ref_shape = vv_db.shape
                        mmap_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
                        mmap_arr = np.lib.format.open_memmap(
                            mmap_file.name, mode="w+", dtype=np.float16,
                            shape=(total, ref_shape[0], ref_shape[1]),
                        )
                        logger.info("Allocated dry-season mmap: %d × %dx%d float16 (%.0f MB)",
                                    total, ref_shape[0], ref_shape[1],
                                    total * ref_shape[0] * ref_shape[1] * 2 / 1e6)
                        # Flush any results that arrived before shape was known
                        for pv, pn in pending:
                            _write_to_mmap(pv, pn, "buffered")
                        pending.clear()
                        _write_to_mmap(vv_db, n_valid, item.id)
                    elif mmap_arr is None:
                        pending.append((vv_db.astype(np.float16), n_valid))
                    else:
                        _write_to_mmap(vv_db, n_valid, item.id)

            except Exception as exc:
                logger.warning("Dry-season scene %s failed (%d/%d): %s",
                               item.id, completed, total, exc)
            # Always submit next after result is consumed
            _dry_submit_next()

    if mmap_arr is None or mmap_idx == 0:
        logger.warning("No dry-season scenes processed — reference mask unavailable")
        if mmap_file:
            os.unlink(mmap_file.name)
        return None

    valid_stack = mmap_arr[:mmap_idx]  # shape (n_scenes, H, W), mmap-backed
    n_scenes, h, w = valid_stack.shape
    logger.info("Computing reference mask median from %d scenes (chunked, mmap)...", n_scenes)
    # Process in row chunks so peak RAM ≈ chunk_rows × W × n_scenes × 4 bytes (float32)
    # 256 rows × 9046 cols × 39 scenes × 4 B ≈ 360 MB — safe on any instance.
    CHUNK_ROWS = 256
    median_vv = np.empty((h, w), dtype=np.float32)
    for row_start in range(0, h, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, h)
        chunk = valid_stack[:, row_start:row_end, :].astype(np.float32)
        median_vv[row_start:row_end, :] = np.nanmedian(chunk, axis=0)
        del chunk
    del valid_stack, mmap_arr
    os.unlink(mmap_file.name)

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
        window = rasterio.windows.Window(
            col_off=col_min, row_off=row_min,
            width=col_max - col_min, height=row_max - row_min,
        )
        with rasterio.open(meas_asset.href) as src:
            # Clamp window to actual dataset bounds
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            # Oversample factor: read at ~2× the output resolution to avoid aliasing,
            # but no finer than needed.  S1 GRD native pixel ≈ 10 m; output is
            # `resolution` metres.  Decimate during the read to cap memory use.
            native_m = 10  # S1 GRD IW native ground resolution (metres)
            out_factor = max(1, resolution // native_m)  # e.g. 50 m → factor 5
            win_h = max(1, round(window.height / out_factor))
            win_w = max(1, round(window.width  / out_factor))
            src_data = src.read(
                1, window=window,
                out_shape=(win_h, win_w),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)
            win_col_off = int(window.col_off)
            win_row_off = int(window.row_off)

        # Adjust GCP pixel/line coordinates to be relative to the window,
        # then scale down to match the decimated src_data pixel grid.
        windowed_gcps = [
            rasterio.control.GroundControlPoint(
                row=(g.row - win_row_off) / out_factor,
                col=(g.col - win_col_off) / out_factor,
                x=g.x, y=g.y, z=g.z,
            )
            for g in gcps
        ]
        win_height, win_width = src_data.shape
        logger.info("Windowed read %s %s: %dx%d px (%.1f MB)",
                    item.id, pol, win_width, win_height,
                    win_width * win_height * 4 / 1e6)

        src_transform = rasterio.transform.from_gcps(windowed_gcps)
        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_crs=target_crs,
            dst_transform=dst_transform,
            resampling=rasterio.warp.Resampling.bilinear,
            src_nodata=0,
            dst_nodata=np.nan,
            num_threads=1,
        )
        del src_data

        # Convert DN to sigma-naught linear scale (S1 GRD: sigma0 = (DN^2) / cal_factor)
        # Both operations in-place on dst_data to avoid a second 272 MB allocation.
        with np.errstate(invalid="ignore"):
            np.multiply(dst_data, dst_data, out=dst_data)
            dst_data /= 1e6

        x_coords = np.linspace(dst_bounds[0], dst_bounds[2], dst_width)
        y_coords = np.linspace(dst_bounds[3], dst_bounds[1], dst_height)
        bands[pol] = xr.DataArray(dst_data, dims=["y", "x"],
                                  coords={"x": x_coords, "y": y_coords})
        del dst_data

    if not bands:
        raise ValueError(f"No valid polarisations for {item.id}")

    ds = xr.Dataset(bands)
    logger.info("GCP-warped S1 scene %s: shape=%s", item.id, ds["VV"].shape if "VV" in ds else "VV missing")
    return ds
