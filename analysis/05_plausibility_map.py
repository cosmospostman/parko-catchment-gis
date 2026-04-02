"""
Stage 05 — Rule-based plausibility map from NDVI anomaly, flowering index, and HAND.

Produces:
  plausibility_map_{year}.tif     — continuous plausibility score, Float32 0–1
  plausibility_zones_{year}.gpkg  — polygons where score >= PLAUSIBILITY_THRESHOLD

Approach
--------
Three ecologically grounded signals are combined with equal weighting:
  1. ndvi_anomaly   — persistent dry-season greenness above native grass baseline
  2. flowering_index — Aug–Oct yellow flower flush independent of NDVI anomaly
  3. hand (inverted) — low topographic position in drainage network

Each signal is percentile-scaled to [0, 1] then averaged:
  plausibility = (ndvi_norm + flower_norm + (1 − hand_norm)) / 3

This is an interim product for drone survey zone selection only.  It is not a
calibrated probability and must not be used for eradication/management decisions.
Once ground-truth drone data is available, Stage 6 (trained classifier) supersedes
this output.

Equal weighting is a deliberate starting assumption.  The drone survey validation
will reveal whether any signal warrants heavier weight before Stage 6 is built.
"""

import logging
import subprocess
import sys
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.features import shapes

logger = logging.getLogger(__name__)

PLAUSIBILITY_THRESHOLD = 0.60   # controls polygon output only; raster always 0–1
MIN_PATCH_HA = 0.25             # consistent with Stage 6 MMU


# ---------------------------------------------------------------------------
# Pure functions (tested independently)
# ---------------------------------------------------------------------------

def percentile_scale(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Scale arr to [0, 1] using percentile clipping. NaN-safe.

    Values below the lo-th percentile map to 0; values above the hi-th map to 1.
    The percentiles are computed only over finite pixels so NaN voids do not bias
    the scaling.  NaN inputs produce NaN outputs.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.full_like(arr, np.nan, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    scaled = (arr - p_lo) / (p_hi - p_lo + 1e-9)
    return np.clip(scaled, 0.0, 1.0)


def compute_plausibility(
    ndvi_arr: np.ndarray,
    flower_arr: np.ndarray,
    hand_arr: np.ndarray,
) -> np.ndarray:
    """Compute equal-weight plausibility score from three normalised inputs.

    Parameters
    ----------
    ndvi_arr, flower_arr, hand_arr : np.ndarray
        Float32 arrays with the same shape.  NaN = no-data.

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1].  NaN where any input is NaN.
    """
    valid = np.isfinite(ndvi_arr) & np.isfinite(flower_arr) & np.isfinite(hand_arr)

    ndvi_norm   = percentile_scale(ndvi_arr)
    flower_norm = percentile_scale(flower_arr)
    hand_norm   = percentile_scale(hand_arr)

    plausibility = np.full_like(ndvi_arr, np.nan, dtype=np.float32)
    plausibility[valid] = (
        ndvi_norm[valid]
        + flower_norm[valid]
        + (1.0 - hand_norm[valid])
    ) / 3.0
    return plausibility


def apply_threshold(plausibility: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean array: True where plausibility >= threshold, False elsewhere.

    NaN pixels are always False.
    """
    return np.where(np.isfinite(plausibility), plausibility >= threshold, False)


def vectorise_zones(
    binary: np.ndarray,
    transform,
    crs,
    min_patch_ha: float,
) -> gpd.GeoDataFrame:
    """Convert a binary uint8 array to a GeoDataFrame of plausibility zone polygons.

    Patches smaller than min_patch_ha are removed.  Returns an empty GeoDataFrame
    (with the correct CRS) if no patches survive the size filter.
    """
    from shapely.geometry import shape as _shape

    geoms = [
        _shape(geom_dict)
        for geom_dict, val in shapes(binary.astype(np.uint8), mask=binary.astype(np.uint8), transform=transform)
        if val == 1
    ]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gdf = gdf[gdf["area_ha"] >= min_patch_ha].reset_index(drop=True)
    return gdf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _compute_percentiles(path, nodata, sample_rows: int = 200) -> tuple[float, float]:
    """Compute p2/p98 over a systematic row sample without loading the full raster."""
    with rasterio.open(str(path)) as src:
        h, w = src.height, src.width
        step = max(1, h // sample_rows)
        rows = range(0, h, step)
        chunks = []
        for row in rows:
            window = rasterio.windows.Window(0, row, w, min(step, h - row))
            arr = src.read(1, window=window).astype(np.float32)
            if nodata is not None and np.isfinite(nodata):
                arr[arr == nodata] = np.nan
            valid = arr[np.isfinite(arr)]
            if valid.size:
                chunks.append(valid)
    all_valid = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if all_valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(all_valid, 2)), float(np.percentile(all_valid, 98))


def _warp_hand_to_ndvi(hand_path, ndvi_path) -> str:
    """Reproject HAND to match NDVI grid via gdalwarp. Returns temp file path."""
    with rasterio.open(str(ndvi_path)) as src:
        ndvi_crs = src.crs.to_string()
        t = src.transform
        h, w = src.height, src.width
    xmin = t.c
    ymax = t.f
    xmax = xmin + t.a * w
    ymin = ymax + t.e * h
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False,
                                      dir=str(hand_path.parent))
    tmp.close()
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs", ndvi_crs,
            "-te", str(xmin), str(ymin), str(xmax), str(ymax),
            "-ts", str(w), str(h),
            "-r", "bilinear",
            "-srcnodata", "-9999",
            "-dstnodata", "-9999",
            "-ot", "Float32",
            "-co", "COMPRESS=DEFLATE",
            "-overwrite",
            str(hand_path),
            tmp.name,
        ],
        check=True,
    )
    return tmp.name


# Number of rows to process per chunk — tuned so three float32 chunks fit well
# within available memory even on a 32 GB instance.
_CHUNK_ROWS = 512


def main() -> None:
    import config
    from pathlib import Path
    from utils.io import configure_logging, ensure_output_dirs

    configure_logging()
    ensure_output_dirs(config.YEAR)

    year = config.YEAR
    logger.info("Stage 05 — plausibility map (year=%d)", year)

    # ── Resolve input paths ──────────────────────────────────────────────────
    ndvi_path   = config.ndvi_anomaly_path(year)
    flower_path = config.flowering_index_path(year)
    hand_path   = config.hand_raster_path(year)

    for p in (ndvi_path, flower_path, hand_path):
        if not p.exists():
            logger.error("Required input not found: %s", p)
            sys.exit(1)

    # ── Align HAND grid to NDVI if needed ───────────────────────────────────
    with rasterio.open(str(ndvi_path)) as ndvi_src, \
         rasterio.open(str(hand_path)) as hand_src:
        need_warp = (
            (hand_src.height, hand_src.width) != (ndvi_src.height, ndvi_src.width)
            or hand_src.transform != ndvi_src.transform
        )

    warped_hand_tmp = None
    if need_warp:
        logger.info("HAND grid differs from NDVI — reprojecting via gdalwarp")
        warped_hand_tmp = _warp_hand_to_ndvi(hand_path, ndvi_path)
        aligned_hand_path = Path(warped_hand_tmp)
    else:
        aligned_hand_path = hand_path

    try:
        # ── First pass: compute global percentiles for scaling ────────────────
        logger.info("Computing global percentiles for scaling")
        with rasterio.open(str(ndvi_path)) as src:
            ndvi_nodata = src.nodata
        with rasterio.open(str(flower_path)) as src:
            flower_nodata = src.nodata
        with rasterio.open(str(aligned_hand_path)) as src:
            hand_nodata = src.nodata

        ndvi_p2,   ndvi_p98   = _compute_percentiles(ndvi_path,          ndvi_nodata)
        flower_p2, flower_p98 = _compute_percentiles(flower_path,        flower_nodata)
        hand_p2,   hand_p98   = _compute_percentiles(aligned_hand_path,  hand_nodata)
        logger.info("NDVI   p2=%.4f  p98=%.4f", ndvi_p2,   ndvi_p98)
        logger.info("Flower p2=%.4f  p98=%.4f", flower_p2, flower_p98)
        logger.info("HAND   p2=%.4f  p98=%.4f", hand_p2,   hand_p98)

        # ── Open all sources + output for chunked processing ─────────────────
        raster_path = config.plausibility_map_path(year)
        raster_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(ndvi_path)) as ndvi_src, \
             rasterio.open(str(flower_path)) as flower_src, \
             rasterio.open(str(aligned_hand_path)) as hand_src:

            profile = ndvi_src.profile.copy()
            profile.update(dtype="float32", count=1, compress="deflate",
                           nodata=np.nan, tiled=True, blockxsize=512, blockysize=512)

            h, w = ndvi_src.height, ndvi_src.width

            with rasterio.open(str(raster_path), "w", **profile) as dst:
                binary_rows = []
                valid_count = 0

                for row_off in range(0, h, _CHUNK_ROWS):
                    n_rows = min(_CHUNK_ROWS, h - row_off)
                    window = rasterio.windows.Window(0, row_off, w, n_rows)

                    ndvi_chunk   = ndvi_src.read(1,   window=window).astype(np.float32)
                    flower_chunk = flower_src.read(1, window=window).astype(np.float32)
                    hand_chunk   = hand_src.read(1,   window=window).astype(np.float32)

                    # Replace nodata with NaN
                    for arr, nd in [(ndvi_chunk, ndvi_nodata),
                                    (flower_chunk, flower_nodata),
                                    (hand_chunk, hand_nodata)]:
                        if nd is not None and np.isfinite(nd):
                            arr[arr == nd] = np.nan

                    # Scale using global percentiles
                    def _scale(arr, p_lo, p_hi):
                        scaled = (arr - p_lo) / (p_hi - p_lo + 1e-9)
                        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

                    ndvi_norm   = _scale(ndvi_chunk,   ndvi_p2,   ndvi_p98)
                    flower_norm = _scale(flower_chunk, flower_p2, flower_p98)
                    hand_norm   = _scale(hand_chunk,   hand_p2,   hand_p98)

                    valid = np.isfinite(ndvi_chunk) & np.isfinite(flower_chunk) & np.isfinite(hand_chunk)
                    plaus = np.full((n_rows, w), np.nan, dtype=np.float32)
                    plaus[valid] = (
                        ndvi_norm[valid] + flower_norm[valid] + (1.0 - hand_norm[valid])
                    ) / 3.0

                    dst.write(plaus, 1, window=window)
                    valid_count += int(valid.sum())
                    binary_rows.append((row_off, apply_threshold(plaus, PLAUSIBILITY_THRESHOLD)))

                    if (row_off // _CHUNK_ROWS) % 10 == 0:
                        logger.info("  processed rows %d–%d / %d", row_off, row_off + n_rows, h)

        nan_frac = 1.0 - valid_count / (h * w)
        logger.info("Plausibility computed: %d valid pixels, %.1f%% NaN", valid_count, nan_frac * 100)
        logger.info("Plausibility raster written to %s", raster_path)

        # ── Assemble full binary mask for vectorisation ───────────────────────
        binary = np.zeros((h, w), dtype=bool)
        for row_off, chunk in binary_rows:
            binary[row_off:row_off + chunk.shape[0], :] = chunk
        del binary_rows

        with rasterio.open(str(ndvi_path)) as src:
            transform = src.transform
            crs = src.crs

        gdf = vectorise_zones(binary, transform, crs, MIN_PATCH_HA)
        logger.info(
            "Plausibility zones: %d polygon(s) above threshold %.2f (min patch %.2f ha)",
            len(gdf), PLAUSIBILITY_THRESHOLD, MIN_PATCH_HA,
        )

        zones_path = config.plausibility_zones_path(year)
        gdf.to_file(str(zones_path), driver="GPKG")
        logger.info("Plausibility zones written to %s", zones_path)

    finally:
        if warped_hand_tmp:
            Path(warped_hand_tmp).unlink(missing_ok=True)

    logger.info("Stage 05 complete")


if __name__ == "__main__":
    main()
