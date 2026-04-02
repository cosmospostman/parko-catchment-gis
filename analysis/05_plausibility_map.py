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
import sys

import geopandas as gpd
import numpy as np
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

def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs

    configure_logging()
    ensure_output_dirs(config.OUTPUTS_DIR, config.YEAR)

    year = config.YEAR
    logger.info("Stage 05 — plausibility map (year=%d)", year)

    # ── Load inputs ──────────────────────────────────────────────────────────
    ndvi_path   = config.ndvi_anomaly_path(year)
    flower_path = config.flowering_index_path(year)
    hand_path   = config.hand_raster_path(year)

    for p in (ndvi_path, flower_path, hand_path):
        if not p.exists():
            logger.error("Required input not found: %s", p)
            sys.exit(1)

    logger.info("Loading NDVI anomaly from %s", ndvi_path)
    ndvi_da = rxr.open_rasterio(str(ndvi_path)).squeeze()

    logger.info("Loading flowering index from %s", flower_path)
    flower_da = rxr.open_rasterio(str(flower_path)).squeeze()

    logger.info("Loading HAND from %s", hand_path)
    hand_da = rxr.open_rasterio(str(hand_path)).squeeze()

    if hand_da.shape != ndvi_da.shape or hand_da.rio.transform() != ndvi_da.rio.transform():
        logger.info(
            "HAND grid %s does not match NDVI grid %s — reprojecting to match",
            hand_da.shape, ndvi_da.shape,
        )
        import rasterio
        hand_da = hand_da.rio.reproject_match(
            ndvi_da,
            resampling=rasterio.enums.Resampling.bilinear,
        )

    ndvi_arr   = ndvi_da.values.astype(np.float32)
    flower_arr = flower_da.values.astype(np.float32)
    hand_arr   = hand_da.values.astype(np.float32)

    # Replace any nodata sentinel values with NaN
    for da, arr in [(ndvi_da, ndvi_arr), (flower_da, flower_arr), (hand_da, hand_arr)]:
        nodata = da.rio.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[arr == nodata] = np.nan

    # ── Compute plausibility ─────────────────────────────────────────────────
    logger.info("Computing plausibility score")
    plausibility = compute_plausibility(ndvi_arr, flower_arr, hand_arr)

    valid_count = int(np.isfinite(plausibility).sum())
    nan_frac = np.isnan(plausibility).mean()
    logger.info(
        "Plausibility computed: %d valid pixels, %.1f%% NaN",
        valid_count, nan_frac * 100,
    )

    # ── Write continuous raster ───────────────────────────────────────────────
    out_da = ndvi_da.copy(data=plausibility.astype(np.float32))
    out_da.rio.write_nodata(np.nan, inplace=True)

    raster_path = config.plausibility_map_path(year)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    out_da.rio.to_raster(str(raster_path), dtype="float32", compress="deflate")
    logger.info("Plausibility raster written to %s", raster_path)

    # ── Vectorise threshold zones ─────────────────────────────────────────────
    binary = apply_threshold(plausibility, PLAUSIBILITY_THRESHOLD)
    transform = ndvi_da.rio.transform()
    crs = ndvi_da.rio.crs

    gdf = vectorise_zones(binary, transform, crs, MIN_PATCH_HA)
    logger.info(
        "Plausibility zones: %d polygon(s) above threshold %.2f (min patch %.2f ha)",
        len(gdf), PLAUSIBILITY_THRESHOLD, MIN_PATCH_HA,
    )

    zones_path = config.plausibility_zones_path(year)
    gdf.to_file(str(zones_path), driver="GPKG")
    logger.info("Plausibility zones written to %s", zones_path)

    logger.info("Stage 05 complete")


if __name__ == "__main__":
    main()
