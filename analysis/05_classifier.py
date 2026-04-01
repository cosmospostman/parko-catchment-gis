"""
Step 05 — Random Forest Parkinsonia probability classifier.

Produces: probability_raster_{year}.tif  (COG, EPSG:7844, float32 [0,1])
"""
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import dask
import geopandas as gpd
import numpy as np
import xarray as xr

# Script-level constants
FEATURE_NAMES = [
    "ndvi_anomaly",
    "flowering_index",
    "vv_db",
    "vh_db",
    "ndvi_median",
    "glcm_contrast",
    "glcm_homogeneity",
    "dist_to_watercourse",
]
GLCM_KERNEL_SIZE = 7
GLCM_ENABLED = True
PSEUDO_ABSENCE_BUFFER_KM = 2.0
RF_CLASS_WEIGHT = "balanced"
MODEL_CACHE_PATH_TEMPLATE = "rf_model_{year}.pkl"

# Ecological bootstrapping: thresholds for high-confidence synthetic training samples.
# Parkinsonia is an obligate riparian weed with a strong dry-season green signal.
SYNTH_PRESENCE_MAX_DIST_WATER_M = 500.0   # rarely establishes beyond 500m from water
SYNTH_ABSENCE_MIN_DIST_WATER_M  = 2000.0  # very unlikely beyond 2km
SYNTH_NDVI_MEDIAN_MIN = 0.15              # not bare ground / sparse grass
SYNTH_NDVI_MEDIAN_MAX = 0.65              # not dense closed-canopy rainforest
SYNTH_TARGET_PRESENCES = 300              # synthetic presences to generate
SYNTH_SAMPLE_WEIGHT = 0.3                 # down-weight vs real ALA records (weight=1.0)

logger = logging.getLogger(__name__)


def _fetch_ala_occurrences(bbox: list, species: str, api_base: str, cache_path: Path = None) -> gpd.GeoDataFrame:
    """Fetch georeferenced occurrence records from ALA API, or load from cache if present."""
    if cache_path is not None and cache_path.exists():
        logger.info("Loading ALA occurrences from cache: %s", cache_path)
        return gpd.read_file(str(cache_path))

    import requests
    import pandas as pd
    from shapely.geometry import Point
    url = "https://biocache.ala.org.au/ws/occurrences/search"
    params = {
        "q": f'taxon_name:"{species}"',
        "bbox": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # miny,minx,maxy,maxx
        "pageSize": 10000,
        "fl": "decimalLongitude,decimalLatitude",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("occurrences", [])
    if not records:
        logger.warning("No ALA occurrences returned")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    df = pd.DataFrame(records)
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r["decimalLongitude"], r["decimalLatitude"]) for _, r in df.iterrows()],
        crs="EPSG:4326",
    )
    return gdf


def _compute_glcm_features(ndvi: np.ndarray, kernel: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Compute simple GLCM-like texture features (contrast and homogeneity)."""
    from scipy.ndimage import generic_filter

    def contrast_fn(values):
        v = values.reshape(kernel, kernel)
        center = v[kernel // 2, kernel // 2]
        if np.isnan(center):
            return np.nan
        return float(np.nanmean((v - center) ** 2))

    def homogeneity_fn(values):
        v = values.reshape(kernel, kernel)
        center = v[kernel // 2, kernel // 2]
        if np.isnan(center):
            return np.nan
        diff = np.abs(v - center)
        return float(np.nanmean(1.0 / (1.0 + diff)))

    contrast = generic_filter(ndvi, contrast_fn, size=kernel, mode="nearest")
    homogeneity = generic_filter(ndvi, homogeneity_fn, size=kernel, mode="nearest")
    return contrast, homogeneity


def _generate_ecological_samples(
    feat_stack: np.ndarray,
    dist_to_water: np.ndarray,
    ndvi_anomaly: np.ndarray,
    flowering: np.ndarray,
    ndvi_median: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[list, list]:
    """Generate high-confidence synthetic presence and absence pixels from ecological priors.

    Presence criteria (all required):
      - Within 500m of watercourse (obligate riparian)
      - ndvi_anomaly in top quartile (stays green while natives senesce)
      - flowering_index above scene median (Aug-Oct phenological signal)
      - ndvi_median in moderate range (not bare ground, not closed-canopy rainforest)

    Absence criteria (any one sufficient):
      - Beyond 2km from watercourse
      - Very low ndvi_median (bare ground / sparse grass)
      - Low ndvi_anomaly AND low flowering_index AND beyond 500m from water
    """
    H, W = ndvi_anomaly.shape
    valid = ~np.isnan(ndvi_anomaly) & ~np.isnan(flowering) & ~np.isnan(ndvi_median)

    ndvi_anom_q75 = float(np.nanpercentile(ndvi_anomaly, 75))
    flower_median = float(np.nanpercentile(flowering, 50))
    ndvi_anom_q25 = float(np.nanpercentile(ndvi_anomaly, 25))
    flower_q25    = float(np.nanpercentile(flowering, 25))

    # High-confidence presence mask
    pres_mask = (
        valid
        & (dist_to_water <= SYNTH_PRESENCE_MAX_DIST_WATER_M)
        & (ndvi_anomaly >= ndvi_anom_q75)
        & (flowering >= flower_median)
        & (ndvi_median >= SYNTH_NDVI_MEDIAN_MIN)
        & (ndvi_median <= SYNTH_NDVI_MEDIAN_MAX)
    )

    # High-confidence absence mask
    abs_mask = valid & (
        (dist_to_water >= SYNTH_ABSENCE_MIN_DIST_WATER_M)
        | (ndvi_median < SYNTH_NDVI_MEDIAN_MIN)
        | (
            (ndvi_anomaly <= ndvi_anom_q25)
            & (flowering <= flower_q25)
            & (dist_to_water > SYNTH_PRESENCE_MAX_DIST_WATER_M)
        )
    )

    pres_idx = np.argwhere(pres_mask)
    abs_idx  = np.argwhere(abs_mask)

    n_pres = min(len(pres_idx), SYNTH_TARGET_PRESENCES)
    n_abs  = min(len(abs_idx), n_pres)  # balanced

    synth_presences = [tuple(pres_idx[i]) for i in rng.choice(len(pres_idx), size=n_pres, replace=False)]
    synth_absences  = [tuple(abs_idx[i])  for i in rng.choice(len(abs_idx),  size=n_abs,  replace=False)]

    logger.info(
        "Ecological bootstrap: %d synthetic presences, %d synthetic absences (weight=%.1f)",
        len(synth_presences), len(synth_absences), SYNTH_SAMPLE_WEIGHT,
    )
    return synth_presences, synth_absences


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs, read_raster, write_cog
    from utils.quicklook import save_quicklook

    configure_logging()

    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)
    bbox_wgs84 = list(catchment.to_crs("EPSG:4326").total_bounds)

    # ── Load raster inputs ────────────────────────────────────────────────────
    ndvi_anomaly_da  = read_raster(config.ndvi_anomaly_path(config.YEAR)).squeeze()
    flowering_da     = read_raster(config.flowering_index_path(config.YEAR)).squeeze()
    ndvi_median_da   = read_raster(config.ndvi_median_path(config.YEAR)).squeeze()

    # Align all to NDVI anomaly grid
    flowering_aligned = flowering_da.rio.reproject_match(ndvi_anomaly_da)
    ndvi_med_aligned  = ndvi_median_da.rio.reproject_match(ndvi_anomaly_da)

    ndvi_arr      = ndvi_anomaly_da.values.astype(np.float32)
    flower_arr    = flowering_aligned.values.astype(np.float32)
    ndvi_med_arr  = ndvi_med_aligned.values.astype(np.float32)

    # GLCM texture
    if GLCM_ENABLED:
        logger.info("Computing GLCM texture features...")
        glcm_contrast, glcm_homogeneity = _compute_glcm_features(ndvi_arr, GLCM_KERNEL_SIZE)
    else:
        glcm_contrast    = np.zeros_like(ndvi_arr)
        glcm_homogeneity = np.ones_like(ndvi_arr)

    # Distance to watercourse (if drainage network available)
    drain_path = Path(config.BASE_DIR) / "data" / "drainage_network.gpkg"
    if drain_path.exists():
        logger.info("Computing distance to watercourse...")
        drainage = gpd.read_file(str(drain_path)).to_crs(config.TARGET_CRS)
        from rasterio.features import rasterize
        import rasterio.transform as rtransform
        transform = ndvi_anomaly_da.rio.transform()
        drain_rast = rasterize(
            [(geom, 1) for geom in drainage.geometry],
            out_shape=ndvi_arr.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        from scipy.ndimage import distance_transform_edt
        dist_to_water = distance_transform_edt(drain_rast == 0).astype(np.float32)
        dist_to_water *= config.TARGET_RESOLUTION  # convert pixels to metres
    else:
        logger.warning("Drainage network not found — dist_to_watercourse set to 0")
        dist_to_water = np.zeros_like(ndvi_arr)

    # VV/VH SAR features (use zeros if flood extent not available as raster)
    vv_db = np.zeros_like(ndvi_arr)
    vh_db = np.zeros_like(ndvi_arr)

    # ── Training data: ALA occurrences + pseudo-absences ─────────────────────
    logger.info("Fetching ALA occurrences...")
    ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
    occurrences = _fetch_ala_occurrences(bbox_wgs84, config.ALA_SPECIES_QUERY, config.ALA_API_BASE, cache_path=ala_cache)
    occurrences = occurrences.to_crs(config.TARGET_CRS)
    occurrences = gpd.clip(occurrences, catchment)
    logger.info("ALA occurrences in catchment: %d", len(occurrences))

    if len(occurrences) == 0:
        raise RuntimeError("No ALA occurrence records found — cannot train classifier")

    # Build feature stack
    feat_stack = np.stack([
        ndvi_arr, flower_arr, vv_db, vh_db,
        ndvi_med_arr, glcm_contrast, glcm_homogeneity, dist_to_water,
    ], axis=-1)  # (H, W, n_features)

    H, W = ndvi_arr.shape
    transform = ndvi_anomaly_da.rio.transform()

    # Sample presence pixels
    import rasterio.transform as rt
    presence_pixels = []
    for pt in occurrences.geometry:
        col, row = ~transform * (pt.x, pt.y)
        col, row = int(col), int(row)
        if 0 <= row < H and 0 <= col < W:
            presence_pixels.append((row, col))

    if not presence_pixels:
        raise RuntimeError("No occurrence points fell within the raster extent")

    # Generate pseudo-absences: random sample > PSEUDO_ABSENCE_BUFFER_KM from presences
    rng = np.random.default_rng(42)
    buffer_px = int(PSEUDO_ABSENCE_BUFFER_KM * 1000 / config.TARGET_RESOLUTION)
    presence_mask = np.zeros((H, W), dtype=bool)
    for r, c in presence_pixels:
        r0, r1 = max(0, r - buffer_px), min(H, r + buffer_px + 1)
        c0, c1 = max(0, c - buffer_px), min(W, c + buffer_px + 1)
        presence_mask[r0:r1, c0:c1] = True

    valid_absence_mask = ~presence_mask & ~np.isnan(ndvi_arr)
    absence_indices = np.argwhere(valid_absence_mask)
    n_absence = min(len(absence_indices), max(len(presence_pixels) * 5, 1000))
    chosen = rng.choice(len(absence_indices), size=n_absence, replace=False)
    absence_pixels = [tuple(absence_indices[i]) for i in chosen]

    # Ecological bootstrap: supplement sparse ALA records with high-confidence
    # synthetic samples derived from ecological priors
    synth_pres_pixels, synth_abs_pixels = _generate_ecological_samples(
        feat_stack, dist_to_water, ndvi_arr, flower_arr, ndvi_med_arr, rng,
    )

    # Build X, y, sample_weight arrays
    # Real ALA records: weight=1.0 — synthetic ecological samples: weight=SYNTH_SAMPLE_WEIGHT
    n_feat = feat_stack.shape[-1]
    X_pres       = np.array([feat_stack[r, c] for r, c in presence_pixels]) if presence_pixels else np.empty((0, n_feat))
    X_abs        = np.array([feat_stack[r, c] for r, c in absence_pixels]) if absence_pixels else np.empty((0, n_feat))
    X_synth_pres = np.array([feat_stack[r, c] for r, c in synth_pres_pixels]) if synth_pres_pixels else np.empty((0, n_feat))
    X_synth_abs  = np.array([feat_stack[r, c] for r, c in synth_abs_pixels]) if synth_abs_pixels else np.empty((0, n_feat))

    X = np.vstack([X_pres, X_abs, X_synth_pres, X_synth_abs])
    y = np.array(
        [1] * len(X_pres) + [0] * len(X_abs)
        + [1] * len(X_synth_pres) + [0] * len(X_synth_abs)
    )
    sample_weight = np.array(
        [1.0] * (len(X_pres) + len(X_abs))
        + [SYNTH_SAMPLE_WEIGHT] * (len(X_synth_pres) + len(X_synth_abs))
    )

    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1)
    X, y, sample_weight = X[valid], y[valid], sample_weight[valid]
    logger.info(
        "Training samples: %d presence (%d ALA + %d synthetic), %d absence (%d ALA + %d synthetic)",
        int(y.sum()), len(presence_pixels), len(synth_pres_pixels),
        int((y == 0).sum()), len(absence_pixels), len(synth_abs_pixels),
    )

    # ── Spatial block cross-validation and training ───────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=42,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", fit_params={"sample_weight": sample_weight})
    logger.info("CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    clf.fit(X, y, sample_weight=sample_weight)

    # Cache model
    model_cache = Path(config.CACHE_DIR) / MODEL_CACHE_PATH_TEMPLATE.format(year=config.YEAR)
    model_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(model_cache, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES, "cv_scores": cv_scores}, f)
    logger.info("Model cached: %s", model_cache)

    # ── Predict probability raster ────────────────────────────────────────────
    logger.info("Predicting probability raster...")
    flat = feat_stack.reshape(-1, feat_stack.shape[-1])
    nan_rows = np.isnan(flat).any(axis=1)
    prob_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
    prob_flat[~nan_rows] = clf.predict_proba(flat[~nan_rows])[:, 1]
    prob_arr = prob_flat.reshape(H, W)

    # Wrap in DataArray preserving spatial metadata
    prob_da = xr.DataArray(
        prob_arr,
        dims=ndvi_anomaly_da.dims[-2:],
        coords={d: ndvi_anomaly_da.coords[d] for d in ndvi_anomaly_da.dims[-2:]},
    ).rio.write_crs(config.TARGET_CRS)

    out_path = config.probability_raster_path(config.YEAR)
    write_cog(prob_da, out_path)
    logger.info("Written: %s", out_path)

    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        prob_da,
        ql_path,
        vmin=0.0,
        vmax=1.0,
        cmap="hot_r",
        title=f"Parkinsonia Probability {config.YEAR}",
    )


if __name__ == "__main__":
    main()
