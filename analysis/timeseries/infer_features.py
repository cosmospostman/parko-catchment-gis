"""Stage 12: inference feature stack assembly from raster tiles.

Produces a 2-D feature array (rows = pixels, cols = features) from a
composite raster chip set. The column order matches the `feature_names`
contract written by `train.py` — this is the correctness invariant that
ensures the RF receives features in exactly the order it was trained on.

`assemble_infer_feature_stack` is a pure function — no I/O, no global state.
It accepts numpy arrays for all inputs so callers can unit-test it without
touching the filesystem.

Feature layout
--------------
The `feature_names` list from training defines the column order. For the
standard training schema (feature_names_fixture.json):

    peak_value, peak_doy, spike_duration, peak_doy_mean, peak_doy_sd,
    years_detected, HAND, dist_to_water, mean_quality

Waveform features (peak_value … years_detected) are computed from the
composite index raster. HAND and dist_to_water come from pre-computed GIS
rasters passed in directly. mean_quality is the per-pixel mean of the
quality weights used to build the composite.

Waveform features from the composite
-------------------------------------
In inference the composite gives one index value per pixel — there is no
multi-year time series. The waveform features collapse to:

    peak_value     = composite flowering_index value (per pixel)
    peak_doy       = DOY of the composite centre date (scalar, broadcast)
    spike_duration = 1.0 (single composite — no duration to measure)
    peak_doy_mean  = same as peak_doy
    peak_doy_sd    = 0.0 (single composite)
    years_detected = 1.0 (single composite)

These are approximations — the model was trained on multi-year waveform
features, but for a single seasonal composite they degenerate to the above.
The caller should be aware that inference accuracy will reflect the degree to
which the composite peak captures the training signal.

This is documented here so inference output can be interpreted correctly.
"""

from __future__ import annotations

import numpy as np

from analysis.primitives.indices import apply_index, flowering_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_infer_feature_stack(
    composite_bands: dict[str, np.ndarray],
    hand_raster: np.ndarray,
    dist_to_water_raster: np.ndarray,
    quality_weights: list[float],
    feature_names: list[str],
    composite_doy: float = 270.0,
) -> np.ndarray:
    """Assemble inference feature array from composite raster inputs.

    Parameters
    ----------
    composite_bands:
        Mapping of band name → 2-D numpy array for the composite image.
        Must include all bands required by flowering_index (B05, B07, B08, B11).
        All arrays must have the same shape.
    hand_raster:
        2-D numpy array of HAND (Height Above Nearest Drainage) values.
        Must have the same shape as composite_bands arrays.
    dist_to_water_raster:
        2-D numpy array of distance-to-water values (metres).
        Must have the same shape as composite_bands arrays.
    quality_weights:
        List of per-acquisition quality weights used to build the composite.
        mean_quality is the mean of these weights (broadcast to all pixels).
    feature_names:
        Ordered list of feature names from training (feature_names_{run_id}.json).
        Determines the column order of the output array.
    composite_doy:
        Day-of-year representing the composite. Used for peak_doy,
        peak_doy_mean. Defaults to 270 (late-September, mid-flowering window).

    Returns
    -------
    np.ndarray
        2-D float64 array of shape (n_pixels, n_features), where n_pixels =
        rows * cols of the input rasters, and n_features = len(feature_names).
        Pixels are in row-major (C) order.

    Raises
    ------
    ValueError
        If composite_bands is empty, shapes are inconsistent, or feature_names
        contains a name that cannot be populated.
    """
    if not composite_bands:
        raise ValueError("composite_bands must not be empty")

    # Determine shape
    first_band = next(iter(composite_bands))
    shape = composite_bands[first_band].shape
    rows, cols = shape

    # Validate shape consistency
    for band, arr in composite_bands.items():
        if arr.shape != shape:
            raise ValueError(
                f"composite_bands['{band}'] has shape {arr.shape}; expected {shape}"
            )
    if hand_raster.shape != shape:
        raise ValueError(
            f"hand_raster has shape {hand_raster.shape}; expected {shape}"
        )
    if dist_to_water_raster.shape != shape:
        raise ValueError(
            f"dist_to_water_raster has shape {dist_to_water_raster.shape}; expected {shape}"
        )

    # Compute flowering index over the composite (2-D)
    index_map = apply_index(flowering_index, composite_bands)  # shape: (rows, cols)

    # mean_quality: scalar broadcast across all pixels
    mean_q = float(np.mean(quality_weights)) if quality_weights else 0.0

    # Build a mapping of feature name → 2-D array
    n_pixels = rows * cols
    feature_map: dict[str, np.ndarray] = {
        "peak_value":     index_map,
        "peak_doy":       np.full(shape, composite_doy, dtype=np.float64),
        "spike_duration": np.ones(shape, dtype=np.float64),
        "peak_doy_mean":  np.full(shape, composite_doy, dtype=np.float64),
        "peak_doy_sd":    np.zeros(shape, dtype=np.float64),
        "years_detected": np.ones(shape, dtype=np.float64),
        "HAND":           hand_raster.astype(np.float64),
        "dist_to_water":  dist_to_water_raster.astype(np.float64),
        "mean_quality":   np.full(shape, mean_q, dtype=np.float64),
    }

    # Check all requested feature names are available
    unknown = [name for name in feature_names if name not in feature_map]
    if unknown:
        raise ValueError(
            f"assemble_infer_feature_stack: unknown feature name(s): {unknown}. "
            f"Available: {list(feature_map.keys())}"
        )

    # Assemble output in the order dictated by feature_names
    columns = [feature_map[name].ravel() for name in feature_names]
    return np.column_stack(columns).astype(np.float64)
