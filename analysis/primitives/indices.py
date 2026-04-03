"""Spectral index primitives shared by training and inference pipelines.

The canonical implementations live here. Both pipelines import from this
module — neither owns a copy. This is a correctness requirement: if
flowering_index were computed differently in train vs. infer (different band
order, a missing clamp, a changed coefficient), the model would be applied to
a feature distribution that differs from training. The probability raster
would look plausible and be silently wrong.

flowering_index
---------------
A red-edge enhanced greenness index tuned for *Parkinsonia aculeata*.
Parkinsonia has a distinctive spectral signature at peak leaf-flush: high
red-edge reflectance (B05, B06, B07) relative to red (B04), combined with
high NIR (B08) relative to SWIR (B11). This combination distinguishes the
dense, fine-leafed canopy flush from background vegetation and bare soil.

Index definition:

    re_slope = (B07 - B05) / (B07 + B05 + ε)    # red-edge steepness
    nir_swir  = (B08 - B11) / (B08 + B11 + ε)   # NIR-SWIR contrast
    flowering_index = (re_slope + nir_swir) / 2  # clamped to [-1, 1]

    ε = 1e-9  # avoids division by zero at very dark or saturated pixels

Both components are normalised differences and individually lie in [-1, 1],
so the average also lies in [-1, 1].

apply_index
-----------
Vectorised wrapper that calls index_fn over a numpy band-stack array.
Inference calls this to apply flowering_index pixel-wise across raster tiles
without leaving the numpy layer. Training calls flowering_index directly in a
per-observation loop — the result is identical.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-9   # division-by-zero guard; too small to affect reflectance math


# ---------------------------------------------------------------------------
# flowering_index
# ---------------------------------------------------------------------------

def flowering_index(bands: dict[str, float]) -> float:
    """Compute the Parkinsonia flowering index for one observation.

    Parameters
    ----------
    bands:
        Mapping of band name → surface reflectance value, e.g.
        {"B04": 0.05, "B05": 0.08, "B07": 0.22, "B08": 0.35, "B11": 0.12}.
        Missing bands are treated as 0.0 (a mild penalty — absence of a band
        is a data-quality issue).

    Returns
    -------
    float
        Index value clamped to [-1, 1]. Higher values indicate stronger
        Parkinsonia-like spectral signature.
    """
    b05 = bands.get("B05", 0.0)
    b07 = bands.get("B07", 0.0)
    b08 = bands.get("B08", 0.0)
    b11 = bands.get("B11", 0.0)

    re_slope = (b07 - b05) / (b07 + b05 + _EPS)
    nir_swir = (b08 - b11) / (b08 + b11 + _EPS)

    raw = (re_slope + nir_swir) / 2.0
    return float(max(-1.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# apply_index — vectorised for inference raster tiles
# ---------------------------------------------------------------------------

def apply_index(
    index_fn: Callable[[dict[str, float]], float],
    band_stack: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply index_fn pixel-wise over a stack of 2-D band arrays.

    Parameters
    ----------
    index_fn:
        Any function with the same signature as flowering_index:
        takes a band dict, returns a float. Typically flowering_index.
    band_stack:
        Mapping of band name → 2-D numpy array of the same shape.
        All arrays must have identical shape.

    Returns
    -------
    np.ndarray
        2-D float64 array of index values, same shape as the input arrays.

    Notes
    -----
    Uses np.vectorize over pixel coordinates rather than hand-unrolling band
    math. This keeps the inference path calling the same index_fn as training
    — no separate vectorised reimplementation that could drift.
    """
    # Determine shape from the first array present
    shapes = {arr.shape for arr in band_stack.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"apply_index: all band arrays must have the same shape, got {shapes}"
        )
    shape = next(iter(shapes))
    rows, cols = shape

    result = np.empty(shape, dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            pixel_bands = {band: float(arr[r, c]) for band, arr in band_stack.items()}
            result[r, c] = index_fn(pixel_bands)
    return result
