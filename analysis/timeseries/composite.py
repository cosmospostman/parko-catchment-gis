"""Stage 12: quality-weighted composite of raster chip stacks.

Takes a stack of raster chips (one per acquisition) and per-acquisition quality
weights, and returns a single-value-per-pixel composite per band.

Algorithm
---------
For each pixel position, the composite value for a band is the weighted average
of that pixel's values across all acquisitions, where the weight is the
acquisition's quality score:

    composite[band][r, c] = sum(weight_i * value_i[r, c]) / sum(weight_i)

If all weights for a pixel are zero, the composite falls back to the unweighted
mean (to avoid division by zero at zero-quality stacks).

This is a pure function — no I/O, no global state.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quality_weighted_composite(
    band_stacks: dict[str, list[np.ndarray]],
    quality_weights: list[float],
) -> dict[str, np.ndarray]:
    """Compute a quality-weighted composite from a raster chip stack.

    Parameters
    ----------
    band_stacks:
        Mapping of band name → list of 2-D numpy arrays, one array per
        acquisition. All arrays must have the same shape.
    quality_weights:
        Per-acquisition quality weights, one float per acquisition.
        Must have the same length as each list in band_stacks.
        Values are typically in [0, 1] (output of ObservationQuality.score),
        but the function does not enforce this — any non-negative weights work.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of band name → 2-D float64 composite array. Same spatial
        shape as the input arrays.

    Raises
    ------
    ValueError
        If band_stacks is empty, or quality_weights length does not match
        the number of acquisitions, or arrays within a band have different shapes.
    """
    if not band_stacks:
        raise ValueError("band_stacks must not be empty")

    n_acq = len(quality_weights)

    # Validate all lists have the same length
    for band, stack in band_stacks.items():
        if len(stack) != n_acq:
            raise ValueError(
                f"band '{band}' has {len(stack)} arrays but quality_weights "
                f"has {n_acq} entries; they must match"
            )

    # Determine shape from the first array
    first_band = next(iter(band_stacks))
    if not band_stacks[first_band]:
        raise ValueError("Each band stack must contain at least one array")
    shape = band_stacks[first_band][0].shape

    # Validate shape consistency across all arrays
    for band, stack in band_stacks.items():
        for i, arr in enumerate(stack):
            if arr.shape != shape:
                raise ValueError(
                    f"band '{band}' acquisition {i} has shape {arr.shape}; "
                    f"expected {shape}"
                )

    # Build weight array: shape (n_acq,) for broadcasting
    weights = np.array(quality_weights, dtype=np.float64)  # shape: (n_acq,)

    result: dict[str, np.ndarray] = {}

    for band, stack in band_stacks.items():
        # Stack into (n_acq, rows, cols)
        arr_stack = np.stack([a.astype(np.float64) for a in stack], axis=0)

        # Broadcast weights to (n_acq, 1, 1) for element-wise multiplication
        w = weights[:, np.newaxis, np.newaxis]

        weighted_sum = np.sum(w * arr_stack, axis=0)   # shape: (rows, cols)
        weight_sum = np.sum(w, axis=0)                 # scalar (uniform weights) or (rows, cols)

        # Fallback: where total weight is zero, use unweighted mean
        zero_mask = weight_sum == 0.0
        if np.any(zero_mask):
            unweighted = np.mean(arr_stack, axis=0)
            composite = np.where(zero_mask, unweighted, weighted_sum / np.where(zero_mask, 1.0, weight_sum))
        else:
            composite = weighted_sum / weight_sum

        result[band] = composite

    return result
