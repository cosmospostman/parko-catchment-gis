"""Unit tests for analysis/timeseries/infer_features.py.

Tests
-----
IF1. Feature stack column order matches feature_names_fixture.json.
IF2. flowering_index via apply_index() gives the same result as calling
     flowering_index directly for matching inputs (shared primitive check).
IF3. Output shape is (n_pixels, n_features).
IF4. HAND and dist_to_water rasters are placed in the correct columns.
IF5. mean_quality is the mean of the supplied quality_weights.
IF6. Unknown feature names raise ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.primitives.indices import apply_index, flowering_index
from analysis.timeseries.infer_features import assemble_infer_feature_stack

# ---------------------------------------------------------------------------
# Load feature_names fixture
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "feature_names_fixture.json"
_FEATURE_NAMES = json.loads(_FIXTURE_PATH.read_text())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Parkinsonia-like band values (same as integration test presence scene)
_PRESENCE_BANDS: dict[str, float] = {
    "B05": 0.10, "B07": 0.40, "B08": 0.50, "B11": 0.08,
}

# Additional bands present in the composite (others in BANDS but not used by flowering_index)
_EXTRA_BANDS: dict[str, float] = {
    "B02": 0.04, "B03": 0.05, "B04": 0.06,
    "B06": 0.18, "B8A": 0.48, "B12": 0.06,
}

_ALL_BANDS = {**_PRESENCE_BANDS, **_EXTRA_BANDS}


def _make_composite(shape: tuple[int, int], band_values: dict[str, float]) -> dict[str, np.ndarray]:
    """Build a constant composite with the given band values."""
    return {band: np.full(shape, val, dtype=np.float32) for band, val in band_values.items()}


def _make_structural(shape: tuple[int, int], hand: float = 2.5, dtw: float = 300.0):
    hand_raster = np.full(shape, hand, dtype=np.float32)
    dist_raster = np.full(shape, dtw, dtype=np.float32)
    return hand_raster, dist_raster


# ---------------------------------------------------------------------------
# IF1: Column order matches feature_names_fixture.json
# ---------------------------------------------------------------------------

def test_IF1_column_order_matches_fixture():
    """Feature stack column order matches feature_names_fixture.json."""
    shape = (3, 4)
    composite = _make_composite(shape, _ALL_BANDS)
    hand, dtw = _make_structural(shape)
    quality_weights = [0.9, 0.7, 0.85]

    stack = assemble_infer_feature_stack(
        composite_bands=composite,
        hand_raster=hand,
        dist_to_water_raster=dtw,
        quality_weights=quality_weights,
        feature_names=_FEATURE_NAMES,
    )

    assert stack.shape[1] == len(_FEATURE_NAMES), (
        f"Expected {len(_FEATURE_NAMES)} columns, got {stack.shape[1]}"
    )

    # HAND column index
    hand_col = _FEATURE_NAMES.index("HAND")
    np.testing.assert_allclose(stack[:, hand_col], 2.5, rtol=1e-5,
                               err_msg="HAND column has wrong values")

    # dist_to_water column index
    dtw_col = _FEATURE_NAMES.index("dist_to_water")
    np.testing.assert_allclose(stack[:, dtw_col], 300.0, rtol=1e-5,
                               err_msg="dist_to_water column has wrong values")

    # peak_value column should be the flowering_index value
    pv_col = _FEATURE_NAMES.index("peak_value")
    expected_fi = flowering_index(_PRESENCE_BANDS)
    np.testing.assert_allclose(stack[:, pv_col], expected_fi, atol=1e-6,
                               err_msg="peak_value column should equal flowering_index")


# ---------------------------------------------------------------------------
# IF2: flowering_index via apply_index gives same result as direct call
# ---------------------------------------------------------------------------

def test_IF2_apply_index_matches_direct_flowering_index():
    """apply_index(flowering_index, ...) gives the same result as flowering_index() directly.

    This is the shared primitive correctness check: the vectorised inference
    path and the per-observation training path must produce identical values.
    """
    shape = (4, 5)

    # Build a composite with known, non-constant band values so we test more than
    # a trivial single-value case
    rng = np.random.default_rng(42)
    band_arrays = {
        "B05": rng.uniform(0.05, 0.30, shape).astype(np.float32),
        "B07": rng.uniform(0.20, 0.50, shape).astype(np.float32),
        "B08": rng.uniform(0.30, 0.60, shape).astype(np.float32),
        "B11": rng.uniform(0.05, 0.20, shape).astype(np.float32),
    }

    # Vectorised path (as used by infer_features)
    vectorised = apply_index(flowering_index, band_arrays)

    # Direct per-pixel path (as used in training waveform loop)
    direct = np.empty(shape, dtype=np.float64)
    for r in range(shape[0]):
        for c in range(shape[1]):
            pixel = {b: float(band_arrays[b][r, c]) for b in band_arrays}
            direct[r, c] = flowering_index(pixel)

    np.testing.assert_allclose(vectorised, direct, rtol=1e-12,
        err_msg="apply_index result must match direct flowering_index calls pixel-by-pixel")


# ---------------------------------------------------------------------------
# IF3: Output shape is (n_pixels, n_features)
# ---------------------------------------------------------------------------

def test_IF3_output_shape():
    """Output is a 2-D array of shape (n_pixels, n_features)."""
    shape = (3, 4)
    n_pixels = 3 * 4
    composite = _make_composite(shape, _ALL_BANDS)
    hand, dtw = _make_structural(shape)

    stack = assemble_infer_feature_stack(
        composite_bands=composite,
        hand_raster=hand,
        dist_to_water_raster=dtw,
        quality_weights=[0.8, 0.6],
        feature_names=_FEATURE_NAMES,
    )

    assert stack.shape == (n_pixels, len(_FEATURE_NAMES)), (
        f"Expected shape ({n_pixels}, {len(_FEATURE_NAMES)}), got {stack.shape}"
    )
    assert stack.dtype == np.float64


# ---------------------------------------------------------------------------
# IF4: Structural rasters placed in correct columns
# ---------------------------------------------------------------------------

def test_IF4_structural_rasters_in_correct_columns():
    """HAND and dist_to_water appear in the columns matching feature_names."""
    shape = (2, 2)
    composite = _make_composite(shape, _ALL_BANDS)

    hand_val = 7.3
    dtw_val = 450.0
    hand = np.full(shape, hand_val, dtype=np.float64)
    dtw = np.full(shape, dtw_val, dtype=np.float64)

    stack = assemble_infer_feature_stack(
        composite_bands=composite,
        hand_raster=hand,
        dist_to_water_raster=dtw,
        quality_weights=[1.0],
        feature_names=_FEATURE_NAMES,
    )

    hand_col = _FEATURE_NAMES.index("HAND")
    dtw_col = _FEATURE_NAMES.index("dist_to_water")

    np.testing.assert_allclose(stack[:, hand_col], hand_val, rtol=1e-6)
    np.testing.assert_allclose(stack[:, dtw_col], dtw_val, rtol=1e-6)


# ---------------------------------------------------------------------------
# IF5: mean_quality equals mean of quality_weights
# ---------------------------------------------------------------------------

def test_IF5_mean_quality_is_mean_of_weights():
    """mean_quality column equals the mean of the supplied quality_weights."""
    shape = (2, 3)
    composite = _make_composite(shape, _ALL_BANDS)
    hand, dtw = _make_structural(shape)
    quality_weights = [0.6, 0.8, 1.0]
    expected_mq = sum(quality_weights) / len(quality_weights)

    stack = assemble_infer_feature_stack(
        composite_bands=composite,
        hand_raster=hand,
        dist_to_water_raster=dtw,
        quality_weights=quality_weights,
        feature_names=_FEATURE_NAMES,
    )

    mq_col = _FEATURE_NAMES.index("mean_quality")
    np.testing.assert_allclose(stack[:, mq_col], expected_mq, rtol=1e-6,
                               err_msg="mean_quality column must equal mean of quality_weights")


# ---------------------------------------------------------------------------
# IF6: Unknown feature names raise ValueError
# ---------------------------------------------------------------------------

def test_IF6_unknown_feature_name_raises():
    """ValueError raised when feature_names contains an unrecognised feature."""
    shape = (2, 2)
    composite = _make_composite(shape, _ALL_BANDS)
    hand, dtw = _make_structural(shape)

    bad_names = _FEATURE_NAMES + ["nonexistent_feature"]

    with pytest.raises(ValueError, match="nonexistent_feature"):
        assemble_infer_feature_stack(
            composite_bands=composite,
            hand_raster=hand,
            dist_to_water_raster=dtw,
            quality_weights=[1.0],
            feature_names=bad_names,
        )
