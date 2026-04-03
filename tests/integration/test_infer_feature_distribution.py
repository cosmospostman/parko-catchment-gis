"""Integration test: inference-path feature distribution matches training-path.

Session 13 goal: scientific assumption test — confirm that feature values at
known presence locations computed via the inference path are consistent with
those computed via the training path.

Design
------
Both pipelines share the same primitive: flowering_index in
analysis/primitives/indices.py. The training path calls flowering_index()
per-observation (scalar dict → float). The inference path calls
apply_index(flowering_index, composite_bands) over a raster (2-D arrays).

This test confirms that for the same spectral band values:

    training path: flowering_index(bands_dict) → float
    inference path: assemble_infer_feature_stack(composite_bands, ...) → array
                    where peak_value column = apply_index(flowering_index, ...)

The two paths must agree on peak_value (flowering_index) to within floating
point tolerance. This guards against accidental drift if either the training
loop or the inference raster path changes the index formula.

Assertions
----------
ID1. peak_value from inference path matches flowering_index from training path
     for presence-class band values (within tolerance).
ID2. peak_value from inference path matches flowering_index from training path
     for absence-class band values (within tolerance).
ID3. Feature stack has no all-zero columns for presence-class input.
ID4. Feature stack has no NaN columns.
ID5. Feature stack column order matches feature_names_fixture.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.primitives.indices import flowering_index
from analysis.timeseries.infer_features import assemble_infer_feature_stack

# ---------------------------------------------------------------------------
# Load feature_names fixture (column order contract)
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "feature_names_fixture.json"
_FEATURE_NAMES: list[str] = json.loads(_FIXTURE_PATH.read_text())

# ---------------------------------------------------------------------------
# Synthetic band values — same as test_feature_pipeline.py presence/absence
# scenes, so the "known presence locations" concept is grounded in the same
# spectral signature used throughout the test suite.
# ---------------------------------------------------------------------------

# Parkinsonia-like signature: high B07/B05 ratio (re_slope > 0),
# high B08/B11 ratio (nir_swir > 0) → flowering_index ≈ 0.6
_PRESENCE_BANDS: dict[str, float] = {
    "B02": 0.04, "B03": 0.05, "B04": 0.06,
    "B05": 0.10, "B06": 0.18, "B07": 0.40,
    "B08": 0.50, "B8A": 0.48,
    "B11": 0.08, "B12": 0.06,
}

# Non-Parkinsonia signature: low B07/B05, low B08/B11 → flowering_index < 0
_ABSENCE_BANDS: dict[str, float] = {
    "B02": 0.10, "B03": 0.12, "B04": 0.14,
    "B05": 0.30, "B06": 0.25, "B07": 0.20,
    "B08": 0.15, "B8A": 0.14,
    "B11": 0.35, "B12": 0.30,
}

# Spatial shape for the synthetic raster (rows, cols)
_SHAPE = (4, 5)
_N_PIXELS = _SHAPE[0] * _SHAPE[1]

# Quality weights for the composite (arbitrary but non-trivial)
_QUALITY_WEIGHTS = [0.9, 0.8, 0.75]

# Structural raster values (arbitrary, realistic)
_HAND_VALUE = 2.5
_DTW_VALUE = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_composite(
    shape: tuple[int, int],
    band_values: dict[str, float],
) -> dict[str, np.ndarray]:
    """Build a spatially constant composite raster from scalar band values."""
    return {band: np.full(shape, val, dtype=np.float32) for band, val in band_values.items()}


def _make_structural(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    hand = np.full(shape, _HAND_VALUE, dtype=np.float32)
    dtw = np.full(shape, _DTW_VALUE, dtype=np.float32)
    return hand, dtw


def _run_inference(band_values: dict[str, float]) -> np.ndarray:
    """Run assemble_infer_feature_stack and return the feature array."""
    composite = _make_composite(_SHAPE, band_values)
    hand, dtw = _make_structural(_SHAPE)
    return assemble_infer_feature_stack(
        composite_bands=composite,
        hand_raster=hand,
        dist_to_water_raster=dtw,
        quality_weights=_QUALITY_WEIGHTS,
        feature_names=_FEATURE_NAMES,
    )


# ---------------------------------------------------------------------------
# Training-path reference values
#
# The training path calls flowering_index(bands) per observation. For a
# presence point with constant band values across the composite, the
# flowering_index value is the same for every acquisition. The inference-path
# peak_value should equal this value (they share the same primitive).
# ---------------------------------------------------------------------------

_TRAINING_FI_PRESENCE: float = flowering_index(_PRESENCE_BANDS)
_TRAINING_FI_ABSENCE: float = flowering_index(_ABSENCE_BANDS)

_PEAK_VALUE_COL: int = _FEATURE_NAMES.index("peak_value")


# ---------------------------------------------------------------------------
# ID1: peak_value (inference) matches flowering_index (training) for presence
# ---------------------------------------------------------------------------

def test_ID1_inference_peak_value_matches_training_for_presence():
    """peak_value from inference path matches training-path flowering_index at presence locations.

    Both paths call the same flowering_index primitive. This asserts that
    the vectorised inference path (apply_index over 2-D arrays) produces
    the same value as the scalar training-path call for identical band inputs.
    """
    stack = _run_inference(_PRESENCE_BANDS)

    infer_peak_values = stack[:, _PEAK_VALUE_COL]

    # All pixels are constant, so peak_value must be identical for every pixel
    np.testing.assert_allclose(
        infer_peak_values,
        _TRAINING_FI_PRESENCE,
        atol=1e-6,
        err_msg=(
            f"Inference peak_value {infer_peak_values[0]:.6f} does not match "
            f"training flowering_index {_TRAINING_FI_PRESENCE:.6f} for presence bands. "
            "The two pipelines are using inconsistent index computations."
        ),
    )


# ---------------------------------------------------------------------------
# ID2: peak_value (inference) matches flowering_index (training) for absence
# ---------------------------------------------------------------------------

def test_ID2_inference_peak_value_matches_training_for_absence():
    """peak_value from inference path matches training-path flowering_index at absence locations."""
    stack = _run_inference(_ABSENCE_BANDS)

    infer_peak_values = stack[:, _PEAK_VALUE_COL]

    np.testing.assert_allclose(
        infer_peak_values,
        _TRAINING_FI_ABSENCE,
        atol=1e-6,
        err_msg=(
            f"Inference peak_value {infer_peak_values[0]:.6f} does not match "
            f"training flowering_index {_TRAINING_FI_ABSENCE:.6f} for absence bands."
        ),
    )


# ---------------------------------------------------------------------------
# ID3: No all-zero columns for presence-class input
# ---------------------------------------------------------------------------

# peak_doy_sd is intentionally all-zero in single-composite inference: it
# represents the standard deviation of peak DOY across years, which collapses
# to 0.0 when there is only one composite (no multi-year time series).
# This is documented in infer_features.py and is expected behaviour, not a defect.
_KNOWN_ZERO_COLUMNS: frozenset[str] = frozenset({"peak_doy_sd"})


def test_ID3_no_all_zero_columns_for_presence():
    """Feature stack has no unexpectedly all-zero columns for presence-class input.

    A degenerate all-zero column would indicate a feature was silently dropped
    or incorrectly initialised — which would produce silently wrong RF predictions.

    Columns in _KNOWN_ZERO_COLUMNS are exempt: they are designed to be zero in
    single-composite inference (e.g. peak_doy_sd has no multi-year std dev).
    """
    stack = _run_inference(_PRESENCE_BANDS)

    assert stack.shape == (_N_PIXELS, len(_FEATURE_NAMES)), (
        f"Unexpected stack shape {stack.shape}; expected ({_N_PIXELS}, {len(_FEATURE_NAMES)})"
    )

    all_zero_cols = [
        _FEATURE_NAMES[col_idx]
        for col_idx in range(stack.shape[1])
        if np.all(stack[:, col_idx] == 0.0)
        and _FEATURE_NAMES[col_idx] not in _KNOWN_ZERO_COLUMNS
    ]
    assert not all_zero_cols, (
        f"Feature columns are unexpectedly all-zero for presence input: {all_zero_cols}. "
        "This is degenerate — the column carries no information."
    )


# ---------------------------------------------------------------------------
# ID4: No NaN columns
# ---------------------------------------------------------------------------

def test_ID4_no_nan_columns():
    """Feature stack contains no NaN values for either presence or absence input."""
    for label, band_values in [("presence", _PRESENCE_BANDS), ("absence", _ABSENCE_BANDS)]:
        stack = _run_inference(band_values)
        nan_cols = [
            _FEATURE_NAMES[col_idx]
            for col_idx in range(stack.shape[1])
            if np.any(np.isnan(stack[:, col_idx]))
        ]
        assert not nan_cols, (
            f"NaN values found in feature columns for {label} input: {nan_cols}"
        )


# ---------------------------------------------------------------------------
# ID5: Column order matches feature_names_fixture.json
# ---------------------------------------------------------------------------

def test_ID5_column_order_matches_fixture():
    """Feature stack column order matches feature_names_fixture.json.

    This is the column order contract: inference must produce features in
    exactly the same order as training so the RF receives the correct inputs.
    """
    stack = _run_inference(_PRESENCE_BANDS)

    assert stack.shape[1] == len(_FEATURE_NAMES), (
        f"Expected {len(_FEATURE_NAMES)} columns (from fixture), got {stack.shape[1]}"
    )

    # Spot-check key structural columns by name
    hand_col = _FEATURE_NAMES.index("HAND")
    dtw_col = _FEATURE_NAMES.index("dist_to_water")
    mq_col = _FEATURE_NAMES.index("mean_quality")

    np.testing.assert_allclose(
        stack[:, hand_col], _HAND_VALUE, rtol=1e-5,
        err_msg=f"HAND column (index {hand_col}) has wrong values",
    )
    np.testing.assert_allclose(
        stack[:, dtw_col], _DTW_VALUE, rtol=1e-5,
        err_msg=f"dist_to_water column (index {dtw_col}) has wrong values",
    )

    expected_mq = float(np.mean(_QUALITY_WEIGHTS))
    np.testing.assert_allclose(
        stack[:, mq_col], expected_mq, rtol=1e-5,
        err_msg=f"mean_quality column (index {mq_col}) has wrong value",
    )
