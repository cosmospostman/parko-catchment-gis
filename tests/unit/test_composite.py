"""Unit tests for analysis/timeseries/composite.py.

Tests
-----
C1. Quality-weighted composite gives higher weight to clear observations.
C2. Single-observation input returns that observation's values unchanged.
C3. All-zero weights fall back to unweighted mean (no NaN output).
C4. Shape contract: output shape matches input array shape.
C5. Mismatched quality_weights length raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.timeseries.composite import quality_weighted_composite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const(value: float, shape: tuple[int, int] = (3, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# C1: Quality-weighted composite gives higher weight to clear observations
# ---------------------------------------------------------------------------

def test_C1_higher_weight_to_clear_observation():
    """High-quality acquisition dominates the composite.

    Two acquisitions with band values 0.1 and 0.9. The first has weight 0.9
    (clear), the second has weight 0.1 (cloudy). The composite should be
    closer to 0.1 than 0.9.
    """
    band_stacks = {
        "B04": [_const(0.1), _const(0.9)],
    }
    quality_weights = [0.9, 0.1]

    result = quality_weighted_composite(band_stacks, quality_weights)

    composite_val = result["B04"][1, 1]
    # Weighted mean: (0.9 * 0.1 + 0.1 * 0.9) / (0.9 + 0.1) = 0.18 / 1.0 = 0.18
    expected = (0.9 * 0.1 + 0.1 * 0.9) / (0.9 + 0.1)
    assert abs(composite_val - expected) < 1e-6, (
        f"Expected composite ~{expected:.4f}, got {composite_val:.4f}"
    )
    # Must be closer to the high-weight value (0.1) than to the low-weight (0.9)
    assert composite_val < 0.5, (
        f"Composite {composite_val:.4f} should be closer to 0.1 (high-weight acquisition)"
    )


def test_C1_weighted_mean_formula_two_bands():
    """Weighted mean is computed correctly for multiple bands simultaneously."""
    band_stacks = {
        "B05": [_const(0.2), _const(0.8)],
        "B07": [_const(0.3), _const(0.7)],
    }
    quality_weights = [0.8, 0.2]

    result = quality_weighted_composite(band_stacks, quality_weights)

    for band, low, high in [("B05", 0.2, 0.8), ("B07", 0.3, 0.7)]:
        expected = (0.8 * low + 0.2 * high) / 1.0
        got = result[band][0, 0]
        assert abs(got - expected) < 1e-6, (
            f"Band {band}: expected {expected:.4f}, got {got:.4f}"
        )


# ---------------------------------------------------------------------------
# C2: Single-observation input returns that observation's values
# ---------------------------------------------------------------------------

def test_C2_single_observation_returns_input_unchanged():
    """With one acquisition and weight=1.0, composite equals the input array."""
    data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    band_stacks = {"B08": [data]}
    quality_weights = [1.0]

    result = quality_weighted_composite(band_stacks, quality_weights)

    np.testing.assert_allclose(result["B08"], data.astype(np.float64), rtol=1e-6)


def test_C2_single_observation_any_weight():
    """With one acquisition and any positive weight, composite still equals input."""
    data = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    band_stacks = {"B11": [data]}
    quality_weights = [0.3]

    result = quality_weighted_composite(band_stacks, quality_weights)

    np.testing.assert_allclose(result["B11"], data.astype(np.float64), rtol=1e-6)


# ---------------------------------------------------------------------------
# C3: All-zero weights fall back to unweighted mean (no NaN)
# ---------------------------------------------------------------------------

def test_C3_zero_weights_no_nan():
    """When all quality weights are zero, result is unweighted mean (not NaN)."""
    band_stacks = {
        "B02": [_const(0.1), _const(0.3)],
    }
    quality_weights = [0.0, 0.0]

    result = quality_weighted_composite(band_stacks, quality_weights)

    assert not np.any(np.isnan(result["B02"])), "Composite must not produce NaN with zero weights"
    expected_fallback = 0.2  # unweighted mean of 0.1 and 0.3
    np.testing.assert_allclose(result["B02"], expected_fallback, atol=1e-6)


# ---------------------------------------------------------------------------
# C4: Output shape matches input array shape
# ---------------------------------------------------------------------------

def test_C4_output_shape_matches_input():
    """Output array has the same spatial shape as the input arrays."""
    shape = (4, 5)
    band_stacks = {
        "B03": [np.random.rand(*shape).astype(np.float32) for _ in range(3)],
        "B04": [np.random.rand(*shape).astype(np.float32) for _ in range(3)],
    }
    quality_weights = [0.5, 0.8, 0.3]

    result = quality_weighted_composite(band_stacks, quality_weights)

    for band in ["B03", "B04"]:
        assert result[band].shape == shape, (
            f"Band {band}: expected shape {shape}, got {result[band].shape}"
        )


# ---------------------------------------------------------------------------
# C5: Error cases
# ---------------------------------------------------------------------------

def test_C5_mismatched_weights_length_raises():
    """ValueError raised when quality_weights length differs from stack length."""
    band_stacks = {
        "B08": [_const(0.5), _const(0.6)],  # 2 acquisitions
    }
    quality_weights = [1.0]  # only 1 weight

    with pytest.raises(ValueError, match="quality_weights"):
        quality_weighted_composite(band_stacks, quality_weights)


def test_C5_empty_band_stacks_raises():
    """ValueError raised when band_stacks is empty."""
    with pytest.raises(ValueError):
        quality_weighted_composite({}, [1.0])
