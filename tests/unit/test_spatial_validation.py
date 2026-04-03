"""Unit tests for validate_spatial() and ValidationResult.

All tests use synthetic data constructed in-memory.
No pre-staged fixture data is required.

Contract tests
--------------
  C1.  Returns ValidationResult with all fields populated
  C2.  AUC is in [0, 1]
  C3.  Perfectly discriminable data returns AUC == 1.0
  C4.  Random (chance) data returns AUC near 0.5
  C5.  Fixture-like presence/absence separation exceeds SPATIAL_VALIDATION_THRESHOLD
         (encodes the scientific claim: well-separated presence/absence data
          must clear the 0.85 AUC gate before inference is permitted)
  C6.  precision is in [0, 1]
  C7.  recall is in [0, 1]
  C8.  calibration_error is in [0, 1]
  C9.  confusion_matrix is (tn, fp, fn, tp) with tn + fp + fn + tp == n_samples
  C10. n_presence and n_absence match input labels
  C11. passes_gate returns True when AUC meets threshold, False when below
  C12. validate_spatial raises ValueError on mismatched lengths
  C13. validate_spatial raises ValueError with no positive samples
  C14. validate_spatial raises ValueError with no negative samples
  C15. validate_spatial raises ValueError on out-of-range probabilities
  C16. validate_spatial is a pure function (does not mutate input lists)
  C17. Inverted predictions (all presence scored 0, absence scored 1) return AUC near 0
"""

from __future__ import annotations

import math
import random

import pytest

from analysis.constants import SPATIAL_VALIDATION_THRESHOLD
from analysis.primitives.validation import ValidationResult, validate_spatial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_data(n_each: int = 50) -> tuple[list[int], list[float]]:
    """Presence and absence perfectly separated: presence=1.0, absence=0.0."""
    labels = [1] * n_each + [0] * n_each
    probs = [1.0] * n_each + [0.0] * n_each
    return labels, probs


def _make_chance_data(n: int = 200, seed: int = 42) -> tuple[list[int], list[float]]:
    """Labels and probabilities are independent (random)."""
    rng = random.Random(seed)
    labels = [rng.randint(0, 1) for _ in range(n)]
    probs = [rng.random() for _ in range(n)]
    return labels, probs


def _make_separated_data(
    n_each: int = 100,
    presence_mean: float = 0.75,
    absence_mean: float = 0.25,
    spread: float = 0.10,
    seed: int = 42,
) -> tuple[list[int], list[float]]:
    """Well-separated presence/absence: presence probs centered near presence_mean,
    absence probs centered near absence_mean. Simulates a trained model output
    that the validation gate should pass (AUC > SPATIAL_VALIDATION_THRESHOLD).
    """
    rng = random.Random(seed)
    presence_probs = [
        max(0.0, min(1.0, presence_mean + rng.gauss(0, spread)))
        for _ in range(n_each)
    ]
    absence_probs = [
        max(0.0, min(1.0, absence_mean + rng.gauss(0, spread)))
        for _ in range(n_each)
    ]
    labels = [1] * n_each + [0] * n_each
    probs = presence_probs + absence_probs
    return labels, probs


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_C1_returns_validation_result_with_all_fields():
    """validate_spatial returns a ValidationResult with all fields populated."""
    labels, probs = _make_perfect_data(20)
    result = validate_spatial(labels, probs)

    assert isinstance(result, ValidationResult)
    assert hasattr(result, "auc")
    assert hasattr(result, "precision")
    assert hasattr(result, "recall")
    assert hasattr(result, "calibration_error")
    assert hasattr(result, "confusion_matrix")
    assert hasattr(result, "n_presence")
    assert hasattr(result, "n_absence")

    # No field is None or NaN
    for field_name in ("auc", "precision", "recall", "calibration_error"):
        val = getattr(result, field_name)
        assert val is not None, f"{field_name} must not be None"
        assert not math.isnan(val), f"{field_name} must not be NaN"


def test_C2_auc_in_unit_interval():
    """AUC is always in [0, 1]."""
    for labels, probs in [
        _make_perfect_data(30),
        _make_chance_data(100),
        _make_separated_data(50),
    ]:
        result = validate_spatial(labels, probs)
        assert 0.0 <= result.auc <= 1.0, f"AUC={result.auc} out of [0, 1]"


def test_C3_perfect_data_returns_auc_1():
    """Perfectly discriminable data returns AUC == 1.0."""
    labels, probs = _make_perfect_data(50)
    result = validate_spatial(labels, probs)
    assert result.auc == pytest.approx(1.0), (
        f"Perfect separation should yield AUC=1.0, got {result.auc}"
    )


def test_C4_chance_data_returns_auc_near_half():
    """Random labels and probabilities yield AUC near 0.5."""
    labels, probs = _make_chance_data(500, seed=0)
    result = validate_spatial(labels, probs)
    # With 500 samples and independent labels/probs, AUC should be within 0.1 of 0.5
    assert abs(result.auc - 0.5) < 0.1, (
        f"Chance data should yield AUC near 0.5, got {result.auc:.4f}"
    )


def test_C5_separated_data_exceeds_validation_threshold():
    """Well-separated fixture-like data must exceed SPATIAL_VALIDATION_THRESHOLD.

    This encodes the scientific claim: if real training data produces
    well-separated presence/absence probabilities, the model must clear
    the AUC gate before inference is permitted.
    """
    labels, probs = _make_separated_data(
        n_each=100,
        presence_mean=0.80,
        absence_mean=0.20,
        spread=0.08,
        seed=42,
    )
    result = validate_spatial(labels, probs)
    assert result.auc >= SPATIAL_VALIDATION_THRESHOLD, (
        f"Well-separated data should yield AUC >= {SPATIAL_VALIDATION_THRESHOLD}, "
        f"got {result.auc:.4f}. This gate protects inference from undertrained models."
    )


def test_C6_precision_in_unit_interval():
    """Precision is in [0, 1]."""
    labels, probs = _make_separated_data(50)
    result = validate_spatial(labels, probs)
    assert 0.0 <= result.precision <= 1.0, (
        f"precision={result.precision} out of [0, 1]"
    )


def test_C7_recall_in_unit_interval():
    """Recall is in [0, 1]."""
    labels, probs = _make_separated_data(50)
    result = validate_spatial(labels, probs)
    assert 0.0 <= result.recall <= 1.0, (
        f"recall={result.recall} out of [0, 1]"
    )


def test_C8_calibration_error_in_unit_interval():
    """Calibration error (ECE) is in [0, 1]."""
    for labels, probs in [
        _make_perfect_data(30),
        _make_chance_data(100),
        _make_separated_data(50),
    ]:
        result = validate_spatial(labels, probs)
        assert 0.0 <= result.calibration_error <= 1.0, (
            f"calibration_error={result.calibration_error} out of [0, 1]"
        )


def test_C9_confusion_matrix_counts_sum_to_n_samples():
    """tn + fp + fn + tp == total number of samples."""
    n_each = 40
    labels, probs = _make_separated_data(n_each)
    result = validate_spatial(labels, probs)
    tn, fp, fn, tp = result.confusion_matrix
    total = tn + fp + fn + tp
    assert total == n_each * 2, (
        f"Confusion matrix counts should sum to {n_each * 2}, got {total}"
    )


def test_C10_n_presence_n_absence_match_labels():
    """n_presence and n_absence match the input label counts."""
    labels = [1, 1, 1, 0, 0]
    probs = [0.9, 0.8, 0.7, 0.3, 0.2]
    result = validate_spatial(labels, probs)
    assert result.n_presence == 3
    assert result.n_absence == 2


def test_C11_passes_gate():
    """passes_gate returns True when AUC >= threshold, False when below."""
    labels, probs = _make_perfect_data(20)
    result = validate_spatial(labels, probs)  # AUC = 1.0

    assert result.passes_gate(SPATIAL_VALIDATION_THRESHOLD) is True
    assert result.passes_gate(1.0) is True
    assert result.passes_gate(1.01) is False  # nothing passes > 1.0


def test_C12_raises_on_mismatched_lengths():
    """ValueError raised when labels and probabilities have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        validate_spatial([1, 0, 1], [0.9, 0.1])


def test_C13_raises_on_no_positive_samples():
    """ValueError raised when all labels are 0 (no presence samples)."""
    with pytest.raises(ValueError, match="no positive"):
        validate_spatial([0, 0, 0], [0.5, 0.3, 0.4])


def test_C14_raises_on_no_negative_samples():
    """ValueError raised when all labels are 1 (no absence samples)."""
    with pytest.raises(ValueError, match="no negative"):
        validate_spatial([1, 1, 1], [0.8, 0.9, 0.7])


def test_C15_raises_on_out_of_range_probabilities():
    """ValueError raised when any probability is outside [0, 1]."""
    with pytest.raises(ValueError, match="out-of-range"):
        validate_spatial([1, 0], [1.5, 0.3])

    with pytest.raises(ValueError, match="out-of-range"):
        validate_spatial([1, 0], [0.9, -0.1])


def test_C16_pure_function_does_not_mutate_inputs():
    """validate_spatial does not mutate the input lists."""
    labels = [1, 1, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.1]
    labels_copy = labels[:]
    probs_copy = probs[:]

    validate_spatial(labels, probs)

    assert labels == labels_copy, "labels list was mutated"
    assert probs == probs_copy, "probabilities list was mutated"


def test_C17_inverted_predictions_return_low_auc():
    """When presence is always scored low and absence scored high, AUC is near 0."""
    n_each = 50
    labels = [1] * n_each + [0] * n_each
    probs = [0.0] * n_each + [1.0] * n_each  # perfectly wrong
    result = validate_spatial(labels, probs)
    assert result.auc == pytest.approx(0.0), (
        f"Perfectly inverted predictions should yield AUC=0.0, got {result.auc}"
    )
