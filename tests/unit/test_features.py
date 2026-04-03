"""Unit tests for assemble_feature_vector().

All tests use synthetic data constructed in-memory.
No pre-staged fixture data is required.

Contract tests
--------------
  C1.  Feature vector contains all expected keys in expected order
  C2.  No NaN values in assembled output
  C3.  mean_quality is in [0, 1]
  C4.  Structural feature join does not silently drop rows
         (row count is preserved — missing a structural key raises ValueError,
          not a silent partial vector)
  C5.  Empty waveform_features raises ValueError (caller contract)
  C6.  Missing waveform key raises ValueError
  C7.  Missing structural key raises ValueError
  C8.  Extra structural keys are included in output
  C9.  Empty observations list yields mean_quality == 0.0
  C10. mean_quality reflects actual quality scores (not always 1.0)
  C11. Key order: waveform keys come first, then structural, then mean_quality
  C12. All values are float (not int, not None)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from analysis.constants import Q_FULL
from analysis.timeseries.features import (
    STRUCTURAL_KEYS,
    WAVEFORM_KEYS,
    assemble_feature_vector,
)
from analysis.timeseries.observation import Observation, ObservationQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_waveform_features(
    peak_value: float = 0.45,
    peak_doy: float = 270.0,
    spike_duration: float = 3.0,
    peak_doy_mean: float = 268.0,
    peak_doy_sd: float = 5.0,
    years_detected: float = 4.0,
) -> dict[str, float]:
    return {
        "peak_value": peak_value,
        "peak_doy": peak_doy,
        "spike_duration": spike_duration,
        "peak_doy_mean": peak_doy_mean,
        "peak_doy_sd": peak_doy_sd,
        "years_detected": years_detected,
    }


def _make_structural_features(
    hand: float = 1.5,
    dist_to_water: float = 250.0,
) -> dict[str, float]:
    return {"HAND": hand, "dist_to_water": dist_to_water}


def _make_obs(
    quality_score: float = 0.80,
    point_id: str = "pt_001",
) -> Observation:
    """Create an Observation whose Q_FULL score is approximately quality_score.

    We set all five components to quality_score^(1/5) so the product equals
    quality_score (approximately, via floating-point).
    """
    component = quality_score ** (1 / 5)
    q = ObservationQuality(
        scl_purity=component,
        aot=component,
        view_zenith=component,
        sun_zenith=component,
        greenness_z=component,
    )
    return Observation(
        point_id=point_id,
        date=datetime(2022, 9, 27),
        bands={"B05": 0.10, "B07": 0.35, "B08": 0.45, "B11": 0.08},
        quality=q,
    )


def _make_obs_sequence(scores: list[float]) -> list[Observation]:
    return [_make_obs(s) for s in scores]


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_C1_feature_vector_contains_all_expected_keys():
    """Feature vector contains all required waveform + structural + mean_quality keys."""
    result = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        _make_obs_sequence([0.80, 0.85, 0.90]),
    )
    expected_keys = set(WAVEFORM_KEYS) | set(STRUCTURAL_KEYS) | {"mean_quality"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}, "
        f"extra keys: {set(result.keys()) - expected_keys}"
    )


def test_C2_no_nan_in_assembled_output():
    """No NaN values in the assembled feature vector."""
    import math
    result = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        _make_obs_sequence([0.80, 0.85]),
    )
    nan_keys = [k for k, v in result.items() if math.isnan(v)]
    assert not nan_keys, f"NaN values found for keys: {nan_keys}"


def test_C3_mean_quality_in_unit_interval():
    """mean_quality is in [0, 1]."""
    result = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        _make_obs_sequence([0.60, 0.70, 0.80, 0.90]),
    )
    assert 0.0 <= result["mean_quality"] <= 1.0, (
        f"mean_quality={result['mean_quality']} out of [0, 1]"
    )


def test_C4_structural_join_preserves_row_count():
    """Missing a structural key raises ValueError — no silent row drop.

    Row count is preserved by the caller checking all required structural keys
    are present. If a key is missing the function raises rather than returning
    a partial vector that would silently shrink the training matrix.
    """
    bad_structural = {"HAND": 2.0}  # missing dist_to_water
    with pytest.raises(ValueError, match="dist_to_water"):
        assemble_feature_vector(
            _make_waveform_features(),
            bad_structural,
            _make_obs_sequence([0.80]),
        )


def test_C5_empty_waveform_raises():
    """Empty waveform_features dict raises ValueError with a helpful message."""
    with pytest.raises(ValueError, match="empty waveform_features"):
        assemble_feature_vector(
            {},
            _make_structural_features(),
            _make_obs_sequence([0.80]),
        )


def test_C6_missing_waveform_key_raises():
    """Waveform dict missing a required key raises ValueError."""
    bad_waveform = _make_waveform_features()
    del bad_waveform["peak_value"]
    with pytest.raises(ValueError, match="peak_value"):
        assemble_feature_vector(
            bad_waveform,
            _make_structural_features(),
            _make_obs_sequence([0.80]),
        )


def test_C7_missing_structural_key_raises():
    """Structural dict missing a required key raises ValueError."""
    bad_structural = {"dist_to_water": 100.0}  # missing HAND
    with pytest.raises(ValueError, match="HAND"):
        assemble_feature_vector(
            _make_waveform_features(),
            bad_structural,
            _make_obs_sequence([0.80]),
        )


def test_C8_extra_structural_keys_included():
    """Extra keys in structural_features appear in the output."""
    structural = _make_structural_features()
    structural["slope_pct"] = 3.5  # extra key not in STRUCTURAL_KEYS

    result = assemble_feature_vector(
        _make_waveform_features(),
        structural,
        _make_obs_sequence([0.80]),
    )
    assert "slope_pct" in result, "Extra structural key should be present in output"
    assert result["slope_pct"] == pytest.approx(3.5)


def test_C9_empty_observations_yields_zero_mean_quality():
    """Empty observations list yields mean_quality == 0.0."""
    result = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        [],
    )
    assert result["mean_quality"] == pytest.approx(0.0)


def test_C10_mean_quality_reflects_actual_scores():
    """mean_quality is computed from observation quality scores, not always 1.0."""
    # Observations with low quality scores
    low_obs = _make_obs_sequence([0.20, 0.25, 0.30])
    result_low = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        low_obs,
    )
    high_obs = _make_obs_sequence([0.80, 0.85, 0.90])
    result_high = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        high_obs,
    )
    assert result_low["mean_quality"] < result_high["mean_quality"], (
        f"Low-quality obs should yield lower mean_quality: "
        f"low={result_low['mean_quality']:.4f}, high={result_high['mean_quality']:.4f}"
    )


def test_C11_key_order_waveform_then_structural_then_quality():
    """Key order: waveform keys first, then structural, then mean_quality last."""
    structural = _make_structural_features()
    result = assemble_feature_vector(
        _make_waveform_features(),
        structural,
        _make_obs_sequence([0.80]),
    )
    keys = list(result.keys())
    waveform_positions = [keys.index(k) for k in WAVEFORM_KEYS]
    structural_positions = [keys.index(k) for k in structural.keys()]
    quality_position = keys.index("mean_quality")

    assert max(waveform_positions) < min(structural_positions), (
        "All waveform keys must appear before structural keys"
    )
    assert max(structural_positions) < quality_position, (
        "All structural keys must appear before mean_quality"
    )
    assert keys[-1] == "mean_quality", "mean_quality must be the last key"


def test_C12_all_values_are_float():
    """All values in the assembled vector are Python floats."""
    result = assemble_feature_vector(
        _make_waveform_features(),
        _make_structural_features(),
        _make_obs_sequence([0.80, 0.85]),
    )
    non_float = {k: type(v).__name__ for k, v in result.items() if not isinstance(v, float)}
    assert not non_float, f"Non-float values found: {non_float}"
