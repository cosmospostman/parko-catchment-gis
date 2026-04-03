"""Unit tests for extract_waveform_features() and flowering_index().

All tests use synthetic Observation objects constructed in-memory.
No pre-staged fixture data is required.

Scientific assumption tests (marked with "Scientific:")
-------------------------------------------------------
These encode the core biological claim of the pipeline as executable
assertions. A failure means either the code is wrong, or the threshold /
window constants in analysis/constants.py need revisiting.

  S1. Presence-like observations have peak_value > FLOWERING_THRESHOLD
  S2. Peak DOY falls within FLOWERING_WINDOW for presence-like observations
  S3. Median presence peak_value > median absence peak_value

Behavioural tests
-----------------
  B1.  Fewer than min_years of data returns {}
  B2.  All-cloud (all low quality) sequence does not manufacture a peak
  B3.  Exactly min_years of detected data returns features (boundary)
  B4.  All feature keys are present in the output dict
  B5.  years_detected matches the count of years with a real peak
  B6.  peak_doy is within the requested window
  B7.  spike_duration >= 1 when a peak is detected
  B8.  peak_doy_sd == 0.0 when all years have the same peak DOY
  B9.  Quality weighting: low-quality obs with high index loses to
       high-quality obs with moderate index
  B10. Observations outside the DOY window are ignored
  B11. apply_index produces the same result as calling flowering_index
       per-pixel (shared primitive correctness)
"""

from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np
import pytest

from analysis.constants import (
    FLOWERING_THRESHOLD,
    FLOWERING_WINDOW,
    Q_FULL,
)
from analysis.primitives.indices import apply_index, flowering_index
from analysis.timeseries.observation import Observation, ObservationQuality
from analysis.timeseries.waveform import extract_waveform_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality(
    scl_purity: float = 0.95,
    aot: float = 0.95,
    view_zenith: float = 0.95,
    sun_zenith: float = 0.90,
    greenness_z: float = 0.90,
) -> ObservationQuality:
    return ObservationQuality(
        scl_purity=scl_purity,
        aot=aot,
        view_zenith=view_zenith,
        sun_zenith=sun_zenith,
        greenness_z=greenness_z,
    )


def _high_quality() -> ObservationQuality:
    """Quality profile that scores near 1.0 on Q_FULL."""
    return _quality(
        scl_purity=0.98, aot=0.97, view_zenith=0.97,
        sun_zenith=0.95, greenness_z=0.95,
    )


def _low_quality() -> ObservationQuality:
    """Quality profile that scores below min_quality=0.3 on Q_FULL."""
    return ObservationQuality(
        scl_purity=0.20, aot=0.20, view_zenith=0.20,
        sun_zenith=0.20, greenness_z=0.20,
    )


def _presence_bands() -> dict[str, float]:
    """Band values that produce a high flowering_index (presence-like).

    Target: re_slope ~ 0.55, nir_swir ~ 0.55  → index ~ 0.55
    B07=0.35, B05=0.10 → re_slope = 0.25/0.45 = 0.556
    B08=0.45, B11=0.08 → nir_swir = 0.37/0.53 = 0.698
    index = (0.556 + 0.698) / 2 = 0.627
    """
    return {"B05": 0.10, "B07": 0.35, "B08": 0.45, "B11": 0.08}


def _absence_bands() -> dict[str, float]:
    """Band values that produce a low flowering_index (absence-like).

    B07=0.12, B05=0.11 → re_slope = 0.01/0.23 = 0.043
    B08=0.20, B11=0.18 → nir_swir = 0.02/0.38 = 0.053
    index = (0.043 + 0.053) / 2 = 0.048
    """
    return {"B05": 0.11, "B07": 0.12, "B08": 0.20, "B11": 0.18}


def _doy_to_date(year: int, doy: int) -> datetime:
    """Convert year + day-of-year to a naive datetime."""
    return datetime(year, 1, 1) + __import__("datetime").timedelta(days=doy - 1)


def _make_obs(
    year: int,
    doy: int,
    bands: dict[str, float],
    quality: ObservationQuality | None = None,
    point_id: str = "pt_001",
) -> Observation:
    if quality is None:
        quality = _high_quality()
    return Observation(
        point_id=point_id,
        date=_doy_to_date(year, doy),
        bands=bands,
        quality=quality,
    )


def _presence_series(
    point_id: str = "pt_presence",
    years: list[int] | None = None,
    peak_doy: int = 270,
) -> list[Observation]:
    """Multi-year presence-like observation sequence.

    Each year gets: one peak observation at peak_doy (presence bands, high
    quality) plus two off-peak low-value observations for context.
    """
    if years is None:
        years = [2020, 2021, 2022, 2023]
    obs = []
    for year in years:
        # Peak observation — strong signal, inside flowering window
        obs.append(_make_obs(year, peak_doy, _presence_bands(), _high_quality(), point_id))
        # Off-peak observations — low index, same quality
        obs.append(_make_obs(year, peak_doy - 60, _absence_bands(), _high_quality(), point_id))
        obs.append(_make_obs(year, peak_doy + 40, _absence_bands(), _high_quality(), point_id))
    return obs


def _absence_series(
    point_id: str = "pt_absence",
    years: list[int] | None = None,
) -> list[Observation]:
    """Multi-year absence-like observation sequence. No peak above threshold."""
    if years is None:
        years = [2020, 2021, 2022, 2023]
    obs = []
    for year in years:
        for doy in [210, 250, 290]:
            obs.append(_make_obs(year, doy, _absence_bands(), _high_quality(), point_id))
    return obs


# ---------------------------------------------------------------------------
# Scientific assumption tests
# ---------------------------------------------------------------------------

def test_S1_presence_peak_value_exceeds_threshold():
    """Scientific: presence series has peak_value > FLOWERING_THRESHOLD."""
    features = extract_waveform_features(
        _presence_series(), flowering_index, window=FLOWERING_WINDOW
    )
    assert features, "Expected non-empty feature dict for presence series"
    assert features["peak_value"] > FLOWERING_THRESHOLD, (
        f"peak_value={features['peak_value']:.4f} <= "
        f"FLOWERING_THRESHOLD={FLOWERING_THRESHOLD}"
    )


def test_S2_presence_peak_doy_within_flowering_window():
    """Scientific: presence peak DOY is within FLOWERING_WINDOW."""
    features = extract_waveform_features(
        _presence_series(), flowering_index, window=FLOWERING_WINDOW
    )
    assert features, "Expected non-empty feature dict for presence series"
    doy = features["peak_doy"]
    doy_start, doy_end = FLOWERING_WINDOW
    assert doy_start <= doy <= doy_end, (
        f"peak_doy={doy} outside FLOWERING_WINDOW={FLOWERING_WINDOW}"
    )


def test_S3_presence_median_peak_higher_than_absence():
    """Scientific: median presence peak_value > median absence peak_value."""
    presence_points = [
        _presence_series(point_id=f"pt_p{i}") for i in range(5)
    ]
    absence_points = [
        _absence_series(point_id=f"pt_a{i}") for i in range(5)
    ]

    presence_peaks = []
    for obs in presence_points:
        f = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
        if f:
            presence_peaks.append(f["peak_value"])

    absence_peaks = []
    for obs in absence_points:
        f = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
        if f:
            absence_peaks.append(f.get("peak_value", 0.0))

    # Absence series may not produce features at all (no peak above threshold)
    # so absence_peaks may be empty — absence peak values are effectively 0
    if not absence_peaks:
        absence_median = 0.0
    else:
        absence_median = statistics.median(absence_peaks)

    assert presence_peaks, "No presence features extracted — check presence_bands()"
    presence_median = statistics.median(presence_peaks)

    assert presence_median > absence_median, (
        f"presence median={presence_median:.4f} not > absence median={absence_median:.4f}"
    )


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

def test_B1_fewer_than_min_years_returns_empty():
    """Fewer than min_years of data returns {}."""
    # Only 2 years of observations, min_years=3 (default)
    obs = _presence_series(years=[2020, 2021])
    features = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
    assert features == {}


def test_B2_all_low_quality_does_not_manufacture_peak():
    """All-cloud (all low quality) acquisitions do not produce a peak."""
    obs = []
    years = [2020, 2021, 2022, 2023]
    for year in years:
        for doy in [220, 250, 280]:
            # Use presence bands but very low quality — score will be below min_quality
            obs.append(_make_obs(year, doy, _presence_bands(), _low_quality()))

    features = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
    assert features == {}, (
        "Low-quality observations should not manufacture a peak; got non-empty features"
    )


def test_B3_exactly_min_years_detected_returns_features():
    """Exactly min_years of detected data returns a non-empty feature dict."""
    # 3 years of data, 3 detected peaks — exactly min_years=3
    obs = _presence_series(years=[2020, 2021, 2022])
    features = extract_waveform_features(
        obs, flowering_index, window=FLOWERING_WINDOW, min_years=3
    )
    assert features != {}, "Exactly min_years should return features"


def test_B4_all_feature_keys_present():
    """Output dict contains all required feature keys."""
    features = extract_waveform_features(
        _presence_series(), flowering_index, window=FLOWERING_WINDOW
    )
    assert features, "Expected non-empty features"
    expected_keys = {
        "peak_value", "peak_doy", "spike_duration",
        "peak_doy_mean", "peak_doy_sd", "years_detected",
    }
    assert set(features.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(features.keys())}, "
        f"extra keys: {set(features.keys()) - expected_keys}"
    )


def test_B5_years_detected_matches_peak_count():
    """years_detected equals the count of years with a peak above threshold."""
    # 4 years present, all should detect
    features = extract_waveform_features(
        _presence_series(years=[2020, 2021, 2022, 2023]),
        flowering_index, window=FLOWERING_WINDOW,
    )
    assert features
    assert features["years_detected"] == pytest.approx(4.0)


def test_B6_peak_doy_within_requested_window():
    """peak_doy is always within the specified window."""
    features = extract_waveform_features(
        _presence_series(), flowering_index, window=FLOWERING_WINDOW
    )
    assert features
    doy_start, doy_end = FLOWERING_WINDOW
    assert doy_start <= features["peak_doy"] <= doy_end


def test_B7_spike_duration_at_least_one_when_detected():
    """spike_duration >= 1 when a peak is detected."""
    features = extract_waveform_features(
        _presence_series(), flowering_index, window=FLOWERING_WINDOW
    )
    assert features
    assert features["spike_duration"] >= 1.0


def test_B8_peak_doy_sd_zero_with_consistent_timing():
    """peak_doy_sd == 0 when all detected years have identical peak DOY."""
    # All observations peak at the same DOY — stdev should be 0
    obs = _presence_series(years=[2020, 2021, 2022, 2023], peak_doy=260)
    features = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
    assert features
    assert features["peak_doy_sd"] == pytest.approx(0.0, abs=1e-9)


def test_B9_quality_weighting_prefers_high_quality_moderate_index():
    """High-quality moderate-index obs beats low-quality high-index obs."""
    # Two observations per year: one with high index + low quality,
    # one with moderate index + high quality.
    # Quality weighting should favour the high-quality one.
    years = [2020, 2021, 2022, 2023]
    obs = []
    for year in years:
        # High-index but low quality
        low_q = ObservationQuality(
            scl_purity=0.30, aot=0.30, view_zenith=0.30,
            sun_zenith=0.30, greenness_z=0.30,
        )
        obs.append(_make_obs(year, 260, _presence_bands(), low_q))

        # Lower-index but high quality — weighted value should win
        # flowering_index(_absence_bands()) is ~ 0.048; quality~0.75
        # weighted ≈ 0.048 * 0.75^5 ≈ 0.012 — below threshold
        # So let's use a band set that gives a moderate but above-threshold index:
        # B07=0.25, B05=0.10 → re_slope = 0.15/0.35 = 0.429
        # B08=0.35, B11=0.10 → nir_swir = 0.25/0.45 = 0.556
        # index = (0.429 + 0.556)/2 = 0.493
        moderate_bands = {"B05": 0.10, "B07": 0.25, "B08": 0.35, "B11": 0.10}
        obs.append(_make_obs(year, 265, moderate_bands, _high_quality()))

    features = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
    # The key check: we get features (did not degenerate), and the peak comes
    # from the high-quality moderate observation (DOY 265), not the low-quality
    # high-index one (DOY 260).
    assert features, "Expected features from quality-weighted series"
    # peak_doy should be 265 (the high-quality observation)
    assert features["peak_doy"] == pytest.approx(265.0), (
        f"Expected peak_doy=265 (high-quality obs), got {features['peak_doy']}"
    )


def test_B10_observations_outside_window_are_ignored():
    """Observations outside the DOY window do not contribute to peak detection."""
    # Put the presence signal outside the window (DOY 100), inside only absence bands
    years = [2020, 2021, 2022, 2023]
    obs = []
    for year in years:
        # Strong signal but DOY 100 — outside FLOWERING_WINDOW (200-340)
        obs.append(_make_obs(year, 100, _presence_bands(), _high_quality()))
        # Inside window: only absence-level signal
        for doy in [220, 260, 300]:
            obs.append(_make_obs(year, doy, _absence_bands(), _high_quality()))

    features = extract_waveform_features(obs, flowering_index, window=FLOWERING_WINDOW)
    # Should return {} because absence bands give index < FLOWERING_THRESHOLD,
    # and the strong signal at DOY 100 is excluded
    assert features == {}, (
        "Observations outside the DOY window should not contribute to peak detection"
    )


# ---------------------------------------------------------------------------
# apply_index correctness test
# ---------------------------------------------------------------------------

def test_B11_apply_index_matches_per_pixel_flowering_index():
    """apply_index gives the same result as calling flowering_index per-pixel."""
    rng = np.random.default_rng(42)
    shape = (4, 4)
    band_stack = {
        "B05": rng.uniform(0.05, 0.20, shape),
        "B07": rng.uniform(0.15, 0.45, shape),
        "B08": rng.uniform(0.25, 0.55, shape),
        "B11": rng.uniform(0.05, 0.25, shape),
    }

    result = apply_index(flowering_index, band_stack)

    # Compare pixel-by-pixel
    for r in range(shape[0]):
        for c in range(shape[1]):
            pixel = {band: float(arr[r, c]) for band, arr in band_stack.items()}
            expected = flowering_index(pixel)
            assert result[r, c] == pytest.approx(expected, abs=1e-12), (
                f"Mismatch at ({r},{c}): apply_index={result[r,c]:.8f}, "
                f"per-pixel={expected:.8f}"
            )


def test_apply_index_raises_on_mismatched_shapes():
    """apply_index raises ValueError when band arrays have different shapes."""
    band_stack = {
        "B05": np.ones((4, 4)),
        "B07": np.ones((3, 4)),  # wrong shape
    }
    with pytest.raises(ValueError, match="same shape"):
        apply_index(flowering_index, band_stack)
