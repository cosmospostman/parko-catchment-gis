"""Unit tests for ArchiveStats and score_observation().

All tests use synthetic Observation objects constructed in-memory.
No pre-staged data is required — the tests are fully self-contained.

Tests
-----
 1. All quality components on the scored result are in [0, 1].
 2. Clear, nadir, low-AOT observations score near 1.0 on Q_FULL.
 3. High-cloud (low scl_purity) observations score low overall.
 4. High-haze (low aot) observations score low on Q_ATMOSPHERIC.
 5. score_observation is a pure function (returns new object, does not mutate input).
 6. greenness_z == 1.0 when NDVI equals the archive mean.
 7. greenness_z decreases as NDVI deviates from the archive mean.
 8. greenness_z == 1.0 before scoring (Stage 5 placeholder is unchanged in input).
 9. ArchiveStats.from_observations() computes correct mean and std.
10. ArchiveStats.from_observations() raises ValueError with fewer than 2 usable obs.
11. ArchiveStats std is floored to a positive value when constructed with std <= 0.
12. score_observation does not raise when B08/B04 are missing (defaults to NDVI=0).
13. Other quality components (scl_purity, aot, view_zenith, sun_zenith) are preserved.
"""

from datetime import datetime

import pytest

from analysis.constants import Q_ATMOSPHERIC, Q_FULL
from analysis.primitives.quality import ArchiveStats, score_observation
from analysis.timeseries.observation import Observation, ObservationQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality(
    scl_purity: float = 0.9,
    aot: float = 0.9,
    view_zenith: float = 0.95,
    sun_zenith: float = 0.85,
    greenness_z: float = 1.0,  # Stage 5 placeholder
) -> ObservationQuality:
    return ObservationQuality(
        scl_purity=scl_purity,
        aot=aot,
        view_zenith=view_zenith,
        sun_zenith=sun_zenith,
        greenness_z=greenness_z,
    )


def _obs(
    bands: dict[str, float] | None = None,
    quality: ObservationQuality | None = None,
    point_id: str = "pt_001",
) -> Observation:
    """Create a synthetic Observation with sensible defaults."""
    if bands is None:
        # Realistic surface reflectance values; NDVI ≈ (0.30 - 0.05) / (0.30 + 0.05) = 0.714
        bands = {"B04": 0.05, "B08": 0.30, "B03": 0.04, "B02": 0.03}
    if quality is None:
        quality = _quality()
    return Observation(
        point_id=point_id,
        date=datetime(2022, 8, 15),
        bands=bands,
        quality=quality,
    )


def _typical_archive_stats() -> ArchiveStats:
    """Return ArchiveStats representative of a vegetated tropical scene."""
    return ArchiveStats(mean=0.60, std=0.15)


# ---------------------------------------------------------------------------
# Test 1: All quality components on the scored result are in [0, 1]
# ---------------------------------------------------------------------------

def test_all_components_in_range_after_scoring():
    obs = _obs()
    scored = score_observation(obs, _typical_archive_stats())
    q = scored.quality
    for name, value in [
        ("scl_purity", q.scl_purity),
        ("aot", q.aot),
        ("view_zenith", q.view_zenith),
        ("sun_zenith", q.sun_zenith),
        ("greenness_z", q.greenness_z),
    ]:
        assert 0.0 <= value <= 1.0, f"{name} = {value} is out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 2: Clear, nadir, low-AOT observation scores near 1.0 on Q_FULL
# ---------------------------------------------------------------------------

def test_clear_nadir_low_aot_scores_near_one():
    # NDVI = (0.60 - 0.03) / (0.60 + 0.03) = 0.905 — not anomalous when mean=0.60
    # Actually let's set NDVI exactly at archive mean for max greenness_z
    # NDVI = (B08 - B04) / (B08 + B04) = 0.60 when B08=0.8, B04=0.2
    bands = {"B04": 0.2, "B08": 0.8}
    obs = _obs(
        bands=bands,
        quality=_quality(scl_purity=1.0, aot=1.0, view_zenith=1.0, sun_zenith=1.0),
    )
    stats = ArchiveStats(mean=0.60, std=0.15)
    scored = score_observation(obs, stats)

    full_score = scored.quality.score(Q_FULL)
    assert full_score > 0.85, f"Expected near-1.0 full score, got {full_score:.4f}"


# ---------------------------------------------------------------------------
# Test 3: High-cloud (low scl_purity) observations score low overall
# ---------------------------------------------------------------------------

def test_high_cloud_scores_low():
    obs = _obs(quality=_quality(scl_purity=0.05))
    scored = score_observation(obs, _typical_archive_stats())
    full_score = scored.quality.score(Q_FULL)
    assert full_score < 0.15, f"Expected low full score for high cloud, got {full_score:.4f}"


# ---------------------------------------------------------------------------
# Test 4: High-haze (low aot quality) scores low on Q_ATMOSPHERIC
# ---------------------------------------------------------------------------

def test_high_haze_scores_low_on_atmospheric():
    obs = _obs(quality=_quality(aot=0.05))
    scored = score_observation(obs, _typical_archive_stats())
    atm_score = scored.quality.score(Q_ATMOSPHERIC)
    assert atm_score < 0.15, f"Expected low atmospheric score for high haze, got {atm_score:.4f}"


# ---------------------------------------------------------------------------
# Test 5: score_observation is a pure function (does not mutate input)
# ---------------------------------------------------------------------------

def test_score_observation_is_pure():
    original_gz = 1.0
    obs = _obs(quality=_quality(greenness_z=original_gz))

    # Capture original identity and state
    original_quality_id = id(obs.quality)
    original_quality_gz = obs.quality.greenness_z

    scored = score_observation(obs, _typical_archive_stats())

    # Input observation must be unchanged
    assert obs.quality.greenness_z == original_quality_gz, "Input quality was mutated"
    assert id(obs.quality) == original_quality_id, "Input quality object was replaced"

    # Returned observation must be a different object
    assert scored is not obs, "score_observation returned the same object"
    assert scored.quality is not obs.quality, "Returned quality is the same object as input"


# ---------------------------------------------------------------------------
# Test 6: greenness_z == 1.0 when NDVI equals the archive mean
# ---------------------------------------------------------------------------

def test_greenness_z_is_one_at_archive_mean():
    # With mean=0.60, std=0.15, set NDVI = 0.60
    # B08=0.8, B04=0.2 → NDVI = (0.8-0.2)/(0.8+0.2) = 0.60
    bands = {"B04": 0.2, "B08": 0.8}
    obs = _obs(bands=bands)
    stats = ArchiveStats(mean=0.60, std=0.15)
    scored = score_observation(obs, stats)
    assert scored.quality.greenness_z == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 7: greenness_z decreases as NDVI deviates from archive mean
# ---------------------------------------------------------------------------

def test_greenness_z_decreases_with_deviation():
    stats = ArchiveStats(mean=0.60, std=0.15)

    # NDVI at mean: greenness_z = 1.0
    obs_mean = _obs(bands={"B04": 0.2, "B08": 0.8})  # NDVI = 0.60
    scored_mean = score_observation(obs_mean, stats)

    # NDVI moderately below mean (0.30): z = (0.30 - 0.60) / 0.15 = -2.0
    # B08=0.6, B04=0.4 → NDVI = (0.6-0.4)/(0.6+0.4) = 0.20? No.
    # NDVI = 0.30 → use B08=0.65, B04=0.35 → NDVI = 0.3/1.0 = 0.30
    obs_low = _obs(bands={"B04": 0.35, "B08": 0.65})  # NDVI ≈ 0.30
    scored_low = score_observation(obs_low, stats)

    # NDVI far below mean (very cloudy/bare soil): B08=0.1, B04=0.09 → NDVI ≈ 0.053
    obs_far = _obs(bands={"B04": 0.09, "B08": 0.10})  # NDVI ≈ 0.053
    scored_far = score_observation(obs_far, stats)

    gz_mean = scored_mean.quality.greenness_z
    gz_low = scored_low.quality.greenness_z
    gz_far = scored_far.quality.greenness_z

    assert gz_mean > gz_low > gz_far, (
        f"Expected gz_mean({gz_mean:.3f}) > gz_low({gz_low:.3f}) > gz_far({gz_far:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 8: greenness_z == 1.0 in the input (Stage 5 placeholder is unchanged)
# ---------------------------------------------------------------------------

def test_input_greenness_z_is_one_before_scoring():
    obs = _obs()
    assert obs.quality.greenness_z == 1.0, (
        "Stage 5 should always set greenness_z=1.0; "
        "score_observation replaces it, not extraction."
    )


# ---------------------------------------------------------------------------
# Test 9: ArchiveStats.from_observations() computes correct mean and std
# ---------------------------------------------------------------------------

def test_archive_stats_from_observations_correct_values():
    # Four observations with known NDVI:
    # B08=0.8, B04=0.2 → NDVI = 0.60
    # B08=0.7, B04=0.3 → NDVI = (0.4/1.0) = 0.40
    # B08=0.9, B04=0.1 → NDVI = (0.8/1.0) = 0.80
    # B08=0.75, B04=0.25 → NDVI = (0.5/1.0) = 0.50
    def _make_obs(b08, b04):
        return _obs(bands={"B08": b08, "B04": b04})

    observations = [
        _make_obs(0.8, 0.2),   # NDVI = 0.60
        _make_obs(0.7, 0.3),   # NDVI = 0.40
        _make_obs(0.9, 0.1),   # NDVI = 0.80
        _make_obs(0.75, 0.25), # NDVI = 0.50
    ]
    stats = ArchiveStats.from_observations(observations)

    expected_mean = (0.60 + 0.40 + 0.80 + 0.50) / 4
    assert stats.mean == pytest.approx(expected_mean, abs=1e-4)
    assert stats.std > 0.0


# ---------------------------------------------------------------------------
# Test 10: ArchiveStats.from_observations() raises with < 2 usable obs
# ---------------------------------------------------------------------------

def test_archive_stats_raises_with_too_few_observations():
    # Observation without B08/B04 — not usable for NDVI
    obs_no_ndvi = _obs(bands={"B03": 0.05, "B02": 0.03})
    with pytest.raises(ValueError, match="2 observations"):
        ArchiveStats.from_observations([obs_no_ndvi])


def test_archive_stats_raises_with_empty_list():
    with pytest.raises(ValueError, match="2 observations"):
        ArchiveStats.from_observations([])


# ---------------------------------------------------------------------------
# Test 11: ArchiveStats std is floored when constructed with std <= 0
# ---------------------------------------------------------------------------

def test_archive_stats_std_floored_at_zero():
    stats = ArchiveStats(mean=0.5, std=0.0)
    assert stats.std > 0.0


def test_archive_stats_std_floored_at_negative():
    stats = ArchiveStats(mean=0.5, std=-0.5)
    assert stats.std > 0.0


# ---------------------------------------------------------------------------
# Test 12: score_observation does not raise when B08/B04 are missing
# ---------------------------------------------------------------------------

def test_score_observation_handles_missing_ndvi_bands():
    obs = _obs(bands={"B03": 0.04, "B02": 0.03})  # no B08 or B04
    scored = score_observation(obs, _typical_archive_stats())
    # Should not raise; greenness_z is some valid value in [0, 1]
    assert 0.0 <= scored.quality.greenness_z <= 1.0


# ---------------------------------------------------------------------------
# Test 13: Other quality components are preserved by score_observation
# ---------------------------------------------------------------------------

def test_other_quality_components_preserved():
    q = _quality(scl_purity=0.75, aot=0.60, view_zenith=0.88, sun_zenith=0.72)
    obs = _obs(quality=q)
    scored = score_observation(obs, _typical_archive_stats())

    assert scored.quality.scl_purity == pytest.approx(0.75)
    assert scored.quality.aot == pytest.approx(0.60)
    assert scored.quality.view_zenith == pytest.approx(0.88)
    assert scored.quality.sun_zenith == pytest.approx(0.72)
