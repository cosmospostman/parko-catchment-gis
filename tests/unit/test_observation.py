"""Unit tests for ObservationQuality and Observation dataclasses.

Tests
-----
1. score(mask=None) returns the product of all five components.
2. score(Q_ATMOSPHERIC) excludes view_zenith, sun_zenith, greenness_z.
3. score() with any component = 0 returns 0.
4. Quality components clamp correctly to [0, 1].
"""

from datetime import datetime

import pytest

from analysis.constants import Q_ATMOSPHERIC
from analysis.timeseries.observation import Observation, ObservationQuality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality(**overrides) -> ObservationQuality:
    """Return an ObservationQuality with sensible defaults, overridable per-test."""
    defaults = dict(
        scl_purity=0.9,
        aot=0.8,
        view_zenith=0.7,
        sun_zenith=0.6,
        greenness_z=0.5,
    )
    defaults.update(overrides)
    return ObservationQuality(**defaults)


def _observation(quality: ObservationQuality | None = None) -> Observation:
    if quality is None:
        quality = _quality()
    return Observation(
        point_id="pt_001",
        date=datetime(2022, 8, 15),
        bands={"B03": 0.043, "B04": 0.021, "B08": 0.182},
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Test 1: score(mask=None) returns product of all five components
# ---------------------------------------------------------------------------

def test_score_no_mask_is_product_of_all_five():
    q = _quality(scl_purity=0.9, aot=0.8, view_zenith=0.7, sun_zenith=0.6, greenness_z=0.5)
    expected = 0.9 * 0.8 * 0.7 * 0.6 * 0.5
    assert q.score() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 2: score(Q_ATMOSPHERIC) excludes view_zenith, sun_zenith, greenness_z
# Q_ATMOSPHERIC = {"scl_purity", "aot"}
# ---------------------------------------------------------------------------

def test_score_q_atmospheric_excludes_geometric_and_greenness():
    q = _quality(scl_purity=0.9, aot=0.8, view_zenith=0.1, sun_zenith=0.1, greenness_z=0.1)
    # Only scl_purity and aot should contribute; the 0.1 components must be ignored
    expected = 0.9 * 0.8
    assert q.score(mask=Q_ATMOSPHERIC) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3: score() with any single component = 0 returns 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zero_component", [
    "scl_purity",
    "aot",
    "view_zenith",
    "sun_zenith",
    "greenness_z",
])
def test_score_zero_component_returns_zero(zero_component):
    q = _quality(**{zero_component: 0.0})
    assert q.score() == 0.0


# ---------------------------------------------------------------------------
# Test 4: Quality components clamp to [0, 1]
# ---------------------------------------------------------------------------

def test_quality_components_clamp_above_one():
    q = ObservationQuality(
        scl_purity=1.5,
        aot=2.0,
        view_zenith=99.0,
        sun_zenith=1.1,
        greenness_z=3.7,
    )
    assert q.scl_purity == 1.0
    assert q.aot == 1.0
    assert q.view_zenith == 1.0
    assert q.sun_zenith == 1.0
    assert q.greenness_z == 1.0


def test_quality_components_clamp_below_zero():
    q = ObservationQuality(
        scl_purity=-0.5,
        aot=-1.0,
        view_zenith=-99.0,
        sun_zenith=-0.1,
        greenness_z=-3.0,
    )
    assert q.scl_purity == 0.0
    assert q.aot == 0.0
    assert q.view_zenith == 0.0
    assert q.sun_zenith == 0.0
    assert q.greenness_z == 0.0


# ---------------------------------------------------------------------------
# Bonus: Observation dataclass smoke test (instantiation and field access)
# ---------------------------------------------------------------------------

def test_observation_fields_accessible():
    obs = _observation()
    assert obs.point_id == "pt_001"
    assert obs.date == datetime(2022, 8, 15)
    assert obs.bands["B03"] == pytest.approx(0.043)
    assert isinstance(obs.quality, ObservationQuality)
    assert obs.meta == {}
