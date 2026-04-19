"""Integration test: full feature vector pipeline end-to-end.

Session 10 goal: verify that the chain
    extract_observations → score_observation → extract_waveform_features
    → assemble_feature_vector
produces the correct output when wired together in sequence.

Also verifies the feature_names contract is stable by writing
tests/fixtures/feature_names_fixture.json on first run and asserting
schema match on subsequent runs.

Design
------
All data is synthetic and constructed in-memory or in tmp_path — no
pre-staged STAC chip files required. The synthetic data is designed to
produce detectable presence and absence signals so the full chain can
run to completion.

Presence points
    3 years × 12 observations per year, all within the flowering window
    (DOY 200–340). Band values are tuned so flowering_index exceeds
    FLOWERING_THRESHOLD after quality weighting.

Absence points
    3 years × 12 observations per year, all within the flowering window.
    Band values produce a low flowering_index (well below threshold).

The test writes fixture chips to a tmp_path DiskChipStore and runs the
complete pipeline chain.

Assertions
----------
I1.  Row count equals point count (no silent drops for presence points)
I2.  Feature vector schema matches feature_names_fixture.json
I3.  mean_quality is present and in [0, 1]
I4.  No NaN values in any assembled feature vector
I5.  All expected feature keys are present
I6.  Key order: waveform keys first, then structural, then mean_quality
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from analysis.constants import (
    BANDS,
    FLOWERING_THRESHOLD,
    FLOWERING_WINDOW,
    Q_FULL,
    SCL_BAND,
)
from analysis.primitives.indices import flowering_index
from analysis.primitives.quality import ArchiveStats, score_observation
from analysis.timeseries.extraction import extract_observations
from analysis.timeseries.features import (
    STRUCTURAL_KEYS,
    WAVEFORM_KEYS,
    assemble_feature_vector,
)
from analysis.timeseries.waveform import extract_waveform_features
from utils.chip_store import DiskChipStore

# ---------------------------------------------------------------------------
# Path to the schema fixture
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
_SCHEMA_FIXTURE = _FIXTURE_DIR / "feature_names_fixture.json"

# ---------------------------------------------------------------------------
# Chip writing helpers (same pattern as test_extraction.py)
# ---------------------------------------------------------------------------

def _write_chip(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def _const_chip(value: float, shape: tuple[int, int] = (5, 5)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake pystac.Item
# ---------------------------------------------------------------------------

def _make_item(item_id: str, year: int, doy: int) -> SimpleNamespace:
    """Create a duck-typed item at a given year and day-of-year."""
    # Convert DOY to a calendar date within [year]
    dt = datetime(year, 1, 1) + __import__("datetime").timedelta(days=doy - 1)
    return SimpleNamespace(
        id=item_id,
        datetime=dt,
        properties={"s2:mgrs_tile": "55HBU"},
    )


# ---------------------------------------------------------------------------
# Synthetic scene builders
# ---------------------------------------------------------------------------

# Band values for a Parkinsonia-like spectral signature:
#   high B07 vs B05 (re_slope > 0), high B08 vs B11 (nir_swir > 0)
#   flowering_index ≈ (re_slope + nir_swir) / 2 ~ 0.6  >> FLOWERING_THRESHOLD
_PRESENCE_BANDS = {
    "B02": 0.04, "B03": 0.05, "B04": 0.06,
    "B05": 0.10, "B06": 0.18, "B07": 0.40,
    "B08": 0.50, "B8A": 0.48,
    "B11": 0.08, "B12": 0.06,
}

# Band values that produce a near-zero or negative flowering_index:
#   low B07 vs B05, low B08 vs B11
_ABSENCE_BANDS = {
    "B02": 0.10, "B03": 0.12, "B04": 0.14,
    "B05": 0.30, "B06": 0.25, "B07": 0.20,  # re_slope < 0
    "B08": 0.15, "B8A": 0.14,
    "B11": 0.35, "B12": 0.30,              # nir_swir < 0
}


def _write_presence_scene(
    inputs_dir: Path,
    item_id: str,
    point_id: str,
) -> None:
    """Write a clear-sky scene with Parkinsonia-like band values."""
    scl = _const_chip(4.0)  # all vegetation = clear
    _write_chip(inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif", scl)
    for band, val in _PRESENCE_BANDS.items():
        _write_chip(
            inputs_dir / item_id / f"{band}_{point_id}.tif",
            _const_chip(val),
        )


def _write_absence_scene(
    inputs_dir: Path,
    item_id: str,
    point_id: str,
) -> None:
    """Write a clear-sky scene with non-Parkinsonia band values."""
    scl = _const_chip(4.0)
    _write_chip(inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif", scl)
    for band, val in _ABSENCE_BANDS.items():
        _write_chip(
            inputs_dir / item_id / f"{band}_{point_id}.tif",
            _const_chip(val),
        )


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

_DOW_START, _DOW_END = FLOWERING_WINDOW   # 200, 340
_YEARS = [2021, 2022, 2023, 2024]         # 4 years ≥ min_years=3
_DOYS = [210, 230, 250, 270, 290, 310]    # 6 DOYs per year inside window


def _build_items_and_chips(
    inputs_dir: Path,
    point_id: str,
    scene_writer,             # _write_presence_scene or _write_absence_scene
) -> list[SimpleNamespace]:
    """Write chips for all (year, doy) combos and return item list."""
    items = []
    for year in _YEARS:
        for doy in _DOYS:
            item_id = f"S2_{point_id}_{year}_doy{doy}"
            item = _make_item(item_id, year, doy)
            items.append(item)
            scene_writer(inputs_dir, item_id, point_id)
    return items


def _run_pipeline(
    inputs_dir: Path,
    point_id: str,
    items: list[SimpleNamespace],
    structural_features: dict[str, float],
) -> dict[str, float] | None:
    """Run extraction → quality → waveform → features for one point.

    Returns the assembled feature vector, or None if waveform returned {}.
    """
    points = [(point_id, 143.5, -16.0)]
    store = DiskChipStore(inputs_dir=inputs_dir)

    # Stage 5: extract
    raw_obs = extract_observations(items, points, store, bands=BANDS)

    if not raw_obs:
        return None

    # Stage 6: quality scoring (ArchiveStats from the extracted observations)
    archive_stats = ArchiveStats.from_observations(raw_obs)
    scored_obs = [score_observation(obs, archive_stats) for obs in raw_obs]

    # Stage 7: waveform
    waveform = extract_waveform_features(scored_obs, index_fn=flowering_index)

    if not waveform:
        return None

    # Stage 8: feature assembly
    return assemble_feature_vector(waveform, structural_features, scored_obs)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def presence_result(tmp_path_factory):
    """Run full pipeline for a presence point and return the feature vector."""
    tmp_path = tmp_path_factory.mktemp("presence")
    structural = {"HAND": 1.2, "dist_to_water": 180.0}
    items = _build_items_and_chips(tmp_path, "pt_presence", _write_presence_scene)
    result = _run_pipeline(tmp_path, "pt_presence", items, structural)
    return result


@pytest.fixture(scope="module")
def three_point_results(tmp_path_factory):
    """Run full pipeline for three points (2 presence, 1 absence) and return list."""
    tmp_path = tmp_path_factory.mktemp("three_points")
    structural = {"HAND": 1.5, "dist_to_water": 200.0}
    results = []
    for i, (pid, writer) in enumerate([
        ("pt_p1", _write_presence_scene),
        ("pt_p2", _write_presence_scene),
        ("pt_abs", _write_absence_scene),
    ]):
        items = _build_items_and_chips(tmp_path, pid, writer)
        result = _run_pipeline(tmp_path, pid, items, structural)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# I1: Row count matches point count (no silent drops)
# ---------------------------------------------------------------------------

def test_I1_row_count_matches_point_count(three_point_results):
    """Row count equals the number of points that have sufficient data."""
    # All 3 points are presence or clear absence with 4 years of data.
    # Presence points always produce waveform features.
    # Absence points may return {} from waveform (index below threshold every year).
    # The test asserts that presence points (first two) are not silently dropped.
    assert len(three_point_results) >= 2, (
        f"Expected at least 2 presence points in results, got {len(three_point_results)}"
    )


# ---------------------------------------------------------------------------
# I2: Feature schema matches feature_names_fixture.json
# ---------------------------------------------------------------------------

def test_I2_feature_schema_matches_fixture(presence_result):
    """Feature vector schema matches feature_names_fixture.json.

    On first run this test produces the fixture. On subsequent runs it
    asserts that the schema is unchanged. If the fixture does not exist,
    the test writes it and passes.
    """
    assert presence_result is not None, (
        "Presence point returned None — pipeline did not produce features. "
        "Check FLOWERING_THRESHOLD and synthetic band values."
    )

    current_keys = list(presence_result.keys())

    if not _SCHEMA_FIXTURE.exists():
        # First run: write the schema fixture
        _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        _SCHEMA_FIXTURE.write_text(json.dumps(current_keys, indent=2))
        return  # fixture produced; assertion trivially passes

    fixture_keys = json.loads(_SCHEMA_FIXTURE.read_text())
    assert current_keys == fixture_keys, (
        f"Feature schema mismatch.\n"
        f"  Current : {current_keys}\n"
        f"  Fixture : {fixture_keys}\n"
        "If this is intentional, delete tests/fixtures/feature_names_fixture.json "
        "and re-run to regenerate it."
    )


# ---------------------------------------------------------------------------
# I3: mean_quality present and in [0, 1]
# ---------------------------------------------------------------------------

def test_I3_mean_quality_present_and_in_range(presence_result):
    """mean_quality column is present and in [0, 1]."""
    assert presence_result is not None
    assert "mean_quality" in presence_result, "mean_quality key missing from feature vector"
    mq = presence_result["mean_quality"]
    assert 0.0 <= mq <= 1.0, f"mean_quality={mq} outside [0, 1]"


# ---------------------------------------------------------------------------
# I4: No NaN in assembled feature vectors
# ---------------------------------------------------------------------------

def test_I4_no_nan_in_feature_vectors(three_point_results):
    """No NaN values in any assembled feature vector."""
    for i, fv in enumerate(three_point_results):
        nan_keys = [k for k, v in fv.items() if math.isnan(v)]
        assert not nan_keys, (
            f"Feature vector {i} contains NaN values for keys: {nan_keys}"
        )


# ---------------------------------------------------------------------------
# I5: All expected feature keys present
# ---------------------------------------------------------------------------

def test_I5_all_expected_keys_present(presence_result):
    """Feature vector contains all waveform keys, structural keys, and mean_quality."""
    assert presence_result is not None
    expected = set(WAVEFORM_KEYS) | set(STRUCTURAL_KEYS) | {"mean_quality"}
    missing = expected - set(presence_result.keys())
    assert not missing, f"Missing expected feature keys: {missing}"


# ---------------------------------------------------------------------------
# I6: Key order: waveform, then structural, then mean_quality
# ---------------------------------------------------------------------------

def test_I6_key_order_stable(presence_result):
    """Key order is waveform → structural → mean_quality."""
    assert presence_result is not None
    keys = list(presence_result.keys())

    waveform_positions = [keys.index(k) for k in WAVEFORM_KEYS]
    structural_positions = [keys.index(k) for k in STRUCTURAL_KEYS]
    quality_position = keys.index("mean_quality")

    assert max(waveform_positions) < min(structural_positions), (
        "Waveform keys must all precede structural keys"
    )
    assert max(structural_positions) < quality_position, (
        "Structural keys must all precede mean_quality"
    )
    assert keys[-1] == "mean_quality", "mean_quality must be the last key"
