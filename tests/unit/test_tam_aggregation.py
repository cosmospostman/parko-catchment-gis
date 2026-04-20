"""Unit tests for TAM year-probability aggregation."""
import math
import numpy as np
import pytest
from tam.core.score import aggregate_year_probs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(pixel_year_probs: dict) -> tuple[list, list, list]:
    """Convert {pid: {year: [probs]}} → (all_pids, all_years, all_probs) lists."""
    all_pids, all_years, all_probs = [], [], []
    for pid, year_map in pixel_year_probs.items():
        for year, probs in year_map.items():
            all_pids.append(np.array([pid] * len(probs)))
            all_years.append(np.array([year] * len(probs), dtype=np.int32))
            all_probs.append(np.array(probs, dtype=np.float32))
    return all_pids, all_years, all_probs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_year_returns_mean_prob():
    pids, years, probs = _make_inputs({"px1": {2023: [0.8, 0.6]}})
    df = aggregate_year_probs(pids, years, probs, end_year=2023)
    assert len(df) == 1
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(0.7)


def test_end_year_gets_full_weight():
    """When only one year exists and it equals end_year, weight=exp(0)=1."""
    pids, years, probs = _make_inputs({"px1": {2024: [0.9]}})
    df = aggregate_year_probs(pids, years, probs, end_year=2024, decay=0.7)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(0.9)


def test_older_year_discounted():
    """Recent signal outweighs older signal under exponential decay."""
    decay = 0.7
    end_year = 2024
    pids, years, probs = _make_inputs({
        "recent": {2024: [0.9], 2021: [0.1]},
        "old":    {2024: [0.1], 2021: [0.9]},
    })
    df = aggregate_year_probs(pids, years, probs, end_year=end_year, decay=decay)
    recent_prob = df.loc[df["point_id"] == "recent", "prob_tam"].iloc[0]
    old_prob    = df.loc[df["point_id"] == "old",    "prob_tam"].iloc[0]
    assert recent_prob > old_prob


def test_weighted_average_correctness():
    """Manually verify the weighted mean formula for two years."""
    decay = 0.7
    end_year = 2024
    pids, years, probs = _make_inputs({"px1": {2022: [0.4], 2024: [1.0]}})
    w_2022 = math.exp(-decay * (2024 - 2022))
    w_2024 = math.exp(-decay * 0)
    expected = (w_2022 * 0.4 + w_2024 * 1.0) / (w_2022 + w_2024)
    df = aggregate_year_probs(pids, years, probs, end_year=end_year, decay=decay)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_multiple_pixels_independent():
    pids, years, probs = _make_inputs({"a": {2023: [0.2]}, "b": {2023: [0.9]}})
    df = aggregate_year_probs(pids, years, probs, end_year=2023)
    a = df.loc[df["point_id"] == "a", "prob_tam"].iloc[0]
    b = df.loc[df["point_id"] == "b", "prob_tam"].iloc[0]
    assert b > a
    assert a == pytest.approx(0.2)
    assert b == pytest.approx(0.9)


def test_empty_input_returns_empty_dataframe():
    df = aggregate_year_probs([], [], [], end_year=2024)
    assert df.empty
    assert "point_id" in df.columns
    assert "prob_tam" in df.columns


def test_future_year_does_not_crash():
    """Years after end_year get exp(positive) weight — verify no crash."""
    pids, years, probs = _make_inputs({"px1": {2025: [0.5], 2024: [0.5]}})
    df = aggregate_year_probs(pids, years, probs, end_year=2024, decay=0.7)
    assert not df.empty
    assert 0.0 <= df["prob_tam"].iloc[0] <= 1.0


def test_decay_zero_equals_uniform_mean():
    """With decay=0 all years are equally weighted — result equals simple mean."""
    pids, years, probs = _make_inputs({"px1": {2020: [0.2], 2022: [0.6], 2024: [1.0]}})
    df = aggregate_year_probs(pids, years, probs, end_year=2024, decay=0.0)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(
        (0.2 + 0.6 + 1.0) / 3, rel=1e-5
    )
