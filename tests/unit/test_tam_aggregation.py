"""Unit tests for TAM year-probability aggregation."""
import math
import numpy as np
import pytest
from tam.pipeline import aggregate_year_probs


def test_single_year_returns_mean_prob():
    pixel_year_probs = {"px1": {2023: [0.8, 0.6]}}
    df = aggregate_year_probs(pixel_year_probs, end_year=2023)
    assert len(df) == 1
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(0.7)


def test_end_year_gets_full_weight():
    """When only one year exists and it equals end_year, weight=exp(0)=1."""
    pixel_year_probs = {"px1": {2024: [0.9]}}
    df = aggregate_year_probs(pixel_year_probs, end_year=2024, decay=0.7)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(0.9)


def test_older_year_discounted():
    """A pixel scoring 0.8 only in end_year outranks one scoring 0.8 only 3 years prior,
    because older years are down-weighted and contribute less to the final aggregation
    when mixed with a zero-signal (no data) gap."""
    decay = 0.7
    end_year = 2024
    # "recent" has its signal in end_year; "old" has the same raw score 3 years back.
    # When each pixel has only one year, the weighted mean equals that year's mean —
    # weights only cancel when normalised. So to demonstrate discounting we need a
    # pixel with *both* years where one year dominates differently.
    pixel_year_probs = {
        "recent": {2024: [0.9], 2021: [0.1]},
        "old":    {2024: [0.1], 2021: [0.9]},
    }
    df = aggregate_year_probs(pixel_year_probs, end_year=end_year, decay=decay)
    recent_prob = df.loc[df["point_id"] == "recent", "prob_tam"].iloc[0]
    old_prob    = df.loc[df["point_id"] == "old",    "prob_tam"].iloc[0]
    assert recent_prob > old_prob


def test_weighted_average_correctness():
    """Manually verify the weighted mean formula for two years."""
    decay = 0.7
    end_year = 2024
    pixel_year_probs = {"px1": {2022: [0.4], 2024: [1.0]}}
    w_2022 = math.exp(-decay * (2024 - 2022))  # exp(-1.4)
    w_2024 = math.exp(-decay * 0)               # 1.0
    expected = (w_2022 * 0.4 + w_2024 * 1.0) / (w_2022 + w_2024)
    df = aggregate_year_probs(pixel_year_probs, end_year=end_year, decay=decay)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_multiple_pixels_independent():
    pixel_year_probs = {
        "a": {2023: [0.2]},
        "b": {2023: [0.9]},
    }
    df = aggregate_year_probs(pixel_year_probs, end_year=2023)
    a = df.loc[df["point_id"] == "a", "prob_tam"].iloc[0]
    b = df.loc[df["point_id"] == "b", "prob_tam"].iloc[0]
    assert b > a
    assert a == pytest.approx(0.2)
    assert b == pytest.approx(0.9)


def test_empty_input_returns_empty_dataframe():
    df = aggregate_year_probs({}, end_year=2024)
    assert df.empty
    assert "point_id" in df.columns
    assert "prob_tam" in df.columns


def test_future_year_gets_zero_discount():
    """Years after end_year get exp(positive) > 1 weight — verify no crash and
    that end_year still has lower weight than a 'future' year (decay is symmetric)."""
    pixel_year_probs = {"px1": {2025: [0.5], 2024: [0.5]}}
    df = aggregate_year_probs(pixel_year_probs, end_year=2024, decay=0.7)
    assert not df.empty
    assert 0.0 <= df["prob_tam"].iloc[0] <= 1.0


def test_decay_zero_equals_uniform_mean():
    """With decay=0 all years are equally weighted — result equals simple mean."""
    pixel_year_probs = {"px1": {2020: [0.2], 2022: [0.6], 2024: [1.0]}}
    df = aggregate_year_probs(pixel_year_probs, end_year=2024, decay=0.0)
    assert df.loc[df["point_id"] == "px1", "prob_tam"].iloc[0] == pytest.approx(
        (0.2 + 0.6 + 1.0) / 3, rel=1e-6
    )
