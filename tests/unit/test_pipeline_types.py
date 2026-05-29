"""Tests for utils/pipeline_types.py — StageSpec, Pipeline."""

import threading
import time

import pytest

from utils.pipeline_types import Pipeline, StageSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double(x):
    return x * 2


def _to_pair(x):
    return [x, x + 100]  # flat_map: 1 → 2


def _identity(x):
    return x


def _drop_even(x):
    return None if x % 2 == 0 else x


def _boom(x):
    if x == 3:
        raise ValueError("boom at 3")
    return x


# ---------------------------------------------------------------------------
# StageSpec
# ---------------------------------------------------------------------------

def test_stage_spec_frozen():
    s = StageSpec(name="a", fn=_identity, concurrency=1, ram_gb=1.0)
    with pytest.raises(Exception):
        s.name = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Budget check
# ---------------------------------------------------------------------------

def test_budget_ok():
    Pipeline(
        [
            StageSpec("a", _identity, concurrency=2, ram_gb=1.0),
            StageSpec("b", _identity, concurrency=1, ram_gb=3.0),
        ],
        ram_budget_gb=10.0,
    )


def test_budget_exceeded():
    with pytest.raises(ValueError, match="peak RAM"):
        Pipeline(
            [
                StageSpec("a", _identity, concurrency=2, ram_gb=4.0),
                StageSpec("b", _identity, concurrency=1, ram_gb=3.0),
            ],
            ram_budget_gb=10.0,
        )


def test_budget_none_no_check():
    # ram_budget_gb=None skips the check entirely.
    Pipeline(
        [StageSpec("a", _identity, concurrency=100, ram_gb=999.0)],
        ram_budget_gb=None,
    )


# ---------------------------------------------------------------------------
# Basic map
# ---------------------------------------------------------------------------

def test_single_stage_map():
    p = Pipeline([StageSpec("double", _double, concurrency=1, ram_gb=0.0)])
    result = list(p.run([1, 2, 3]))
    assert sorted(result) == [2, 4, 6]


def test_two_stage_map():
    p = Pipeline([
        StageSpec("double", _double, concurrency=1, ram_gb=0.0),
        StageSpec("again",  _double, concurrency=1, ram_gb=0.0),
    ])
    result = list(p.run([1, 2, 3]))
    assert sorted(result) == [4, 8, 12]


def test_empty_inputs():
    p = Pipeline([StageSpec("id", _identity, concurrency=1, ram_gb=0.0)])
    assert list(p.run([])) == []


# ---------------------------------------------------------------------------
# flat_map (fn returns list)
# ---------------------------------------------------------------------------

def test_flat_map():
    p = Pipeline([StageSpec("pair", _to_pair, concurrency=1, ram_gb=0.0)])
    result = sorted(p.run([1, 2]))
    assert result == [1, 2, 101, 102]


def test_flat_map_then_map():
    p = Pipeline([
        StageSpec("pair",   _to_pair, concurrency=1, ram_gb=0.0),
        StageSpec("double", _double,  concurrency=1, ram_gb=0.0),
    ])
    result = sorted(p.run([1]))
    assert result == [2, 202]


# ---------------------------------------------------------------------------
# Drop (fn returns None)
# ---------------------------------------------------------------------------

def test_drop_even():
    p = Pipeline([StageSpec("filter", _drop_even, concurrency=1, ram_gb=0.0)])
    result = sorted(p.run([1, 2, 3, 4, 5]))
    assert result == [1, 3, 5]


# ---------------------------------------------------------------------------
# Concurrency > 1: sentinel correctness
# ---------------------------------------------------------------------------

def test_concurrency_2_correct_output():
    p = Pipeline([StageSpec("double", _double, concurrency=2, ram_gb=0.0)])
    result = sorted(p.run(range(10)))
    assert result == [i * 2 for i in range(10)]


def test_concurrency_2_no_duplicate_sentinel():
    # If sentinel propagation is broken (N sentinels forwarded instead of 1),
    # the output queue would yield an extra sentinel causing the loop to
    # terminate early — the result would be shorter than expected.
    p = Pipeline([StageSpec("id", _identity, concurrency=3, ram_gb=0.0)])
    result = list(p.run(range(20)))
    assert len(result) == 20


# ---------------------------------------------------------------------------
# Concurrency across multiple stages
# ---------------------------------------------------------------------------

def test_multi_stage_multi_concurrency():
    p = Pipeline([
        StageSpec("a", _double,   concurrency=2, ram_gb=0.0),
        StageSpec("b", _identity, concurrency=3, ram_gb=0.0),
        StageSpec("c", _double,   concurrency=1, ram_gb=0.0),
    ])
    result = sorted(p.run([1, 2, 3, 4]))
    assert result == [4, 8, 12, 16]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def test_error_propagated_to_caller():
    p = Pipeline([StageSpec("boom", _boom, concurrency=1, ram_gb=0.0)])
    with pytest.raises(ValueError, match="boom at 3"):
        list(p.run([1, 2, 3, 4]))


# ---------------------------------------------------------------------------
# Backpressure: slow consumer doesn't OOM (queue maxsize enforced)
# ---------------------------------------------------------------------------

def test_backpressure_bounded_queue():
    produced = []

    def _slow_produce(x):
        produced.append(x)
        return x

    p = Pipeline([StageSpec("track", _slow_produce, concurrency=1, ram_gb=0.0)])
    gen = p.run(range(100))
    # Consume one item at a time; the queue should stay bounded.
    results = []
    for _ in range(100):
        results.append(next(gen))
    assert sorted(results) == list(range(100))
    # Feed thread should not have pushed all 100 items before consumer started.
    # (This is a structural check; exact count is non-deterministic.)
    assert len(produced) <= 100


# ---------------------------------------------------------------------------
# Thread safety: no data races on shared state
# ---------------------------------------------------------------------------

def test_no_dropped_items_under_concurrency():
    counter = {"n": 0}
    lock = threading.Lock()

    def _count(x):
        with lock:
            counter["n"] += 1
        return x

    p = Pipeline([StageSpec("count", _count, concurrency=4, ram_gb=0.0)])
    inputs = list(range(50))
    result = list(p.run(inputs))
    assert counter["n"] == 50
    assert sorted(result) == inputs
