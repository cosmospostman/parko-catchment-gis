"""Train/score numerical-parity registry.

Several statistics are computed once at training time (usually in Polars, inside
`tam/core/train.py` / `tam/_prep_worker.py`) and again at inference time (usually in a
hand-written numba kernel, inside `tam/core/score.py` / `tam/core/_preprocess_numba.py`),
because `score`'s streaming/chunked architecture can't reuse the training-side
implementation directly. Any numerical divergence between the two — even a tiny one from
a differing library default (quantile interpolation, std ddof, float32 vs float64
accumulation, NaN handling, sort order) — silently shifts every pixel's features at
inference relative to what the model saw in training, producing confident, uniform
mispredictions that look like "the model is just bad" rather than a plumbing bug.

Three such bugs were found in the "annual feature" path during the Mitchell
false-positive investigation (see `docs/MITCHELL-DEBUG.md`); the third (a
`quantile`/`std` method mismatch) was caught immediately by a test of exactly this
shape, on the first attempt at writing one.

To prevent recurrence for *future* shared statistics: register a case below whenever you
add a feature that is computed independently on both the train side and the score side.
Each case provides a `train_fn` and `score_fn` that both consume the *same* synthetic
raw-observation array and must produce numerically identical output
(`np.testing.assert_allclose`). The parametrized test below then covers every
registered case automatically — no new test function needed per feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
import pytest


@dataclass(frozen=True)
class ParityCase:
    name: str
    # Builds the shared synthetic input. Returns whatever `train_fn`/`score_fn` need —
    # typically a dict so both sides can pull out the views/columns they require.
    make_input: Callable[[], dict]
    # Each fn takes the dict from make_input() and returns an (W, K) float array of
    # per-window statistics, in the same column order, comparable via assert_allclose.
    train_fn: Callable[[dict], np.ndarray]
    score_fn: Callable[[dict], np.ndarray]
    rtol: float = 1e-5
    atol: float = 1e-6


# ---------------------------------------------------------------------------
# Shared synthetic input: one pixel-year window's worth of S2 observations,
# wide enough to exercise NaNs, percentile interpolation, and variance.
# ---------------------------------------------------------------------------

def _make_band_summary_input() -> dict:
    rng = np.random.default_rng(7)
    n = 47
    feat = rng.uniform(0.05, 0.4, size=(n, 3)).astype(np.float32)
    # Sprinkle some NaNs — both sides must drop them identically.
    feat[rng.integers(0, n, size=4), rng.integers(0, 3, size=4)] = np.nan
    cols = ["B08", "B04", "NDVI"]
    return {
        "feat": feat,
        "cols": cols,
        "df": pl.DataFrame({c: feat[:, i] for i, c in enumerate(cols)}),
    }


def _train_band_summaries(data: dict) -> np.ndarray:
    """Mirrors `_compute_band_summaries`'s aggregation expressions directly —
    NaN -> null, then quantile(interpolation="linear") and std(ddof=0) — without the
    group-by/lazy-plan machinery, since this fixture is already a single window's
    worth of rows."""
    df, cols = data["df"], data["cols"]
    df = df.with_columns([pl.col(c).fill_nan(None) for c in cols])
    row = df.select(
        [pl.col(c).quantile(0.05, interpolation="linear").alias(f"{c}_p5")  for c in cols] +
        [pl.col(c).quantile(0.95, interpolation="linear").alias(f"{c}_p95") for c in cols] +
        [pl.col(c).std(ddof=0).alias(f"{c}_std")                             for c in cols]
    )
    out = np.empty((1, len(cols) * 3), dtype=np.float64)
    for i, c in enumerate(cols):
        out[0, i * 3 + 0] = row[f"{c}_p5"][0]
        out[0, i * 3 + 1] = row[f"{c}_p95"][0]
        out[0, i * 3 + 2] = row[f"{c}_std"][0]
    return out


def _score_band_summaries(data: dict) -> np.ndarray:
    from tam.core._preprocess_numba import compute_band_summaries

    feat = data["feat"]
    boundaries = np.array([0], dtype=np.int64)
    ends = np.array([len(feat)], dtype=np.int64)
    out = np.empty((1, feat.shape[1] * 3), dtype=np.float32)
    compute_band_summaries(feat, boundaries, ends, out)
    return out.astype(np.float64)


# ---------------------------------------------------------------------------
# Registry — add one entry here per feature computed independently on both sides.
# ---------------------------------------------------------------------------

PARITY_CASES: list[ParityCase] = [
    ParityCase(
        name="band_summaries_p5_p95_std",
        make_input=_make_band_summary_input,
        train_fn=_train_band_summaries,
        score_fn=_score_band_summaries,
    ),
]


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda c: c.name)
def test_train_score_numerical_parity(case: ParityCase):
    """Each registered (train_fn, score_fn) pair must produce numerically identical
    output for the same synthetic raw observations — not just matching shape/schema.

    A failure here means the train-time and score-time implementations of `case.name`
    disagree on how to compute the same statistic (e.g. differing library defaults for
    quantile interpolation, std ddof, NaN handling) — exactly the class of bug that
    causes silent, uniform train/inference distribution shifts. See the module
    docstring and `docs/MITCHELL-DEBUG.md` for the three real-world bugs this guards
    against.
    """
    data = case.make_input()
    expected = case.train_fn(data)
    actual = case.score_fn(data)
    np.testing.assert_allclose(
        actual, expected, rtol=case.rtol, atol=case.atol,
        err_msg=f"train/score parity mismatch for '{case.name}' — the train-side and "
                f"score-side implementations compute this statistic differently. "
                f"Check for differing library defaults (quantile interpolation, std "
                f"ddof, float precision, NaN handling).")
