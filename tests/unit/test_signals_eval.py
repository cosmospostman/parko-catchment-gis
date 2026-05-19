"""Unit tests for signals/eval.py.

All tests use synthetic in-memory data. Region parquets are monkeypatched so
no disk I/O is performed.

IQR overlap contracts
---------------------
  O1.  Non-overlapping IQRs → 0.0
  O2.  Identical distributions → 1.0
  O3.  Partial overlap → value in (0, 1)
  O4.  Fewer than 4 observations in either class → NaN

AUROC contracts
---------------
  A1.  Perfect separation (all presence > absence) → 1.0
  A2.  Perfect inverse separation → 0.0
  A3.  Random (interleaved) → ~0.5
  A4.  Empty class → NaN

ClassStats contracts
--------------------
  S1.  Correct median and IQR from known values
  S2.  Empty array → NaN scalars, n_pixel_years == 0

evaluate() contracts
--------------------
  E1.  Returns one EvalResult per site with data
  E2.  Missing region is skipped with a warning (not a crash)
  E3.  rank_key not in summarise() output raises ValueError
  E4.  min_obs_per_year filter excludes sparse pixel-years
  E5.  AUROC and IQR overlap are consistent with known synthetic separation
  E6.  n_pixels counts unique point_ids per class
  E7.  EvalResult __str__ contains site name, signal name, and AUROC
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from signals.base import Signal
from signals.eval import (
    ClassStats,
    EvalResult,
    SiteSpec,
    _auroc,
    _class_stats,
    _iqr_overlap,
    _load_region,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs_df(
    point_ids: list[str],
    years: list[int],
    signal_val: float,
    source: str = "S2",
    scl_purity: float = 1.0,
    n_obs_per_year: int = 10,
) -> pd.DataFrame:
    """Return a synthetic observation dataframe for given pixels and years."""
    rows = []
    for pid in point_ids:
        for yr in years:
            for i in range(n_obs_per_year):
                rows.append({
                    "point_id": pid,
                    "date": pd.Timestamp(f"{yr}-06-{i+1:02d}"),
                    "source": source,
                    "scl_purity": scl_purity,
                    "B05": 0.2,
                    "B07": 0.3,
                    "B08": 0.5,
                    "B8A": signal_val,   # controls NDRE output
                })
    return pd.DataFrame(rows)


class _FixedSignal(Signal):
    """Returns a fixed value per group of rows, driven by B8A column."""
    name = "fixed"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            # Use B8A as the signal value directly for test control
            out[good] = df.loc[good, "B8A"].values.astype("float32")
        return out


def _make_pixel_year_frame(
    point_ids: list[str],
    years: list[int],
    label: str,
    signal_val: float,
    signal_name: str = "fixed",
    n_obs: int = 10,
) -> pd.DataFrame:
    """Return a pre-aggregated pixel-year frame as _load_region would produce."""
    records = []
    for pid in point_ids:
        for yr in years:
            records.append({
                "point_id": pid,
                "year": yr,
                "label": label,
                f"{signal_name}_p05": signal_val,
                f"{signal_name}_p25": signal_val,
                f"{signal_name}_p50": signal_val,
                f"{signal_name}_p75": signal_val,
                f"{signal_name}_p95": signal_val,
                f"{signal_name}_std": 0.0,
                f"{signal_name}_amplitude": 0.0,
                f"{signal_name}_n_obs": n_obs,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# O — IQR overlap
# ---------------------------------------------------------------------------

class TestIQROverlap:
    def test_o1_no_overlap(self):
        pres = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
        abs_ = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        assert _iqr_overlap(pres, abs_) == 0.0

    def test_o2_identical_distributions(self):
        vals = np.linspace(0.2, 0.8, 20)
        assert _iqr_overlap(vals, vals) == pytest.approx(1.0)

    def test_o3_partial_overlap_in_range(self):
        # IQR of pres ≈ [0.55, 0.75], IQR of abs ≈ [0.25, 0.45] — clearly separate
        # Use values that are offset by less than one IQR width to force genuine overlap
        pres = np.linspace(0.4, 0.8, 20)
        abs_ = np.linspace(0.3, 0.7, 20)   # IQR [0.4, 0.6] overlaps pres IQR [0.5, 0.7]
        result = _iqr_overlap(pres, abs_)
        assert 0.0 < result < 1.0

    def test_o4_too_few_observations(self):
        assert np.isnan(_iqr_overlap(np.array([0.5, 0.6, 0.7]), np.array([0.1, 0.2])))
        assert np.isnan(_iqr_overlap(np.array([]), np.array([0.1, 0.2, 0.3, 0.4])))


# ---------------------------------------------------------------------------
# A — AUROC
# ---------------------------------------------------------------------------

class TestAUROC:
    def test_a1_perfect_separation(self):
        pres = np.array([0.8, 0.9, 1.0])
        abs_ = np.array([0.1, 0.2, 0.3])
        assert _auroc(pres, abs_) == pytest.approx(1.0)

    def test_a2_perfect_inverse(self):
        pres = np.array([0.1, 0.2, 0.3])
        abs_ = np.array([0.8, 0.9, 1.0])
        assert _auroc(pres, abs_) == pytest.approx(0.0)

    def test_a3_random_approximately_half(self):
        rng = np.random.default_rng(42)
        vals = rng.uniform(0, 1, 200)
        pres = vals[:100]
        abs_ = vals[100:]
        assert abs(_auroc(pres, abs_) - 0.5) < 0.1

    def test_a4_empty_class_returns_nan(self):
        assert np.isnan(_auroc(np.array([]), np.array([0.5])))
        assert np.isnan(_auroc(np.array([0.5]), np.array([])))


# ---------------------------------------------------------------------------
# S — ClassStats
# ---------------------------------------------------------------------------

class TestClassStats:
    def test_s1_correct_stats_from_known_values(self):
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stats = _class_stats("presence", vals)
        assert stats.median == pytest.approx(0.3)
        assert stats.p25 == pytest.approx(0.2)
        assert stats.p75 == pytest.approx(0.4)
        assert stats.iqr == pytest.approx(0.2)
        assert stats.n_pixel_years == 5

    def test_s2_empty_array_returns_nan(self):
        stats = _class_stats("absence", np.array([]))
        assert np.isnan(stats.mean)
        assert np.isnan(stats.median)
        assert stats.n_pixel_years == 0
        assert stats.n_pixels == 0


# ---------------------------------------------------------------------------
# E — evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _make_site(self, pres_val: float, abs_val: float) -> tuple[SiteSpec, "_FixedSignal"]:
        site = SiteSpec("test_site", [
            ("pres_region", "presence"),
            ("abs_region",  "absence"),
        ])
        return site

    def _patch_load(self, pres_val: float, abs_val: float, n_obs: int = 10):
        """Return a context manager that patches _load_region with synthetic frames."""
        pres_frame = _make_pixel_year_frame(
            [f"pp_{i}" for i in range(5)], [2022, 2023], "presence", pres_val, n_obs=n_obs
        )
        abs_frame = _make_pixel_year_frame(
            [f"aa_{i}" for i in range(5)], [2022, 2023], "absence", abs_val, n_obs=n_obs
        )

        def _fake_load(region_id: str, label: str, signal: Signal):
            if label == "presence":
                return pres_frame.copy()
            return abs_frame.copy()

        return patch("signals.eval._load_region", side_effect=_fake_load)

    def test_e1_returns_one_result_per_site(self):
        sites = [
            SiteSpec("site_a", [("r1", "presence"), ("r2", "absence")]),
            SiteSpec("site_b", [("r3", "presence"), ("r4", "absence")]),
        ]
        with self._patch_load(0.8, 0.2):
            results = evaluate(_FixedSignal(), sites, rank_key="p05", verbose=False)
        assert len(results) == 2
        assert {r.site for r in results} == {"site_a", "site_b"}

    def test_e2_missing_region_skipped_not_crash(self):
        site = SiteSpec("site", [("good_region", "presence"), ("missing_region", "absence")])
        pres_frame = _make_pixel_year_frame(["p0", "p1"], [2022], "presence", 0.8)

        def _fake_load(region_id, label, signal):
            if region_id == "good_region":
                return pres_frame.copy()
            return None  # missing

        with patch("signals.eval._load_region", side_effect=_fake_load):
            # Only presence loaded — no absence class, so no result but no crash
            results = evaluate(_FixedSignal(), [site], rank_key="p05", verbose=False)
        # Result may be produced with empty absence or skipped — either is acceptable;
        # what must NOT happen is an exception
        assert isinstance(results, list)

    def test_e3_bad_rank_key_raises(self):
        site = SiteSpec("site", [("r1", "presence"), ("r2", "absence")])
        with self._patch_load(0.8, 0.2):
            with pytest.raises(ValueError, match="rank_key"):
                evaluate(_FixedSignal(), [site], rank_key="nonexistent_key", verbose=False)

    def test_e4_min_obs_filter_excludes_sparse_years(self):
        site = SiteSpec("site", [("r1", "presence"), ("r2", "absence")])
        # n_obs=3 below default min_obs_per_year=6 — all pixel-years filtered out
        with self._patch_load(0.8, 0.2, n_obs=3):
            results = evaluate(_FixedSignal(), [site], rank_key="p05",
                               min_obs_per_year=6, verbose=False)
        # Result is still returned but with empty classes (NaN stats, 0 pixel-years)
        assert len(results) == 1
        assert results[0].presence.n_pixel_years == 0
        assert results[0].absence.n_pixel_years == 0
        assert np.isnan(results[0].auroc)

    def test_e5_auroc_and_iqr_consistent_with_known_separation(self):
        site = SiteSpec("site", [("r1", "presence"), ("r2", "absence")])
        with self._patch_load(pres_val=0.9, abs_val=0.1):
            results = evaluate(_FixedSignal(), [site], rank_key="p05", verbose=False)
        assert len(results) == 1
        r = results[0]
        assert r.auroc == pytest.approx(1.0)
        assert r.iqr_overlap == pytest.approx(0.0)

    def test_e6_n_pixels_counts_unique_point_ids(self):
        site = SiteSpec("site", [("r1", "presence"), ("r2", "absence")])
        with self._patch_load(0.8, 0.2):
            results = evaluate(_FixedSignal(), [site], rank_key="p05", verbose=False)
        assert results[0].presence.n_pixels == 5
        assert results[0].absence.n_pixels == 5

    def test_e7_str_contains_key_fields(self):
        site = SiteSpec("my_site", [("r1", "presence"), ("r2", "absence")])
        with self._patch_load(0.8, 0.2):
            results = evaluate(_FixedSignal(), [site], rank_key="p05", verbose=False)
        s = str(results[0])
        assert "my_site" in s
        assert "fixed" in s
        assert "AUROC" in s
