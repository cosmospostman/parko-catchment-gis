"""Unit tests for the signals/ package.

All tests use synthetic in-memory DataFrames. No parquet files are read.

Signal base class contracts
---------------------------
  B1.  quality_mask passes S2 rows with scl_purity >= 0.5
  B2.  quality_mask rejects rows where source != "S2"
  B3.  quality_mask rejects rows where scl_purity < 0.5
  B4.  quality_mask is permissive when source/scl_purity columns are absent
  B5.  quality_mask accepts a custom scl_purity_min threshold
  B6.  Default summarise returns all required keys
  B7.  Default summarise returns NaN scalars when all observations are NaN
  B8.  Default summarise n_obs counts only non-NaN values
  B9.  Default summarise amplitude == p95 - p05
  B10. Default summarise uses df_slice but does not require it to have extra cols

NDRESignal contracts
--------------------
  N1.  compute returns NaN for S1 rows
  N2.  compute returns NaN for low-scl_purity rows
  N3.  compute returns correct NDRE value: (B8A - B05) / (B8A + B05)
  N4.  compute returns NaN when denominator is zero (B8A + B05 == 0)
  N5.  compute output dtype is float32
  N6.  compute length matches input df

CIRESignal contracts
--------------------
  C1.  compute returns correct CI_RE value: (B07 / B05) - 1
  C2.  compute returns NaN when B05 == 0
  C3.  compute returns NaN for quality-failed rows
  C4.  compute output dtype is float32
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from signals.base import Signal
from signals.ndre import CIRESignal, NDRESignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 10,
    source: str = "S2",
    scl_purity: float = 1.0,
    B05: float = 0.2,
    B07: float = 0.3,
    B08: float = 0.5,
    B8A: float = 0.4,
) -> pl.DataFrame:
    """Return a minimal observation dataframe with uniform values."""
    dates = pl.date_range(
        pl.date(2023, 1, 1), pl.date(2023, 1, 1).dt.offset_by(f"{(n-1)*5}d"),
        interval="5d", eager=True,
    )
    return pl.DataFrame({
        "point_id":   [f"px_{i:04d}" for i in range(n)],
        "date":       dates,
        "source":     [source] * n,
        "scl_purity": [scl_purity] * n,
        "B05":        [float(B05)] * n,
        "B07":        [float(B07)] * n,
        "B08":        [float(B08)] * n,
        "B8A":        [float(B8A)] * n,
    })


def _mixed_df() -> pl.DataFrame:
    """Return a dataframe with a mix of S2/S1 rows and varying scl_purity."""
    return pl.DataFrame({
        "point_id":   ["p0", "p0", "p0", "p0"],
        "date":       [pl.date(2023, 6, 1), pl.date(2023, 6, 6),
                       pl.date(2023, 6, 11), pl.date(2023, 6, 16)],
        "source":     ["S2", "S1", "S2", "S2"],
        "scl_purity": [1.0, None, 0.3, 0.8],
        "B05":        [0.2, None, 0.2, 0.1],
        "B07":        [0.3, None, 0.3, 0.4],
        "B08":        [0.5, None, 0.5, 0.6],
        "B8A":        [0.4, None, 0.4, 0.5],
    })


# ---------------------------------------------------------------------------
# Concrete minimal Signal subclass for base class tests
# ---------------------------------------------------------------------------

class _ConstantSignal(Signal):
    """Returns a fixed value for every quality-passing row."""
    name = "constant"

    def __init__(self, value: float = 1.0):
        self._value = value

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        out[good] = self._value
        return pl.Series(out)


# ---------------------------------------------------------------------------
# B — Signal base class / quality_mask
# ---------------------------------------------------------------------------

class TestQualityMask:
    def test_b1_passes_good_s2_rows(self):
        df = _make_df(source="S2", scl_purity=1.0)
        mask = Signal.quality_mask(df)
        assert mask.to_numpy().all()

    def test_b2_rejects_s1_rows(self):
        df = _make_df(source="S1", scl_purity=1.0)
        mask = Signal.quality_mask(df)
        assert not mask.to_numpy().any()

    def test_b3_rejects_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.3)
        mask = Signal.quality_mask(df)
        assert not mask.to_numpy().any()

    def test_b4_permissive_without_columns(self):
        df = pl.DataFrame({"B05": [0.2, 0.3], "B8A": [0.4, 0.5]})
        mask = Signal.quality_mask(df)
        assert mask.to_numpy().all()

    def test_b5_custom_threshold(self):
        df = _make_df(source="S2", scl_purity=0.6)
        assert Signal.quality_mask(df, scl_purity_min=0.5).to_numpy().all()
        assert not Signal.quality_mask(df, scl_purity_min=0.7).to_numpy().any()


class TestDefaultSummarise:
    def setup_method(self):
        self.sig = _ConstantSignal(value=0.5)
        self.df = _make_df(n=20)

    def test_b6_returns_required_keys(self):
        ts = self.sig.compute(self.df)
        result = self.sig.summarise(ts, self.df)
        for key in ("p05", "p25", "p50", "p75", "p95", "std", "amplitude", "n_obs"):
            assert key in result, f"missing key: {key}"

    def test_b7_all_nan_input_returns_nan_scalars(self):
        ts = pl.Series(np.full(len(self.df), np.nan, dtype="float32"))
        result = self.sig.summarise(ts, self.df)
        for key in ("p05", "p25", "p50", "p75", "p95", "std", "amplitude"):
            assert np.isnan(result[key]), f"{key} should be NaN"
        assert result["n_obs"] == 0

    def test_b8_n_obs_counts_non_nan(self):
        ts_arr = self.sig.compute(self.df).to_numpy().copy()
        ts_arr[:5] = np.nan
        ts = pl.Series(ts_arr)
        result = self.sig.summarise(ts, self.df)
        assert result["n_obs"] == 15

    def test_b9_amplitude_equals_p95_minus_p05(self):
        df = _make_df(n=20)
        ts = pl.Series(np.linspace(0.1, 0.9, 20).astype("float32"))
        result = self.sig.summarise(ts, df)
        assert abs(result["amplitude"] - (result["p95"] - result["p05"])) < 1e-6

    def test_b10_summarise_does_not_require_extra_df_cols(self):
        df_minimal = pl.DataFrame({"point_id": ["p0"] * 10})
        ts = pl.Series([0.5] * 10)
        result = self.sig.summarise(ts, df_minimal)
        assert result["n_obs"] == 10


# ---------------------------------------------------------------------------
# N — NDRESignal
# ---------------------------------------------------------------------------

class TestNDRESignal:
    def setup_method(self):
        self.sig = NDRESignal()

    def test_n1_nan_for_s1_rows(self):
        df = _make_df(source="S1")
        ts = self.sig.compute(df)
        assert np.isnan(ts.to_numpy()).all()

    def test_n2_nan_for_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.1)
        ts = self.sig.compute(df)
        assert np.isnan(ts.to_numpy()).all()

    def test_n3_correct_ndre_value(self):
        b8a, b05 = 0.4, 0.2
        expected = (b8a - b05) / (b8a + b05)
        df = _make_df(B8A=b8a, B05=b05)
        ts = self.sig.compute(df)
        valid = ts.to_numpy()
        valid = valid[~np.isnan(valid)]
        assert np.allclose(valid, expected, atol=1e-5)

    def test_n4_nan_when_denominator_zero(self):
        df = _make_df(B8A=0.0, B05=0.0)
        ts = self.sig.compute(df)
        assert np.isnan(ts.to_numpy()).all()

    def test_n5_output_dtype_float32(self):
        df = _make_df()
        ts = self.sig.compute(df)
        assert ts.dtype == pl.Float32

    def test_n6_length_matches_input(self):
        df = _make_df(n=15)
        ts = self.sig.compute(df)
        assert len(ts) == len(df)

    def test_n_mixed_rows_only_good_s2_computed(self):
        df = _mixed_df()
        ts = self.sig.compute(df).to_numpy()
        # row 0: good S2 → value
        assert not np.isnan(ts[0])
        # row 1: S1 → NaN
        assert np.isnan(ts[1])
        # row 2: low scl_purity → NaN
        assert np.isnan(ts[2])
        # row 3: good S2 → value
        assert not np.isnan(ts[3])


# ---------------------------------------------------------------------------
# C — CIRESignal
# ---------------------------------------------------------------------------

class TestCIRESignal:
    def setup_method(self):
        self.sig = CIRESignal()

    def test_c1_correct_ci_re_value(self):
        b07, b05 = 0.3, 0.2
        expected = b07 / b05 - 1.0
        df = _make_df(B07=b07, B05=b05)
        ts = self.sig.compute(df)
        valid = ts.to_numpy()
        valid = valid[~np.isnan(valid)]
        assert np.allclose(valid, expected, atol=1e-5)

    def test_c2_nan_when_b05_zero(self):
        df = _make_df(B05=0.0, B07=0.3)
        ts = self.sig.compute(df)
        assert np.isnan(ts.to_numpy()).all()

    def test_c3_nan_for_quality_failed_rows(self):
        df = _make_df(source="S2", scl_purity=0.2)
        ts = self.sig.compute(df)
        assert np.isnan(ts.to_numpy()).all()

    def test_c4_output_dtype_float32(self):
        df = _make_df()
        ts = self.sig.compute(df)
        assert ts.dtype == pl.Float32
