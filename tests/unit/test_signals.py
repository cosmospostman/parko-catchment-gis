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
  N6.  compute index matches input df index

CIRESignal contracts
--------------------
  C1.  compute returns correct CI_RE value: (B07 / B05) - 1
  C2.  compute returns NaN when B05 == 0
  C3.  compute returns NaN for quality-failed rows
  C4.  compute output dtype is float32
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
) -> pd.DataFrame:
    """Return a minimal observation dataframe with uniform values."""
    return pd.DataFrame({
        "point_id":   [f"px_{i:04d}" for i in range(n)],
        "date":       pd.date_range("2023-01-01", periods=n, freq="5D"),
        "source":     source,
        "scl_purity": scl_purity,
        "B05":        float(B05),
        "B07":        float(B07),
        "B08":        float(B08),
        "B8A":        float(B8A),
    })


def _mixed_df() -> pd.DataFrame:
    """Return a dataframe with a mix of S2/S1 rows and varying scl_purity."""
    rows = [
        dict(point_id="p0", date=pd.Timestamp("2023-06-01"), source="S2",  scl_purity=1.0,  B05=0.2, B07=0.3, B08=0.5, B8A=0.4),
        dict(point_id="p0", date=pd.Timestamp("2023-06-06"), source="S1",  scl_purity=np.nan, B05=np.nan, B07=np.nan, B08=np.nan, B8A=np.nan),
        dict(point_id="p0", date=pd.Timestamp("2023-06-11"), source="S2",  scl_purity=0.3,  B05=0.2, B07=0.3, B08=0.5, B8A=0.4),
        dict(point_id="p0", date=pd.Timestamp("2023-06-16"), source="S2",  scl_purity=0.8,  B05=0.1, B07=0.4, B08=0.6, B8A=0.5),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Concrete minimal Signal subclass for base class tests
# ---------------------------------------------------------------------------

class _ConstantSignal(Signal):
    """Returns a fixed value for every quality-passing row."""
    name = "constant"

    def __init__(self, value: float = 1.0):
        self._value = value

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        out[good] = self._value
        return out


# ---------------------------------------------------------------------------
# B — Signal base class / quality_mask
# ---------------------------------------------------------------------------

class TestQualityMask:
    def test_b1_passes_good_s2_rows(self):
        df = _make_df(source="S2", scl_purity=1.0)
        mask = Signal.quality_mask(df)
        assert mask.all()

    def test_b2_rejects_s1_rows(self):
        df = _make_df(source="S1", scl_purity=1.0)
        mask = Signal.quality_mask(df)
        assert not mask.any()

    def test_b3_rejects_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.3)
        mask = Signal.quality_mask(df)
        assert not mask.any()

    def test_b4_permissive_without_columns(self):
        df = pd.DataFrame({"B05": [0.2, 0.3], "B8A": [0.4, 0.5]})
        mask = Signal.quality_mask(df)
        assert mask.all()

    def test_b5_custom_threshold(self):
        df = _make_df(source="S2", scl_purity=0.6)
        assert Signal.quality_mask(df, scl_purity_min=0.5).all()
        assert not Signal.quality_mask(df, scl_purity_min=0.7).any()


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
        ts = pd.Series(np.nan, index=self.df.index, dtype="float32")
        result = self.sig.summarise(ts, self.df)
        for key in ("p05", "p25", "p50", "p75", "p95", "std", "amplitude"):
            assert np.isnan(result[key]), f"{key} should be NaN"
        assert result["n_obs"] == 0

    def test_b8_n_obs_counts_non_nan(self):
        ts = self.sig.compute(self.df)
        # Manually null half
        ts.iloc[:5] = np.nan
        result = self.sig.summarise(ts, self.df)
        assert result["n_obs"] == 15

    def test_b9_amplitude_equals_p95_minus_p05(self):
        # Use varying values so percentiles differ
        df = _make_df(n=20)
        ts = pd.Series(np.linspace(0.1, 0.9, 20), index=df.index, dtype="float32")
        result = self.sig.summarise(ts, df)
        assert abs(result["amplitude"] - (result["p95"] - result["p05"])) < 1e-6

    def test_b10_summarise_does_not_require_extra_df_cols(self):
        df_minimal = pd.DataFrame({"point_id": ["p0"] * 10})
        ts = pd.Series([0.5] * 10, dtype="float32")
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
        assert ts.isna().all()

    def test_n2_nan_for_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.1)
        ts = self.sig.compute(df)
        assert ts.isna().all()

    def test_n3_correct_ndre_value(self):
        b8a, b05 = 0.4, 0.2
        expected = (b8a - b05) / (b8a + b05)
        df = _make_df(B8A=b8a, B05=b05)
        ts = self.sig.compute(df)
        assert np.allclose(ts.dropna(), expected, atol=1e-5)

    def test_n4_nan_when_denominator_zero(self):
        df = _make_df(B8A=0.0, B05=0.0)
        ts = self.sig.compute(df)
        assert ts.isna().all()

    def test_n5_output_dtype_float32(self):
        df = _make_df()
        ts = self.sig.compute(df)
        assert ts.dtype == np.float32

    def test_n6_index_matches_input(self):
        df = _make_df(n=15)
        df.index = range(100, 115)
        ts = self.sig.compute(df)
        assert list(ts.index) == list(df.index)

    def test_n_mixed_rows_only_good_s2_computed(self):
        df = _mixed_df()
        ts = self.sig.compute(df)
        # row 0: good S2 → value
        assert not np.isnan(ts.iloc[0])
        # row 1: S1 → NaN
        assert np.isnan(ts.iloc[1])
        # row 2: low scl_purity → NaN
        assert np.isnan(ts.iloc[2])
        # row 3: good S2 → value
        assert not np.isnan(ts.iloc[3])


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
        assert np.allclose(ts.dropna(), expected, atol=1e-5)

    def test_c2_nan_when_b05_zero(self):
        df = _make_df(B05=0.0, B07=0.3)
        ts = self.sig.compute(df)
        assert ts.isna().all()

    def test_c3_nan_for_quality_failed_rows(self):
        df = _make_df(source="S2", scl_purity=0.2)
        ts = self.sig.compute(df)
        assert ts.isna().all()

    def test_c4_output_dtype_float32(self):
        df = _make_df()
        ts = self.sig.compute(df)
        assert ts.dtype == np.float32
