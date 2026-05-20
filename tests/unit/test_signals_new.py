"""Unit tests for signals/mavi.py, signals/ndsvi.py, signals/temporal.py.

All tests use synthetic in-memory DataFrames. No parquet files are read.

MAVISignal contracts
--------------------
  M1.  compute returns correct value: (B08 - B04) / (B08 + B04 + B11)
  M2.  compute returns NaN when denominator is zero (B08 + B04 + B11 == 0)
  M3.  compute returns NaN for S1 rows
  M4.  compute returns NaN for low-scl_purity rows
  M5.  compute output dtype is float32
  M6.  compute length matches input df

NDSVISignal contracts
---------------------
  D1.  compute returns correct value: (B11 - B04) / (B11 + B04)
  D2.  compute returns NaN when denominator is zero
  D3.  compute returns NaN for quality-failed rows
  D4.  compute output dtype is float32

B12B11Signal contracts
----------------------
  W1.  compute returns correct value: B12 / B11
  W2.  compute returns NaN when B11 == 0
  W3.  compute returns NaN for quality-failed rows
  W4.  compute output dtype is float32

TemporalVarianceSignal contracts
---------------------------------
  T1.  compute passes NDVI values through for quality-passing rows
  T2.  compute returns NaN for S1 rows
  T3.  compute returns NaN for low-scl_purity rows
  T4.  compute output dtype is float32
  T5.  summarise std is the discriminative key and is non-zero for varying NDVI
  T6.  summarise std is zero for constant NDVI series
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from signals.mavi import MAVISignal
from signals.ndsvi import B12B11Signal, NDSVISignal
from signals.temporal import TemporalVarianceSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 10,
    source: str = "S2",
    scl_purity: float = 1.0,
    B04: float = 0.1,
    B08: float = 0.5,
    B11: float = 0.2,
    B12: float = 0.15,
    NDVI: float = 0.6,
) -> pl.DataFrame:
    dates = pl.date_range(
        pl.date(2023, 1, 1), pl.date(2023, 1, 1).dt.offset_by(f"{(n-1)*5}d"),
        interval="5d", eager=True,
    )
    return pl.DataFrame({
        "point_id":   [f"px_{i:04d}" for i in range(n)],
        "date":       dates,
        "source":     [source] * n,
        "scl_purity": [scl_purity] * n,
        "B04":        [float(B04)] * n,
        "B08":        [float(B08)] * n,
        "B11":        [float(B11)] * n,
        "B12":        [float(B12)] * n,
        "NDVI":       [float(NDVI)] * n,
    })


def _valid(ts: pl.Series) -> np.ndarray:
    arr = ts.to_numpy()
    return arr[~np.isnan(arr)]


def _all_nan(ts: pl.Series) -> bool:
    return np.isnan(ts.to_numpy()).all()


# ---------------------------------------------------------------------------
# M — MAVISignal
# ---------------------------------------------------------------------------

class TestMAVISignal:
    def setup_method(self):
        self.sig = MAVISignal()

    def test_m1_correct_value(self):
        b08, b04, b11 = 0.5, 0.1, 0.2
        expected = (b08 - b04) / (b08 + b04 + b11)
        df = _make_df(B08=b08, B04=b04, B11=b11)
        ts = self.sig.compute(df)
        assert np.allclose(_valid(ts), expected, atol=1e-5)

    def test_m2_nan_when_denominator_zero(self):
        df = _make_df(B08=0.0, B04=0.0, B11=0.0)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_m3_nan_for_s1_rows(self):
        df = _make_df(source="S1")
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_m4_nan_for_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.1)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_m5_output_dtype_float32(self):
        ts = self.sig.compute(_make_df())
        assert ts.dtype == pl.Float32

    def test_m6_length_matches_input(self):
        df = _make_df(n=12)
        ts = self.sig.compute(df)
        assert len(ts) == len(df)


# ---------------------------------------------------------------------------
# D — NDSVISignal
# ---------------------------------------------------------------------------

class TestNDSVISignal:
    def setup_method(self):
        self.sig = NDSVISignal()

    def test_d1_correct_value(self):
        b11, b04 = 0.3, 0.1
        expected = (b11 - b04) / (b11 + b04)
        df = _make_df(B11=b11, B04=b04)
        ts = self.sig.compute(df)
        assert np.allclose(_valid(ts), expected, atol=1e-5)

    def test_d2_nan_when_denominator_zero(self):
        df = _make_df(B11=0.0, B04=0.0)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_d3_nan_for_quality_failed(self):
        df = _make_df(source="S2", scl_purity=0.2)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_d4_output_dtype_float32(self):
        ts = self.sig.compute(_make_df())
        assert ts.dtype == pl.Float32


# ---------------------------------------------------------------------------
# W — B12B11Signal
# ---------------------------------------------------------------------------

class TestB12B11Signal:
    def setup_method(self):
        self.sig = B12B11Signal()

    def test_w1_correct_value(self):
        b12, b11 = 0.15, 0.2
        expected = b12 / b11
        df = _make_df(B12=b12, B11=b11)
        ts = self.sig.compute(df)
        assert np.allclose(_valid(ts), expected, atol=1e-5)

    def test_w2_nan_when_b11_zero(self):
        df = _make_df(B11=0.0, B12=0.1)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_w3_nan_for_quality_failed(self):
        df = _make_df(source="S1", scl_purity=1.0)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_w4_output_dtype_float32(self):
        ts = self.sig.compute(_make_df())
        assert ts.dtype == pl.Float32


# ---------------------------------------------------------------------------
# T — TemporalVarianceSignal
# ---------------------------------------------------------------------------

class TestTemporalVarianceSignal:
    def setup_method(self):
        self.sig = TemporalVarianceSignal()

    def test_t1_passes_ndvi_through(self):
        ndvi_val = 0.72
        df = _make_df(NDVI=ndvi_val)
        ts = self.sig.compute(df)
        assert np.allclose(_valid(ts), ndvi_val, atol=1e-5)

    def test_t2_nan_for_s1_rows(self):
        df = _make_df(source="S1")
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_t3_nan_for_low_scl_purity(self):
        df = _make_df(source="S2", scl_purity=0.3)
        ts = self.sig.compute(df)
        assert _all_nan(ts)

    def test_t4_output_dtype_float32(self):
        ts = self.sig.compute(_make_df())
        assert ts.dtype == pl.Float32

    def test_t5_std_nonzero_for_varying_ndvi(self):
        df = _make_df(n=20).with_columns(
            pl.Series("NDVI", np.linspace(0.1, 0.9, 20).astype("float32"))
        )
        ts = self.sig.compute(df)
        stats = self.sig.summarise(ts, df)
        assert stats["std"] > 0.0

    def test_t6_std_zero_for_constant_ndvi(self):
        df = _make_df(n=20, NDVI=0.5)
        ts = self.sig.compute(df)
        stats = self.sig.summarise(ts, df)
        assert stats["std"] == pytest.approx(0.0, abs=1e-5)
