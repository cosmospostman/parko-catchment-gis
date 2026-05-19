"""signals/ — Derived spectral and temporal signals for Parkinsonia discrimination.

Each Signal transforms raw per-observation band data into a derived time series
and per-pixel-year summary statistics. Signals are consumed by:

  - The evaluation harness (signals/eval.py) for discriminability assessment
  - The TAM dataset loader, which appends compute() output as additional band
    columns in the observation sequence
  - Simple classifiers that use summarise() scalars as tabular features

Usage
-----
    from signals.ndre import NDRESignal

    sig = NDRESignal()
    ts = sig.compute(df)               # per-observation Series, NaN where quality fails
    stats = sig.summarise(ts, df)      # dict: p05 p25 p50 p75 p95 std amplitude n_obs
"""

from signals.base import Signal
from signals.ndvi import NDVISignal, NDWISignal, EVISignal

__all__ = ["Signal", "NDVISignal", "NDWISignal", "EVISignal"]
