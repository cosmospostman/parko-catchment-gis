"""signals/ndvi.py — NDVI, NDWI, and EVI Signal implementations.

These are the standard vegetation and water indices used as TAM training
features. Implementing them as Signal subclasses makes them available in the
evaluation harness alongside candidate discriminators (MAVI, NDRE, etc.),
so they can be prototyped and compared before being plugged into the pipeline.

Formulations
------------
    NDVI = (B08 - B04) / (B08 + B04)
    NDWI = (B03 - B08) / (B03 + B08)
    EVI  = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)

All three apply quality_mask and return NaN for quality-failed rows and
for rows where the denominator is zero.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from signals.base import Signal


class NDVISignal(Signal):
    """Normalised Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""

    name = "ndvi"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b08 = df["B08"].to_numpy().astype("float32")
            b04 = df["B04"].to_numpy().astype("float32")
            denom = b08 + b04
            valid = good & (denom != 0)
            out[valid] = (b08[valid] - b04[valid]) / denom[valid]
        return pl.Series(out)


class NDWISignal(Signal):
    """Normalised Difference Water Index: (B03 - B08) / (B03 + B08)."""

    name = "ndwi"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b03 = df["B03"].to_numpy().astype("float32")
            b08 = df["B08"].to_numpy().astype("float32")
            denom = b03 + b08
            valid = good & (denom != 0)
            out[valid] = (b03[valid] - b08[valid]) / denom[valid]
        return pl.Series(out)


class EVISignal(Signal):
    """Enhanced Vegetation Index: 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)."""

    name = "evi"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b02 = df["B02"].to_numpy().astype("float32")
            b04 = df["B04"].to_numpy().astype("float32")
            b08 = df["B08"].to_numpy().astype("float32")
            denom = b08 + 6 * b04 - 7.5 * b02 + 1
            valid = good & (denom != 0)
            out[valid] = 2.5 * (b08[valid] - b04[valid]) / denom[valid]
        return pl.Series(out)
