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
import pandas as pd

from signals.base import Signal


class NDVISignal(Signal):
    """Normalised Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""

    name = "ndvi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b08 = df.loc[good, "B08"].values.astype("float32")
            b04 = df.loc[good, "B04"].values.astype("float32")
            denom = b08 + b04
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom != 0
            result[valid] = (b08[valid] - b04[valid]) / denom[valid]
            out[good] = result
        return out


class NDWISignal(Signal):
    """Normalised Difference Water Index: (B03 - B08) / (B03 + B08)."""

    name = "ndwi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b03 = df.loc[good, "B03"].values.astype("float32")
            b08 = df.loc[good, "B08"].values.astype("float32")
            denom = b03 + b08
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom != 0
            result[valid] = (b03[valid] - b08[valid]) / denom[valid]
            out[good] = result
        return out


class EVISignal(Signal):
    """Enhanced Vegetation Index: 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)."""

    name = "evi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b02 = df.loc[good, "B02"].values.astype("float32")
            b04 = df.loc[good, "B04"].values.astype("float32")
            b08 = df.loc[good, "B08"].values.astype("float32")
            denom = b08 + 6 * b04 - 7.5 * b02 + 1
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom != 0
            result[valid] = 2.5 * (b08[valid] - b04[valid]) / denom[valid]
            out[good] = result
        return out
