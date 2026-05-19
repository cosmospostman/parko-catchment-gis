"""signals/ndsvi.py — NDSVI and B12/B11 structural decoupling signals.

Hypothesis
----------
NDSVI substitutes B04 (red, chlorophyll-absorbing) for B08 in the denominator,
making it sensitive to the chlorophyll-absorption side of the green-to-senescent
transition. Parkinsonia's photosynthetic stems suppress the red reflectance rise
during senescence, causing NDSVI to diverge from senescing grass pixels.

B12/B11 decoupling: B11 (~1610 nm) tracks liquid water; B12 (~2200 nm) tracks
dry cellulose/lignin. In a woody skeleton, B12 stays elevated by structural
carbon while B11 varies with moisture. The B12/B11 ratio (or its temporal
variance) separates structurally woody pixels from bare soil even when NDVI
is similar.

Formulations
------------
    NDSVI   = (B11 - B04) / (B11 + B04)
    B12_B11 = B12 / B11

Bands required: B04, B11, B12 — all present in training tile parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Signal


class NDSVISignal(Signal):
    """Normalised Difference Senescent Vegetation Index: (B11 - B04) / (B11 + B04)."""

    name = "ndsvi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b11 = df.loc[good, "B11"].values.astype("float32")
            b04 = df.loc[good, "B04"].values.astype("float32")
            denom = b11 + b04
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom != 0
            result[valid] = (b11[valid] - b04[valid]) / denom[valid]
            out[good] = result
        return out


class B12B11Signal(Signal):
    """SWIR structural decoupling ratio: B12 / B11.

    Elevated values indicate dry cellulose/lignin-dominated pixels (woody
    skeleton). Temporal variance of this ratio separates structurally woody
    pixels from bare soil and senescing grass even when NDVI is similar.
    Evaluate using rank_key="std" to target the decoupling effect.
    """

    name = "b12_b11"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b12 = df.loc[good, "B12"].values.astype("float32")
            b11 = df.loc[good, "B11"].values.astype("float32")
            result = np.full(b11.shape, np.nan, dtype="float32")
            valid = b11 != 0
            result[valid] = b12[valid] / b11[valid]
            out[good] = result
        return out
