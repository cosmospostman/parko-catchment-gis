"""signals/temporal.py — Temporal variance signal (σ_t).

Hypothesis
----------
Annual grasses respond to rainfall pulses with large, rapid NDVI spikes then
rapid decay. Parkinsonia, buffered by deep roots, exhibits a flatter, more
stable response curve. Per-pixel-year NDVI standard deviation captures this
volatility directly without requiring rainfall calendar data.

Formulation
-----------
    sigma_t = std(NDVI) across qualifying S2 observations in the year

The base summarise() already returns std; this signal just passes through the
pre-computed NDVI column. Evaluate using rank_key="std".

Note: lower sigma_t = more stable = more consistent with Parkinsonia. The
AUROC convention (higher score = presence) means direction may flip — check
whether presence AUROC < 0.5 and consider 1 - AUROC as the discriminability
score if so.

Bands required: NDVI column — pre-computed in all training tile parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Signal


class TemporalVarianceSignal(Signal):
    """Per-pixel-year NDVI standard deviation as a stability signal.

    Passes the pre-computed NDVI column through the quality mask unchanged.
    The discriminative summary is std (lower = more stable = more likely
    Parkinsonia). Use rank_key="std" with evaluate().
    """

    name = "ndvi_sigma_t"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            out[good] = df.loc[good, "NDVI"].values.astype("float32")
        return out
