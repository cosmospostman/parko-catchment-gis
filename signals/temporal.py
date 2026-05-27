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

Bands required: B08, B04 — NDVI is computed inline from bands.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from signals.base import Signal


class TemporalVarianceSignal(Signal):
    """Per-pixel-year NDVI standard deviation as a stability signal.

    Computes NDVI inline from B08/B04, then passes it through the quality mask.
    The discriminative summary is std (lower = more stable = more likely
    Parkinsonia). Use rank_key="std" with evaluate().
    """

    name = "ndvi_sigma_t"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b08 = df["B08"].to_numpy().astype("float32")
            b04 = df["B04"].to_numpy().astype("float32")
            denom = b08 + b04
            with np.errstate(invalid="ignore"):
                ndvi = np.where(denom != 0, (b08 - b04) / denom, np.nan)
            out[good] = ndvi[good]
        return pl.Series(out)
