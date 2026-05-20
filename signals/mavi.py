"""signals/mavi.py — Moisture-Adjusted Vegetation Index (MAVI).

Hypothesis
----------
Sparse Parkinsonia draws down soil moisture in a radial zone beyond its canopy
via deep roots, producing a SWIR "drawdown halo". MAVI makes this explicit by
placing B11 (SWIR-1, soil and canopy water sensitive) in the denominator.

At sparse sites the discrimination window is Apr–May (wet-to-dry transition),
where within-bbox distributions are bimodal: a low-MAVI grass/soil cluster and
a high-MAVI canopy cluster. The harness evaluates p05 as the dry-season floor.

Formulation
-----------
    MAVI = (B08 - B04) / (B08 + B04 + B11)

Bands required: B04, B08, B11 — all present in training tile parquets.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from signals.base import Signal


class MAVISignal(Signal):
    """Moisture-Adjusted Vegetation Index: (B08 - B04) / (B08 + B04 + B11)."""

    name = "mavi"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b08 = df["B08"].to_numpy().astype("float32")
            b04 = df["B04"].to_numpy().astype("float32")
            b11 = df["B11"].to_numpy().astype("float32")
            denom = b08 + b04 + b11
            valid = good & (denom != 0)
            out[valid] = (b08[valid] - b04[valid]) / denom[valid]
        return pl.Series(out)
