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
import pandas as pd

from signals.base import Signal


class MAVISignal(Signal):
    """Moisture-Adjusted Vegetation Index: (B08 - B04) / (B08 + B04 + B11)."""

    name = "mavi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b08 = df.loc[good, "B08"].values.astype("float32")
            b04 = df.loc[good, "B04"].values.astype("float32")
            b11 = df.loc[good, "B11"].values.astype("float32")
            denom = b08 + b04 + b11
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom != 0
            result[valid] = (b08[valid] - b04[valid]) / denom[valid]
            out[good] = result
        return out
