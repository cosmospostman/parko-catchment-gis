"""signals/ndre.py — NDRE and CI_RE red-edge chlorophyll signals.

Hypothesis
----------
Parkinsonia's photosynthetic stems retain chlorophyll through the dry season.
When leaves are absent and grasses are fully senescent, the stem chlorophyll
maintains an elevated red-edge signature that collapses in purely herbaceous
pixels. This is distinct from the wet-season chlorophyll flush captured by
re_p10 — the discrimination window here is late dry season (Jul–Sep).

Formulations
------------
    NDRE  = (B8A - B05) / (B8A + B05)
    CI_RE = (B07 / B05) - 1

NDRE is the primary signal (normalised, cross-site comparable). CI_RE is a
ratio form that is more sensitive at high chlorophyll concentrations and is
included as a secondary variant. Both are implemented as separate Signal
subclasses so the harness can evaluate them independently.

Bands required: B05, B07, B8A — all present in training tile parquets.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from signals.base import Signal


class NDRESignal(Signal):
    """Normalised Difference Red-Edge index: (B8A - B05) / (B8A + B05)."""

    name = "ndre"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b8a = df["B8A"].to_numpy().astype("float32")
            b05 = df["B05"].to_numpy().astype("float32")
            denom = b8a + b05
            valid = good & (denom != 0)
            out[valid] = (b8a[valid] - b05[valid]) / denom[valid]
        return pl.Series(out)


class CIRESignal(Signal):
    """Chlorophyll Index Red-Edge: (B07 / B05) - 1.

    Ratio form; more sensitive than NDRE at high chlorophyll concentrations.
    Values near 0 indicate senescent or bare pixels; higher values indicate
    active chlorophyll.
    """

    name = "ci_re"

    def compute(self, df: pl.DataFrame) -> pl.Series:
        good = self.quality_mask(df).to_numpy()
        out = np.full(len(df), np.nan, dtype="float32")
        if good.any():
            b07 = df["B07"].to_numpy().astype("float32")
            b05 = df["B05"].to_numpy().astype("float32")
            valid = good & (b05 != 0)
            out[valid] = b07[valid] / b05[valid] - 1.0
        return pl.Series(out)
