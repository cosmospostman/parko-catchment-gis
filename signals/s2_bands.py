"""signals/s2_bands.py — Raw S2 band pass-through signals.

Each Signal subclass returns the raw reflectance value for one band, with the
shared quality mask applied (S2 rows, scl_purity >= 0.5). No arithmetic beyond
the ratio signals at the bottom of this file.

Raw band signals
----------------
    B02Signal   Blue        ~490 nm
    B03Signal   Green       ~560 nm
    B04Signal   Red         ~665 nm
    B05Signal   Red-edge 1  ~705 nm
    B07Signal   Red-edge 3  ~783 nm
    B08Signal   NIR broad   ~842 nm
    B8ASignal   NIR narrow  ~865 nm
    B11Signal   SWIR 1      ~1610 nm
    B12Signal   SWIR 2      ~2190 nm

Ratio signals (structural decoupling, hypothesised in ndsvi.py)
--------------------------------------------------------------
    B12B11Signal   B12 / B11  — dry cellulose/lignin vs. liquid water
    B11B08Signal   B11 / B08  — SWIR moisture loading vs. NIR canopy structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Signal


def _band_signal(band: str) -> type[Signal]:
    """Factory: return a Signal subclass that passes through *band* unchanged."""

    class _BandSignal(Signal):
        name = band.lower()

        def compute(self, df: pd.DataFrame) -> pd.Series:
            good = self.quality_mask(df)
            out = pd.Series(np.nan, index=df.index, dtype="float32")
            if good.any():
                out[good] = df.loc[good, band].values.astype("float32")
            return out

    _BandSignal.__name__ = f"{band}Signal"
    _BandSignal.__qualname__ = f"{band}Signal"
    return _BandSignal


B02Signal = _band_signal("B02")
B03Signal = _band_signal("B03")
B04Signal = _band_signal("B04")
B05Signal = _band_signal("B05")
B07Signal = _band_signal("B07")
B08Signal = _band_signal("B08")
B8ASignal = _band_signal("B8A")
B11Signal = _band_signal("B11")
B12Signal = _band_signal("B12")


class B12B11Signal(Signal):
    """SWIR structural decoupling ratio: B12 / B11.

    Dry cellulose/lignin (B12) vs. liquid water (B11). Separates structurally
    woody pixels from bare soil and senescing grass even when NDVI is similar.
    Duplicated here from ndsvi.py so raw-band and ratio signals are co-located.
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


class B11B08Signal(Signal):
    """SWIR/NIR moisture-structure ratio: B11 / B08.

    B11 (~1610 nm) loads on liquid water content; B08 (~842 nm) tracks canopy
    structure. The ratio captures moisture loading relative to canopy density,
    expected to diverge between Parkinsonia's photosynthetic stems (low moisture,
    high NIR) and surrounding grass (high moisture, moderate NIR).
    """

    name = "b11_b08"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any():
            b11 = df.loc[good, "B11"].values.astype("float32")
            b08 = df.loc[good, "B08"].values.astype("float32")
            result = np.full(b08.shape, np.nan, dtype="float32")
            valid = b08 != 0
            result[valid] = b11[valid] / b08[valid]
            out[good] = result
        return out
