"""signals/s1.py — S1 SAR Signal implementations.

Four signals matching the S1_FEATURE_COLS used in the TAM pipeline:

    VHSignal    — mean dry-season VH in dB
    VVSignal    — mean dry-season VV in dB
    VHVVSignal  — VH_db − VV_db (cross-pol ratio)
    RVISignal   — 4·VH_lin / (VH_lin + VV_lin)

All four:
  - filter to source == "S1" (no SCL purity filter — S1 has no cloud mask)
  - override summarise() to window on dry-season DOY (121–304) for the
    primary dry_mean/dry_std/dry_n_obs keys, plus full-year percentiles
  - use dry_mean as the default rank key
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Signal
from tam.core.constants import DRY_DOY_MIN, DRY_DOY_MAX


def _get_doy(df_slice: pd.DataFrame) -> pd.Series:
    if "doy" in df_slice.columns:
        return df_slice["doy"]
    return pd.to_datetime(df_slice["date"]).dt.day_of_year


def _dry_window_summarise(ts: pd.Series, df_slice: pd.DataFrame) -> dict:
    """Shared summarise for S1 signals: dry-season primary keys + full-year percentiles."""
    doy = _get_doy(df_slice)
    dry_mask = doy.between(DRY_DOY_MIN, DRY_DOY_MAX)

    dry_valid = ts[dry_mask].dropna()
    dry_n = len(dry_valid)
    if dry_n > 0:
        dry_mean = float(dry_valid.mean())
        dry_std = float(dry_valid.std()) if dry_n > 1 else 0.0
    else:
        dry_mean = np.nan
        dry_std = np.nan

    full_valid = ts.dropna()
    n = len(full_valid)
    if n == 0:
        return dict(dry_mean=np.nan, dry_std=np.nan, dry_n_obs=0,
                    p05=np.nan, p25=np.nan, p50=np.nan, p75=np.nan,
                    p95=np.nan, n_obs=0)

    p05, p25, p50, p75, p95 = np.nanpercentile(full_valid, [5, 25, 50, 75, 95])
    return dict(
        dry_mean=dry_mean,
        dry_std=dry_std,
        dry_n_obs=dry_n,
        p05=float(p05),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
        n_obs=n,
    )


class VHSignal(Signal):
    """VH backscatter in dB: 10·log10(vh_linear). Dry-season windowed summarise."""

    name = "vh_db"

    @staticmethod
    def quality_mask(df: pd.DataFrame, **_) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if "source" in df.columns:
            mask &= df["source"].eq("S1")
        return mask

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any() and "vh" in df.columns:
            vh_lin = df.loc[good, "vh"].values.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                db = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan).astype("float32")
            out[good] = db
        return out

    def summarise(self, ts: pd.Series, df_slice: pd.DataFrame) -> dict:
        return _dry_window_summarise(ts, df_slice)


class VVSignal(Signal):
    """VV backscatter in dB: 10·log10(vv_linear). Dry-season windowed summarise."""

    name = "vv_db"

    @staticmethod
    def quality_mask(df: pd.DataFrame, **_) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if "source" in df.columns:
            mask &= df["source"].eq("S1")
        return mask

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any() and "vv" in df.columns:
            vv_lin = df.loc[good, "vv"].values.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                db = np.where(vv_lin > 0, 10.0 * np.log10(vv_lin), np.nan).astype("float32")
            out[good] = db
        return out

    def summarise(self, ts: pd.Series, df_slice: pd.DataFrame) -> dict:
        return _dry_window_summarise(ts, df_slice)


class VHVVSignal(Signal):
    """VH−VV cross-pol ratio in dB. Both bands converted to dB independently."""

    name = "vh_vv"

    @staticmethod
    def quality_mask(df: pd.DataFrame, **_) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if "source" in df.columns:
            mask &= df["source"].eq("S1")
        return mask

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any() and "vh" in df.columns and "vv" in df.columns:
            vh_lin = df.loc[good, "vh"].values.astype("float32")
            vv_lin = df.loc[good, "vv"].values.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                vh_db = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan).astype("float32")
                vv_db = np.where(vv_lin > 0, 10.0 * np.log10(vv_lin), np.nan).astype("float32")
            out[good] = (vh_db - vv_db).astype("float32")
        return out

    def summarise(self, ts: pd.Series, df_slice: pd.DataFrame) -> dict:
        return _dry_window_summarise(ts, df_slice)


class RVISignal(Signal):
    """Radar Vegetation Index: 4·VH_lin / (VH_lin + VV_lin).

    Returns NaN where the denominator is zero or either band is NaN.
    """

    name = "rvi"

    @staticmethod
    def quality_mask(df: pd.DataFrame, **_) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if "source" in df.columns:
            mask &= df["source"].eq("S1")
        return mask

    def compute(self, df: pd.DataFrame) -> pd.Series:
        good = self.quality_mask(df)
        out = pd.Series(np.nan, index=df.index, dtype="float32")
        if good.any() and "vh" in df.columns and "vv" in df.columns:
            vh_lin = df.loc[good, "vh"].values.astype("float32")
            vv_lin = df.loc[good, "vv"].values.astype("float32")
            denom = vh_lin + vv_lin
            result = np.full(denom.shape, np.nan, dtype="float32")
            valid = denom > 0
            result[valid] = 4.0 * vh_lin[valid] / denom[valid]
            out[good] = result
        return out

    def summarise(self, ts: pd.Series, df_slice: pd.DataFrame) -> dict:
        return _dry_window_summarise(ts, df_slice)
