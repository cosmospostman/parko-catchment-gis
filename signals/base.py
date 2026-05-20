"""signals/base.py — Signal base class and shared quality filtering.

A Signal transforms raw per-observation band data into a derived time series
and per-pixel-year summary statistics. It has no opinion about z-scoring,
class comparison, or visualisation — those are the harness's responsibility.

Interface
---------
    signal.compute(df)               -> Series  (per-observation, null where quality fails)
    signal.summarise(ts, df_slice)   -> dict    (per pixel-year scalars)

Quality filtering
-----------------
Call ``Signal.quality_mask(df)`` to get a boolean Series that is True for
usable S2 observations. Signals must apply this in ``compute()`` and return
null for rows that fail. The shared implementation filters on:
    - source == "S2"         (exclude S1-only rows)
    - scl_purity >= threshold (default 0.5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl


_DEFAULT_SCL_PURITY_MIN: float = 0.5


class Signal(ABC):
    """Abstract base for all derived spectral/temporal signals.

    Subclasses must implement ``compute``. They may override ``summarise``
    when the default percentile set is insufficient (e.g. signals that need
    DOY-windowed aggregation).
    """

    #: Short unique identifier used as a column name and in output filenames.
    name: str

    # ------------------------------------------------------------------
    # Quality filtering (shared, call from compute())
    # ------------------------------------------------------------------

    @staticmethod
    def quality_mask(df: pl.DataFrame, scl_purity_min: float = _DEFAULT_SCL_PURITY_MIN) -> pl.Series:
        """Return a boolean mask: True for usable S2 observations.

        Filters:
            - ``source == "S2"`` if the column is present
            - ``scl_purity >= scl_purity_min`` if the column is present

        Rows missing either column are treated as usable (permissive default
        for parquets that pre-filtered before writing).
        """
        mask = pl.Series([True] * len(df))
        if "source" in df.columns:
            mask = mask & (df["source"] == "S2")
        if "scl_purity" in df.columns:
            mask = mask & (df["scl_purity"] >= scl_purity_min)
        return mask

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Derive the signal value for every row in *df*.

        Parameters
        ----------
        df:
            Raw observation dataframe. Contains all S2 band columns plus
            ``source``, ``scl_purity``, ``date``, ``point_id``.

        Returns
        -------
        Series with the same length as *df*. Must be null for rows that fail
        ``quality_mask``. Implementations should call::

            good = self.quality_mask(df)
            out = pl.Series(np.full(len(df), np.nan, dtype="float32"))
            out = out.set(pl.arg_where(good), ...)  # compute only on good rows
            return out
        """

    def summarise(self, ts: pl.Series, df_slice: pl.DataFrame) -> dict:
        """Aggregate a per-observation time series into per-pixel-year scalars.

        Parameters
        ----------
        ts:
            Output of ``compute()`` sliced to one (point_id, year) group.
            May contain null/NaN (quality-failed observations).
        df_slice:
            The corresponding rows of the source dataframe, same length as *ts*.
            Available for DOY access, seasonal windowing, or raw band inspection.

        Returns
        -------
        dict with at minimum:
            p05, p25, p50, p75, p95  — percentiles of valid observations
            std                       — standard deviation
            amplitude                 — p95 - p05
            n_obs                     — count of non-null values
        """
        valid = ts.drop_nulls().to_numpy()
        # Also drop NaN that slipped through as float values
        valid = valid[~np.isnan(valid)]
        n = len(valid)
        if n == 0:
            return dict(p05=np.nan, p25=np.nan, p50=np.nan, p75=np.nan,
                        p95=np.nan, std=np.nan, amplitude=np.nan, n_obs=0)
        p05, p25, p50, p75, p95 = np.nanpercentile(valid, [5, 25, 50, 75, 95])
        return dict(
            p05=float(p05),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p95=float(p95),
            std=float(np.std(valid)),
            amplitude=float(p95 - p05),
            n_obs=n,
        )
