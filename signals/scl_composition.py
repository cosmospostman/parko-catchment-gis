"""signals/scl_composition.py — Monthly SCL class composition signal.

For each pixel, computes the fraction of cloud-free observations in each SCL
class, grouped by calendar month, averaged across all years.  Produces a
12-month profile per pixel per class — the seasonal shape of surface
composition (water / vegetation / bare soil / other).

Intended for describe.py (descriptive), not explore.py (discriminative).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import polars as pl

from signals import QualityParams
from signals._shared import load_and_filter


class SclCompositionSignal:
    """Monthly SCL class composition signal."""

    @dataclass
    class Params:
        quality: QualityParams = field(default_factory=QualityParams)

    def __init__(self, params: SclCompositionSignal.Params | None = None) -> None:
        self.params = params or SclCompositionSignal.Params()

    def compute(self, pixel_df: pd.DataFrame, loc: object) -> pd.DataFrame:
        """Compute monthly SCL class fractions per pixel.

        Parameters
        ----------
        pixel_df:
            Raw observation DataFrame for this location.  Must contain an
            ``scl`` column (forward-only from frenchs re-collection).  If the
            column is absent the method returns an empty DataFrame.
        loc:
            ``utils.location.Location`` (unused beyond signature consistency).

        Returns
        -------
        Long-format DataFrame, one row per (pixel, month):
        ``[point_id, lon, lat, month, n_obs, scl_veg, scl_bare, scl_water, scl_other]``
        Fractions sum to 1.0 per row.  ``scl_other`` = classes 7 + 11.
        """
        if isinstance(pixel_df, pd.DataFrame) and "scl" not in pixel_df.columns:
            print("  scl_composition: skipping — parquet predates raw SCL storage")
            return pd.DataFrame()

        p = self.params
        df = load_and_filter(
            pixel_df,
            p.quality.scl_purity_min,
            load_cols=["point_id", "lon", "lat", "date", "scl_purity", "scl"],
        )

        if "scl" not in df.columns:
            print("  scl_composition: skipping — parquet predates raw SCL storage")
            return pd.DataFrame()

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        monthly = (
            df.group_by(["point_id", "month"])
            .agg([
                pl.len().alias("n_obs"),
                (pl.col("scl") == 4).mean().alias("scl_veg"),
                (pl.col("scl") == 5).mean().alias("scl_bare"),
                (pl.col("scl") == 6).mean().alias("scl_water"),
                pl.col("scl").is_in([7, 11]).mean().alias("scl_other"),
            ])
            .sort(["point_id", "month"])
            .with_columns([
                pl.col("n_obs").cast(pl.Int32),
                pl.col("month").cast(pl.Int8),
                pl.col("scl_veg").cast(pl.Float32),
                pl.col("scl_bare").cast(pl.Float32),
                pl.col("scl_water").cast(pl.Float32),
                pl.col("scl_other").cast(pl.Float32),
            ])
            .to_pandas()
        )

        result = monthly.merge(coords, on="point_id", how="left")
        result = result[["point_id", "lon", "lat", "month", "n_obs",
                         "scl_veg", "scl_bare", "scl_water", "scl_other"]]
        return result

    def compute_timeseries(self, pixel_df: pd.DataFrame, loc: object) -> pd.DataFrame:
        """Compute SCL class fractions per pixel per (year, month).

        Same as ``compute`` but preserves the year dimension — one row per
        (pixel, year, month) — so the caller can plot actual temporal evolution
        rather than a climatological seasonal cycle.

        Returns
        -------
        Long-format DataFrame:
        ``[point_id, lon, lat, year, month, n_obs, scl_veg, scl_bare, scl_water, scl_other]``
        """
        if isinstance(pixel_df, pd.DataFrame) and "scl" not in pixel_df.columns:
            print("  scl_composition: skipping — parquet predates raw SCL storage")
            return pd.DataFrame()

        p = self.params
        df = load_and_filter(
            pixel_df,
            p.quality.scl_purity_min,
            load_cols=["point_id", "lon", "lat", "date", "scl_purity", "scl"],
        )

        if "scl" not in df.columns:
            print("  scl_composition: skipping — parquet predates raw SCL storage")
            return pd.DataFrame()

        coords = (
            df.select(["point_id", "lon", "lat"])
            .unique("point_id")
            .to_pandas()
        )

        ts = (
            df.group_by(["point_id", "year", "month"])
            .agg([
                pl.len().alias("n_obs"),
                (pl.col("scl") == 4).mean().alias("scl_veg"),
                (pl.col("scl") == 5).mean().alias("scl_bare"),
                (pl.col("scl") == 6).mean().alias("scl_water"),
                pl.col("scl").is_in([7, 11]).mean().alias("scl_other"),
            ])
            .sort(["point_id", "year", "month"])
            .with_columns([
                pl.col("n_obs").cast(pl.Int32),
                pl.col("year").cast(pl.Int16),
                pl.col("month").cast(pl.Int8),
                pl.col("scl_veg").cast(pl.Float32),
                pl.col("scl_bare").cast(pl.Float32),
                pl.col("scl_water").cast(pl.Float32),
                pl.col("scl_other").cast(pl.Float32),
            ])
            .to_pandas()
        )

        result = ts.merge(coords, on="point_id", how="left")
        result = result[["point_id", "lon", "lat", "year", "month", "n_obs",
                         "scl_veg", "scl_bare", "scl_water", "scl_other"]]
        return result
