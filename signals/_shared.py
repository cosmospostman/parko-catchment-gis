"""signals/_shared.py — Shared computational kernels and site-param loader.

Three pure functions used by two or more signal classes:
  load_and_filter   — parquet load + SCL purity filter + year/month columns
  annual_percentile — per-pixel per-year percentile, averaged across years
  dry_season_cv     — per-pixel inter-annual CV of dry-season median

All three kernels operate on Polars DataFrames internally for parallel
groupby/agg performance on large parquets (tens of millions of rows).
Inputs may be a pandas DataFrame or a Path; outputs are always pandas.

Plus load_signal_params() which reads signal overrides from a Location's YAML
and returns a populated Params instance merged with the signal's defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Computational kernels
# ---------------------------------------------------------------------------

def load_and_filter(
    path_or_df: Union[Path, pd.DataFrame],
    scl_purity_min: float,
    load_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Load a pixel parquet (or accept a pre-loaded DataFrame), apply SCL
    purity filter, and add ``year`` / ``month`` integer columns.

    Parameters
    ----------
    path_or_df:
        Path to a parquet file, or a pandas DataFrame.
    scl_purity_min:
        Minimum ``scl_purity`` fraction to retain a row.
    load_cols:
        If given, only these columns are loaded from disk (ignored when a
        DataFrame is passed directly).

    Returns
    -------
    Polars DataFrame with ``year`` and ``month`` columns added.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = pl.from_pandas(path_or_df)
    else:
        df = pl.read_parquet(path_or_df, columns=load_cols)

    df = df.filter(pl.col("scl_purity") >= scl_purity_min)
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])
    return df


def annual_percentile(
    df: pl.DataFrame,
    value_col: str,
    percentile: float,
    min_obs_per_year: int,
) -> pd.DataFrame:
    """Per-pixel per-year percentile, averaged across qualifying years.

    Parameters
    ----------
    df:
        Polars observation-level DataFrame with ``point_id``, ``year``, and
        ``value_col`` columns.
    value_col:
        Column name to compute percentile on.
    percentile:
        Percentile in [0, 1], e.g. 0.10 for the 10th percentile.
    min_obs_per_year:
        Minimum observations per (pixel, year) to include that year.

    Returns
    -------
    Pandas DataFrame with columns ``[point_id, <value_col>_p<n>, n_years]``
    where ``<n>`` is the percentile as an integer (e.g. ``re_ratio_p10``).
    """
    n = int(round(percentile * 100))
    out_col = f"{value_col}_p{n}"

    # Annual percentile per (pixel, year), filtering sparse years
    annual = (
        df.group_by(["point_id", "year"])
        .agg([
            pl.col(value_col).quantile(percentile, interpolation="linear").alias(out_col),
            pl.col(value_col).count().alias("_n_obs"),
        ])
        .filter(pl.col("_n_obs") >= min_obs_per_year)
    )

    # Mean across years + year count
    result = (
        annual.group_by("point_id")
        .agg([
            pl.col(out_col).mean(),
            pl.col(out_col).count().alias("n_years"),
        ])
    )

    return result.to_pandas()


def dry_season_cv(
    df: pl.DataFrame,
    value_col: str,
    dry_months: list[int],
    min_obs_dry: int,
) -> pd.DataFrame:
    """Per-pixel inter-annual coefficient of variation of dry-season median.

    Parameters
    ----------
    df:
        Polars observation-level DataFrame with ``point_id``, ``year``,
        ``month``, and ``value_col`` columns.
    value_col:
        Column name to compute CV on (typically ``B08`` for NIR).
    dry_months:
        Month numbers that define the dry season (from ``loc.dry_months``).
    min_obs_dry:
        Minimum observations per (pixel, year) within dry months to include
        that year.

    Returns
    -------
    Pandas DataFrame with columns
    ``[point_id, <value_col>_mean, <value_col>_std, <value_col>_cv, n_years]``.
    """
    df_dry = df.filter(pl.col("month").is_in(dry_months))

    # Annual dry-season median per (pixel, year), filtering sparse years
    medians = (
        df_dry.group_by(["point_id", "year"])
        .agg([
            pl.col(value_col).median().alias("_annual_med"),
            pl.col(value_col).count().alias("_n_obs"),
        ])
        .filter(pl.col("_n_obs") >= min_obs_dry)
    )

    # Inter-annual mean, std, CV
    result = (
        medians.group_by("point_id")
        .agg([
            pl.col("_annual_med").mean().alias(f"{value_col}_mean"),
            pl.col("_annual_med").std().alias(f"{value_col}_std"),
            pl.col("_annual_med").count().alias("n_years"),
        ])
        .with_columns([
            (pl.col(f"{value_col}_std") / pl.col(f"{value_col}_mean")).alias(f"{value_col}_cv"),
        ])
    )

    return result.to_pandas()


# ---------------------------------------------------------------------------
# Site-param loader
# ---------------------------------------------------------------------------

def load_signal_params(loc: object, signal: str) -> object:
    """Return a signal-specific Params instance for a given location.

    Reads the ``signals:`` section from the Location's YAML (exposed as
    ``loc.signal_params``), merges the site overrides with signal defaults,
    and returns a populated ``<SignalClass>.Params`` dataclass.

    Parameters
    ----------
    loc:
        A ``utils.location.Location`` instance.
    signal:
        One of ``'nir_cv'``, ``'rec_p'``, ``'red_edge'``, ``'swir'``,
        ``'flowering'``.

    Returns
    -------
    A ``<SignalClass>.Params`` dataclass instance with site overrides applied.
    """
    # Import here to avoid circular imports at module load time
    from signals.nir_cv import NirCvSignal
    from signals.wet_dry_amp import RecPSignal
    from signals.red_edge import RedEdgeSignal
    from signals.swir import SwirSignal
    from signals.flowering import FloweringSignal
    from signals import QualityParams

    _signal_map = {
        "nir_cv": NirCvSignal,
        "rec_p": RecPSignal,
        "red_edge": RedEdgeSignal,
        "swir": SwirSignal,
        "flowering": FloweringSignal,
    }

    if signal not in _signal_map:
        raise ValueError(f"Unknown signal '{signal}'. Choose from: {list(_signal_map)}")

    cls = _signal_map[signal]
    site_overrides: dict = (loc.signal_params or {}).get(signal, {})

    if not site_overrides:
        return cls.Params()

    # Separate quality-related keys from signal-level keys
    quality_keys = {"scl_purity_min", "min_obs_per_year", "min_obs_dry"}
    quality_overrides = {k: v for k, v in site_overrides.items() if k in quality_keys}
    signal_overrides = {k: v for k, v in site_overrides.items() if k not in quality_keys}

    quality = QualityParams(**quality_overrides) if quality_overrides else QualityParams()

    return cls.Params(quality=quality, **signal_overrides)
