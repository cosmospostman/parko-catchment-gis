"""signals/_shared.py — Shared computational kernels and site-param loader.

Three pure functions used by two or more signal classes:
  load_and_filter   — parquet load + SCL purity filter + year/month columns
  annual_percentile — per-pixel per-year percentile, averaged across years
  dry_season_cv     — per-pixel inter-annual CV of dry-season median

Plus load_signal_params() which reads signal overrides from a Location's YAML
and returns a populated Params instance merged with the signal's defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


# ---------------------------------------------------------------------------
# Computational kernels
# ---------------------------------------------------------------------------

def load_and_filter(
    path_or_df: Union[Path, pd.DataFrame],
    scl_purity_min: float,
    load_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a pixel parquet (or accept a pre-loaded DataFrame), apply SCL
    purity filter, and add ``year`` / ``month`` integer columns.

    Parameters
    ----------
    path_or_df:
        Path to a parquet file, or an already-loaded DataFrame.
    scl_purity_min:
        Minimum ``scl_purity`` fraction to retain a row.
    load_cols:
        If given, only these columns are loaded from disk (ignored when a
        DataFrame is passed directly).

    Returns
    -------
    Filtered DataFrame with ``year`` and ``month`` columns added.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_parquet(path_or_df, columns=load_cols)

    df = df[df["scl_purity"] >= scl_purity_min].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def annual_percentile(
    df: pd.DataFrame,
    value_col: str,
    percentile: float,
    min_obs_per_year: int,
) -> pd.DataFrame:
    """Per-pixel per-year percentile, averaged across qualifying years.

    Parameters
    ----------
    df:
        Observation-level DataFrame with ``point_id``, ``year``, and
        ``value_col`` columns.
    value_col:
        Column name to compute percentile on.
    percentile:
        Percentile in [0, 1], e.g. 0.10 for the 10th percentile.
    min_obs_per_year:
        Minimum observations per (pixel, year) to include that year.

    Returns
    -------
    DataFrame with columns ``[point_id, <value_col>_p<n>, n_years]``
    where ``<n>`` is the percentile as an integer (e.g. ``re_ratio_p10``).
    """
    n = int(round(percentile * 100))
    out_col = f"{value_col}_p{n}"

    # Drop (pixel, year) pairs with too few observations
    counts = df.groupby(["point_id", "year"])[value_col].count()
    valid = counts[counts >= min_obs_per_year].index
    df_valid = df.set_index(["point_id", "year"]).loc[valid].reset_index()

    # Annual percentile per pixel
    annual = (
        df_valid.groupby(["point_id", "year"])[value_col]
        .quantile(percentile)
        .rename(out_col)
        .reset_index()
    )

    # Mean across years + year count
    result = (
        annual.groupby("point_id")[out_col]
        .agg(**{out_col: "mean", "n_years": "count"})
        .reset_index()
    )
    return result


def dry_season_cv(
    df: pd.DataFrame,
    value_col: str,
    dry_months: list[int],
    min_obs_dry: int,
) -> pd.DataFrame:
    """Per-pixel inter-annual coefficient of variation of dry-season median.

    Parameters
    ----------
    df:
        Observation-level DataFrame with ``point_id``, ``year``, ``month``,
        and ``value_col`` columns.
    value_col:
        Column name to compute CV on (typically ``B08`` for NIR).
    dry_months:
        Month numbers that define the dry season (from ``loc.dry_months``).
    min_obs_dry:
        Minimum observations per (pixel, year) within dry months to include
        that year.

    Returns
    -------
    DataFrame with columns
    ``[point_id, <value_col>_mean, <value_col>_std, <value_col>_cv, n_years]``.
    """
    df_dry = df[df["month"].isin(dry_months)].copy()

    # Drop (pixel, year) pairs with too few dry-season observations
    counts = df_dry.groupby(["point_id", "year"])[value_col].count()
    valid = counts[counts >= min_obs_dry].index
    df_valid = df_dry.set_index(["point_id", "year"]).loc[valid].reset_index()

    # Annual dry-season median
    medians = (
        df_valid.groupby(["point_id", "year"])[value_col]
        .median()
        .rename("_annual_med")
        .reset_index()
    )

    # Inter-annual mean, std, CV
    result = (
        medians.groupby("point_id")["_annual_med"]
        .agg(
            **{
                f"{value_col}_mean": "mean",
                f"{value_col}_std": "std",
                "n_years": "count",
            }
        )
        .reset_index()
    )
    result[f"{value_col}_cv"] = result[f"{value_col}_std"] / result[f"{value_col}_mean"]
    return result


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
