"""utils/polars_utils.py — Shared polars utility functions.

Functions here encapsulate common polars patterns used throughout the pipeline
so callers stay concise and consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


def scan_region_parquet(
    path: Path,
    filter_expr: pl.Expr | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load a region parquet lazily, applying optional column projection and filter.

    Prefer this over pl.read_parquet for region-sized files — scan_parquet pushes
    column selection and filter predicates down to the Parquet reader, avoiding
    loading unused columns or rows into RAM.

    Parameters
    ----------
    path:
        Path to the parquet file.
    filter_expr:
        Optional polars Expr to filter rows before collect.
    columns:
        Optional list of column names to select. If None, all columns are loaded.

    Returns
    -------
    Collected DataFrame.
    """
    lf: pl.LazyFrame = pl.scan_parquet(path)
    if columns is not None:
        lf = lf.select(columns)
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    return lf.collect()


def add_year_doy(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Add ``year`` (Int32) and ``doy`` (Int32) columns derived from a date column.

    Parameters
    ----------
    df:
        Input DataFrame containing ``date_col``.
    date_col:
        Name of the date column. Must be castable to pl.Date.

    Returns
    -------
    DataFrame with two additional columns: ``year`` and ``doy``.
    """
    return df.with_columns([
        pl.col(date_col).cast(pl.Date).dt.year().alias("year"),
        pl.col(date_col).cast(pl.Date).dt.ordinal_day().alias("doy"),
    ])


def series_to_f32(s: pl.Series, fill_null: float | None = None) -> np.ndarray:
    """Return a Series as a float32 numpy array, dropping nulls.

    Parameters
    ----------
    s:
        Input Series. May contain nulls or NaN float values.
    fill_null:
        If provided, fill nulls with this value before converting (no drop).
        If None, nulls are dropped.

    Returns
    -------
    1-D float32 numpy array.
    """
    if fill_null is not None:
        return s.fill_null(fill_null).to_numpy().astype("float32")
    return s.drop_nulls().to_numpy().astype("float32")
