"""tam/core/global_features.py — Per-pixel annual statistics used as global features.

Features are computed from the full multi-year pixel time series and appended to the
transformer's pooled representation before the classification head. This gives the model
access to physically-motivated signals that the transformer may not reliably extract from
raw observation sequences alone.

Features (based on Longreach signal analysis — see docs/research/LONGREACH-STAGE2.md):

  nir_cv      Inter-annual CV of dry-season B08 median. Lower = stable evergreen = Parkinsonia.
  rec_p       Mean annual NDVI amplitude (p90 − p10). Higher = deeper wet/dry swing = Parkinsonia.
  peak_doy    Mean annual DOY of NDVI peak. Parkinsonia peaks consistently earlier in wet season.
  peak_doy_cv Per-pixel SD of annual peak DOY. Lower = more consistent timing = Parkinsonia.

Dry season: May–October (DOY 121–304). Research confirmed this window gives stronger class
separation than the theoretical April–September prior.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

GLOBAL_FEATURE_NAMES: list[str] = ["nir_cv", "rec_p", "peak_doy", "peak_doy_cv", "dry_ndvi"]

_DRY_DOY_MIN = 121   # May 1
_DRY_DOY_MAX = 304   # October 31
_MIN_DRY_OBS = 5
_MIN_OBS     = 10
_SMOOTH_DAYS = 30


def compute_global_features(pixel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pixel global features from the full observation time series.

    Parameters
    ----------
    pixel_df:
        Raw observations. Must contain: point_id, date (or doy), year, B08, B04.

    Returns
    -------
    DataFrame indexed by point_id with columns [nir_cv, rec_p, peak_doy, peak_doy_cv].
    Pixels with insufficient data have NaN for the affected feature.
    """
    df = pixel_df.copy()
    if "doy" not in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

    df["_ndvi"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"] + 1e-6)

    nir_cv, dry_ndvi = _compute_nir_cv_and_dry_ndvi(df)
    rec_p      = _compute_rec_p(df)
    peak_stats = _compute_peak_doy(df)

    result = pd.DataFrame(index=df["point_id"].unique())
    result.index.name = "point_id"
    result["nir_cv"]      = nir_cv
    result["rec_p"]       = rec_p
    result["peak_doy"]    = peak_stats["peak_doy"]
    result["peak_doy_cv"] = peak_stats["peak_doy_cv"]
    result["dry_ndvi"]    = dry_ndvi

    return result[GLOBAL_FEATURE_NAMES]


def _compute_nir_cv_and_dry_ndvi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute nir_cv and dry_ndvi together — both need the dry-season filter."""
    dry = df[(df["doy"] >= _DRY_DOY_MIN) & (df["doy"] <= _DRY_DOY_MAX)].copy()
    dry["_ndvi_dry"] = (dry["B08"] - dry["B04"]) / (dry["B08"] + dry["B04"] + 1e-6)

    annual = (
        dry.groupby(["point_id", "year"])[["B08", "_ndvi_dry"]]
        .agg(
            nir_median=("B08", "median"),
            nir_count=("B08", "count"),
            ndvi_median=("_ndvi_dry", "median"),
        )
        .reset_index()
    )
    annual = annual[annual["nir_count"] >= _MIN_DRY_OBS]

    # nir_cv
    per_pixel = annual.groupby("point_id")["nir_median"].agg(["mean", "std", "count"])
    per_pixel = per_pixel[per_pixel["count"] >= 2]
    nir_cv = (per_pixel["std"] / per_pixel["mean"].clip(lower=1e-6)).rename("nir_cv")

    # dry_ndvi — median of annual dry-season NDVI medians
    dry_ndvi = annual.groupby("point_id")["ndvi_median"].median().rename("dry_ndvi")

    return nir_cv, dry_ndvi


def _compute_rec_p(df: pd.DataFrame) -> pd.Series:
    annual = (
        df.groupby(["point_id", "year"])["_ndvi"]
        .agg(
            p90=lambda x: np.percentile(x, 90) if len(x) >= _MIN_OBS else np.nan,
            p10=lambda x: np.percentile(x, 10) if len(x) >= _MIN_OBS else np.nan,
        )
        .reset_index()
    )
    annual["amp"] = annual["p90"] - annual["p10"]
    return annual.groupby("point_id")["amp"].mean().rename("rec_p")


def _peak_doy_chunk(chunk_df: pd.DataFrame) -> list[dict]:
    results = []
    for (pid, yr), grp in chunk_df.groupby(["point_id", "year"]):
        grp = grp.sort_values("doy")
        if len(grp) < _MIN_OBS:
            continue
        s = grp.set_index("doy")["_ndvi"].sort_index()
        smoothed = s.rolling(
            window=min(_SMOOTH_DAYS, len(s)), min_periods=3, center=True
        ).median()
        if not smoothed.isna().all():
            results.append({"point_id": pid, "year": yr, "peak_doy": int(smoothed.idxmax())})
    return results


def _compute_peak_doy(df: pd.DataFrame) -> pd.DataFrame:
    from joblib import Parallel, delayed

    n_jobs = os.cpu_count() or 1
    pixels = df["point_id"].unique()
    chunks = np.array_split(pixels, n_jobs)
    chunk_dfs = [df[df["point_id"].isin(c)] for c in chunks if len(c) > 0]

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_peak_doy_chunk)(c) for c in chunk_dfs
    )
    results = [row for rows in nested for row in rows]

    if not results:
        return pd.DataFrame(columns=["peak_doy", "peak_doy_cv"], index=pd.Index([], name="point_id"))

    annual = pd.DataFrame(results)
    return annual.groupby("point_id")["peak_doy"].agg(
        peak_doy="mean",
        peak_doy_cv="std",
    )
