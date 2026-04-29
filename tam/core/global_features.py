"""tam/core/global_features.py — Per-pixel annual statistics used as global features.

Features are computed from the full multi-year pixel time series and appended to the
transformer's pooled representation before the classification head. This gives the model
access to physically-motivated signals that the transformer may not reliably extract from
raw observation sequences alone.

S2-derived features (based on Longreach signal analysis — see docs/research/LONGREACH-STAGE2.md):

  nir_cv      Inter-annual CV of dry-season B08 median. Lower = stable evergreen = Parkinsonia.
  rec_p       Mean annual NDVI amplitude (p90 − p10). Higher = deeper wet/dry swing = Parkinsonia.
  peak_doy    Mean annual DOY of NDVI peak. Parkinsonia peaks consistently earlier in wet season.
  peak_doy_cv Per-pixel SD of annual peak DOY. Lower = more consistent timing = Parkinsonia.
  dry_ndvi    Median dry-season NDVI.

S1-derived features (structural, intended to aid cross-climate-zone transferability):

  s1_mean_vh_dry       Mean VH backscatter (dB) over dry season (May–Oct). Woody stem density.
  s1_vh_contrast       Mean VH wet-season (dB) minus mean VH dry-season (dB). Grass fluctuates
                       strongly; woody Parkinsonia is stable.
  s1_vh_std            Temporal std of VH (dB) across all observations. Low = stable canopy.
  s1_mean_rvi          Mean Radar Vegetation Index (4·VH_lin / (VV_lin + VH_lin)) across all
                       observations. Vegetation density, normalised out soil moisture.

Dry season: May–October (DOY 121–304). Wet season: November–April (DOY 305–365 + 1–120).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

GLOBAL_FEATURE_NAMES: list[str] = [
    "s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi",
    "nir_cv", "rec_p", "peak_doy", "peak_doy_cv", "dry_ndvi",
]

_DRY_DOY_MIN = 121   # May 1
_DRY_DOY_MAX = 304   # October 31
_MIN_DRY_OBS = 5
_MIN_OBS     = 10
_SMOOTH_DAYS = 30


_WET_DOY_RANGES = [(305, 365), (1, 120)]   # Nov–Apr wraps around year boundary


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

    result = pd.DataFrame(index=df["point_id"].unique())
    result.index.name = "point_id"

    if "B08" in pixel_df.columns and "B04" in pixel_df.columns:
        df["_ndvi"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"] + 1e-6)
        nir_cv, dry_ndvi = _compute_nir_cv_and_dry_ndvi(df)
        rec_p      = _compute_rec_p(df)
        peak_stats = _compute_peak_doy(df)
        result["nir_cv"]      = nir_cv
        result["rec_p"]       = rec_p
        result["peak_doy"]    = peak_stats["peak_doy"]
        result["peak_doy_cv"] = peak_stats["peak_doy_cv"]
        result["dry_ndvi"]    = dry_ndvi
    else:
        for col in ["nir_cv", "rec_p", "peak_doy", "peak_doy_cv", "dry_ndvi"]:
            result[col] = np.nan

    if "vh" in pixel_df.columns and "vv" in pixel_df.columns:
        s1_feats = _compute_s1_globals(pixel_df)
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            result[col] = s1_feats.get(col, pd.Series(dtype=float))
    else:
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            result[col] = np.nan

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


def _compute_s1_globals(pixel_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute per-pixel S1 global features from S1 rows in pixel_df.

    Expects rows with source="S1" (or any row where vh/vv are non-null).
    All backscatter values stored as linear power; converted to dB here.
    """
    if "doy" not in pixel_df.columns:
        df = pixel_df.copy()
        df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year
    else:
        df = pixel_df

    # Keep only S1 rows with at least one valid band
    if "source" in df.columns:
        s1 = df[df["source"] == "S1"].copy()
    else:
        s1 = df[df["vh"].notna() | df["vv"].notna()].copy()

    if s1.empty:
        empty = pd.Series(dtype=float)
        return {k: empty for k in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]}

    # Convert linear power → dB (guard against zero/negative)
    s1["_vh_db"] = 10 * np.log10(s1["vh"].clip(lower=1e-10))
    s1["_vv_db"] = 10 * np.log10(s1["vv"].clip(lower=1e-10))

    # RVI: 4·VH_lin / (VV_lin + VH_lin); requires both bands
    both = s1["vh"].notna() & s1["vv"].notna()
    s1["_rvi"] = np.nan
    s1.loc[both, "_rvi"] = (
        4 * s1.loc[both, "vh"] / (s1.loc[both, "vv"] + s1.loc[both, "vh"])
    )

    dry_mask = (s1["doy"] >= _DRY_DOY_MIN) & (s1["doy"] <= _DRY_DOY_MAX)
    wet_mask = (s1["doy"] > _DRY_DOY_MAX) | (s1["doy"] < _DRY_DOY_MIN)

    dry = s1[dry_mask & s1["_vh_db"].notna()]
    wet = s1[wet_mask & s1["_vh_db"].notna()]

    mean_vh_dry = dry.groupby("point_id")["_vh_db"].mean().rename("s1_mean_vh_dry")
    mean_vh_wet = wet.groupby("point_id")["_vh_db"].mean()
    vh_contrast = (mean_vh_wet - mean_vh_dry).rename("s1_vh_contrast")
    vh_std      = s1[s1["_vh_db"].notna()].groupby("point_id")["_vh_db"].std().rename("s1_vh_std")
    mean_rvi    = s1[s1["_rvi"].notna()].groupby("point_id")["_rvi"].mean().rename("s1_mean_rvi")

    return {
        "s1_mean_vh_dry": mean_vh_dry,
        "s1_vh_contrast": vh_contrast,
        "s1_vh_std":      vh_std,
        "s1_mean_rvi":    mean_rvi,
    }
