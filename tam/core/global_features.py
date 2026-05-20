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
import polars as pl

from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX

GLOBAL_FEATURE_NAMES: list[str] = [
    "s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi",
    "nir_cv", "rec_p", "peak_doy", "peak_doy_cv", "dry_ndvi",
]
_MIN_DRY_OBS = 5
_MIN_OBS     = 10
_SMOOTH_DAYS = 30


_WET_DOY_RANGES = [(305, 365), (1, 120)]   # Nov–Apr wraps around year boundary


def compute_global_features(pixel_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-pixel global features from the full observation time series.

    Parameters
    ----------
    pixel_df:
        Raw observations. Must contain: point_id, date (or doy), year, B08, B04.

    Returns
    -------
    DataFrame with point_id column and global feature columns.
    Pixels with insufficient data have null for the affected feature.
    """
    if "doy" not in pixel_df.columns:
        pixel_df = pixel_df.with_columns(
            pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
        )

    all_pids = pixel_df["point_id"].unique().to_list()
    result = pl.DataFrame({"point_id": all_pids})

    if "B08" in pixel_df.columns and "B04" in pixel_df.columns:
        pixel_df = pixel_df.with_columns(
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04") + 1e-6)).alias("_ndvi")
        )
        nir_cv_df, dry_ndvi_df = _compute_nir_cv_and_dry_ndvi(pixel_df)
        rec_p_df   = _compute_rec_p(pixel_df)
        peak_stats = _compute_peak_doy(pixel_df)
        result = result.join(nir_cv_df, on="point_id", how="left")
        result = result.join(dry_ndvi_df, on="point_id", how="left")
        result = result.join(rec_p_df, on="point_id", how="left")
        result = result.join(peak_stats, on="point_id", how="left")
    else:
        for col in ["nir_cv", "rec_p", "peak_doy", "peak_doy_cv", "dry_ndvi"]:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    if "vh" in pixel_df.columns and "vv" in pixel_df.columns:
        s1_feats = _compute_s1_globals(pixel_df)
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            if col in s1_feats.columns:
                result = result.join(s1_feats.select(["point_id", col]), on="point_id", how="left")
            else:
                result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    else:
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result.select(["point_id"] + GLOBAL_FEATURE_NAMES)


def _compute_nir_cv_and_dry_ndvi(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute nir_cv and dry_ndvi together — both need the dry-season filter."""
    dry = df.filter(
        (pl.col("doy") >= _DRY_DOY_MIN) & (pl.col("doy") <= _DRY_DOY_MAX)
    ).with_columns(
        ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04") + 1e-6)).alias("_ndvi_dry")
    )

    annual = (
        dry.group_by(["point_id", "year"])
        .agg([
            pl.col("B08").median().alias("nir_median"),
            pl.col("B08").count().alias("nir_count"),
            pl.col("_ndvi_dry").median().alias("ndvi_median"),
        ])
        .filter(pl.col("nir_count") >= _MIN_DRY_OBS)
    )

    # nir_cv
    per_pixel = (
        annual.group_by("point_id")
        .agg([
            pl.col("nir_median").mean().alias("nir_mean"),
            pl.col("nir_median").std().alias("nir_std"),
            pl.col("nir_median").count().alias("nir_year_count"),
        ])
        .filter(pl.col("nir_year_count") >= 2)
        .with_columns(
            (pl.col("nir_std") / pl.col("nir_mean").clip(lower_bound=1e-6)).alias("nir_cv")
        )
        .select(["point_id", "nir_cv"])
    )

    # dry_ndvi
    dry_ndvi = (
        annual.group_by("point_id")
        .agg(pl.col("ndvi_median").median().alias("dry_ndvi"))
    )

    return per_pixel, dry_ndvi


def _compute_rec_p(df: pl.DataFrame) -> pl.DataFrame:
    annual = (
        df.group_by(["point_id", "year"])
        .agg([
            pl.col("_ndvi").quantile(0.90).alias("p90"),
            pl.col("_ndvi").quantile(0.10).alias("p10"),
            pl.col("_ndvi").count().alias("n_obs"),
        ])
        .filter(pl.col("n_obs") >= _MIN_OBS)
        .with_columns((pl.col("p90") - pl.col("p10")).alias("amp"))
    )
    return (
        annual.group_by("point_id")
        .agg(pl.col("amp").mean().alias("rec_p"))
    )


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median with min_periods=3; returns NaN where insufficient data."""
    n = len(values)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        patch = values[lo:hi]
        valid = patch[~np.isnan(patch)]
        if len(valid) >= 3:
            out[i] = np.median(valid)
    return out


def _peak_doy_chunk(chunk_df: pl.DataFrame) -> list[dict]:
    """Compute peak DOY for each (point_id, year) group using numpy rolling median."""
    results = []
    for (pid, yr), grp in chunk_df.sort(["point_id", "year", "doy"]).group_by(
        ["point_id", "year"], maintain_order=True
    ):
        if len(grp) < _MIN_OBS:
            continue
        doys = grp["doy"].to_numpy()
        ndvi = grp["_ndvi"].to_numpy().astype(np.float64)
        w = min(_SMOOTH_DAYS, len(ndvi))
        smoothed = _rolling_median(ndvi, w)
        valid = ~np.isnan(smoothed)
        if valid.any():
            results.append({"point_id": pid, "year": yr, "peak_doy": int(doys[np.argmax(smoothed)])})
    return results


def _compute_peak_doy(df: pl.DataFrame) -> pl.DataFrame:
    from joblib import Parallel, delayed

    n_jobs = os.cpu_count() or 1
    pixels = df["point_id"].unique().to_list()
    chunk_size = max(1, len(pixels) // n_jobs)
    chunks = [
        df.filter(pl.col("point_id").is_in(pixels[i:i + chunk_size]))
        for i in range(0, len(pixels), chunk_size)
        if pixels[i:i + chunk_size]
    ]

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_peak_doy_chunk)(c) for c in chunks
    )
    results = [row for rows in nested for row in rows]

    if not results:
        return pl.DataFrame({"point_id": pl.Series([], dtype=pl.Utf8),
                              "peak_doy": pl.Series([], dtype=pl.Float64),
                              "peak_doy_cv": pl.Series([], dtype=pl.Float64)})

    annual = pl.DataFrame(results)
    return (
        annual.group_by("point_id")
        .agg([
            pl.col("peak_doy").mean().alias("peak_doy"),
            pl.col("peak_doy").std().alias("peak_doy_cv"),
        ])
    )


def _compute_s1_globals(pixel_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-pixel S1 global features from S1 rows in pixel_df."""
    if "doy" not in pixel_df.columns:
        df = pixel_df.with_columns(
            pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
        )
    else:
        df = pixel_df

    if "source" in df.columns:
        s1 = df.filter(pl.col("source") == "S1")
    else:
        s1 = df.filter(pl.col("vh").is_not_null() | pl.col("vv").is_not_null())

    if s1.is_empty():
        return pl.DataFrame({"point_id": pl.Series([], dtype=pl.Utf8)})

    vh_lin = s1["vh"].to_numpy().astype("float32")
    vv_lin = s1["vv"].to_numpy().astype("float32")
    vh_db = (10 * np.log10(np.clip(vh_lin, 1e-10, None))).astype("float32")
    vv_db = (10 * np.log10(np.clip(vv_lin, 1e-10, None))).astype("float32")

    both = (~np.isnan(vh_lin)) & (~np.isnan(vv_lin))
    rvi = np.full(len(s1), np.nan, dtype="float32")
    denom = vh_lin[both] + vv_lin[both]
    rvi[both] = np.where(denom > 0, 4 * vh_lin[both] / denom, np.nan)

    s1 = s1.with_columns([
        pl.Series("_vh_db", vh_db),
        pl.Series("_vv_db", vv_db),
        pl.Series("_rvi", rvi),
    ])

    dry = s1.filter(
        (pl.col("doy") >= _DRY_DOY_MIN) & (pl.col("doy") <= _DRY_DOY_MAX) &
        pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan()
    )
    wet = s1.filter(
        ((pl.col("doy") > _DRY_DOY_MAX) | (pl.col("doy") < _DRY_DOY_MIN)) &
        pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan()
    )

    mean_vh_dry = dry.group_by("point_id").agg(pl.col("_vh_db").mean().alias("s1_mean_vh_dry"))
    mean_vh_wet = wet.group_by("point_id").agg(pl.col("_vh_db").mean().alias("_vh_wet"))
    vh_std_df = (
        s1.filter(pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan())
        .group_by("point_id")
        .agg(pl.col("_vh_db").std().alias("s1_vh_std"))
    )
    mean_rvi = (
        s1.filter(pl.col("_rvi").is_not_null() & pl.col("_rvi").is_not_nan())
        .group_by("point_id")
        .agg(pl.col("_rvi").mean().alias("s1_mean_rvi"))
    )

    result = mean_vh_dry
    result = result.join(mean_vh_wet, on="point_id", how="left")
    result = result.with_columns(
        (pl.col("_vh_wet") - pl.col("s1_mean_vh_dry")).alias("s1_vh_contrast")
    ).drop("_vh_wet")
    result = result.join(vh_std_df, on="point_id", how="left")
    result = result.join(mean_rvi, on="point_id", how="left")

    return result
