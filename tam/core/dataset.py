"""tam/core/dataset.py — PyTorch dataset for TAM temporal attention model.

Each dataset item is one (pixel, year) annual window: a padded sequence of
Sentinel-2 observations with DOY positional indices and a binary presence label.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SPECTRAL_INDEX_COLS, add_spectral_indices

BAND_COLS: list[str] = list(BANDS)   # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
INDEX_COLS: list[str] = SPECTRAL_INDEX_COLS
ALL_FEATURE_COLS: list[str] = BAND_COLS + INDEX_COLS

# S1-derived per-observation features (mixed / s1_only modes)
S1_FEATURE_COLS: list[str] = ["s1_vh", "s1_vv", "s1_vh_vv", "s1_rvi"]

# V9-SPECTRAL feature set: B06 excluded (calibration artefacts), EVI excluded
# (additive denominator behaves poorly under pixel z-score), NDWI retained.
# MAVI, NDRE, CI_RE added after signal eval (2026-05-19): all cleared AUROC >= 0.75
# at sparse_stress tier (CI_RE=0.816, MAVI=0.794, NDRE=0.780).
V9_FEATURE_COLS: list[str] = [
    "B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "MAVI", "NDRE", "CI_RE",
]

# V10: joint S1+S2. S2 cols identical to V9; S1 adds VH and VV only.
# vh_vv ratio and RVI are derived; VH+VV alone until their discriminative
# value is confirmed in sweep.
V10_FEATURE_COLS: list[str] = [
    "B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "MAVI", "NDRE", "CI_RE",
]
V10_S1_FEATURE_COLS: list[str] = ["s1_vh", "s1_vv"]
N_BANDS: int = len(ALL_FEATURE_COLS)          # 13: S2 only
N_BANDS_S1: int = len(ALL_FEATURE_COLS) + len(S1_FEATURE_COLS)  # 17: mixed/s1_only S1+S2
N_BANDS_MIXED: int = N_BANDS_S1               # 17: mixed S1+S2 native rows (same width)
MAX_SEQ_LEN: int = 256
MIN_OBS_PER_YEAR: int = 8
MIN_S1_OBS_PER_YEAR: int = 4


def lin_to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear-power SAR backscatter to dB; returns nan where linear <= 0."""
    return np.where(linear > 0, 10.0 * np.log10(linear), np.nan)


class TAMSample(NamedTuple):
    bands:        torch.Tensor   # (MAX_SEQ_LEN, N_BANDS)  float32, normalised, zero-padded
    doy:          torch.Tensor   # (MAX_SEQ_LEN,)           int64, 1–365, 0=padding
    mask:         torch.Tensor   # (MAX_SEQ_LEN,)           bool, True=padding
    n_obs:        torch.Tensor   # ()                       float32, n / MAX_SEQ_LEN
    global_feats: torch.Tensor   # (N_GLOBAL,)              float32, z-scored global features
    label:        torch.Tensor   # ()                       float32 {0, 1}
    weight:       torch.Tensor   # ()                       float32 confidence weight
    is_s1:        torch.Tensor   # (MAX_SEQ_LEN,)           bool, True=S1 observation
    point_id: str
    year:     int


def despeckle_s1(s1: pl.DataFrame, window: int) -> pl.DataFrame:
    """Apply temporal despeckle to S1 linear backscatter via a per-pixel rolling median.

    Operates on the raw linear vh/vv columns. Speckle is independent between
    acquisitions, so a rolling median across N passes suppresses it without spatial
    blurring. window=3 is conservative; 5 and 7 are also reasonable but risk smoothing
    real wet/dry seasonal transitions.

    Applied before dB conversion so that derived features (VH-VV, RVI) also benefit.
    window=0 or window=1 is a no-op.
    """
    if window < 2:
        return s1
    s1 = s1.sort(["point_id", "date"])
    exprs = []
    for col in ("vh", "vv"):
        if col in s1.columns:
            exprs.append(
                pl.col(col)
                  .rolling_median(window_size=window, min_samples=1, center=True)
                  .over("point_id")
                  .cast(pl.Float32)
                  .alias(col)
            )
    if exprs:
        s1 = s1.with_columns(exprs)
    return s1


def collate_fn(samples: list[TAMSample]) -> dict:
    return {
        "bands":        torch.stack([s.bands        for s in samples]),
        "doy":          torch.stack([s.doy          for s in samples]),
        "mask":         torch.stack([s.mask         for s in samples]),
        "n_obs":        torch.stack([s.n_obs        for s in samples]),
        "global_feats": torch.stack([s.global_feats for s in samples]),
        "label":        torch.stack([s.label        for s in samples]),
        "weight":       torch.stack([s.weight       for s in samples]),
        "is_s1":        torch.stack([s.is_s1        for s in samples]),
        "point_id": [s.point_id for s in samples],
        "year":     [s.year     for s in samples],
    }


class TAMDataset(Dataset):
    """One item = one (pixel, year) annual observation window.

    Parameters
    ----------
    pixel_df:
        Raw observations for the pixels of interest (all years).
        Must contain: point_id, date, scl_purity, year, and all BAND_COLS.
        The ``year`` column must already be present (add via load_and_filter).
    labels:
        One of:
        - dict[tuple[str, int], float] — pixel-year labels (from train_tam)
        - dict[str, float] — pixel-level labels (auto-broadcast in __init__)
        - None — inference mode (all pixels scored, labels returned as 0.0)
    band_mean / band_std:
        Per-band normalisation statistics, shape (N_BANDS,).
        If None, computed from pixel_df (training mode).
    scl_purity_min:
        Minimum scl_purity to retain an observation.
    min_obs_per_year:
        Annual windows with fewer clear observations are skipped.
    """

    def __init__(
        self,
        pixel_df: pl.DataFrame,
        labels: dict[tuple[str, int], float] | dict[str, float] | None,
        band_mean: np.ndarray | None = None,
        band_std: np.ndarray | None = None,
        scl_purity_min: float = 0.5,
        min_obs_per_year: int = MIN_OBS_PER_YEAR,
        doy_jitter: int = 0,
        doy_phase_shift: bool = False,
        band_noise_std: float = 0.0,
        obs_dropout_min: int = 0,
        global_features_df: pl.DataFrame | None = None,
        global_feat_mean: np.ndarray | None = None,
        global_feat_std: np.ndarray | None = None,
        use_s1: bool = False,
        pixel_zscore: bool = False,
        s1_despeckle_window: int = 0,
        feature_cols_override: list[str] | None = None,
        s1_feature_cols_override: list[str] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        presorted: bool = False,
        _log_rss: "Callable[[str], None] | None" = None,
    ) -> None:
        _probe = _log_rss if _log_rss is not None else (lambda _tag: None)

        if use_s1 is True:
            use_s1 = "mixed"
        if use_s1 not in (False, None, "mixed", "s1_only"):
            raise ValueError(f"use_s1 must be True/False, 'mixed', or 's1_only'; got {use_s1!r}")

        # Detect label type: pixel-year dict vs pixel dict
        self._labels_are_pixel_year: bool = False
        if labels is not None and len(labels) > 0:
            first_key = next(iter(labels))
            self._labels_are_pixel_year = isinstance(first_key, tuple)

        # If pixel-level dict, get keep-pids from keys; if pixel-year, from key[0]
        if labels is not None:
            if self._labels_are_pixel_year:
                _keep_pids: set[str] = {k[0] for k in labels}  # type: ignore[union-attr]
            else:
                _keep_pids = set(labels.keys())
        else:
            _keep_pids = set()

        # Build observation sequence depending on mode
        if use_s1 == "mixed":
            if "source" not in pixel_df.columns or "S1" not in pixel_df["source"].unique().to_list():
                raise ValueError("mixed mode requires S1 rows in pixel_df (source='S1')")

            s2_rows = pixel_df.filter(pl.col("source") == "S2")
            s1_rows = pixel_df.filter(pl.col("source") == "S1")

            if "scl_purity" in s2_rows.columns:
                s2_rows = s2_rows.filter(pl.col("scl_purity") >= scl_purity_min)

            s1_rows = despeckle_s1(s1_rows, s1_despeckle_window)
            vh_lin = s1_rows["vh"].cast(pl.Float32).to_numpy() if "vh" in s1_rows.columns else np.full(len(s1_rows), np.nan, dtype=np.float32)
            vv_lin = s1_rows["vv"].cast(pl.Float32).to_numpy() if "vv" in s1_rows.columns else np.full(len(s1_rows), np.nan, dtype=np.float32)
            s1_vh    = lin_to_db(vh_lin).astype(np.float32)
            s1_vv    = lin_to_db(vv_lin).astype(np.float32)
            s1_vh_vv = s1_vh - s1_vv
            denom = vh_lin + vv_lin
            s1_rvi = np.where(denom > 0, 4 * vh_lin / denom, np.nan).astype(np.float32)
            s1_rows = s1_rows.with_columns([
                pl.Series("s1_vh",    s1_vh),
                pl.Series("s1_vv",    s1_vv),
                pl.Series("s1_vh_vv", s1_vh_vv),
                pl.Series("s1_rvi",   s1_rvi),
            ])
            _active_s1_cols = list(s1_feature_cols_override) if s1_feature_cols_override is not None else S1_FEATURE_COLS
            # Drop S1 rows where all active S1 bands are null
            s1_rows = s1_rows.filter(
                pl.any_horizontal([pl.col(c).is_not_null() for c in _active_s1_cols])
            )

            s2_feature_cols = list(feature_cols_override) if feature_cols_override is not None else ALL_FEATURE_COLS
            feature_cols = s2_feature_cols + _active_s1_cols
            for col in s2_feature_cols:
                if col not in s1_rows.columns:
                    s1_rows = s1_rows.with_columns(pl.lit(None).cast(pl.Float32).alias(col))
            for col in _active_s1_cols:
                if col not in s2_rows.columns:
                    s2_rows = s2_rows.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

            _s2_index_cols = [c for c in s2_feature_cols if c in INDEX_COLS]
            if any(c not in s2_rows.columns for c in _s2_index_cols):
                s2_rows = add_spectral_indices(s2_rows)

            # Slim to only the columns needed downstream before concat:
            # sort keys (point_id, year, date, doy, source) + feature_cols.
            # Drops scl_purity, vh, vv, scl, lon, lat which are no longer needed —
            # narrower frame means the sort copy allocates proportionally less.
            _keep_for_sort = {"point_id", "year", "date", "doy", "source"} | set(feature_cols)
            s2_rows = s2_rows.select([c for c in s2_rows.columns if c in _keep_for_sort])
            s1_rows = s1_rows.select([c for c in s1_rows.columns if c in _keep_for_sort])

            # Explicit del before concat: releases the two source frames before
            # the combined frame is alive, halving the peak RSS at this point.
            _concat_parts = [s2_rows, s1_rows]
            del s2_rows, s1_rows
            df = pl.concat(_concat_parts, how="diagonal_relaxed")
            del _concat_parts
            _probe(f"TAMDataset: after concat s2+s1 ({len(df):,} rows)")

            if pixel_zscore:
                s2_zscore_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
                s1_zscore_cols = ["s1_vh", "s1_vv"]
                for zscore_cols, src in [(s2_zscore_cols, "S2"), (s1_zscore_cols, "S1")]:
                    src_mask = pl.col("source") == src
                    if not df.filter(src_mask).height:
                        continue
                    stats = (
                        df.lazy()
                        .filter(src_mask)
                        .select(["point_id"] + zscore_cols)
                        .group_by("point_id")
                        .agg([pl.col(c).mean().alias(f"{c}__mean") for c in zscore_cols] +
                             [pl.col(c).std().alias(f"{c}__std")  for c in zscore_cols])
                    )
                    normed_exprs = [
                        pl.when(src_mask)
                          .then(
                              (pl.col(c) - pl.col(f"{c}__mean").fill_null(0.0)) /
                              pl.col(f"{c}__std").fill_null(1e-6).clip(lower_bound=1e-6)
                          )
                          .otherwise(pl.col(c))
                          .alias(c)
                        for c in zscore_cols
                    ]
                    stat_cols = [f"{c}__mean" for c in zscore_cols] + [f"{c}__std" for c in zscore_cols]
                    df = (
                        df.lazy()
                        .join(stats, on="point_id", how="left")
                        .with_columns(normed_exprs)
                        .drop(stat_cols)
                        .collect()
                    )

        elif use_s1 == "s1_only":
            s1_rows = pixel_df.filter(pl.col("source") == "S1") if "source" in pixel_df.columns else pl.DataFrame()
            if s1_rows.is_empty():
                raise ValueError("s1_only mode requires S1 rows in pixel_df (source='S1')")
            s1_rows = despeckle_s1(s1_rows, s1_despeckle_window)
            vh_lin = s1_rows["vh"].cast(pl.Float32).to_numpy()
            vv_lin = s1_rows["vv"].cast(pl.Float32).to_numpy()
            s1_vh    = lin_to_db(vh_lin).astype(np.float32)
            s1_vv    = lin_to_db(vv_lin).astype(np.float32)
            s1_vh_vv = s1_vh - s1_vv
            denom = vh_lin + vv_lin
            s1_rvi = np.where(denom > 0, 4 * vh_lin / denom, np.nan).astype(np.float32)
            s1_rows = s1_rows.with_columns([
                pl.Series("s1_vh",    s1_vh),
                pl.Series("s1_vv",    s1_vv),
                pl.Series("s1_vh_vv", s1_vh_vv),
                pl.Series("s1_rvi",   s1_rvi),
            ])
            df = s1_rows
            feature_cols = S1_FEATURE_COLS
            # Drop rows where all S1 bands are null
            df = df.filter(
                pl.any_horizontal([pl.col(c).is_not_null() for c in S1_FEATURE_COLS])
            )

            if pixel_zscore:
                s1_zscore_cols = [c for c in ("s1_vh", "s1_vv") if c in df.columns]
                if s1_zscore_cols:
                    stats = (
                        df.lazy()
                        .select(["point_id"] + s1_zscore_cols)
                        .group_by("point_id")
                        .agg([pl.col(c).mean().alias(f"{c}__mean") for c in s1_zscore_cols] +
                             [pl.col(c).std().alias(f"{c}__std")  for c in s1_zscore_cols])
                    )
                    normed_exprs = [
                        ((pl.col(c) - pl.col(f"{c}__mean").fill_null(0.0)) /
                         pl.col(f"{c}__std").fill_null(0.1).clip(lower_bound=0.1)).alias(c)
                        for c in s1_zscore_cols
                    ]
                    stat_cols = [f"{c}__mean" for c in s1_zscore_cols] + [f"{c}__std" for c in s1_zscore_cols]
                    df = (
                        df.lazy()
                        .join(stats, on="point_id", how="left")
                        .with_columns(normed_exprs)
                        .drop(stat_cols)
                        .collect()
                    )
        else:
            feature_cols = feature_cols_override if feature_cols_override is not None else ALL_FEATURE_COLS

            # Filter to labeled pixels first
            if labels is not None:
                src_filter = (pl.col("source") == "S2") if "source" in pixel_df.columns else pl.lit(True)
                df = pixel_df.filter(src_filter & pl.col("point_id").is_in(_keep_pids))
            else:
                if "source" in pixel_df.columns:
                    df = pixel_df.filter(pl.col("source") == "S2")
                else:
                    df = pixel_df

            if any(c not in df.columns for c in feature_cols if c in set(INDEX_COLS)):
                df = add_spectral_indices(df)

            if pixel_zscore:
                s2_zscore_cols = [c for c in feature_cols if c in df.columns]
                if s2_zscore_cols:
                    stats = (
                        df.lazy()
                        .select(["point_id"] + s2_zscore_cols)
                        .group_by("point_id")
                        .agg([pl.col(c).mean().alias(f"{c}__mean") for c in s2_zscore_cols] +
                             [pl.col(c).std().alias(f"{c}__std")  for c in s2_zscore_cols])
                    )
                    normed_exprs = [
                        ((pl.col(c) - pl.col(f"{c}__mean").fill_null(0.0)) /
                         pl.col(f"{c}__std").fill_null(1e-6).clip(lower_bound=1e-6)).alias(c)
                        for c in s2_zscore_cols
                    ]
                    stat_cols = [f"{c}__mean" for c in s2_zscore_cols] + [f"{c}__std" for c in s2_zscore_cols]
                    df = (
                        df.lazy()
                        .join(stats, on="point_id", how="left")
                        .with_columns(normed_exprs)
                        .drop(stat_cols)
                        .collect()
                    )

        if use_s1 not in ("s1_only", "mixed"):
            if any(c not in df.columns for c in feature_cols if c in set(INDEX_COLS)):
                df = add_spectral_indices(df)

            if "scl_purity" in df.columns:
                df = df.filter(pl.col("scl_purity") >= scl_purity_min)

            _band_check = [c for c in feature_cols if c in df.columns]
            null_mask = pl.any_horizontal([pl.col(c).is_null() for c in _band_check])
            if df.filter(null_mask).height > 0:
                df = df.filter(~null_mask)

        # For pixel-year labels, filter to labeled pixels (s1_only path only —
        # mixed path already slimmed above; S2-only path filtered at construction).
        # When called from train_tam, pixel_df is pre-filtered to labeled pids by the
        # caller, so this is a no-op for the train/val split paths. It remains here
        # as a safety guard for external callers that pass an unfiltered pixel_df.
        if not hasattr(self, "_labels_are_pixel_year"):
            self._labels_are_pixel_year = (
                labels is not None and len(labels) > 0
                and isinstance(next(iter(labels)), tuple)
            )
        if labels is not None and use_s1 == "s1_only":
            df = df.filter(pl.col("point_id").is_in(_keep_pids))

        # Compute band stats from training data if not supplied.
        # In mixed mode, compute per-source to avoid materialising a huge sparse matrix:
        # S2 cols are NaN in S1 rows and vice versa, so nanmean/nanstd per column is correct
        # but we avoid the full dense allocation by computing each column independently.
        _band_check_cols = [c for c in feature_cols if c in df.columns]
        if band_mean is None or band_std is None:
            if use_s1 == "mixed" and "source" in df.columns:
                _s1_col_set = set(_active_s1_cols)
                _source_stats = (
                    df.lazy()
                    .group_by("source")
                    .agg(
                        [pl.col(c).mean().alias(f"{c}__mean") for c in _band_check_cols] +
                        [pl.col(c).std().alias(f"{c}__std")  for c in _band_check_cols]
                    )
                    .collect()
                )
                _s2_stats = _source_stats.filter(pl.col("source") == "S2")
                _s1_stats = _source_stats.filter(pl.col("source") == "S1")
                band_mean = np.array([
                    (_s1_stats[f"{c}__mean"][0] if c in _s1_col_set else _s2_stats[f"{c}__mean"][0])
                    for c in _band_check_cols
                ], dtype=np.float32)
                band_std = np.array([
                    (_s1_stats[f"{c}__std"][0] if c in _s1_col_set else _s2_stats[f"{c}__std"][0])
                    for c in _band_check_cols
                ], dtype=np.float32)
            else:
                _col_stats = (
                    df.lazy()
                    .select(
                        [pl.col(c).mean().alias(f"{c}__mean") for c in _band_check_cols] +
                        [pl.col(c).std().alias(f"{c}__std")   for c in _band_check_cols]
                    )
                    .collect()
                )
                band_mean = np.array([_col_stats[f"{c}__mean"][0] for c in _band_check_cols], dtype=np.float32)
                band_std  = np.array([_col_stats[f"{c}__std"][0]  for c in _band_check_cols], dtype=np.float32)
            band_mean = np.where(np.isnan(band_mean), 0.0, band_mean)
            band_std  = np.where(band_std < 1e-6, 1.0, band_std)

        self.band_mean = band_mean.astype(np.float32)
        self.band_std  = band_std.astype(np.float32)

        # Add DOY before sort so it's available in the sorted frame.
        if "doy" not in df.columns:
            df = df.with_columns(
                pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
            )

        _is_mixed = (use_s1 == "mixed")

        # Slim to sort-key + feature cols before the expensive sort.
        # Non-mixed paths may still carry scl_purity, vh, vv, scl which are no
        # longer needed. Narrower frame = smaller sort copy.
        _keep_for_sort = {"point_id", "year", "date", "doy", "source"} | set(feature_cols)
        _surplus = [c for c in df.columns if c not in _keep_for_sort]
        if _surplus:
            df = df.drop(_surplus)

        # Step 1: sort — required so truncation keeps the earliest observations.
        # Skip when caller guarantees data is already sorted by (point_id, year, date).
        # Not safe to skip for mixed mode: concat(s2, s1) interleaves sources, breaking order.
        _probe(f"TAMDataset: before sort ({len(df):,} rows)")
        if not presorted or _is_mixed:
            df = df.sort(["point_id", "year", "date"])
        if "date" in df.columns:
            df = df.drop("date")
        gc.collect()  # release pre-sort arena promptly
        _probe("TAMDataset: after sort")

        # Step 2: min-obs filter — drop windows that don't meet the observation
        # threshold. Use group_by to count once, then semi-join to filter rows.
        if _is_mixed:
            _group_counts = (
                df.group_by(["point_id", "year"], maintain_order=True)
                .agg([
                    pl.len().alias("n_total"),
                    (pl.col("source") == "S1").sum().cast(pl.Int32).alias("n_s1"),
                ])
                .with_columns((pl.col("n_total") - pl.col("n_s1")).alias("n_s2"))
            )
            _group_keep = _group_counts.filter(
                (pl.col("n_s2") >= min_obs_per_year) &
                (pl.col("n_s1") >= MIN_S1_OBS_PER_YEAR)
            ).select(["point_id", "year"])
        else:
            _group_keep = (
                df.group_by(["point_id", "year"], maintain_order=True)
                .len()
                .filter(pl.col("len") >= min_obs_per_year)
                .select(["point_id", "year"])
            )

        df = df.join(_group_keep, on=["point_id", "year"], how="inner")
        del _group_keep

        sizes = (
            df.group_by(["point_id", "year"], maintain_order=True)
            .len()["len"]
            .to_numpy()
        )

        # Step 3: truncate to max_seq_len in Polars — only when any group actually
        # exceeds the cap (avoids the overhead of int_range().over() when all groups
        # are already short, which is the common case after scl_purity filtering).
        _max_grp = int(sizes.max()) if len(sizes) else 0
        if _max_grp > max_seq_len:
            df = (
                df.lazy()
                .with_columns(
                    pl.int_range(pl.len(), dtype=pl.Int32)
                    .over(["point_id", "year"])
                    .alias("_row_nr")
                )
                .filter(pl.col("_row_nr") < max_seq_len)
                .drop("_row_nr")
                .collect()
            )
            sizes = (
                df.group_by(["point_id", "year"], maintain_order=True)
                .len()["len"]
                .to_numpy()
            )

        _probe(f"TAMDataset: after sort+filter+truncate ({len(df):,} rows)")

        # Materialise to numpy.  Polars float32 columns return float32 numpy arrays
        # directly — no extra copy needed.
        _probe(f"TAMDataset: before to_numpy ({len(df):,} rows × {len(feature_cols)} feats)")
        raw_arr = df.select(feature_cols).to_numpy()
        feat_arr = raw_arr if raw_arr.dtype == np.float32 else raw_arr.astype(np.float32)
        np.subtract(feat_arr, self.band_mean, out=feat_arr)
        np.divide(feat_arr, self.band_std, out=feat_arr)
        np.nan_to_num(feat_arr, nan=0.0, copy=False)
        bands_all  = feat_arr
        doy_all    = df["doy"].to_numpy().astype(np.int32)
        pid_all    = df["point_id"].to_numpy()
        yr_all     = df["year"].to_numpy()
        if "source" in df.columns:
            source_all = (df["source"].to_numpy() == "S1").astype(np.int8)
        else:
            source_all = np.zeros(len(df), dtype=np.int8)

        del df, raw_arr
        gc.collect()
        _probe(f"TAMDataset: after to_numpy + del df ({len(bands_all):,} rows)")

        # sizes are already ≤ max_seq_len — truncation happened in Polars above.
        n_windows = len(sizes)
        n_feat    = len(feature_cols)
        starts    = np.empty(n_windows, dtype=np.int64)
        if n_windows:
            starts[0] = 0
            np.cumsum(sizes[:-1], out=starts[1:])

        self._bands   = bands_all
        self._doys    = doy_all
        self._sources = source_all
        self._lengths = sizes.astype(np.int32)
        self._offsets = starts.astype(np.int32)
        self._pids    = pid_all[starts] if n_windows else np.empty(0, dtype=pid_all.dtype)
        self._years   = yr_all[starts].astype(np.int32) if n_windows else np.empty(0, dtype=np.int32)

        del bands_all, doy_all, source_all
        _probe(f"TAMDataset: after truncation ({len(self._pids):,} windows, {len(self._bands):,} rows kept)")

        # For pixel-year labels, drop windows not in the label set.
        # Use string encoding for fast set membership (avoids tuple-hash overhead).
        if labels is not None and self._labels_are_pixel_year:
            valid_py   = set(labels.keys())
            valid_keys = {f"{p}\x00{y}" for p, y in valid_py}
            window_keys = np.array([f"{p}\x00{y}" for p, y in zip(self._pids, self._years)])
            keep     = np.array([k in valid_keys for k in window_keys], dtype=bool)
            kept_idx = np.where(keep)[0]
            # Fast path: if every window survived the label filter, skip the copy.
            if len(kept_idx) == len(self._pids):
                pass
            elif len(kept_idx):
                lens = self._lengths[kept_idx]
                # Vectorised row mask: map each kept window's rows via np.repeat.
                # kept_offsets_old[i] + arange(lens[i]) gives row indices for window i;
                # np.repeat + cumsum avoids a Python loop over windows.
                kept_offsets_old = self._offsets[kept_idx]
                row_indices = (
                    np.repeat(kept_offsets_old.astype(np.int64), lens) +
                    np.arange(lens.sum(), dtype=np.int64) -
                    np.repeat(np.concatenate([[0], np.cumsum(lens[:-1])]), lens)
                )
                row_mask = np.zeros(len(self._bands), dtype=bool)
                row_mask[row_indices] = True
                new_offsets = (np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32)
                               if len(kept_idx) > 1 else np.array([0], dtype=np.int32))
                self._bands   = self._bands[row_mask]
                self._doys    = self._doys[row_mask]
                self._sources = self._sources[row_mask]
                self._offsets = new_offsets
                self._lengths = lens.astype(np.int32)
                self._pids    = self._pids[kept_idx]
                self._years   = self._years[kept_idx]
            else:
                self._bands   = self._bands[:0]
                self._doys    = self._doys[:0]
                self._sources = self._sources[:0]
                self._offsets = np.empty(0, dtype=np.int32)
                self._lengths = np.empty(0, dtype=np.int32)
                self._pids    = self._pids[:0]
                self._years   = self._years[:0]
        # For pixel-level labels, broadcast: auto-expands when __getitem__ is called
        # (stored as dict[str, float]; lookup by pid works for both dict types)
        _probe(f"TAMDataset: after label-filter ({len(self._pids):,} windows final)")

        self._n_features = len(feature_cols)
        self._labels = labels
        self._doy_jitter = doy_jitter
        self._doy_phase_shift = doy_phase_shift
        self._band_noise_std = band_noise_std
        self._obs_dropout_min = obs_dropout_min
        self._max_seq_len = max_seq_len

        # Global features: per-pixel scalars z-scored and stored as float32 arrays
        if global_features_df is not None and len(global_features_df) > 0:
            feat_cols = [c for c in global_features_df.columns if c != "point_id"]
            n_global = len(feat_cols)

            gf_numpy  = global_features_df.select(feat_cols).to_numpy().astype(np.float32)
            gf_pids   = global_features_df["point_id"].to_numpy()
            sorter    = np.argsort(gf_pids)
            gf_pids_s = gf_pids[sorter]
            gf_numpy_s = gf_numpy[sorter]

            # Vectorised lookup: searchsorted + validity mask — no Python loop
            n_windows = len(self._pids)
            pos   = np.searchsorted(gf_pids_s, self._pids)
            pos   = np.clip(pos, 0, len(gf_pids_s) - 1)
            found = gf_pids_s[pos] == self._pids
            feat_vals = np.where(found[:, None], gf_numpy_s[pos], np.nan).astype(np.float32)

            if global_feat_mean is None or global_feat_std is None:
                global_feat_mean = np.nanmean(feat_vals, axis=0)
                global_feat_std  = np.nanstd(feat_vals, axis=0)
                global_feat_mean = np.where(np.isnan(global_feat_mean), 0.0, global_feat_mean)
                global_feat_std  = np.where(global_feat_std < 1e-6, 1.0, global_feat_std)

            self.global_feat_mean = global_feat_mean.astype(np.float32)
            self.global_feat_std  = global_feat_std.astype(np.float32)

            normed = (feat_vals - self.global_feat_mean) / self.global_feat_std
            self._global_feat_arr = np.where(np.isnan(normed), 0.0, normed).astype(np.float32)
            self._n_global = n_global
        else:
            self.global_feat_mean = np.zeros(0, dtype=np.float32)
            self.global_feat_std  = np.ones(0,  dtype=np.float32)
            self._global_feat_arr = np.empty((0, 0), dtype=np.float32)
            self._n_global = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pids)

    def __getitem__(self, idx: int) -> TAMSample:
        pid   = self._pids[idx]
        yr    = int(self._years[idx])
        start = int(self._offsets[idx])
        n_obs = int(self._lengths[idx])
        bands_np = self._bands  [start:start+n_obs].copy()
        doy_np   = self._doys   [start:start+n_obs].copy()
        src_np   = self._sources[start:start+n_obs].copy()
        n = n_obs

        seq_cap = self._max_seq_len
        if n > seq_cap or (self._obs_dropout_min > 0 and n > self._obs_dropout_min):
            lo = self._obs_dropout_min if self._obs_dropout_min > 0 else seq_cap
            lo = min(lo, seq_cap)
            hi = min(n, seq_cap)
            keep = np.random.randint(lo, hi + 1) if lo < hi else hi
            idx_keep = np.sort(np.random.choice(n, keep, replace=False))
            bands_np = bands_np[idx_keep]
            doy_np   = doy_np[idx_keep]
            src_np   = src_np[idx_keep]
            n        = keep

        if self._band_noise_std > 0.0:
            n_s2 = min(len(ALL_FEATURE_COLS), self._n_features)
            n_s1 = self._n_features - n_s2
            if n_s1 > 0 and src_np.any():
                noise  = np.random.randn(n, self._n_features).astype(np.float32) * self._band_noise_std
                is_s1  = src_np.astype(bool)
                noise[~is_s1, n_s2:] = 0.0
                noise[is_s1,  :n_s2] = 0.0
                bands_np = bands_np + noise
            else:
                offset = np.zeros(self._n_features, dtype=np.float32)
                offset[:n_s2] = np.random.randn(n_s2).astype(np.float32) * self._band_noise_std
                bands_np = bands_np + offset

        doy_vals = doy_np.astype(np.int64)
        if self._doy_phase_shift:
            phase_offset = np.random.randint(0, 365)
            doy_vals = ((doy_vals - 1 + phase_offset) % 365) + 1
            sort_idx = np.argsort(doy_vals, kind="stable")
            doy_vals = doy_vals[sort_idx]
            bands_np = bands_np[sort_idx]
            src_np   = src_np[sort_idx]
        elif self._doy_jitter > 0:
            offset = np.random.randint(-self._doy_jitter, self._doy_jitter + 1)
            doy_vals = np.clip(doy_vals + offset, 1, 365)

        bands = np.zeros((seq_cap, self._n_features), dtype=np.float32)
        bands[:n] = bands_np

        doy = np.zeros(seq_cap, dtype=np.int64)
        doy[:n] = doy_vals

        mask = np.ones(seq_cap, dtype=bool)
        mask[:n] = False

        is_s1 = np.zeros(seq_cap, dtype=bool)
        is_s1[:n] = src_np.astype(bool)

        if self._labels is None:
            label = 0.0
        elif self._labels_are_pixel_year:
            label = float(self._labels.get((pid, yr), 0.0))  # type: ignore[union-attr]
        else:
            label = float(self._labels[pid])  # type: ignore[index]
        weight = 1.0

        gf = self._global_feat_arr[idx] if self._n_global > 0 else np.zeros(0, dtype=np.float32)

        return TAMSample(
            bands        = torch.from_numpy(bands),
            doy          = torch.from_numpy(doy),
            mask         = torch.from_numpy(mask),
            n_obs        = torch.tensor(n / seq_cap, dtype=torch.float32),
            global_feats = torch.from_numpy(gf),
            label        = torch.tensor(label,  dtype=torch.float32),
            weight       = torch.tensor(weight, dtype=torch.float32),
            is_s1        = torch.from_numpy(is_s1),
            point_id     = pid,
            year         = yr,
        )

    # ------------------------------------------------------------------
    def to_files(self, path: Path) -> None:
        """Write all numpy arrays + metadata to *path* (must be an empty directory).

        Used by the subprocess-isolation path in train_tam: the child process
        builds TAMDataset (paying the jemalloc arena cost), calls to_files, then
        exits — the OS reclaims all arenas unconditionally.  The parent reloads
        with from_files, which only touches the small output arrays.
        """
        import json
        path = Path(path)
        np.save(path / "bands.npy",   self._bands)
        np.save(path / "doys.npy",    self._doys)
        np.save(path / "sources.npy", self._sources)
        np.save(path / "lengths.npy", self._lengths)
        np.save(path / "offsets.npy", self._offsets)
        np.save(path / "pids.npy",    self._pids)
        np.save(path / "years.npy",   self._years)
        np.save(path / "band_mean.npy", self.band_mean)
        np.save(path / "band_std.npy",  self.band_std)
        np.save(path / "global_feat_mean.npy", self.global_feat_mean)
        np.save(path / "global_feat_std.npy",  self.global_feat_std)
        if self._n_global > 0:
            np.save(path / "global_feat_arr.npy", self._global_feat_arr)
        meta = {
            "n_features":      self._n_features,
            "n_global":        self._n_global,
            "doy_jitter":      self._doy_jitter,
            "doy_phase_shift": self._doy_phase_shift,
            "band_noise_std":  self._band_noise_std,
            "obs_dropout_min": self._obs_dropout_min,
            "max_seq_len":     self._max_seq_len,
            "labels_are_pixel_year": self._labels_are_pixel_year,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def from_files(
        cls,
        path: Path,
        labels: "dict[tuple[str, int], float] | dict[str, float] | None",
        doy_jitter: int = 0,
        doy_phase_shift: bool = False,
        band_noise_std: float = 0.0,
        obs_dropout_min: int = 0,
    ) -> "TAMDataset":
        """Reconstruct a TAMDataset from files written by to_files.

        Does not touch Polars — only numpy loads.  Called in the parent process
        after the child (which paid the Polars arena cost) has exited.
        """
        import json
        path = Path(path)
        obj = cls.__new__(cls)

        obj._bands   = np.load(path / "bands.npy",   mmap_mode="r+")
        obj._doys    = np.load(path / "doys.npy",    mmap_mode="r+")
        obj._sources = np.load(path / "sources.npy", mmap_mode="r+")
        obj._lengths = np.load(path / "lengths.npy", mmap_mode="r+")
        obj._offsets = np.load(path / "offsets.npy", mmap_mode="r+")
        obj._pids    = np.load(path / "pids.npy",    allow_pickle=True)
        obj._years   = np.load(path / "years.npy",   mmap_mode="r+")
        obj.band_mean = np.load(path / "band_mean.npy")
        obj.band_std  = np.load(path / "band_std.npy")
        obj.global_feat_mean = np.load(path / "global_feat_mean.npy")
        obj.global_feat_std  = np.load(path / "global_feat_std.npy")

        with open(path / "meta.json") as f:
            meta = json.load(f)

        obj._n_features      = meta["n_features"]
        obj._n_global        = meta["n_global"]
        obj._doy_jitter      = doy_jitter
        obj._doy_phase_shift = doy_phase_shift
        obj._band_noise_std  = band_noise_std
        obj._obs_dropout_min = obs_dropout_min
        obj._max_seq_len     = meta["max_seq_len"]
        obj._labels_are_pixel_year = meta["labels_are_pixel_year"]
        obj._labels = labels

        if obj._n_global > 0:
            obj._global_feat_arr = np.load(path / "global_feat_arr.npy", mmap_mode="r+")
        else:
            obj._global_feat_arr = np.empty((0, 0), dtype=np.float32)

        return obj

    # ------------------------------------------------------------------
    @classmethod
    def merge_shards(
        cls,
        shards: "list[TAMDataset]",
        labels: "dict[tuple[str, int], float] | dict[str, float] | None",
        band_mean: np.ndarray,
        band_std: np.ndarray,
        global_feat_mean: np.ndarray,
        global_feat_std: np.ndarray,
        doy_jitter: int = 0,
        doy_phase_shift: bool = False,
        band_noise_std: float = 0.0,
        obs_dropout_min: int = 0,
    ) -> "TAMDataset":
        """Merge a list of TAMDataset shards into a single dataset.

        Each shard was built with shared normalisation stats (band_mean/std,
        global_feat_mean/std), so the per-band values are already in the same
        space.  Merging is pure numpy: concatenate flat arrays and fix offsets.
        """
        if not shards:
            raise ValueError("merge_shards requires at least one shard")

        obj = cls.__new__(cls)

        # Concatenate observation-level arrays
        obj._bands   = np.concatenate([s._bands   for s in shards], axis=0)
        obj._doys    = np.concatenate([s._doys    for s in shards])
        obj._sources = np.concatenate([s._sources for s in shards])

        # Concatenate window-level arrays, adjusting offsets by cumulative band count
        lengths_list = [s._lengths for s in shards]
        pids_list    = [s._pids    for s in shards]
        years_list   = [s._years   for s in shards]

        # Compute per-shard band offset (number of obs rows before this shard)
        band_cumsum = np.concatenate([[0], np.cumsum([len(s._bands) for s in shards[:-1]])])
        offsets_list = [s._offsets + int(band_cumsum[i]) for i, s in enumerate(shards)]

        obj._lengths = np.concatenate(lengths_list)
        obj._offsets = np.concatenate(offsets_list)
        obj._pids    = np.concatenate(pids_list)
        obj._years   = np.concatenate(years_list)

        # Global features: concatenate per-window rows (already normalised with shared stats)
        if shards[0]._n_global > 0:
            obj._global_feat_arr = np.concatenate([s._global_feat_arr for s in shards], axis=0)
            obj._n_global = shards[0]._n_global
        else:
            obj._global_feat_arr = np.empty((0, 0), dtype=np.float32)
            obj._n_global = 0

        # Shared stats (passed in from the stats subprocess)
        obj.band_mean        = band_mean.astype(np.float32)
        obj.band_std         = band_std.astype(np.float32)
        obj.global_feat_mean = global_feat_mean.astype(np.float32)
        obj.global_feat_std  = global_feat_std.astype(np.float32)

        # Training hyperparams
        obj._n_features      = shards[0]._n_features
        obj._doy_jitter      = doy_jitter
        obj._doy_phase_shift = doy_phase_shift
        obj._band_noise_std  = band_noise_std
        obj._obs_dropout_min = obs_dropout_min
        obj._max_seq_len     = shards[0]._max_seq_len
        obj._labels_are_pixel_year = shards[0]._labels_are_pixel_year
        obj._labels          = labels

        return obj

    # ------------------------------------------------------------------
    @property
    def band_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return self.band_mean.copy(), self.band_std.copy()

    def unique_pixels(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for pid in self._pids:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out
