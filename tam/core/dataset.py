"""tam/core/dataset.py — PyTorch dataset for TAM temporal attention model.

Each dataset item is one (pixel, year) annual window: a padded sequence of
Sentinel-2 observations with DOY positional indices and a binary presence label.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import NamedTuple

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
MAX_SEQ_LEN: int = 128
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
    ) -> None:
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

            df = pl.concat([s2_rows, s1_rows], how="diagonal_relaxed")

            if labels is not None:
                df = df.filter(pl.col("point_id").is_in(_keep_pids))

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

        # For pixel-year labels, filter to labeled pixels (s1_only/mixed paths)
        if not hasattr(self, "_labels_are_pixel_year"):
            self._labels_are_pixel_year = (
                labels is not None and len(labels) > 0
                and isinstance(next(iter(labels)), tuple)
            )
        if labels is not None and use_s1 in ("s1_only", "mixed"):
            df = df.filter(pl.col("point_id").is_in(_keep_pids))

        # Compute band stats from training data if not supplied.
        # In mixed mode, compute per-source to avoid materialising a huge sparse matrix:
        # S2 cols are NaN in S1 rows and vice versa, so nanmean/nanstd per column is correct
        # but we avoid the full dense allocation by computing each column independently.
        _band_check_cols = [c for c in feature_cols if c in df.columns]
        if band_mean is None or band_std is None:
            if use_s1 == "mixed" and "source" in df.columns:
                _s1_col_set = set(_active_s1_cols)
                band_mean = np.array([
                    df.filter(pl.col("source") == ("S1" if c in _s1_col_set else "S2"))[c]
                    .drop_nulls().mean()
                    for c in _band_check_cols
                ], dtype=np.float32)
                band_std = np.array([
                    df.filter(pl.col("source") == ("S1" if c in _s1_col_set else "S2"))[c]
                    .drop_nulls().std()
                    for c in _band_check_cols
                ], dtype=np.float32)
            else:
                vals = df.select(_band_check_cols).to_numpy().astype(np.float32)
                band_mean = np.nanmean(vals, axis=0)
                band_std  = np.nanstd(vals, axis=0)
            band_mean = np.where(np.isnan(band_mean), 0.0, band_mean)
            band_std  = np.where(band_std < 1e-6, 1.0, band_std)

        self.band_mean = band_mean.astype(np.float32)
        self.band_std  = band_std.astype(np.float32)

        # Sort and compute DOY
        df = df.sort(["point_id", "year", "date"])

        if "doy" not in df.columns:
            df = df.with_columns(
                pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
            )

        # Vectorised extract: normalise bands in bulk, split on group boundaries.
        # Use in-place ops to avoid a second full-size allocation.
        feat_arr = df.select(feature_cols).to_numpy().astype(np.float32)
        np.subtract(feat_arr, self.band_mean, out=feat_arr)
        np.divide(feat_arr, self.band_std, out=feat_arr)
        np.nan_to_num(feat_arr, nan=0.0, copy=False)
        bands_all = feat_arr
        doy_all   = df["doy"].to_numpy().astype(np.int32)
        pid_all   = df["point_id"].to_numpy()
        yr_all    = df["year"].to_numpy()
        if "source" in df.columns:
            source_all = (df["source"].to_numpy() == "S1").astype(np.int8)
        else:
            source_all = np.zeros(len(df), dtype=np.int8)

        _is_mixed = (use_s1 == "mixed")
        sizes = (
            df.group_by(["point_id", "year"], maintain_order=True)
            .len()["len"]
            .to_numpy()
        )
        split_points = np.cumsum(sizes)[:-1]
        del df
        gc.collect()

        bands_split  = np.split(bands_all,  split_points)
        doy_split    = np.split(doy_all,    split_points)
        pid_split    = np.split(pid_all,    split_points)
        yr_split     = np.split(yr_all,     split_points)
        source_split = np.split(source_all, split_points)

        _w_pids: list[str] = []
        _w_years: list[int] = []
        _w_offsets: list[int] = []
        _w_lengths: list[int] = []
        _out_bands: list[np.ndarray] = []
        _out_doys: list[np.ndarray] = []
        _out_sources: list[np.ndarray] = []

        write_ptr = 0
        for b, d, p, y, src in zip(bands_split, doy_split, pid_split, yr_split, source_split):
            if _is_mixed:
                n_s2 = int((src == 0).sum())
                n_s1 = int((src == 1).sum())
                if n_s2 < min_obs_per_year or n_s1 < MIN_S1_OBS_PER_YEAR:
                    continue
            else:
                if len(b) < min_obs_per_year:
                    continue
            n = min(len(b), MAX_SEQ_LEN)
            _out_bands.append(b[:n])
            _out_doys.append(d[:n])
            _out_sources.append(src[:n])
            _w_pids.append(p[0])
            _w_years.append(int(y[0]))
            _w_offsets.append(write_ptr)
            _w_lengths.append(n)
            write_ptr += n

        n_feat = len(feature_cols)
        self._bands   = np.concatenate(_out_bands,   axis=0) if _out_bands   else np.empty((0, n_feat), dtype=np.float32)
        self._doys    = np.concatenate(_out_doys,    axis=0) if _out_doys    else np.empty(0, dtype=np.int32)
        self._sources = np.concatenate(_out_sources, axis=0) if _out_sources else np.empty(0, dtype=np.int8)
        self._offsets = np.array(_w_offsets, dtype=np.int32)
        self._lengths = np.array(_w_lengths, dtype=np.int32)
        self._pids    = np.array(_w_pids,    dtype=object)
        self._years   = np.array(_w_years,   dtype=np.int32)

        # For pixel-year labels, drop windows not in the label set
        if labels is not None and self._labels_are_pixel_year:
            valid_py = set(labels.keys())
            keep = np.array(
                [(self._pids[i], int(self._years[i])) in valid_py for i in range(len(self._pids))],
                dtype=bool,
            )
            kept_idx = np.where(keep)[0]
            if len(kept_idx):
                new_bands   = np.concatenate([self._bands  [self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0)
                new_doys    = np.concatenate([self._doys   [self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0)
                new_sources = np.concatenate([self._sources[self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0)
                lens = self._lengths[kept_idx]
                new_offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32) if len(kept_idx) > 1 else np.array([0], dtype=np.int32)
            else:
                new_bands   = self._bands[:0]
                new_doys    = self._doys[:0]
                new_sources = self._sources[:0]
                new_offsets = np.empty(0, dtype=np.int32)
                lens        = np.empty(0, dtype=np.int32)
            self._bands   = new_bands
            self._doys    = new_doys
            self._sources = new_sources
            self._offsets = new_offsets
            self._lengths = lens if len(kept_idx) else np.empty(0, dtype=np.int32)
            self._pids    = self._pids[kept_idx]
            self._years   = self._years[kept_idx]
        # For pixel-level labels, broadcast: auto-expands when __getitem__ is called
        # (stored as dict[str, float]; lookup by pid works for both dict types)

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

            gf_lookup: dict[str, int] = {
                pid: i for i, pid in enumerate(global_features_df["point_id"].to_list())
            }
            n_windows = len(self._pids)
            feat_vals = np.full((n_windows, n_global), np.nan, dtype=np.float32)
            gf_numpy  = global_features_df.select(feat_cols).to_numpy().astype(np.float32)
            for i, pid in enumerate(self._pids):
                if pid in gf_lookup:
                    feat_vals[i] = gf_numpy[gf_lookup[pid]]

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
