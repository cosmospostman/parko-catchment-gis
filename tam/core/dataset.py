"""tam/core/dataset.py — PyTorch dataset for TAM temporal attention model.

Each dataset item is one (pixel, year) annual window: a padded sequence of
Sentinel-2 observations with DOY positional indices and a binary presence label.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SPECTRAL_INDEX_COLS, add_spectral_indices

BAND_COLS: list[str] = list(BANDS)   # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
INDEX_COLS: list[str] = SPECTRAL_INDEX_COLS
ALL_FEATURE_COLS: list[str] = BAND_COLS + INDEX_COLS
N_BANDS: int = len(ALL_FEATURE_COLS)  # 13 (10 bands + NDVI + NDWI + EVI)
MAX_SEQ_LEN: int = 128       # canonical value lives in TAMConfig; kept here for model.py import
MIN_OBS_PER_YEAR: int = 8


class TAMSample(NamedTuple):
    bands:        torch.Tensor   # (MAX_SEQ_LEN, N_BANDS)  float32, normalised, zero-padded
    doy:          torch.Tensor   # (MAX_SEQ_LEN,)           int64, 1–365, 0=padding
    mask:         torch.Tensor   # (MAX_SEQ_LEN,)           bool, True=padding
    n_obs:        torch.Tensor   # ()                       float32, n / MAX_SEQ_LEN
    global_feats: torch.Tensor   # (N_GLOBAL,)              float32, z-scored global features
    label:        torch.Tensor   # ()                       float32 {0, 1}
    weight:       torch.Tensor   # ()                       float32 confidence weight
    point_id: str
    year:     int


def collate_fn(samples: list[TAMSample]) -> dict:
    return {
        "bands":        torch.stack([s.bands        for s in samples]),
        "doy":          torch.stack([s.doy          for s in samples]),
        "mask":         torch.stack([s.mask         for s in samples]),
        "n_obs":        torch.stack([s.n_obs        for s in samples]),
        "global_feats": torch.stack([s.global_feats for s in samples]),
        "label":        torch.stack([s.label        for s in samples]),
        "weight":       torch.stack([s.weight       for s in samples]),
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
        Series indexed by point_id, values in {0.0, 1.0}. Points not in
        this index are excluded from the dataset (inference mode: pass None).
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
        pixel_df: pd.DataFrame,
        labels: pd.Series | None,
        band_mean: np.ndarray | None = None,
        band_std: np.ndarray | None = None,
        scl_purity_min: float = 0.5,
        min_obs_per_year: int = MIN_OBS_PER_YEAR,
        doy_jitter: int = 0,
        band_noise_std: float = 0.0,
        obs_dropout_min: int = 0,
        global_features_df: pd.DataFrame | None = None,
        global_feat_mean: np.ndarray | None = None,
        global_feat_std: np.ndarray | None = None,
    ) -> None:
        # doy_jitter: max ±days to shift all observations in a window (training only).
        # A single offset is drawn per __getitem__ call and applied uniformly so
        # relative timing between observations is preserved.  Set 0 to disable.
        # band_noise_std: std of Gaussian noise added to normalised band values per
        # observation independently (training only).  Set 0.0 to disable.
        # obs_dropout_min: if >0, subsample each window to Uniform(obs_dropout_min, n)
        # per __getitem__ call, teaching the model to be invariant to obs density.
        if any(c not in pixel_df.columns for c in INDEX_COLS):
            df = add_spectral_indices(pixel_df)
        else:
            df = pixel_df
        feature_cols = ALL_FEATURE_COLS

        # SCL filter
        if "scl_purity" in df.columns:
            df = df[df["scl_purity"] >= scl_purity_min]

        # Drop rows with NaN in any band (skip scan if data is clean)
        if df[BAND_COLS].isna().any().any():
            df = df.dropna(subset=BAND_COLS)

        # Restrict to labeled pixels when labels provided
        if labels is not None:
            df = df[df["point_id"].isin(labels.index)]

        # Compute band stats from training data if not supplied
        if band_mean is None or band_std is None:
            vals = df[feature_cols].values.astype(np.float32)
            band_mean = np.nanmean(vals, axis=0)
            band_std  = np.nanstd(vals, axis=0)
            band_std  = np.where(band_std < 1e-6, 1.0, band_std)

        self.band_mean = band_mean.astype(np.float32)
        self.band_std  = band_std.astype(np.float32)

        # Build index: pre-extract bands and DOY as numpy arrays to avoid
        # pandas overhead in __getitem__ (called from DataLoader workers on CPU).
        #
        # Vectorised path: sort once, normalise in bulk, then split on group
        # boundaries — avoids per-group Python overhead with 100k+ windows.
        df = df.sort_values(["point_id", "year", "date"])

        if "doy" not in df.columns:
            df = df.copy()
            df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

        bands_all = ((df[feature_cols].values.astype(np.float32) - self.band_mean) / self.band_std)
        doy_all   = df["doy"].values.astype(np.int32)
        pid_all   = df["point_id"].values
        yr_all    = df["year"].values

        group_sizes = df.groupby(["point_id", "year"], sort=False).size()
        split_points = np.cumsum(group_sizes.values)[:-1]

        bands_split = np.split(bands_all, split_points)
        doy_split   = np.split(doy_all,   split_points)
        pid_split   = np.split(pid_all,   split_points)
        yr_split    = np.split(yr_all,    split_points)

        self._windows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        for b, d, p, y in zip(bands_split, doy_split, pid_split, yr_split):
            if len(b) < min_obs_per_year:
                continue
            n = min(len(b), MAX_SEQ_LEN)
            self._windows.append((p[0], int(y[0]), b[:n], d[:n]))

        self._labels = labels
        self._doy_jitter = doy_jitter
        self._band_noise_std = band_noise_std
        self._obs_dropout_min = obs_dropout_min

        # Global features: per-pixel scalars z-scored and stored as float32 arrays
        if global_features_df is not None and len(global_features_df) > 0:
            n_global = global_features_df.shape[1]
            feat_vals = global_features_df.reindex(
                [pid for pid, *_ in self._windows]
            ).values.astype(np.float32)  # (n_windows, n_global) — NaN for missing

            if global_feat_mean is None or global_feat_std is None:
                # Training mode: compute stats ignoring NaN
                global_feat_mean = np.nanmean(feat_vals, axis=0)
                global_feat_std  = np.nanstd(feat_vals, axis=0)
                global_feat_std  = np.where(global_feat_std < 1e-6, 1.0, global_feat_std)

            self.global_feat_mean = global_feat_mean.astype(np.float32)
            self.global_feat_std  = global_feat_std.astype(np.float32)

            # Z-score; impute NaN → 0 (feature mean after normalisation)
            normed = (feat_vals - self.global_feat_mean) / self.global_feat_std
            normed = np.where(np.isnan(normed), 0.0, normed).astype(np.float32)
            self._global_feats = {
                pid: normed[i] for i, (pid, *_) in enumerate(self._windows)
            }
            self._n_global = n_global
        else:
            self.global_feat_mean = np.zeros(0, dtype=np.float32)
            self.global_feat_std  = np.ones(0,  dtype=np.float32)
            self._global_feats = {}
            self._n_global = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> TAMSample:
        pid, yr, bands_np, doy_np = self._windows[idx]
        n = len(bands_np)

        if self._obs_dropout_min > 0 and n > self._obs_dropout_min:
            keep = np.random.randint(self._obs_dropout_min, n + 1)
            idx_keep = np.sort(np.random.choice(n, keep, replace=False))
            bands_np = bands_np[idx_keep]
            doy_np   = doy_np[idx_keep]
            n        = keep

        if self._band_noise_std > 0.0:
            offset = np.random.randn(N_BANDS).astype(np.float32) * self._band_noise_std
            bands_np = bands_np + offset

        bands = np.zeros((MAX_SEQ_LEN, N_BANDS), dtype=np.float32)
        bands[:n] = bands_np

        doy_vals = doy_np.astype(np.int64)
        if self._doy_jitter > 0:
            offset = np.random.randint(-self._doy_jitter, self._doy_jitter + 1)
            # Clamp to 1–365 rather than wrapping: wrapping would reorder observations
            # that straddle the year boundary, misaligning the band and DOY sequences.
            doy_vals = np.clip(doy_vals + offset, 1, 365)

        doy = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        doy[:n] = doy_vals

        # Padding mask: True = padding position (PyTorch convention)
        mask = np.ones(MAX_SEQ_LEN, dtype=bool)
        mask[:n] = False

        label  = float(self._labels[pid]) if self._labels is not None else 0.0
        weight = 1.0

        gf = self._global_feats.get(pid, np.zeros(self._n_global, dtype=np.float32))

        return TAMSample(
            bands        = torch.from_numpy(bands),
            doy          = torch.from_numpy(doy),
            mask         = torch.from_numpy(mask),
            n_obs        = torch.tensor(n / MAX_SEQ_LEN, dtype=torch.float32),
            global_feats = torch.from_numpy(gf),
            label        = torch.tensor(label,  dtype=torch.float32),
            weight       = torch.tensor(weight, dtype=torch.float32),
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
        for pid, _, _b, _d in self._windows:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out
