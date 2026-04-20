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
    bands:    torch.Tensor   # (MAX_SEQ_LEN, N_BANDS)  float32, normalised, zero-padded
    doy:      torch.Tensor   # (MAX_SEQ_LEN,)           int64, 1–365, 0=padding
    mask:     torch.Tensor   # (MAX_SEQ_LEN,)           bool, True=padding
    label:    torch.Tensor   # ()                       float32 {0, 1}
    weight:   torch.Tensor   # ()                       float32 confidence weight
    point_id: str
    year:     int


def collate_fn(samples: list[TAMSample]) -> dict:
    return {
        "bands":    torch.stack([s.bands  for s in samples]),
        "doy":      torch.stack([s.doy    for s in samples]),
        "mask":     torch.stack([s.mask   for s in samples]),
        "label":    torch.stack([s.label  for s in samples]),
        "weight":   torch.stack([s.weight for s in samples]),
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
    ) -> None:
        # doy_jitter: max ±days to shift all observations in a window (training only).
        # A single offset is drawn per __getitem__ call and applied uniformly so
        # relative timing between observations is preserved.  Set 0 to disable.
        # band_noise_std: std of Gaussian noise added to normalised band values per
        # observation independently (training only).  Set 0.0 to disable.
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
        self._windows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        for (pid, yr), grp in df.groupby(["point_id", "year"], sort=False):
            grp = grp.sort_values("date")
            if len(grp) < min_obs_per_year:
                continue
            n = min(len(grp), MAX_SEQ_LEN)
            raw = grp[feature_cols].values[:n].astype(np.float32)
            bands_np = (raw - self.band_mean) / self.band_std
            if "doy" in grp.columns:
                doy_np = grp["doy"].values[:n].astype(np.int32)
            else:
                doy_np = pd.to_datetime(grp["date"].values[:n]).day_of_year.values.astype(np.int32)
            self._windows.append((pid, int(yr), bands_np, doy_np))

        self._labels = labels
        self._doy_jitter = doy_jitter

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> TAMSample:
        pid, yr, bands_np, doy_np = self._windows[idx]
        n = len(bands_np)

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

        return TAMSample(
            bands    = torch.from_numpy(bands),
            doy      = torch.from_numpy(doy),
            mask     = torch.from_numpy(mask),
            label    = torch.tensor(label,  dtype=torch.float32),
            weight   = torch.tensor(weight, dtype=torch.float32),
            point_id = pid,
            year     = yr,
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
