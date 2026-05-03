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

# S1-derived per-observation features, snapped to nearest S2 date
S1_FEATURE_COLS: list[str] = ["s1_vh", "s1_vv", "s1_vh_vv", "s1_rvi"]
S1_SNAP_WINDOW_DAYS: int = 7   # max |S1_date - S2_date| to accept a snap

N_BANDS: int = len(ALL_FEATURE_COLS)          # 13: S2 only (no S1 in time series yet)
N_BANDS_S1: int = len(ALL_FEATURE_COLS) + len(S1_FEATURE_COLS)  # 17: S2 + S1
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


def despeckle_s1(s1: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply temporal despeckle to S1 linear backscatter via a per-pixel rolling median.

    Operates on the raw linear vh/vv columns in-place (on a copy). Speckle is
    independent between acquisitions, so a rolling median across N passes suppresses
    it without spatial blurring. window=3 is conservative; 5 and 7 are also reasonable
    but risk smoothing real wet/dry seasonal transitions.

    Applied before dB conversion so that derived features (VH-VV, RVI) also benefit.
    window=0 or window=1 is a no-op.
    """
    if window < 2:
        return s1
    s1 = s1.copy()
    s1 = s1.sort_values(["point_id", "date"])
    for col in ("vh", "vv"):
        if col in s1.columns:
            s1[col] = (
                s1.groupby("point_id")[col]
                .transform(lambda x: x.rolling(window, min_periods=1, center=True).median())
                .astype(np.float32)
            )
    return s1


def snap_s1_to_s2(pixel_df: pd.DataFrame, window_days: int = S1_SNAP_WINDOW_DAYS, despeckle_window: int = 0) -> pd.DataFrame:
    """Snap S1 observations to their nearest S2 date per pixel.

    For each S2 observation, finds the closest S1 observation within ±window_days.
    Returns the S2 DataFrame with four additional columns appended:
        s1_vh, s1_vv  — VH/VV in dB (10·log10 of linear power)
        s1_vh_vv      — VH−VV ratio in dB
        s1_rvi        — Radar Vegetation Index 4·VH_lin/(VV_lin+VH_lin)
    Rows without a S1 snap within the window have NaN for all four columns.

    Operates in-place on a copy; the input DataFrame is not modified.
    Source column must be present ("S2"/"S1") or all rows treated as S2.
    """
    if "source" not in pixel_df.columns or "S1" not in pixel_df["source"].values:
        # No S1 data — return S2 rows with NaN S1 columns
        out = pixel_df.copy() if "source" not in pixel_df.columns else pixel_df[pixel_df["source"] == "S2"].copy()
        for col in S1_FEATURE_COLS:
            out[col] = np.nan
        return out

    s2 = pixel_df[pixel_df["source"] == "S2"].copy()
    s1 = pixel_df[pixel_df["source"] == "S1"].copy()

    if s2.empty:
        for col in S1_FEATURE_COLS:
            s2[col] = np.nan
        return s2

    # Despeckle before dB conversion so derived features (VH-VV, RVI) also benefit
    s1 = despeckle_s1(s1, despeckle_window)

    # Compute dB and derived S1 columns once
    s1 = s1.copy()
    vh_lin = s1["vh"].values.astype(np.float32) if "vh" in s1.columns else np.full(len(s1), np.nan, dtype=np.float32)
    vv_lin = s1["vv"].values.astype(np.float32) if "vv" in s1.columns else np.full(len(s1), np.nan, dtype=np.float32)
    s1["s1_vh"]    = np.where(vh_lin > 0, 10 * np.log10(vh_lin), np.nan).astype(np.float32)
    s1["s1_vv"]    = np.where(vv_lin > 0, 10 * np.log10(vv_lin), np.nan).astype(np.float32)
    s1["s1_vh_vv"] = s1["s1_vh"] - s1["s1_vv"]
    denom = vh_lin + vv_lin
    s1["s1_rvi"]   = np.where(denom > 0, 4 * vh_lin / denom, np.nan).astype(np.float32)

    # Vectorised snap using merge_asof per pixel.
    # merge_asof requires both sides sorted by the key within each group.
    # We apply it per point_id to guarantee monotone dates on both sides.
    s2 = s2.copy()
    s2["_date_dt"] = pd.to_datetime(s2["date"])
    s1["_date_dt"] = pd.to_datetime(s1["date"])

    s1_snap = s1[["point_id", "_date_dt", "s1_vh", "s1_vv", "s1_vh_vv", "s1_rvi"]]
    tol = pd.Timedelta(days=window_days)

    pieces: list[pd.DataFrame] = []
    s1_by_pid = dict(tuple(s1_snap.groupby("point_id")))

    for pid, grp_s2 in s2.groupby("point_id", sort=False):
        grp_s2 = grp_s2.sort_values("_date_dt")
        if pid not in s1_by_pid:
            for col in S1_FEATURE_COLS:
                grp_s2 = grp_s2.copy()
                grp_s2[col] = np.nan
            pieces.append(grp_s2)
            continue

        grp_s1 = s1_by_pid[pid].sort_values("_date_dt")
        left = grp_s2[["_date_dt"]].reset_index()

        grp_s1_b = grp_s1[["_date_dt"] + S1_FEATURE_COLS].rename(columns={"_date_dt": "_s1_date"})
        grp_s1_f = grp_s1_b.copy()

        back = pd.merge_asof(left, grp_s1_b, left_on="_date_dt", right_on="_s1_date",
                             direction="backward", tolerance=tol)
        fwd  = pd.merge_asof(left, grp_s1_f, left_on="_date_dt", right_on="_s1_date",
                             direction="forward",  tolerance=tol)

        _inf = pd.Timedelta(days=window_days + 1)
        # Use the matched S1 date (_s1_date) to compute actual distance; NaT → beyond window
        back_delta = (left["_date_dt"] - back["_s1_date"]).abs().fillna(_inf)
        fwd_delta  = (fwd["_s1_date"]  - left["_date_dt"]).abs().fillna(_inf)
        use_fwd = (fwd_delta < back_delta).values

        grp_s2 = grp_s2.copy()
        for col in S1_FEATURE_COLS:
            bv = back[col].values.astype(np.float32)
            fv = fwd[col].values.astype(np.float32)
            grp_s2[col] = np.where(use_fwd & ~np.isnan(fv), fv, bv)

        pieces.append(grp_s2)

    s2 = pd.concat(pieces).sort_index().drop(columns=["_date_dt"])
    return s2


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
        doy_phase_shift: bool = False,  # if True, randomly shift full time series by 0–364d with wraparound
        band_noise_std: float = 0.0,
        obs_dropout_min: int = 0,
        global_features_df: pd.DataFrame | None = None,
        global_feat_mean: np.ndarray | None = None,
        global_feat_std: np.ndarray | None = None,
        use_s1: bool = False,
        pixel_zscore: bool = False,  # if True, z-score each pixel's S1 bands by its own multi-year mean/std
        s1_despeckle_window: int = 0,
    ) -> None:
        # doy_jitter: max ±days to shift all observations in a window (training only).
        # A single offset is drawn per __getitem__ call and applied uniformly so
        # relative timing between observations is preserved.  Set 0 to disable.
        # band_noise_std: std of Gaussian noise added to normalised band values per
        # observation independently (training only).  Set 0.0 to disable.
        # obs_dropout_min: if >0, subsample each window to Uniform(obs_dropout_min, n)
        # per __getitem__ call, teaching the model to be invariant to obs density.
        # Build observation sequence depending on mode
        if use_s1 == "s1_only":
            # S1-only mode: use S1 rows directly, no snapping to S2 dates.
            # Bands: vh_db, vv_db, vh_vv, rvi — computed from linear vh/vv.
            s1_rows = pixel_df[pixel_df["source"] == "S1"].copy() if "source" in pixel_df.columns else pd.DataFrame()
            if s1_rows.empty:
                raise ValueError("s1_only mode requires S1 rows in pixel_df (source='S1')")
            s1_rows = despeckle_s1(s1_rows, s1_despeckle_window)
            vh_lin = s1_rows["vh"].values.astype(np.float32)
            vv_lin = s1_rows["vv"].values.astype(np.float32)
            s1_rows["s1_vh"]    = np.where(vh_lin > 0, 10 * np.log10(vh_lin), np.nan).astype(np.float32)
            s1_rows["s1_vv"]    = np.where(vv_lin > 0, 10 * np.log10(vv_lin), np.nan).astype(np.float32)
            s1_rows["s1_vh_vv"] = s1_rows["s1_vh"] - s1_rows["s1_vv"]
            denom = vh_lin + vv_lin
            s1_rows["s1_rvi"]   = np.where(denom > 0, 4 * vh_lin / denom, np.nan).astype(np.float32)
            df = s1_rows
            feature_cols = S1_FEATURE_COLS  # 4 bands: vh, vv, vh_vv, rvi
            # No SCL filter for S1 (no cloud mask); drop rows where all S1 bands NaN
            df = df.dropna(subset=S1_FEATURE_COLS, how="all")

            if pixel_zscore:
                # Per-pixel z-scoring: remove each pixel's multi-year backscatter mean/std.
                # Applied to VH and VV only — removes site-level offsets (incidence angle,
                # soil type) that cause between-site shortcuts.
                # VH-VV and RVI are ratios and already self-normalised against geometry;
                # z-scoring them would destroy the real canopy structure signal in their
                # absolute values (e.g. Parkinsonia's more negative VH-VV vs open cover).
                for col, min_std in [("s1_vh", 0.1), ("s1_vv", 0.1)]:
                    if col in df.columns:
                        pix_mean = df.groupby("point_id")[col].transform("mean")
                        pix_std  = df.groupby("point_id")[col].transform("std").clip(lower=min_std)
                        df[col]  = (df[col] - pix_mean) / pix_std
        elif use_s1:
            df = snap_s1_to_s2(pixel_df, despeckle_window=s1_despeckle_window)
            feature_cols = ALL_FEATURE_COLS + S1_FEATURE_COLS
        else:
            if "source" in pixel_df.columns:
                df = pixel_df[pixel_df["source"] == "S2"].copy()
            else:
                df = pixel_df
            feature_cols = ALL_FEATURE_COLS

        if use_s1 != "s1_only":
            if any(c not in df.columns for c in INDEX_COLS):
                df = add_spectral_indices(df)

            # SCL filter
            if "scl_purity" in df.columns:
                df = df[df["scl_purity"] >= scl_purity_min]

            # Drop rows with NaN in any S2 band
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
            # All-NaN columns (e.g. S1 cols when no S1 data present): impute
            # mean→0 and std→1 so normalisation is a no-op for those columns.
            band_mean = np.where(np.isnan(band_mean), 0.0, band_mean)
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
        # Impute NaN → 0 after normalisation. NaN arises in S1 columns for
        # observations that had no S1 acquisition within the snap window.
        # 0 in normalised space = feature mean, i.e. "no information".
        bands_all = np.where(np.isnan(bands_all), 0.0, bands_all).astype(np.float32)
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

        self._n_features = len(feature_cols)
        self._labels = labels
        self._doy_jitter = doy_jitter
        self._doy_phase_shift = doy_phase_shift
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
                # All-NaN columns (e.g. S1 globals when no S1 data): impute mean→0, std→1
                global_feat_mean = np.where(np.isnan(global_feat_mean), 0.0, global_feat_mean)
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
            # Add noise only to S2 columns — S1 columns are NaN-imputed to 0
            # for unsnapped observations and must not be perturbed away from 0.
            n_s2 = len(ALL_FEATURE_COLS)
            offset = np.zeros(self._n_features, dtype=np.float32)
            offset[:n_s2] = np.random.randn(n_s2).astype(np.float32) * self._band_noise_std
            bands_np = bands_np + offset

        doy_vals = doy_np.astype(np.int64)
        if self._doy_phase_shift:
            # Random full-year phase shift with wraparound — forces model to learn
            # shape of curve relative to concurrent bands (e.g. VV) not calendar time.
            phase_offset = np.random.randint(0, 365)
            doy_vals = ((doy_vals - 1 + phase_offset) % 365) + 1
            # Re-sort observations by shifted DOY to maintain temporal order
            sort_idx = np.argsort(doy_vals, kind="stable")
            doy_vals = doy_vals[sort_idx]
            bands_np = bands_np[sort_idx]
        elif self._doy_jitter > 0:
            offset = np.random.randint(-self._doy_jitter, self._doy_jitter + 1)
            # Clamp to 1–365 rather than wrapping: wrapping would reorder observations
            # that straddle the year boundary, misaligning the band and DOY sequences.
            doy_vals = np.clip(doy_vals + offset, 1, 365)

        bands = np.zeros((MAX_SEQ_LEN, self._n_features), dtype=np.float32)
        bands[:n] = bands_np

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
