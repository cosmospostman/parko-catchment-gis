"""tam/core/score.py — Chunked inference for TAMClassifier."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, MAX_SEQ_LEN
from tam.core.model import TAMClassifier

logger = logging.getLogger(__name__)

_N_FEATURES = len(ALL_FEATURE_COLS)
_SENTINEL = object()

# Precomputed offset for fast numpy year/doy extraction (no pd.to_datetime)
_EPOCH_D = np.datetime64("1970-01-01", "D")

# Band column positions in the feature matrix (matches ALL_FEATURE_COLS order)
_BAND_INDICES = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}


def _extract_year_doy(ts_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract year and day-of-year from timestamp[us]-as-int64."""
    ts_day = ts_us // (86_400 * 1_000_000)
    days = _EPOCH_D + ts_day.astype("timedelta64[D]")
    year = days.astype("datetime64[Y]").astype("int32") + 1970
    year_start = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]").astype("int64")
    doy = (ts_day - year_start + 1).astype("int32")
    return year, doy


class _PreparedBatch(NamedTuple):
    bands: torch.Tensor   # (W, MAX_SEQ_LEN, N_FEATURES) float32
    doy:   torch.Tensor   # (W, MAX_SEQ_LEN) int64
    mask:  torch.Tensor   # (W, MAX_SEQ_LEN) bool
    pids:  np.ndarray     # (W,) object
    years: np.ndarray     # (W,) int32


def _preprocess(
    chunk: pd.DataFrame,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float,
    min_obs_per_year: int,
    pin: bool = False,
) -> _PreparedBatch | None:
    """CPU-side preprocessing using numba kernels for maximum throughput.

    Pipeline:
      1. SCL filter (already applied by reader, but guard here too)
      2. Extract features into C-contiguous float32 via numba parallel kernel
      3. Find pixel-year window boundaries with numpy
      4. Fill padded (W, SEQ, F) arrays + normalise via numba parallel kernel
    """
    from tam.core._preprocess_numba import extract_features, fill_windows

    if "scl_purity" in chunk.columns:
        chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
    if chunk.empty:
        return None

    N = len(chunk)

    # Step 1: extract feature matrix — bands are already float32, zero-copy .values
    feat = np.empty((N, _N_FEATURES), dtype=np.float32)
    extract_features(
        chunk["B02"].values, chunk["B03"].values, chunk["B04"].values,
        chunk["B05"].values, chunk["B06"].values, chunk["B07"].values,
        chunk["B08"].values, chunk["B8A"].values, chunk["B11"].values,
        chunk["B12"].values,
        feat,
    )

    # Step 2: find group boundaries (pixel-year windows)
    pid_arr  = chunk["point_id"].values
    year_arr = chunk["year"].values.astype(np.int32)
    doy_arr  = chunk["doy"].values.astype(np.int32)

    pid_change  = np.empty(N, dtype=bool)
    year_change = np.empty(N, dtype=bool)
    pid_change[0] = year_change[0] = True
    pid_change[1:]  = pid_arr[1:]  != pid_arr[:-1]
    year_change[1:] = year_arr[1:] != year_arr[:-1]

    boundaries = np.where(pid_change | year_change)[0]
    ends    = np.append(boundaries[1:], N)
    lengths = ends - boundaries
    valid   = lengths >= min_obs_per_year

    if not valid.any():
        return None

    valid_starts = boundaries[valid].astype(np.int64)
    capped = np.minimum(lengths[valid], MAX_SEQ_LEN).astype(np.int32)
    W = int(valid.sum())

    # Step 3: fill padded tensors with normalisation via numba parallel kernel
    bands_np = np.zeros((W, MAX_SEQ_LEN, _N_FEATURES), dtype=np.float32)
    doy_np   = np.zeros((W, MAX_SEQ_LEN), dtype=np.int64)
    mask_np  = np.ones( (W, MAX_SEQ_LEN), dtype=np.bool_)

    fill_windows(feat, doy_arr, valid_starts, capped, band_mean, band_std,
                 bands_np, doy_np, mask_np)

    pids  = pid_arr[valid_starts]
    years = year_arr[valid_starts]

    bands_th = torch.from_numpy(bands_np)
    doy_th   = torch.from_numpy(doy_np)
    mask_th  = torch.from_numpy(mask_np)
    if pin:
        bands_th = bands_th.pin_memory()
        doy_th   = doy_th.pin_memory()
        mask_th  = mask_th.pin_memory()

    return _PreparedBatch(bands_th, doy_th, mask_th, pids, years)


def _gpu_score(
    prepared: _PreparedBatch,
    model: TAMClassifier,
    all_pids: list,
    all_years: list,
    all_probs: list,
    batch_size: int,
    device: str,
) -> None:
    """GPU-side: transfer tensors, run inference, append (pid, year, prob) to lists."""
    bands_th, doy_th, mask_th, pids, years = prepared
    W = len(pids)
    with torch.inference_mode():
        for start in range(0, W, batch_size):
            end = min(start + batch_size, W)
            prob, _ = model(
                bands_th[start:end].to(device, non_blocking=True),
                doy_th[start:end].to(device, non_blocking=True),
                mask_th[start:end].to(device, non_blocking=True),
            )
            prob_np = prob.cpu().numpy()
            all_pids.append(pids[start:end])
            all_years.append(years[start:end])
            all_probs.append(prob_np)


def aggregate_year_probs(
    all_pids: list,
    all_years: list,
    all_probs: list,
    end_year: int,
    decay: float = 0.7,
) -> pd.DataFrame:
    """Aggregate per-(pixel, year) probabilities into a single score per pixel.

    Vectorised: concatenates all arrays, applies decay weights, then uses
    pandas groupby to compute weighted mean per pixel.
    """
    if not all_pids:
        return pd.DataFrame(columns=["point_id", "prob_tam"])

    pids_np  = np.concatenate(all_pids)
    years_np = np.concatenate(all_years).astype(np.int32)
    probs_np = np.concatenate(all_probs).astype(np.float32)

    weights = np.exp(-decay * (end_year - years_np)).astype(np.float32)
    weighted_probs = weights * probs_np

    df = pd.DataFrame({"point_id": pids_np, "wp": weighted_probs, "w": weights})
    agg = df.groupby("point_id", sort=False)[["wp", "w"]].sum()
    agg["prob_tam"] = agg["wp"] / agg["w"]
    return agg[["prob_tam"]].reset_index()


def score_pixels_chunked(
    parquet: Path,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 8,
    batch_size: int = 4096,
    buffer_row_groups: int = 16,
    n_prep_workers: int = 4,
    device: str | None = None,
    tile_id: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    n_total_pixels: int | None = None,
) -> pd.DataFrame:
    """Score all pixels in parquet with a three-stage concurrent pipeline.

    Stage 1 (reader thread):          reads row groups → raw DataFrame queue
    Stage 2 (n_prep_workers threads): numba preprocessing → pinned tensor queue
    Stage 3 (main thread / GPU):      inference

    Returns a DataFrame with columns: point_id, prob_tam.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from tam.core._preprocess_numba import warmup as _numba_warmup

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.startswith("cuda") and torch.cuda.is_available()
    model.to(device)
    model.eval()

    logger.info("Warming up numba kernels ...")
    _numba_warmup()

    pf = pq.ParquetFile(parquet)
    n_rg = pf.metadata.num_row_groups
    tile_prefix = f"_{tile_id}_" if tile_id else None

    read_cols = (
        ["point_id", "date", "scl_purity"]
        + (["item_id"] if tile_id else [])
        + BAND_COLS
    )

    # Stage 1→2: raw DataFrames
    raw_q: Queue = Queue(maxsize=n_prep_workers * 2)
    # Stage 2→3: prepared batches
    prep_q: Queue = Queue(maxsize=n_prep_workers * 2)

    # --- Stage 1: reader ---
    def _reader() -> None:
        leftover: pd.DataFrame = pd.DataFrame()
        buffer: list[pd.DataFrame] = []

        def _emit(buf: list[pd.DataFrame], lo: pd.DataFrame, is_last: bool) -> pd.DataFrame:
            if not buf:
                return lo
            chunk = pd.concat(buf, ignore_index=True)
            if not lo.empty:
                chunk = pd.concat([lo, chunk], ignore_index=True)
            if not is_last:
                boundary_pid = chunk["point_id"].iloc[-1]
                new_lo = chunk[chunk["point_id"] == boundary_pid].copy()
                chunk = chunk[chunk["point_id"] != boundary_pid]
            else:
                new_lo = pd.DataFrame()
            if not chunk.empty:
                raw_q.put(chunk)
            return new_lo

        for rg in range(n_rg):
            chunk = pf.read_row_group(rg, columns=read_cols).to_pandas()
            if tile_prefix:
                chunk = chunk[chunk["item_id"].str.contains(tile_prefix, regex=False)]
            chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
            if chunk.empty:
                continue
            ts_us = chunk["date"].values.astype("int64")
            chunk["year"], chunk["doy"] = _extract_year_doy(ts_us)
            if end_year:
                chunk = chunk[chunk["year"] <= end_year]
            if chunk.empty:
                continue
            buffer.append(chunk)
            if len(buffer) >= buffer_row_groups:
                leftover = _emit(buffer, leftover, is_last=(rg == n_rg - 1))
                buffer = []

        leftover = _emit(buffer, leftover, is_last=True)
        if not leftover.empty:
            raw_q.put(leftover)
        for _ in range(n_prep_workers):
            raw_q.put(_SENTINEL)

    # --- Stage 2: preprocessor workers ---
    def _preprocessor() -> None:
        while True:
            item = raw_q.get()
            if item is _SENTINEL:
                break
            prepared = _preprocess(item, band_mean, band_std, scl_purity_min, min_obs_per_year, pin=pin)
            if prepared is not None:
                prep_q.put(prepared)
        prep_q.put(_SENTINEL)

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()

    prep_pool = ThreadPoolExecutor(max_workers=n_prep_workers)
    prep_futures = [prep_pool.submit(_preprocessor) for _ in range(n_prep_workers)]

    all_pids:  list[np.ndarray] = []
    all_years: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    n_scored = 0
    sentinels_seen = 0

    while sentinels_seen < n_prep_workers:
        item = prep_q.get()
        if item is _SENTINEL:
            sentinels_seen += 1
            continue
        _gpu_score(item, model, all_pids, all_years, all_probs, batch_size, device)
        n_scored += len(np.unique(item.pids))
        if n_total_pixels:
            logger.info("Scored %.1f%% (%d / %d pixels)", 100 * n_scored / n_total_pixels, n_scored, n_total_pixels)
        else:
            logger.info("Scored %d windows so far", n_scored)

    reader_thread.join()
    for f in prep_futures:
        f.result()
    prep_pool.shutdown(wait=False)

    logger.info("Aggregating scores ...")
    if not end_year:
        end_year = int(np.concatenate(all_years).max())
    return aggregate_year_probs(all_pids, all_years, all_probs, end_year=end_year, decay=decay)
