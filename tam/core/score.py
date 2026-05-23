"""tam/core/score.py — Chunked inference for TAMClassifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple

import numpy as np
import polars as pl
import torch

from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, MAX_SEQ_LEN, S1_FEATURE_COLS, despeckle_s1, lin_to_db
from tam.core.model import TAMClassifier

logger = logging.getLogger(__name__)

_N_FEATURES = len(ALL_FEATURE_COLS)
_N_FEATURES_S1 = len(S1_FEATURE_COLS)
_SENTINEL = object()

# Precomputed offset for fast numpy year/doy extraction (no pd.to_datetime)
_EPOCH_D = np.datetime64("1970-01-01", "D")

# Band column positions in the feature matrix (matches ALL_FEATURE_COLS order)
_BAND_INDICES = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}


def _compute_pixel_s1_stats(parquet: Path) -> tuple[dict, dict, dict, dict]:
    """One-pass pre-read to compute per-pixel VH/VV mean/std for z-scoring.

    Returns (pids, vh_mean, vv_mean, vh_std, vv_std) as parallel arrays indexed
    by unique point_id. Called once before the main scoring pipeline when
    pixel_zscore=True so that _extract_s1_features can normalise on the fly.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet)
    available = set(pf.schema_arrow.names)
    read_cols = [c for c in ["point_id", "source", "vh", "vv"] if c in available]
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pl.from_arrow(pf.read_row_group(rg, columns=read_cols))
        if "source" in chunk.columns:
            chunk = chunk.filter(pl.col("source") == "S1")
        chunk = chunk.drop_nulls(subset=["vh", "vv"])
        if not chunk.is_empty():
            chunks.append(chunk.select(["point_id", "vh", "vv"]))
    if not chunks:
        return {}, {}, {}, {}
    all_s1 = pl.concat(chunks)
    vh_db = lin_to_db(all_s1["vh"].to_numpy().astype(np.float32))
    vv_db = lin_to_db(all_s1["vv"].to_numpy().astype(np.float32))
    all_s1 = all_s1.with_columns([
        pl.Series("vh_db", vh_db),
        pl.Series("vv_db", vv_db),
    ])
    agg = all_s1.group_by("point_id").agg([
        pl.col("vh_db").mean().alias("vh_mean"),
        pl.col("vh_db").std().clip(lower_bound=0.1).alias("vh_std"),
        pl.col("vv_db").mean().alias("vv_mean"),
        pl.col("vv_db").std().clip(lower_bound=0.1).alias("vv_std"),
    ])
    vh_mean_d = dict(zip(agg["point_id"].to_list(), agg["vh_mean"].to_list()))
    vh_std_d  = dict(zip(agg["point_id"].to_list(), agg["vh_std"].to_list()))
    vv_mean_d = dict(zip(agg["point_id"].to_list(), agg["vv_mean"].to_list()))
    vv_std_d  = dict(zip(agg["point_id"].to_list(), agg["vv_std"].to_list()))
    return vh_mean_d, vh_std_d, vv_mean_d, vv_std_d


def _compute_pixel_s1_stats_mixed(
    year_parquets: list[tuple[int, Path]],
    s1_feature_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-pass: compute per-pixel mean/std for each s1_feature_col across all year parquets.

    Returns (pid_mean, pid_std) dicts mapping point_id → float32 array of length len(s1_feature_cols).
    Used in mixed-mode scoring to z-score S1 features by each pixel's own long-run statistics.
    """
    import pyarrow.parquet as pq

    chunks: list[pl.DataFrame] = []
    for _, path in sorted(year_parquets):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in ["point_id", "source", "vh", "vv"] if c in available]
        for rg in range(pf.metadata.num_row_groups):
            chunk = pl.from_arrow(pf.read_row_group(rg, columns=read_cols))
            if "source" in chunk.columns:
                chunk = chunk.filter(pl.col("source") == "S1")
            chunk = chunk.drop_nulls(subset=["vh", "vv"])
            if not chunk.is_empty():
                chunks.append(chunk.select(["point_id", "vh", "vv"]))

    if not chunks:
        return {}, {}

    all_s1 = pl.concat(chunks)
    vh_db = lin_to_db(all_s1["vh"].to_numpy().astype(np.float32))
    vv_db = lin_to_db(all_s1["vv"].to_numpy().astype(np.float32))
    vh_vv = vh_db - vv_db
    vh_lin = all_s1["vh"].to_numpy().astype(np.float32)
    vv_lin = all_s1["vv"].to_numpy().astype(np.float32)
    denom  = vh_lin + vv_lin
    rvi    = np.where(denom > 0, 4 * vh_lin / denom, np.nan).astype(np.float32)

    col_map = {"s1_vh": vh_db, "s1_vv": vv_db, "s1_vh_vv": vh_vv, "s1_rvi": rvi}
    df = all_s1.with_columns([
        pl.Series(col, col_map[col]) for col in s1_feature_cols if col in col_map
    ])

    agg_exprs = []
    for col in s1_feature_cols:
        if col in col_map:
            agg_exprs += [
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().clip(lower_bound=0.1).alias(f"{col}_std"),
            ]
    grp = df.group_by("point_id").agg(agg_exprs)
    pids = grp["point_id"].to_numpy()

    means = [grp[f"{col}_mean"].to_numpy().astype(np.float32) for col in s1_feature_cols if col in col_map]
    stds  = [grp[f"{col}_std"].to_numpy().astype(np.float32)  for col in s1_feature_cols if col in col_map]
    mean_matrix = np.column_stack(means)
    std_matrix  = np.column_stack(stds)

    return (
        {pid: mean_matrix[i] for i, pid in enumerate(pids)},
        {pid: std_matrix[i]  for i, pid in enumerate(pids)},
    )


def _compute_s1_despeckle_lookup(parquet: Path, window: int) -> pl.DataFrame:
    """One-pass pre-read to compute despeckled linear vh/vv per pixel.

    Returns a DataFrame with columns [point_id, date, vh, vv] where vh/vv are
    the temporally smoothed values. Only called when s1_only=True and
    s1_despeckle_window >= 2.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet)
    available = set(pf.schema_arrow.names)
    read_cols = [c for c in ["point_id", "date", "source", "vh", "vv"] if c in available]
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pl.from_arrow(pf.read_row_group(rg, columns=read_cols))
        if "source" in chunk.columns:
            chunk = chunk.filter(pl.col("source") == "S1")
        if "vh" not in chunk.columns or "vv" not in chunk.columns:
            continue
        chunk = chunk.drop_nulls(subset=["vh", "vv"])
        if not chunk.is_empty():
            chunks.append(chunk.select(["point_id", "date", "vh", "vv"]))
    if not chunks:
        return pl.DataFrame(schema={"point_id": pl.Utf8, "date": pl.Datetime,
                                     "vh": pl.Float32, "vv": pl.Float32})
    all_s1 = pl.concat(chunks)
    return despeckle_s1(all_s1, window).select(["point_id", "date", "vh", "vv"])


def _extract_s1_features(
    chunk: pl.DataFrame,
    pixel_zscore: bool = False,
    vh_mean: dict | None = None,
    vh_std:  dict | None = None,
    vv_mean: dict | None = None,
    vv_std:  dict | None = None,
    despeckle_lookup: pl.DataFrame | None = None,
) -> np.ndarray:
    """Extract 4 S1 features (vh_db, vv_db, vh_vv, rvi) from linear vh/vv columns.

    When pixel_zscore=True, VH and VV are z-scored by each pixel's own multi-year
    mean/std (pre-computed via _compute_pixel_s1_stats). VH-VV and RVI are left
    in their natural units — they are self-normalising ratios that carry absolute
    canopy structure signal.

    When despeckle_lookup is provided (a DataFrame with columns [point_id, date,
    vh, vv] from _compute_s1_despeckle_lookup), smoothed linear vh/vv values are
    substituted before dB conversion via a (point_id, date) join.
    """
    if despeckle_lookup is not None and not despeckle_lookup.is_empty() and "date" in chunk.columns:
        chunk = (
            chunk
            .join(
                despeckle_lookup.rename({"vh": "_vh_s", "vv": "_vv_s"}),
                on=["point_id", "date"], how="left",
            )
            .with_columns([
                pl.when(pl.col("_vh_s").is_not_null()).then(pl.col("_vh_s")).otherwise(pl.col("vh")).alias("vh"),
                pl.when(pl.col("_vv_s").is_not_null()).then(pl.col("_vv_s")).otherwise(pl.col("vv")).alias("vv"),
            ])
            .drop(["_vh_s", "_vv_s"])
        )
    vh_lin = chunk["vh"].to_numpy().astype(np.float32)
    vv_lin = chunk["vv"].to_numpy().astype(np.float32)
    vh_db  = lin_to_db(vh_lin)
    vv_db  = lin_to_db(vv_lin)
    vh_vv  = vh_db - vv_db
    denom  = vh_lin + vv_lin
    rvi    = np.where(denom > 0, 4 * vh_lin / denom, np.nan)

    if pixel_zscore and vh_mean is not None:
        pids = chunk["point_id"].values
        p_vh_mean = np.array([vh_mean.get(p, 0.0) for p in pids], dtype=np.float32)
        p_vh_std  = np.array([vh_std.get(p,  1.0) for p in pids], dtype=np.float32)
        p_vv_mean = np.array([vv_mean.get(p, 0.0) for p in pids], dtype=np.float32)
        p_vv_std  = np.array([vv_std.get(p,  1.0) for p in pids], dtype=np.float32)
        vh_db = (vh_db - p_vh_mean) / p_vh_std
        vv_db = (vv_db - p_vv_mean) / p_vv_std

    feat = np.stack([vh_db, vv_db, vh_vv, rvi], axis=1).astype(np.float32)
    feat = np.where(np.isnan(feat), 0.0, feat)
    return feat


def _extract_year_doy(ts_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract year and day-of-year from timestamp[us]-as-int64."""
    ts_day = ts_us // (86_400 * 1_000_000)
    days = _EPOCH_D + ts_day.astype("timedelta64[D]")
    year = days.astype("datetime64[Y]").astype("int32") + 1970
    year_start = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]").astype("int64")
    doy = (ts_day - year_start + 1).astype("int32")
    return year, doy


class _PreparedBatch(NamedTuple):
    bands: torch.Tensor            # (W, MAX_SEQ_LEN, N_FEATURES) float32
    doy:   torch.Tensor            # (W, MAX_SEQ_LEN) int64
    mask:  torch.Tensor            # (W, MAX_SEQ_LEN) bool
    n_obs: torch.Tensor            # (W,) float32, n / MAX_SEQ_LEN
    pids:  np.ndarray              # (W,) object
    years: np.ndarray              # (W,) int32
    is_s1: torch.Tensor | None     # (W, MAX_SEQ_LEN) bool, True=S1 obs; None for S2-only models


def _preprocess(
    chunk: pl.DataFrame,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float,
    min_obs_per_year: int,
    pin: bool = False,
    s1_only: bool = False,
    mixed: bool = False,
    pixel_zscore: bool = False,
    vh_mean: dict | None = None,
    vh_std:  dict | None = None,
    vv_mean: dict | None = None,
    vv_std:  dict | None = None,
    despeckle_lookup: pl.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    pixel_zscore_stats: tuple[dict, dict] | None = None,
    s1_zscore_stats: tuple[dict, dict] | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
) -> _PreparedBatch | None:
    """CPU-side preprocessing using numba kernels for maximum throughput.

    Pipeline:
      1. SCL filter (already applied by reader, but guard here too)
      2. Extract features into C-contiguous float32 via numba parallel kernel
      3. Find pixel-year window boundaries with numpy
      4. Fill padded (W, SEQ, F) arrays + normalise via numba parallel kernel
    """
    from tam.core._preprocess_numba import extract_features, fill_windows
    from tam.core.dataset import MIN_S1_OBS_PER_YEAR

    is_s1_np: np.ndarray | None = None  # (N,) bool — set in mixed mode

    if mixed:
        # --- Mixed S2+S1 mode ---
        # chunk contains interleaved S2 and S1 rows (source column distinguishes them).
        # S2 rows carry spectral bands; S1 rows carry raw linear vh/vv.
        # Feature vector: [s2_feature_cols (n_s2), s1_feature_cols (n_s1)]
        # S2 rows: s1 positions are 0; S1 rows: s2 positions are 0.
        _s2_cols = s2_feature_cols or list(ALL_FEATURE_COLS)
        _s1_cols = s1_feature_cols or ["s1_vh", "s1_vv"]
        n_s2 = len(_s2_cols)
        n_s1 = len(_s1_cols)
        n_feat = n_s2 + n_s1

        has_source = "source" in chunk.columns
        is_s1_bool = (chunk["source"].to_numpy() == "S1") if has_source else np.zeros(len(chunk), dtype=bool)

        # Filter S2 rows by SCL purity; keep all S1 rows
        if "scl_purity" in chunk.columns:
            purity = chunk["scl_purity"].to_numpy()
            keep = is_s1_bool | (purity >= scl_purity_min)
            if not keep.all():
                chunk = chunk.filter(pl.Series(keep))
                is_s1_bool = is_s1_bool[keep]

        if chunk.is_empty():
            return None

        N = len(chunk)

        # Compute s1_vh/s1_vv dB features for S1 rows
        feat = np.zeros((N, n_feat), dtype=np.float32)
        s1_idx = np.where(is_s1_bool)[0]
        s2_idx = np.where(~is_s1_bool)[0]

        # S2 features
        if len(s2_idx) > 0:
            s2_chunk = chunk[s2_idx.tolist()]
            from analysis.constants import add_spectral_indices
            _need = [c for c in _s2_cols if c not in s2_chunk.columns]
            if _need:
                s2_chunk = add_spectral_indices(s2_chunk)
            s2_vals = s2_chunk.select(_s2_cols).to_numpy().astype(np.float32)
            feat[s2_idx, :n_s2] = s2_vals

        # S1 features (compute dB from linear vh/vv)
        if len(s1_idx) > 0:
            s1_chunk = chunk[s1_idx.tolist()]
            vh_lin = s1_chunk["vh"].to_numpy().astype(np.float32) if "vh" in s1_chunk.columns else np.full(len(s1_chunk), np.nan, np.float32)
            vv_lin = s1_chunk["vv"].to_numpy().astype(np.float32) if "vv" in s1_chunk.columns else np.full(len(s1_chunk), np.nan, np.float32)
            s1_db_map = {
                "s1_vh":    lin_to_db(vh_lin),
                "s1_vv":    lin_to_db(vv_lin),
                "s1_vh_vv": lin_to_db(vh_lin) - lin_to_db(vv_lin),
                "s1_rvi":   np.where(vh_lin + vv_lin > 0, 4 * vh_lin / (vh_lin + vv_lin), np.nan).astype(np.float32),
            }
            for j, col in enumerate(_s1_cols):
                feat[s1_idx, n_s2 + j] = s1_db_map.get(col, np.zeros(len(s1_idx), np.float32))

        is_s1_np = is_s1_bool

    elif s1_only:
        if "source" in chunk.columns:
            chunk = chunk.filter(pl.col("source") == "S1")
        chunk = chunk.drop_nulls(subset=["vh", "vv"])
        if chunk.is_empty():
            return None
        N = len(chunk)
        feat = _extract_s1_features(chunk, pixel_zscore=pixel_zscore,
                                    vh_mean=vh_mean, vh_std=vh_std,
                                    vv_mean=vv_mean, vv_std=vv_std,
                                    despeckle_lookup=despeckle_lookup)
        n_feat = _N_FEATURES_S1
    else:
        if "scl_purity" in chunk.columns:
            chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
        if chunk.is_empty():
            return None
        N = len(chunk)
        _use_default_cols = feature_cols is None or feature_cols == list(ALL_FEATURE_COLS)
        if _use_default_cols:
            n_feat = _N_FEATURES
            feat = np.empty((N, n_feat), dtype=np.float32)
            extract_features(
                chunk["B02"].to_numpy(), chunk["B03"].to_numpy(), chunk["B04"].to_numpy(),
                chunk["B05"].to_numpy(), chunk["B06"].to_numpy(), chunk["B07"].to_numpy(),
                chunk["B08"].to_numpy(), chunk["B8A"].to_numpy(), chunk["B11"].to_numpy(),
                chunk["B12"].to_numpy(),
                feat,
            )
        else:
            from analysis.constants import add_spectral_indices
            _cols_needed = set(feature_cols) - set(chunk.columns)
            if _cols_needed:
                chunk = add_spectral_indices(chunk)
            n_feat = len(feature_cols)
            feat = chunk.select(feature_cols).to_numpy().astype(np.float32)

    # Step 2: find group boundaries (pixel-year windows)
    pid_arr  = chunk["point_id"].to_numpy()
    year_arr = chunk["year"].to_numpy().astype(np.int32)
    doy_arr  = chunk["doy"].to_numpy().astype(np.int32)

    pid_change  = np.empty(len(chunk), dtype=bool)
    year_change = np.empty(len(chunk), dtype=bool)
    pid_change[0] = year_change[0] = True
    pid_change[1:]  = pid_arr[1:]  != pid_arr[:-1]
    year_change[1:] = year_arr[1:] != year_arr[:-1]

    boundaries = np.where(pid_change | year_change)[0]
    ends    = np.append(boundaries[1:], len(chunk))
    lengths = ends - boundaries

    if mixed:
        # Require both enough S2 obs and enough S1 obs per window
        valid = np.zeros(len(boundaries), dtype=bool)
        for i, (s, e) in enumerate(zip(boundaries, ends)):
            src_seg = is_s1_np[s:e]
            valid[i] = ((~src_seg).sum() >= min_obs_per_year and
                        src_seg.sum() >= MIN_S1_OBS_PER_YEAR)
    else:
        valid = lengths >= min_obs_per_year

    if not valid.any():
        return None

    valid_starts = boundaries[valid].astype(np.int64)
    capped = np.minimum(lengths[valid], max_seq_len).astype(np.int32)
    W = int(valid.sum())

    # Step 3: fill padded tensors
    bands_np  = np.zeros((W, max_seq_len, n_feat), dtype=np.float32)
    doy_np    = np.zeros((W, max_seq_len), dtype=np.int64)
    mask_np   = np.ones( (W, max_seq_len), dtype=np.bool_)
    is_s1_out = np.zeros((W, max_seq_len), dtype=np.bool_) if mixed else None

    pids_at_starts = pid_arr[valid_starts]

    if mixed:
        # Per-pixel z-score applied separately to S2 and S1 slots
        _s2_zero = np.zeros(n_s2, np.float32)
        _s2_one  = np.ones(n_s2,  np.float32)
        _s1_zero = np.zeros(n_s1, np.float32)
        _s1_one  = np.ones(n_s1,  np.float32)
        p_s2_mean = p_s2_std = p_s1_mean = p_s1_std = None
        if pixel_zscore_stats is not None:
            p_s2_mean, p_s2_std = pixel_zscore_stats
        if s1_zscore_stats is not None:
            p_s1_mean, p_s1_std = s1_zscore_stats

        for k in range(W):
            s = valid_starts[k]
            c = capped[k]
            win_feat = feat[s:s+c].copy()
            win_src  = is_s1_np[s:s+c]
            s2_rows  = ~win_src
            s1_rows  = win_src

            if p_s2_mean is not None and s2_rows.any():
                pid = pids_at_starts[k]
                pm = p_s2_mean.get(pid, _s2_zero)
                ps = p_s2_std.get(pid,  _s2_one)
                win_feat[s2_rows, :n_s2] = (win_feat[s2_rows, :n_s2] - pm) / ps

            if p_s1_mean is not None and s1_rows.any():
                pid = pids_at_starts[k]
                pm1 = p_s1_mean.get(pid, _s1_zero)
                ps1 = p_s1_std.get(pid,  _s1_one)
                win_feat[s1_rows, n_s2:] = (win_feat[s1_rows, n_s2:] - pm1) / ps1

            normed = np.where(np.isnan(win_feat), 0.0, win_feat).astype(np.float32)
            n = min(c, max_seq_len)
            bands_np[k, :n]  = normed[:n]
            doy_np[k,  :n]   = doy_arr[s:s+n]
            mask_np[k, :n]   = False
            is_s1_out[k, :n] = win_src[:n]

    elif pixel_zscore_stats is not None and not s1_only:
        pid_mean_lookup, pid_std_lookup = pixel_zscore_stats
        _zero = np.zeros(n_feat, dtype=np.float32)
        _one  = np.ones(n_feat,  dtype=np.float32)
        for k in range(W):
            s = valid_starts[k]
            c = capped[k]
            pm = pid_mean_lookup.get(pids_at_starts[k], _zero)
            ps = pid_std_lookup.get(pids_at_starts[k],  _one)
            window = (feat[s:s+c] - pm) / ps
            normed = np.where(np.isnan(window), 0.0, window).astype(np.float32)
            n = min(c, max_seq_len)
            bands_np[k, :n] = normed[:n]
            doy_np[k,  :n]  = doy_arr[s:s+n]
            mask_np[k, :n]  = False
    else:
        fill_windows(feat, doy_arr, valid_starts, capped, band_mean, band_std,
                     bands_np, doy_np, mask_np)

    pids  = pid_arr[valid_starts]
    years = year_arr[valid_starts]
    n_obs_np = (capped / max_seq_len).astype(np.float32)

    bands_th = torch.from_numpy(bands_np)
    doy_th   = torch.from_numpy(doy_np)
    mask_th  = torch.from_numpy(mask_np)
    n_obs_th = torch.from_numpy(n_obs_np)
    is_s1_th = torch.from_numpy(is_s1_out) if is_s1_out is not None else None
    if pin:
        bands_th = bands_th.pin_memory()
        doy_th   = doy_th.pin_memory()
        mask_th  = mask_th.pin_memory()
        n_obs_th = n_obs_th.pin_memory()

    return _PreparedBatch(bands_th, doy_th, mask_th, n_obs_th, pids, years, is_s1_th)


def _compute_s2_pixel_zscore_stats(
    year_parquets: list[tuple[int, Path]],
    feature_cols: list[str],
    scl_purity_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-pass: compute per-pixel mean and std for each feature col across all years.

    Returns (pid_mean, pid_std) dicts mapping point_id → float32 array of length n_feat.
    Used to apply pixel z-scoring at inference time, matching the training normalisation.
    """
    import pyarrow.parquet as pq
    from analysis.constants import add_spectral_indices

    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "scl_purity"] + raw_band_cols

    chunks: list[pl.DataFrame] = []
    for _, path in sorted(year_parquets):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in read_cols if c in available]
        for rg in range(pf.metadata.num_row_groups):
            chunk = pl.from_arrow(pf.read_row_group(rg, columns=cols))
            if "scl_purity" in chunk.columns:
                chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
            chunks.append(chunk)

    if not chunks:
        return {}, {}

    df = pl.concat(chunks)
    index_cols = [c for c in feature_cols if c in ("NDVI", "NDWI", "EVI")]
    if index_cols:
        df = add_spectral_indices(df)

    agg_exprs = [pl.col("point_id")]
    for col in feature_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            agg_exprs.append(pl.col(col).std().fill_null(1.0).clip(lower_bound=1e-6).alias(f"{col}_std"))
    grp = df.group_by("point_id").agg([
        *(pl.col(col).mean().alias(f"{col}_mean") for col in feature_cols if col in df.columns),
        *(pl.col(col).std().fill_null(1.0).clip(lower_bound=1e-6).alias(f"{col}_std")
          for col in feature_cols if col in df.columns),
    ])
    pids = grp["point_id"].to_numpy()

    means: list[np.ndarray] = []
    stds:  list[np.ndarray] = []
    for col in feature_cols:
        if col not in df.columns:
            means.append(np.zeros(len(pids), dtype=np.float32))
            stds.append(np.ones(len(pids),  dtype=np.float32))
            continue
        means.append(grp[f"{col}_mean"].fill_null(0.0).to_numpy().astype(np.float32))
        stds.append(grp[f"{col}_std"].fill_null(1.0).to_numpy().astype(np.float32))

    mean_matrix = np.column_stack(means)  # (n_pixels, n_feat)
    std_matrix  = np.column_stack(stds)

    pid_mean = {pid: mean_matrix[i] for i, pid in enumerate(pids)}
    pid_std  = {pid: std_matrix[i]  for i, pid in enumerate(pids)}
    return pid_mean, pid_std


def _compute_band_summaries_from_parquets(
    year_parquets: list[tuple[int, Path]],
    feature_cols: list[str],
    scl_purity_min: float,
    global_feat_mean: np.ndarray,
    global_feat_std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pre-pass: compute per-pixel normalised [p5, p95, std] for each feature col.

    Reads all year parquets, concatenates S2 rows, computes summary stats per pixel,
    normalises using global_feat_mean/std (saved from training), and returns a dict
    {point_id: float32 array}.  Vector order matches _compute_band_summaries in train.py:
    [col0_p5, col0_p95, col0_std, col1_p5, ...] (3 stats × n_feature_cols).
    """
    import pyarrow.parquet as pq
    from analysis.constants import add_spectral_indices

    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "scl_purity"] + raw_band_cols

    chunks: list[pl.DataFrame] = []
    for _, path in sorted(year_parquets):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in read_cols if c in available]
        for rg in range(pf.metadata.num_row_groups):
            chunk = pl.from_arrow(pf.read_row_group(rg, columns=cols))
            if "scl_purity" in chunk.columns:
                chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
            chunks.append(chunk)

    if not chunks:
        return {}

    df = pl.concat(chunks)

    # Compute spectral indices if needed
    index_cols = [c for c in feature_cols if c in ("NDVI", "NDWI", "EVI")]
    if index_cols:
        df = add_spectral_indices(df)

    # Compute raw p5/p95/std per pixel for each feature col (no normalisation yet)
    # NOTE: pixel z-score is intentionally NOT applied here — training computes band
    # summaries from raw reflectances (before z-scoring), so inference must match.
    agg_exprs = []
    for col in feature_cols:
        if col in df.columns:
            agg_exprs += [
                pl.col(col).quantile(0.05).alias(f"{col}_p5"),
                pl.col(col).quantile(0.95).alias(f"{col}_p95"),
                pl.col(col).std().fill_null(0.0).alias(f"{col}_std"),
            ]
    grp = df.group_by("point_id").agg(agg_exprs)
    pids = grp["point_id"].to_numpy()
    summary_cols: list[np.ndarray] = []
    for col in feature_cols:
        if col not in df.columns:
            z = np.zeros(len(pids), dtype=np.float32)
            summary_cols += [z, z, z]
            continue
        p5  = grp[f"{col}_p5"].fill_null(0.0).to_numpy().astype(np.float32)
        p95 = grp[f"{col}_p95"].fill_null(0.0).to_numpy().astype(np.float32)
        std = grp[f"{col}_std"].fill_null(0.0).to_numpy().astype(np.float32)
        summary_cols += [p5, p95, std]

    summary_matrix = np.column_stack(summary_cols)  # (n_pixels, n_cols*3)

    # Normalise with training global_feat_mean/std, matching TAMDataset normalisation
    summary_matrix = (summary_matrix - global_feat_mean) / np.where(global_feat_std < 1e-6, 1.0, global_feat_std)

    return {pid: summary_matrix[i].astype(np.float32) for i, pid in enumerate(pids)}


def _gpu_score(
    prepared: _PreparedBatch,
    model: TAMClassifier,
    all_pids: list,
    all_years: list,
    all_probs: list,
    batch_size: int,
    device: str,
    band_summaries: dict | None = None,
) -> None:
    """GPU-side: transfer tensors, run inference, append (pid, year, prob) to lists."""
    bands_th, doy_th, mask_th, n_obs_th, pids, years, is_s1_th = prepared
    W = len(pids)

    global_feats_th: torch.Tensor | None = None
    if band_summaries and model.n_global_features > 0:
        n_g = model.n_global_features
        gf_np = np.stack([
            band_summaries.get(pid, np.zeros(n_g, dtype=np.float32))
            for pid in pids
        ])
        global_feats_th = torch.from_numpy(gf_np)

    with torch.inference_mode():
        for start in range(0, W, batch_size):
            end = min(start + batch_size, W)
            gf_batch = global_feats_th[start:end].to(device, non_blocking=True) if global_feats_th is not None else None
            is_s1_batch = is_s1_th[start:end].to(device, non_blocking=True) if is_s1_th is not None else None
            prob, _ = model(
                bands_th[start:end].to(device, non_blocking=True),
                doy_th[start:end].to(device, non_blocking=True),
                mask_th[start:end].to(device, non_blocking=True),
                n_obs_th[start:end].to(device, non_blocking=True),
                global_feats=gf_batch,
                is_s1=is_s1_batch,
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
) -> pl.DataFrame:
    """Aggregate per-(pixel, year) probabilities into a single score per pixel.

    Vectorised: concatenates all arrays, applies decay weights, then uses
    Polars groupby to compute weighted mean per pixel.
    """
    if not all_pids:
        return pl.DataFrame({"point_id": pl.Series([], dtype=pl.Utf8),
                              "prob_tam": pl.Series([], dtype=pl.Float32)})

    pids_np  = np.concatenate(all_pids)
    years_np = np.concatenate(all_years).astype(np.int32)
    probs_np = np.concatenate(all_probs).astype(np.float32)

    weights = np.exp(-decay * (end_year - years_np)).astype(np.float32)
    weighted_probs = weights * probs_np

    df = pl.DataFrame({
        "point_id": pids_np,
        "wp": weighted_probs,
        "w": weights,
    })
    agg = df.group_by("point_id", maintain_order=False).agg([
        pl.col("wp").sum().alias("wp_sum"),
        pl.col("w").sum().alias("w_sum"),
    ]).with_columns(
        (pl.col("wp_sum") / pl.col("w_sum")).alias("prob_tam")
    ).select(["point_id", "prob_tam"])
    return agg


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
    s1_only: bool = False,
    mixed: bool = False,
    pixel_zscore: bool = False,
    s1_despeckle_window: int = 0,
    feature_cols: list[str] | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    band_summaries: dict | None = None,
    pixel_zscore_stats: tuple[dict, dict] | None = None,
    s1_zscore_stats: tuple[dict, dict] | None = None,
    # accumulators — pass across years to merge results before final aggregation
    _all_pids:  list | None = None,
    _all_years: list | None = None,
    _all_probs: list | None = None,
) -> pl.DataFrame:
    """Score all pixels in a single-year parquet with a three-stage concurrent pipeline.

    Stage 1 (reader thread):          reads row groups → raw DataFrame queue
    Stage 2 (n_prep_workers threads): numba preprocessing → pinned tensor queue
    Stage 3 (main thread / GPU):      inference

    When called across multiple years, pass the same _all_pids/_all_years/_all_probs
    lists and only call aggregate_year_probs() after the final year.  The convenience
    wrapper score_location_years() handles this automatically.

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

    # Pre-pass: compute per-pixel VH/VV stats for z-scoring (s1_only mode only).
    # Must happen before the main pipeline so chunked preprocessors have full-pixel history.
    _vh_mean = _vh_std = _vv_mean = _vv_std = None
    if s1_only and pixel_zscore:
        logger.info("Computing per-pixel S1 z-score stats (pre-pass) ...")
        _vh_mean, _vh_std, _vv_mean, _vv_std = _compute_pixel_s1_stats(parquet)
        logger.info("Z-score stats computed for %d pixels", len(_vh_mean))

    _despeckle_lookup = None
    if s1_only and s1_despeckle_window >= 2:
        logger.info("Computing S1 despeckle lookup (window=%d, pre-pass) ...", s1_despeckle_window)
        _despeckle_lookup = _compute_s1_despeckle_lookup(parquet, s1_despeckle_window)
        logger.info("Despeckle lookup computed for %d S1 rows", len(_despeckle_lookup))

    pf = pq.ParquetFile(parquet)
    n_rg = pf.metadata.num_row_groups
    tile_prefix = f"_{tile_id}_" if tile_id else None

    _s2_band_cols = [c for c in (s2_feature_cols or feature_cols or BAND_COLS) if c not in ("NDVI", "NDWI", "EVI")]
    if s1_only:
        read_cols = (
            ["point_id", "date", "source"]
            + (["item_id"] if tile_id else [])
            + ["vh", "vv"]
        )
    elif mixed:
        # Read both S2 spectral bands and S1 raw vh/vv; source column distinguishes rows
        read_cols = (
            ["point_id", "date", "source", "scl_purity"]
            + (["item_id"] if tile_id else [])
            + _s2_band_cols
            + ["vh", "vv"]
        )
    else:
        read_cols = (
            ["point_id", "date", "scl_purity"]
            + (["item_id"] if tile_id else [])
            + _s2_band_cols
        )
    # Only read columns that exist in this parquet
    available = set(pf.schema_arrow.names)
    read_cols = [c for c in read_cols if c in available]

    # Stage 1→2: raw DataFrames
    raw_q: Queue = Queue(maxsize=n_prep_workers * 2)
    # Stage 2→3: prepared batches
    prep_q: Queue = Queue(maxsize=n_prep_workers * 2)

    # --- Stage 1: reader ---
    def _reader() -> None:
        leftover: pl.DataFrame = pl.DataFrame()
        buffer: list[pl.DataFrame] = []

        def _emit(buf: list[pl.DataFrame], lo: pl.DataFrame, is_last: bool) -> pl.DataFrame:
            if not buf:
                return lo
            chunk = pl.concat(buf)
            if not lo.is_empty():
                chunk = pl.concat([lo, chunk])
            if not is_last:
                boundary_pid = chunk["point_id"][-1]
                new_lo = chunk.filter(pl.col("point_id") == boundary_pid)
                chunk = chunk.filter(pl.col("point_id") != boundary_pid)
            else:
                new_lo = pl.DataFrame()
            if not chunk.is_empty():
                raw_q.put(chunk)
            return new_lo

        try:
            for rg in range(n_rg):
                chunk = pl.from_arrow(pf.read_row_group(rg, columns=read_cols))
                if tile_prefix:
                    chunk = chunk.filter(pl.col("item_id").str.contains(tile_prefix, literal=True))
                if not s1_only and not mixed and "scl_purity" in chunk.columns:
                    chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
                if chunk.is_empty():
                    continue
                # Extract year/doy from date column
                ts_us = chunk["date"].cast(pl.Datetime("us")).to_numpy(allow_copy=True).astype("int64")
                year_arr, doy_arr = _extract_year_doy(ts_us)
                chunk = chunk.with_columns([
                    pl.Series("year", year_arr.astype(np.int32)),
                    pl.Series("doy",  doy_arr.astype(np.int32)),
                ])
                if end_year:
                    chunk = chunk.filter(pl.col("year") <= end_year)
                if chunk.is_empty():
                    continue
                buffer.append(chunk)
                if len(buffer) >= buffer_row_groups:
                    leftover = _emit(buffer, leftover, is_last=(rg == n_rg - 1))
                    buffer = []

            leftover = _emit(buffer, leftover, is_last=True)
            if not leftover.is_empty():
                raw_q.put(leftover)
        except Exception:
            logger.exception("Reader thread crashed")
        finally:
            for _ in range(n_prep_workers):
                raw_q.put(_SENTINEL)

    # --- Stage 2: preprocessor workers ---
    def _preprocessor() -> None:
        try:
            while True:
                item = raw_q.get()
                if item is _SENTINEL:
                    break
                prepared = _preprocess(item, band_mean, band_std, scl_purity_min, min_obs_per_year, pin=pin,
                                       s1_only=s1_only, mixed=mixed,
                                       pixel_zscore=pixel_zscore, vh_mean=_vh_mean, vh_std=_vh_std,
                                       vv_mean=_vv_mean, vv_std=_vv_std,
                                       despeckle_lookup=_despeckle_lookup,
                                       feature_cols=feature_cols,
                                       pixel_zscore_stats=pixel_zscore_stats,
                                       s1_zscore_stats=s1_zscore_stats,
                                       s2_feature_cols=s2_feature_cols,
                                       s1_feature_cols=s1_feature_cols,
                                       max_seq_len=getattr(model, "_max_seq_len", MAX_SEQ_LEN))
                if prepared is not None:
                    prep_q.put(prepared)
        except Exception:
            logger.exception("Preprocessor worker crashed")
        finally:
            prep_q.put(_SENTINEL)

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()

    prep_pool = ThreadPoolExecutor(max_workers=n_prep_workers)
    prep_futures = [prep_pool.submit(_preprocessor) for _ in range(n_prep_workers)]

    all_pids:  list[np.ndarray] = _all_pids  if _all_pids  is not None else []
    all_years: list[np.ndarray] = _all_years if _all_years is not None else []
    all_probs: list[np.ndarray] = _all_probs if _all_probs is not None else []
    n_scored = 0
    sentinels_seen = 0

    while sentinels_seen < n_prep_workers:
        item = prep_q.get()
        if item is _SENTINEL:
            sentinels_seen += 1
            continue
        _gpu_score(item, model, all_pids, all_years, all_probs, batch_size, device,
                   band_summaries=band_summaries)
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
        end_year = int(np.concatenate(all_years).max()) if all_years else 0
    return aggregate_year_probs(all_pids, all_years, all_probs, end_year=end_year, decay=decay)


def score_location_years(
    year_parquets: list[tuple[int, Path]],
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 8,
    batch_size: int = 4096,
    n_prep_workers: int = 4,
    device: str | None = None,
    tile_id: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    n_total_pixels: int | None = None,
    s1_only: bool = False,
    mixed: bool = False,
    pixel_zscore: bool = False,
    s1_despeckle_window: int = 0,
    feature_cols: list[str] | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    summary_feature_cols: list[str] | None = None,
    global_feat_mean: np.ndarray | None = None,
    global_feat_std: np.ndarray | None = None,
) -> pl.DataFrame:
    """Score a location across multiple annual parquets and aggregate.

    year_parquets: [(year, path), ...] — must be pixel-sorted parquets.

    Accumulates (pid, year, prob) triples across all years then calls
    aggregate_year_probs() once, so decay weighting is globally consistent.
    """
    all_pids:  list[np.ndarray] = []
    all_years: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    # Resolve effective column lists for the three roles:
    # - _s2_cols: S2 spectral features (for pixel z-score pre-pass and _preprocess)
    # - _s1_cols: S1 dB features (for S1 pixel z-score pre-pass; mixed mode only)
    # - _summary_cols: band-summary global features (must match global_feat_mean shape)
    _s2_cols      = s2_feature_cols or feature_cols or list(ALL_FEATURE_COLS)
    _s1_cols      = s1_feature_cols or ["s1_vh", "s1_vv"]
    _summary_cols = summary_feature_cols or feature_cols

    pixel_zscore_stats: tuple[dict, dict] | None = None
    s1_zscore_stats:    tuple[dict, dict] | None = None
    band_summaries: dict | None = None

    if pixel_zscore and not s1_only:
        logger.info("Pre-pass: computing per-pixel S2 z-score stats (%d feature cols) ...", len(_s2_cols))
        pixel_zscore_stats = _compute_s2_pixel_zscore_stats(
            year_parquets=year_parquets,
            feature_cols=_s2_cols,
            scl_purity_min=scl_purity_min,
        )
        logger.info("Z-score stats computed for %d pixels", len(pixel_zscore_stats[0]))

        if mixed:
            logger.info("Pre-pass: computing per-pixel S1 z-score stats (%s) ...", _s1_cols)
            s1_zscore_stats = _compute_pixel_s1_stats_mixed(
                year_parquets=year_parquets,
                s1_feature_cols=_s1_cols,
            )
            logger.info("S1 z-score stats computed for %d pixels", len(s1_zscore_stats[0]))

        if model.n_global_features > 0:
            if global_feat_mean is None or global_feat_std is None:
                raise ValueError(
                    "Model has global features but tam_global_feat_stats.npz was not found. "
                    "Re-train to generate it, or pass global_feat_mean/std explicitly."
                )
            if _summary_cols is None:
                raise ValueError(
                    "Model has global features but summary_feature_cols is unknown. "
                    "Pass feature_cols or summary_feature_cols matching global_feat_mean shape."
                )
            logger.info("Pre-pass: computing per-pixel band summaries (%d feature cols) ...", len(_summary_cols))
            band_summaries = _compute_band_summaries_from_parquets(
                year_parquets=year_parquets,
                feature_cols=_summary_cols,
                scl_purity_min=scl_purity_min,
                global_feat_mean=global_feat_mean,
                global_feat_std=global_feat_std,
            )
            logger.info("Band summaries computed for %d pixels", len(band_summaries))

    _eff_end_year = end_year or max(y for y, _ in year_parquets)

    for year, path in sorted(year_parquets):
        if end_year and year > end_year:
            logger.info("Skipping %d (past end_year=%d)", year, end_year)
            continue
        logger.info("Scoring year %d — %s", year, path.name)
        score_pixels_chunked(
            parquet=path,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            scl_purity_min=scl_purity_min,
            min_obs_per_year=min_obs_per_year,
            batch_size=batch_size,
            n_prep_workers=n_prep_workers,
            device=device,
            tile_id=tile_id,
            end_year=_eff_end_year,
            decay=decay,
            n_total_pixels=n_total_pixels,
            s1_only=s1_only,
            mixed=mixed,
            pixel_zscore=pixel_zscore,
            s1_despeckle_window=s1_despeckle_window,
            feature_cols=None,  # not used in mixed/pixel_zscore paths
            s2_feature_cols=_s2_cols,
            s1_feature_cols=_s1_cols,
            band_summaries=band_summaries,
            pixel_zscore_stats=pixel_zscore_stats,
            s1_zscore_stats=s1_zscore_stats,
            _all_pids=all_pids,
            _all_years=all_years,
            _all_probs=all_probs,
        )

    logger.info("Aggregating scores across %d years ...", len(year_parquets))
    return aggregate_year_probs(all_pids, all_years, all_probs, end_year=_eff_end_year, decay=decay)


_STAGING_WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["point_id"],
    write_statistics=False,
)


def score_tile_year(
    parquet: Path,
    tile_id: str,
    year: int,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    staging_dir: Path,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 8,
    batch_size: int = 4096,
    n_prep_workers: int = 4,
    device: str | None = None,
    s1_only: bool = False,
    s1_despeckle_window: int = 0,
) -> Path:
    """Score a single (tile_id, year) parquet and write a staging parquet.

    The staging parquet has columns: point_id (str), year (int16), prob_tam_raw (float32).
    If the staging file already exists it is returned immediately (idempotent / crash-safe).

    Returns the staging parquet path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{tile_id}_{year}.parquet"
    if out_path.exists():
        logger.info("Staging file exists, skipping %s year=%d", tile_id, year)
        return out_path

    logger.info("Scoring tile=%s year=%d ...", tile_id, year)
    scores = score_pixels_chunked(
        parquet=parquet,
        model=model,
        band_mean=band_mean,
        band_std=band_std,
        scl_purity_min=scl_purity_min,
        min_obs_per_year=min_obs_per_year,
        batch_size=batch_size,
        n_prep_workers=n_prep_workers,
        device=device,
        tile_id=tile_id,
        end_year=year,
        decay=0.0,   # no decay here — aggregation happens in phase 2
        s1_only=s1_only,
        s1_despeckle_window=s1_despeckle_window,
    )

    scores = scores.with_columns(pl.lit(np.int16(year)).alias("year"))
    tbl = scores.select([
        "point_id",
        "year",
        pl.col("prob_tam").alias("prob_tam_raw"),
    ]).to_arrow()
    tmp_path = out_path.with_suffix(".tmp.parquet")
    pq.write_table(tbl, tmp_path, **_STAGING_WRITE_OPTS)
    tmp_path.rename(out_path)  # atomic on POSIX; only visible once complete
    logger.info("Wrote staging %s (%d pixels)", out_path.name, len(scores))
    return out_path


def _score_tile_worker(args: tuple) -> tuple[str, list[Path]]:
    """Worker target for torch.multiprocessing: score one tile across all years.

    Returns (tile_id, staging_paths).
    """
    (
        tile_id, year_parquets, model, band_mean, band_std,
        staging_dir, scl_purity_min, min_obs_per_year,
        batch_size, n_prep_workers, device, n_tile_workers, s1_only,
        s1_despeckle_window,
    ) = args

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // n_tile_workers))

    paths = []
    for year, parquet in year_parquets:
        p = score_tile_year(
            parquet=parquet,
            tile_id=tile_id,
            year=year,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            staging_dir=staging_dir,
            scl_purity_min=scl_purity_min,
            min_obs_per_year=min_obs_per_year,
            batch_size=batch_size,
            n_prep_workers=n_prep_workers,
            device=device,
            s1_only=s1_only,
            s1_despeckle_window=s1_despeckle_window,
        )
        paths.append(p)
    return tile_id, paths


def score_tiles_chunked(
    tile_year_parquets: dict[str, list[tuple[int, Path]]],
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    out_dir: Path,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 8,
    batch_size: int = 4096,
    n_prep_workers: int = 4,
    n_tile_workers: int = 1,
    device: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    s1_only: bool = False,
    s1_despeckle_window: int = 0,
) -> list[Path]:
    """Score each S2 tile independently, writing one parquet per tile.

    Two-phase approach:
      Phase 1 — per-(tile, year): score and write staging parquet
                  (out_dir/staging/{tile_id}_{year}.parquet).
                  Idempotent: existing staging files are reused for crash recovery.
      Phase 2 — per-tile: read staging parquets, apply decay aggregation,
                  convert to uint8, write final parquet (out_dir/{tile_id}.scores.parquet),
                  then remove staging files.

    When n_tile_workers > 1, uses torch.multiprocessing.Pool (spawn) with
    model.share_memory() so workers share weights without copying.

    Returns list of written final parquet paths.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / "staging"

    _eff_end_year = end_year or max(y for yps in tile_year_parquets.values() for y, _ in yps)

    # Filter years beyond end_year
    tile_year_parquets = {
        tid: [(y, p) for y, p in yps if y <= _eff_end_year]
        for tid, yps in tile_year_parquets.items()
    }

    # --- Phase 1: score each (tile, year) ---
    worker_args = [
        (
            tile_id, year_parquets, model, band_mean, band_std,
            staging_dir, scl_purity_min, min_obs_per_year,
            batch_size, n_prep_workers, device, n_tile_workers, s1_only,
            s1_despeckle_window,
        )
        for tile_id, year_parquets in tile_year_parquets.items()
    ]

    staging_by_tile: dict[str, list[Path]] = {}

    if n_tile_workers > 1:
        model.share_memory()
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_tile_workers) as pool:
            for tile_id, paths in pool.imap_unordered(_score_tile_worker, worker_args):
                staging_by_tile[tile_id] = paths
                logger.info("Tile %s complete (%d year files)", tile_id, len(paths))
    else:
        for args in worker_args:
            tile_id, paths = _score_tile_worker(args)
            staging_by_tile[tile_id] = paths
            logger.info("Tile %s complete (%d year files)", tile_id, len(paths))

    # --- Phase 2: aggregate per tile and write final parquet ---
    final_paths: list[Path] = []

    _FINAL_WRITE_OPTS = dict(
        compression="zstd",
        compression_level=3,
        use_dictionary=["point_id"],
        write_statistics=True,
    )

    for tile_id, s_paths in staging_by_tile.items():
        logger.info("Aggregating tile %s across %d years ...", tile_id, len(s_paths))

        # Read all staging parquets for this tile
        raw = pl.concat([
            pl.read_parquet(p, columns=["point_id", "year", "prob_tam_raw"])
            for p in s_paths
        ])

        all_pids  = [raw["point_id"].to_numpy()]
        all_years = [raw["year"].to_numpy().astype(np.int32)]
        all_probs = [raw["prob_tam_raw"].to_numpy().astype(np.float32)]

        agg = aggregate_year_probs(all_pids, all_years, all_probs, end_year=_eff_end_year, decay=decay)

        # Convert float [0,1] → uint8 [0,100]
        agg = agg.with_columns(
            (pl.col("prob_tam").clip(0.0, 1.0) * 100).round().cast(pl.UInt8).alias("prob_tam")
        )

        tbl = agg.select(["point_id", "prob_tam"]).to_arrow()
        final_path = out_dir / f"{tile_id}.scores.parquet"
        pq.write_table(tbl, final_path, **_FINAL_WRITE_OPTS)
        logger.info("Wrote %s (%d pixels)", final_path.name, len(agg))
        final_paths.append(final_path)

        # Clean up staging files for this tile
        for p in s_paths:
            p.unlink(missing_ok=True)

    # Remove staging dir if empty
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    return final_paths
