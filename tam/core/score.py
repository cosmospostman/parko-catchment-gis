"""tam/core/score.py — Chunked inference for TAMClassifier."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple

_nullctx = contextlib.nullcontext

import numpy as np
import polars as pl
import torch

from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, MAX_SEQ_LEN, S1_FEATURE_COLS, despeckle_s1, lin_to_db, prepare_s1_frame, prepare_s2_frame, subsample_obs_indices
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

    s1_paths = []
    for _, path in sorted(year_parquets):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        if "vh" in available and "vv" in available:
            s1_paths.append(str(path))

    if not s1_paths:
        return {}, {}

    lf = (
        pl.scan_parquet(s1_paths, low_memory=True)
        .filter(pl.col("source") == "S1")
        .drop_nulls(subset=["vh", "vv"])
        .select(["point_id", "vh", "vv"])
    )
    all_s1 = lf.collect()

    vh_db  = lin_to_db(all_s1["vh"].to_numpy().astype(np.float32))
    vv_db  = lin_to_db(all_s1["vv"].to_numpy().astype(np.float32))
    vh_vv  = vh_db - vv_db
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
    bands:        torch.Tensor        # (W, MAX_SEQ_LEN, N_FEATURES) float32
    doy:          torch.Tensor        # (W, MAX_SEQ_LEN) int64
    mask:         torch.Tensor        # (W, MAX_SEQ_LEN) bool
    n_obs:        torch.Tensor        # (W,) float32, n / MAX_SEQ_LEN
    pids:         np.ndarray          # (W,) object
    years:        np.ndarray          # (W,) int32
    is_s1:        torch.Tensor | None # (W, MAX_SEQ_LEN) bool; None for S2-only models
    global_feats: np.ndarray | None   # (W, n_global_features) float32; None if no global head


class _TransferredBatch(NamedTuple):
    """Batch with gate tensors already on GPU and full tensors pinned + transferred.

    Produced by the H2D transfer thread so the GPU consumer does zero CPU work
    and zero H2D stalls on the critical scoring path.
    """
    # Gate tensors — on device, ready for the gate forward pass
    gate_bands_dev:  "torch.Tensor"        # (B, T_gate, F) float32 on device
    gate_doy_dev:    "torch.Tensor"        # (B, T_gate) int64 on device
    gate_mask_dev:   "torch.Tensor"        # (B, T_gate) bool on device
    gate_n_obs_dev:  "torch.Tensor"        # (B,) float32 on device
    gate_is_s1_dev:  "torch.Tensor | None" # (B, T_gate) bool on device, or None
    # Full tensors — on device (transferred after gate pass selects survivors)
    bands_dev:       "torch.Tensor"        # (B, T_full, F) on device
    doy_dev:         "torch.Tensor"        # (B, T_full) on device
    mask_dev:        "torch.Tensor"        # (B, T_full) on device
    n_obs_dev:       "torch.Tensor"        # (B,) on device
    is_s1_dev:       "torch.Tensor | None" # (B, T_full) on device, or None
    global_feats_dev: "torch.Tensor | None" # (B, G) on device, or None
    # CPU-side identity (never transferred)
    pids:            np.ndarray
    years:           np.ndarray
    B:               int                   # batch size
    # CUDA event recorded on transfer_stream after all H2D DMAs complete.
    # compute_stream waits on this before launching kernels — GPU-side sync only,
    # no CPU stall.  None when not using CUDA.
    xfer_event:      "torch.cuda.Event | None"


class _RawChunk(NamedTuple):
    """Pre-extracted numpy arrays from a Polars chunk — built in the parser thread.

    Keeps all Polars GIL-holding work single-threaded so that the N prep workers
    receive pure numpy/numba inputs and can run without serialising on the GIL.
    """
    feat:     np.ndarray          # (N, n_feat) float32 — spectral/SAR feature matrix
    is_s1:    np.ndarray | None   # (N,) bool — True for S1 rows; None in S2-only mode
    pid_arr:  np.ndarray          # (N,) object — point_id per row
    year_arr: np.ndarray          # (N,) int32
    doy_arr:  np.ndarray          # (N,) int32
    n_feat:   int
    n_s2:     int                 # width of S2 feature block (0 in s1_only)
    n_s1:     int                 # width of S1 feature block (0 in s2-only)


class _PASlice(NamedTuple):
    """A raw PyArrow table slice for mixed-mode workers.

    In mixed mode the parser puts these on raw_q instead of _RawChunk so that
    workers do the GIL-releasing PyArrow extraction in parallel rather than
    the parser doing it serially with Polars.
    """
    tbl:            "pa.Table"   # slice with year/doy columns already appended
    s2_feature_cols: list[str]
    s1_feature_cols: list[str]
    scl_purity_min:  float


class _ZscoreArrays:
    """Vectorised per-pixel z-score lookup built once per pre-pass result.

    Converts a {pid: float32_array} dict into parallel numpy arrays so that
    batch lookups for an entire chunk reduce to one list comprehension over
    integer indices rather than repeated numpy-array dict fetches.
    """

    __slots__ = ("_pid_to_idx", "_means", "_stds", "_n_feat", "_fallback_idx")

    def __init__(
        self,
        pid_mean: dict[str, np.ndarray],
        pid_std:  dict[str, np.ndarray],
        n_feat: int,
    ) -> None:
        if not pid_mean:
            self._pid_to_idx: dict[str, int] = {}
            self._means = np.zeros((1, n_feat), dtype=np.float32)
            self._stds  = np.ones( (1, n_feat), dtype=np.float32)
        else:
            pids = list(pid_mean.keys())
            self._pid_to_idx = {p: i for i, p in enumerate(pids)}
            self._means = np.stack([pid_mean[p] for p in pids]).astype(np.float32)
            self._stds  = np.stack([pid_std[p]  for p in pids]).astype(np.float32)
        self._n_feat = n_feat
        # Row 0 of an empty lookup is zeros/ones (safe default).
        # For non-empty lookups, append a fallback row at index len(pids).
        self._means = np.vstack([self._means, np.zeros((1, n_feat), np.float32)])
        self._stds  = np.vstack([self._stds,  np.ones( (1, n_feat), np.float32)])
        self._fallback_idx = len(self._means) - 1

    def batch_lookup(self, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (means, stds) arrays of shape (len(pids), n_feat) in one numpy op."""
        idx = np.fromiter(
            (self._pid_to_idx.get(p, self._fallback_idx) for p in pids),
            dtype=np.intp, count=len(pids),
        )
        return self._means[idx], self._stds[idx]


def _extract_mixed_pa(
    pa_slice: "_PASlice",
) -> "_RawChunk | None":
    """Extract mixed-mode feature arrays directly from a PyArrow table slice.

    Pure PyArrow + numpy — releases the GIL throughout so N workers can run
    truly in parallel (unlike the Polars path which serialises on the GIL).
    Called by prep workers in mixed mode; replaces the Polars branch of
    _extract_raw_arrays for that path.
    """
    import pyarrow.compute as _pac

    tbl = pa_slice.tbl
    _s2_cols = pa_slice.s2_feature_cols
    _s1_cols = pa_slice.s1_feature_cols
    scl_purity_min = pa_slice.scl_purity_min
    n_s2 = len(_s2_cols)
    n_s1 = len(_s1_cols)
    n_feat = n_s2 + n_s1

    schema_names = set(tbl.schema.names)
    has_source = "source" in schema_names

    if has_source:
        # Evaluate source column once; reuse the BooleanArray for both the
        # scl_purity filter and the post-filter is_s1 mask.
        src_col = tbl.column("source")
        if src_col.num_chunks > 1:
            src_col = src_col.combine_chunks()
        is_s1_pa = _pac.equal(src_col, "S1")
        if "scl_purity" in schema_names:
            # or_kleene: True OR null = True (keeps S1 rows whose scl_purity is null)
            keep_pa = _pac.or_kleene(is_s1_pa, _pac.greater_equal(tbl.column("scl_purity").combine_chunks(), scl_purity_min))
            tbl = tbl.filter(keep_pa)
            # Filter is_s1_pa in-sync with the table so we don't re-evaluate source.
            is_s1_pa = is_s1_pa.filter(keep_pa)
        is_s1_bool = is_s1_pa.to_numpy(zero_copy_only=False)
    else:
        if "scl_purity" in schema_names:
            tbl = tbl.filter(_pac.greater_equal(tbl.column("scl_purity").combine_chunks(), scl_purity_min))
        is_s1_bool = np.zeros(tbl.num_rows, dtype=bool)

    if tbl.num_rows == 0:
        return None

    N = tbl.num_rows
    feat = np.zeros((N, n_feat), dtype=np.float32)

    # Compute S2 and S1 row indices once from the boolean mask; use take() on
    # both paths so we avoid re-evaluating the filter predicate a second time.
    s2_idx = np.where(~is_s1_bool)[0]
    s1_idx = np.where(is_s1_bool)[0]
    # pa.array wrapping is zero-copy for int64 on little-endian platforms.
    import pyarrow as _pa_local
    s2_idx_pa = _pa_local.array(s2_idx, type=_pa_local.int64())
    s1_idx_pa = _pa_local.array(s1_idx, type=_pa_local.int64())

    # S2 bands: extract the float block via Polars from_arrow → to_numpy.
    # pl.from_arrow zero-copies the Arrow buffers; to_numpy writes a single
    # C-contiguous float32 matrix in one C-level pass — 5× faster than a
    # per-column combine_chunks+asarray loop for the same data.
    if s2_idx.size > 0:
        raw_s2_cols = [c for c in _s2_cols if c not in ("NDVI", "NDWI", "EVI", "MAVI", "NDRE", "CI_RE")]
        present_s2 = [c for c in raw_s2_cols if c in schema_names]
        if present_s2:
            s2_sub = tbl.select(present_s2).take(s2_idx_pa)
            feat[s2_idx, :len(present_s2)] = pl.from_arrow(s2_sub).to_numpy(order="c")

    # S1 features: VH/VV → dB + ratio features
    if s1_idx.size > 0 and "vh" in schema_names and "vv" in schema_names:
        s1_sub = tbl.select(["vh", "vv"]).take(s1_idx_pa)
        vh_s = np.asarray(s1_sub.column("vh"), dtype=np.float32)
        vv_s = np.asarray(s1_sub.column("vv"), dtype=np.float32)
        s1_vh  = lin_to_db(vh_s)
        s1_vv  = lin_to_db(vv_s)
        vh_vv  = (s1_vh - s1_vv).astype(np.float32)
        denom  = vh_s + vv_s
        rvi    = np.where(denom > 0, 4 * vh_s / denom, 0.0).astype(np.float32)
        col_map = {"s1_vh": s1_vh, "s1_vv": s1_vv, "s1_vh_vv": vh_vv, "s1_rvi": rvi}
        for j, col in enumerate(_s1_cols):
            if col in col_map:
                feat[s1_idx, n_s2 + j] = col_map[col]

    pid_arr  = np.asarray(tbl.column("point_id").combine_chunks())
    year_arr = np.asarray(tbl.column("year").combine_chunks(), dtype=np.int32)
    doy_arr  = np.asarray(tbl.column("doy").combine_chunks(),  dtype=np.int32)
    return _RawChunk(feat, is_s1_bool, pid_arr, year_arr, doy_arr, n_feat, n_s2, n_s1)


def _extract_raw_arrays(
    chunk: pl.DataFrame,
    scl_purity_min: float,
    s1_only: bool,
    mixed: bool,
    feature_cols: list[str] | None,
    s2_feature_cols: list[str] | None,
    s1_feature_cols: list[str] | None,
    pixel_zscore: bool,
    vh_mean: dict | None,
    vh_std:  dict | None,
    vv_mean: dict | None,
    vv_std:  dict | None,
    despeckle_lookup,
) -> "_RawChunk | None":
    """Extract feature numpy arrays from a Polars chunk — runs in the parser thread.

    All Polars .filter(), .cast(), .to_numpy() calls are here so that prep workers
    receive plain numpy arrays and need no GIL-holding Polars calls.
    Returns None if the chunk is empty after filtering.
    """
    from tam.core._preprocess_numba import extract_features

    if mixed:
        _s2_cols = s2_feature_cols or list(ALL_FEATURE_COLS)
        _s1_cols = s1_feature_cols or ["s1_vh", "s1_vv"]
        n_s2 = len(_s2_cols)
        n_s1 = len(_s1_cols)
        n_feat = n_s2 + n_s1

        has_source = "source" in chunk.columns
        if has_source:
            is_s1_series = chunk["source"] == "S1"
            if "scl_purity" in chunk.columns:
                keep_series = is_s1_series | (chunk["scl_purity"] >= scl_purity_min)
                if not keep_series.all():
                    chunk = chunk.filter(keep_series)
                    is_s1_series = chunk["source"] == "S1"
            is_s1_bool = is_s1_series.to_numpy()
        else:
            if "scl_purity" in chunk.columns:
                chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
            is_s1_bool = np.zeros(len(chunk), dtype=bool)

        if chunk.is_empty():
            return None

        N = len(chunk)
        feat = np.zeros((N, n_feat), dtype=np.float32)

        s1_mask = pl.Series(is_s1_bool)
        if is_s1_bool.any():
            s1_chunk = prepare_s1_frame(chunk.filter(s1_mask))
            s1_idx = np.where(is_s1_bool)[0]
            for j, col in enumerate(_s1_cols):
                if col in s1_chunk.columns:
                    feat[s1_idx, n_s2 + j] = s1_chunk[col].cast(pl.Float32).to_numpy()

        if (~is_s1_bool).any():
            s2_chunk = prepare_s2_frame(chunk.filter(~s1_mask), scl_purity_min=0.0, feature_cols=_s2_cols)
            s2_vals = s2_chunk.select(
                [pl.col(c).cast(pl.Float32) for c in _s2_cols]
            ).to_numpy()
            s2_idx = np.where(~is_s1_bool)[0]
            feat[s2_idx, :n_s2] = s2_vals

        pid_arr  = chunk["point_id"].to_numpy()
        year_arr = chunk["year"].to_numpy().astype(np.int32)
        doy_arr  = chunk["doy"].to_numpy().astype(np.int32)
        return _RawChunk(feat, is_s1_bool, pid_arr, year_arr, doy_arr, n_feat, n_s2, n_s1)

    elif s1_only:
        if "source" in chunk.columns:
            chunk = chunk.filter(pl.col("source") == "S1")
        chunk = chunk.drop_nulls(subset=["vh", "vv"])
        if chunk.is_empty():
            return None
        feat = _extract_s1_features(chunk, pixel_zscore=pixel_zscore,
                                    vh_mean=vh_mean, vh_std=vh_std,
                                    vv_mean=vv_mean, vv_std=vv_std,
                                    despeckle_lookup=despeckle_lookup)
        n_feat = _N_FEATURES_S1
        pid_arr  = chunk["point_id"].to_numpy()
        year_arr = chunk["year"].to_numpy().astype(np.int32)
        doy_arr  = chunk["doy"].to_numpy().astype(np.int32)
        return _RawChunk(feat, None, pid_arr, year_arr, doy_arr, n_feat, 0, n_feat)

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
            chunk = prepare_s2_frame(chunk, scl_purity_min=0.0, feature_cols=feature_cols)
            n_feat = len(feature_cols)
            feat = chunk.select(
                [pl.col(c).cast(pl.Float32) for c in feature_cols]
            ).to_numpy()

        pid_arr  = chunk["point_id"].to_numpy()
        year_arr = chunk["year"].to_numpy().astype(np.int32)
        doy_arr  = chunk["doy"].to_numpy().astype(np.int32)
        return _RawChunk(feat, None, pid_arr, year_arr, doy_arr, n_feat, n_feat, 0)


def _preprocess(
    raw: "_RawChunk",
    band_mean: np.ndarray,
    band_std: np.ndarray,
    min_obs_per_year: int,
    pin: bool = False,
    mixed: bool = True,
    pixel_zscore: bool = False,
    pixel_zscore_stats: _ZscoreArrays | tuple[dict, dict] | None = None,
    s1_zscore_stats: _ZscoreArrays | tuple[dict, dict] | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = 4096,
    global_feat_mean: np.ndarray | None = None,
    global_feat_std: np.ndarray | None = None,
    n_global_features: int = 0,
) -> list[_PreparedBatch]:
    """CPU-side preprocessing using numba kernels for maximum throughput.

    Receives pre-extracted numpy arrays (_RawChunk built in the parser thread) so
    all Polars GIL-holding work is already done; only numba/numpy runs here.

    Returns a list of _PreparedBatch objects, each at most batch_size windows.
    """
    from tam.core._preprocess_numba import (
        fill_windows,
        fill_windows_mixed, fill_windows_mixed_subsample, fill_windows_zscore,
        count_s2_s1_per_window,
        compute_window_stats, compute_window_stats_s2only, compute_band_summaries,
    )
    from tam.core.dataset import MIN_S1_OBS_PER_YEAR

    feat, is_s1_np, pid_arr, year_arr, doy_arr, n_feat, n_s2, n_s1 = raw

    # Step 2: find group boundaries (pixel-year windows)
    pid_change  = np.empty(len(pid_arr), dtype=bool)
    year_change = np.empty(len(pid_arr), dtype=bool)
    pid_change[0] = year_change[0] = True
    pid_change[1:]  = pid_arr[1:]  != pid_arr[:-1]
    year_change[1:] = year_arr[1:] != year_arr[:-1]

    boundaries = np.where(pid_change | year_change)[0].astype(np.int64)
    ends       = np.append(boundaries[1:], np.int64(len(pid_arr))).astype(np.int64)
    lengths    = (ends - boundaries).astype(np.int32)

    if mixed:
        # Count S2 and S1 obs per window in parallel via numba kernel.
        n_s2_per_win = np.empty(len(boundaries), dtype=np.int32)
        n_s1_per_win = np.empty(len(boundaries), dtype=np.int32)
        count_s2_s1_per_window(is_s1_np, boundaries, ends, n_s2_per_win, n_s1_per_win)
        valid = (n_s2_per_win >= min_obs_per_year) & (n_s1_per_win >= MIN_S1_OBS_PER_YEAR)
    else:
        valid = lengths >= min_obs_per_year

    if not valid.any():
        return []

    valid_starts  = boundaries[valid]          # already int64
    valid_lengths = lengths[valid]
    capped = np.minimum(valid_lengths, max_seq_len).astype(np.int32)
    W = int(valid.sum())

    # Step 3: fill padded tensors
    bands_np  = np.zeros((W, max_seq_len, n_feat), dtype=np.float32)
    doy_np    = np.zeros((W, max_seq_len), dtype=np.int64)
    mask_np   = np.ones( (W, max_seq_len), dtype=np.bool_)
    is_s1_out = np.zeros((W, max_seq_len), dtype=np.bool_) if mixed else None

    pids_at_starts = pid_arr[valid_starts]

    if mixed:
        # Compute per-window z-score stats directly from the chunk's own rows.
        # No pre-pass needed: parquet is pixel-sorted so all obs for a pixel
        # land in the same chunk, making the chunk's stats complete for each pixel.
        s2_pm = np.empty((W, n_s2), dtype=np.float32)
        s2_ps = np.empty((W, n_s2), dtype=np.float32)
        s1_pm = np.empty((W, n_s1), dtype=np.float32)
        s1_ps = np.empty((W, n_s1), dtype=np.float32)
        compute_window_stats(
            feat, is_s1_np, valid_starts, ends[valid],
            np.int64(n_s2), s2_pm, s2_ps, s1_pm, s1_ps,
        )

        # Separate windows that fit within max_seq_len from those needing S1 subsampling.
        # Subsampling is rare (only multi-year windows with high S1 cadence exceed T=128).
        needs_subsample = valid_lengths > max_seq_len

        if needs_subsample.any():
            # Fast path (non-subsampled subset): numba kernel.
            fast_mask = ~needs_subsample
            if fast_mask.any():
                fast_idx = np.where(fast_mask)[0].astype(np.int64)
                fill_windows_mixed(
                    feat, is_s1_np, doy_arr,
                    valid_starts[fast_idx], capped[fast_idx],
                    s2_pm[fast_idx], s2_ps[fast_idx],
                    s1_pm[fast_idx], s1_ps[fast_idx],
                    np.int64(n_s2),
                    bands_np, doy_np, mask_np, is_s1_out,
                )

            # Overlong windows: numba parallel kernel (replaces Python loop).
            sub_idx = np.where(needs_subsample)[0].astype(np.int64)
            fill_windows_mixed_subsample(
                feat, is_s1_np, doy_arr,
                valid_starts[sub_idx], valid_lengths[sub_idx].astype(np.int32),
                s2_pm[sub_idx], s2_ps[sub_idx],
                s1_pm[sub_idx], s1_ps[sub_idx],
                np.int64(n_s2), np.int64(max_seq_len),
                bands_np, doy_np, mask_np, is_s1_out,
            )
        else:
            # All windows fit: run entirely in the numba kernel.
            fill_windows_mixed(
                feat, is_s1_np, doy_arr,
                valid_starts, capped,
                s2_pm, s2_ps, s1_pm, s1_ps,
                np.int64(n_s2),
                bands_np, doy_np, mask_np, is_s1_out,
            )

    elif pixel_zscore and n_s2 > 0:
        # Compute per-window stats from chunk — no pre-pass needed.
        all_pm = np.empty((W, n_feat), dtype=np.float32)
        all_ps = np.empty((W, n_feat), dtype=np.float32)
        compute_window_stats_s2only(feat, valid_starts, ends[valid], all_pm, all_ps)
        fill_windows_zscore(feat, doy_arr, valid_starts, capped,
                            all_pm, all_ps, bands_np, doy_np, mask_np)
    else:
        fill_windows(feat, doy_arr, valid_starts, capped, band_mean, band_std,
                     bands_np, doy_np, mask_np)

    pids  = pid_arr[valid_starts]
    years = year_arr[valid_starts]
    n_obs_np = (capped / max_seq_len).astype(np.float32)

    # Compute per-window band summaries (p5/p95/std) from the assembled pixel data.
    # feat is still in raw (pre-z-score) units here, matching training convention.
    # Only S2 columns are used; in mixed mode we build a compact S2-only view first.
    global_feats_np: np.ndarray | None = None
    if n_global_features > 0 and global_feat_mean is not None and global_feat_std is not None:
        n_sum_cols = n_s2 if mixed else n_feat
        if mixed and is_s1_np is not None:
            # Build a compact array containing only S2 rows for the summary kernel.
            s2_only_feat = feat[~is_s1_np, :n_sum_cols]
            # Per-window S2 row counts via cumsum — O(N) vs O(W*obs) Python loop.
            s2_cs = np.empty(len(is_s1_np) + 1, dtype=np.int64)
            s2_cs[0] = 0
            np.cumsum(~is_s1_np, out=s2_cs[1:])
            ends_v = ends[valid]
            s2_counts = s2_cs[ends_v] - s2_cs[valid_starts]
            s2_starts = np.empty(W, dtype=np.int64)
            s2_starts[0] = 0
            np.cumsum(s2_counts[:-1], out=s2_starts[1:])
            s2_ends = s2_starts + s2_counts
            gf = np.empty((W, n_sum_cols * 3), dtype=np.float32)
            compute_band_summaries(s2_only_feat, s2_starts, s2_ends, gf)
        else:
            gf = np.empty((W, n_sum_cols * 3), dtype=np.float32)
            compute_band_summaries(feat[:, :n_sum_cols], valid_starts, ends[valid], gf)
        safe_std = np.where(global_feat_std < 1e-6, 1.0, global_feat_std)
        global_feats_np = ((gf - global_feat_mean) / safe_std).astype(np.float32)

    # Slice into batch_size-sized pieces before converting to tensors.
    # This bounds the size of each prep_q item regardless of how many windows
    # the raw chunk contained, keeping pinned-memory pressure predictable.
    batches: list[_PreparedBatch] = []
    W = len(pids)
    for s in range(0, W, batch_size):
        e = min(s + batch_size, W)
        # Row slices of C-contiguous arrays are already contiguous — ascontiguousarray
        # is a no-op and avoids the 2ms copy that .copy() would trigger per batch.
        b_bands = torch.from_numpy(np.ascontiguousarray(bands_np[s:e]))
        b_doy   = torch.from_numpy(np.ascontiguousarray(doy_np[s:e]))
        b_mask  = torch.from_numpy(np.ascontiguousarray(mask_np[s:e]))
        b_nobs  = torch.from_numpy(np.ascontiguousarray(n_obs_np[s:e]))
        b_is_s1 = torch.from_numpy(np.ascontiguousarray(is_s1_out[s:e])) if is_s1_out is not None else None
        b_gf    = np.ascontiguousarray(global_feats_np[s:e]) if global_feats_np is not None else None
        if pin:
            b_bands = b_bands.pin_memory()
            b_doy   = b_doy.pin_memory()
            b_mask  = b_mask.pin_memory()
            b_nobs  = b_nobs.pin_memory()
        batches.append(_PreparedBatch(b_bands, b_doy, b_mask, b_nobs, pids[s:e], years[s:e], b_is_s1, b_gf))
    return batches


def _compute_s2_pixel_zscore_stats(
    year_parquets: list[tuple[int, Path]],
    feature_cols: list[str],
    scl_purity_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-pass: compute per-pixel mean and std for each feature col across all years.

    Returns (pid_mean, pid_std) dicts mapping point_id → float32 array of length n_feat.
    Used to apply pixel z-scoring at inference time, matching the training normalisation.
    """
    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "scl_purity"] + raw_band_cols

    # Filter to S2-capable parquets (skip S1-only shards that lack band columns)
    s2_paths = []
    for _, path in sorted(year_parquets):
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(path).schema_arrow.names)
        if any(c in available for c in raw_band_cols):
            s2_paths.append(str(path))

    if not s2_paths:
        return {}, {}

    cols_expr = [c for c in read_cols]
    lf = pl.scan_parquet(s2_paths, low_memory=True)
    # Drop S1 rows if source column present, apply scl_purity filter
    if "source" in lf.collect_schema().names():
        lf = lf.filter(pl.col("source") != "S1")
    if "scl_purity" in lf.collect_schema().names():
        lf = lf.filter(pl.col("scl_purity") >= scl_purity_min)
    # Select only needed columns (schema may vary across files; missing → null)
    available_cols = set(lf.collect_schema().names())
    sel_cols = [c for c in cols_expr if c in available_cols]
    lf = lf.select(sel_cols)

    # Compute S2 index features inline (NDVI, NDWI, EVI) if requested
    extra_exprs: list[pl.Expr] = []
    if "NDVI" in feature_cols and "B08" in available_cols and "B04" in available_cols:
        extra_exprs.append(
            ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04")).clip(lower_bound=1e-6)).alias("NDVI")
        )
    if "NDWI" in feature_cols and "B03" in available_cols and "B08" in available_cols:
        extra_exprs.append(
            ((pl.col("B03") - pl.col("B08")) / (pl.col("B03") + pl.col("B08")).clip(lower_bound=1e-6)).alias("NDWI")
        )
    if "EVI" in feature_cols and "B08" in available_cols and "B04" in available_cols and "B02" in available_cols:
        extra_exprs.append(
            (2.5 * (pl.col("B08") - pl.col("B04")) /
             (pl.col("B08") + 6.0 * pl.col("B04") - 7.5 * pl.col("B02") + 1.0).clip(lower_bound=1e-6)).alias("EVI")
        )
    if extra_exprs:
        lf = lf.with_columns(extra_exprs)

    present_feat_cols = [c for c in feature_cols if c in set(lf.collect_schema().names())]
    grp = lf.group_by("point_id").agg([
        *(pl.col(c).mean().alias(f"{c}_mean") for c in present_feat_cols),
        *(pl.col(c).std().fill_null(1.0).clip(lower_bound=1e-6).alias(f"{c}_std")
          for c in present_feat_cols),
    ]).collect()

    pids = grp["point_id"].to_numpy()
    means: list[np.ndarray] = []
    stds:  list[np.ndarray] = []
    for col in feature_cols:
        if col not in present_feat_cols:
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

    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "scl_purity"] + raw_band_cols

    chunks: list[pl.DataFrame] = []
    for _, path in sorted(year_parquets):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        if not any(c in available for c in raw_band_cols):
            continue  # S1-only shard — no S2 bands
        cols = [c for c in read_cols if c in available]
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pl.from_arrow(pf.read_row_group(rg, columns=cols)))

    if not chunks:
        return {}

    df = prepare_s2_frame(pl.concat(chunks), scl_purity_min, feature_cols)

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


def _has_flash_attn() -> bool:
    """True if flash_attn varlen kernel is importable AND supports the current GPU."""
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return False
        # flash-attn v4 beta requires sm_9x/10x/11x; sm_12x (Blackwell) not yet supported
        major, _ = torch.cuda.get_device_capability()
        return major in (9, 10, 11)
    except (ImportError, AssertionError):
        return False


def _gpu_score(
    prepared: _PreparedBatch,
    model: TAMClassifier,
    device: str,
    gate_threshold: float = 0.0,
    T_gate: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """GPU-side: transfer one batch, run inference, return (pids, years, probs, n_gate_cut).

    Each prepared item is already batch_size or smaller (sliced in _preprocess),
    so this is a single forward pass — no internal loop.

    Uses flash-attn varlen forward (no padding waste) when flash_attn is installed,
    falling back to the standard padded forward otherwise.

    When gate_threshold > 0, runs a cheap T_gate-token forward pass first and skips
    the full forward for pixels scoring below gate_threshold. Rejected pixels are
    returned with prob=0.0.
    """
    bands_th, doy_th, mask_th, n_obs_th, pids, years, is_s1_th, global_feats_np = prepared
    B = len(pids)
    use_varlen = device.startswith("cuda") and _has_flash_attn()

    gf_batch: torch.Tensor | None = None
    if global_feats_np is not None and model.n_global_features > 0:
        gf_batch = torch.from_numpy(global_feats_np).to(device, non_blocking=True)

    n_gate_cut = 0

    with torch.inference_mode():
        if gate_threshold > 0.0:
            # Gate pass: farthest-point DOY sampling → T_gate-token forward.
            # n_obs_gate uses the full max_seq_len denominator so the model's
            # sparsity scalar (n/T) matches what it saw during training.
            from tam.core._preprocess_numba import build_gate_tensors
            T_full = bands_th.shape[1]
            gate_t = min(T_gate, T_full)
            n_feat = bands_th.shape[2]

            bands_np_cpu = bands_th.numpy()
            doy_np_cpu   = doy_th.numpy()
            mask_np_cpu  = mask_th.numpy()
            is_s1_np_cpu = is_s1_th.numpy().astype(bool) if is_s1_th is not None else np.zeros((B, T_full), dtype=bool)

            gate_bands   = np.zeros((B, gate_t, n_feat), dtype=np.float32)
            gate_doy     = np.zeros((B, gate_t), dtype=np.int64)
            gate_is_s1   = np.zeros((B, gate_t), dtype=bool)
            gate_mask_np = np.ones( (B, gate_t), dtype=bool)

            build_gate_tensors(
                bands_np_cpu, doy_np_cpu, mask_np_cpu, is_s1_np_cpu,
                np.int64(gate_t),
                gate_bands, gate_doy, gate_mask_np, gate_is_s1,
            )

            gate_n_obs = np.clip((~gate_mask_np).sum(axis=1), 1, None).astype(np.float32) / T_full
            gate_prob, _ = model(
                torch.from_numpy(gate_bands).to(device, non_blocking=True),
                torch.from_numpy(gate_doy).to(device, non_blocking=True),
                torch.from_numpy(gate_mask_np).to(device, non_blocking=True),
                torch.from_numpy(gate_n_obs).to(device, non_blocking=True),
                global_feats=gf_batch,
                is_s1=torch.from_numpy(gate_is_s1).to(device, non_blocking=True) if is_s1_th is not None else None,
            )

            # Stay on GPU: threshold + index without a CPU round-trip.
            # len() on a CUDA tensor is the one unavoidable sync (need count for branch).
            survivor_idx_dev = torch.where(gate_prob.float() >= gate_threshold)[0]
            n_survivors      = len(survivor_idx_dev)
            n_gate_cut       = B - n_survivors

            probs_out = np.zeros(B, dtype=np.float32)
            if n_survivors > 0:
                # survivor_idx for CPU indexing (bands_th etc. are still on CPU here)
                survivor_idx = survivor_idx_dev.cpu().numpy()
                if use_varlen:
                    mask_surv  = mask_th[survivor_idx]
                    seq_lens   = (~mask_surv).sum(dim=1).to(torch.int32)
                    cu_seqlens = torch.zeros(n_survivors + 1, dtype=torch.int32, device=device)
                    cu_seqlens[1:] = seq_lens.to(device).cumsum(0)
                    max_seqlen = int(seq_lens.max().item())
                    bands_surv = bands_th[survivor_idx].to(device, non_blocking=True)
                    doy_surv   = doy_th[survivor_idx].to(device, non_blocking=True)
                    n_obs_surv = n_obs_th[survivor_idx].to(device, non_blocking=True)
                    is_s1_surv = is_s1_th[survivor_idx].to(device, non_blocking=True) if is_s1_th is not None else None
                    valid_mask = ~mask_surv.to(device)
                    prob, _ = model.forward_varlen(
                        bands_flat=bands_surv[valid_mask],
                        doy_flat=doy_surv[valid_mask],
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                        n_obs=n_obs_surv,
                        global_feats=gf_batch[survivor_idx_dev] if gf_batch is not None else None,
                        is_s1_flat=is_s1_surv[valid_mask] if is_s1_surv is not None else None,
                    )
                else:
                    prob, _ = model(
                        bands_th[survivor_idx].to(device, non_blocking=True),
                        doy_th[survivor_idx].to(device, non_blocking=True),
                        mask_th[survivor_idx].to(device, non_blocking=True),
                        n_obs_th[survivor_idx].to(device, non_blocking=True),
                        global_feats=gf_batch[survivor_idx_dev] if gf_batch is not None else None,
                        is_s1=is_s1_th[survivor_idx].to(device, non_blocking=True) if is_s1_th is not None else None,
                    )
                probs_out[survivor_idx] = prob.cpu().float().numpy()

            return pids, years, probs_out, n_gate_cut

        else:
            if use_varlen:
                seq_lens   = (~mask_th).sum(dim=1).to(torch.int32)
                cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
                cu_seqlens[1:] = seq_lens.to(device).cumsum(0)
                max_seqlen = int(seq_lens.max().item())
                bands_dev  = bands_th.to(device, non_blocking=True)
                doy_dev    = doy_th.to(device, non_blocking=True)
                n_obs_dev  = n_obs_th.to(device, non_blocking=True)
                is_s1_dev  = is_s1_th.to(device, non_blocking=True) if is_s1_th is not None else None
                valid_mask = ~mask_th.to(device)
                prob, _ = model.forward_varlen(
                    bands_flat=bands_dev[valid_mask],
                    doy_flat=doy_dev[valid_mask],
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    n_obs=n_obs_dev,
                    global_feats=gf_batch,
                    is_s1_flat=is_s1_dev[valid_mask] if is_s1_dev is not None else None,
                )
            else:
                prob, _ = model(
                    bands_th.to(device, non_blocking=True),
                    doy_th.to(device, non_blocking=True),
                    mask_th.to(device, non_blocking=True),
                    n_obs_th.to(device, non_blocking=True),
                    global_feats=gf_batch,
                    is_s1=is_s1_th.to(device, non_blocking=True) if is_s1_th is not None else None,
                )
            return pids, years, prob.cpu().float().numpy(), 0


def _gpu_forward(
    tb: "_TransferredBatch",
    model: "TAMClassifier",
    gate_threshold: float,
    compute_stream: "torch.cuda.Stream | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, int]":
    """Pure GPU forward — all tensors already on device, no H2D, no CPU prep.

    Returns (pids, years, probs_out, n_gate_cut).
    """
    use_varlen = _has_flash_attn()
    B = tb.B

    # GPU-side sync: wait for the transfer stream's DMA to finish before
    # launching any compute kernels.  This is a device-side dependency only —
    # the CPU returns immediately.
    if compute_stream is not None and tb.xfer_event is not None:
        compute_stream.wait_event(tb.xfer_event)

    _stream_ctx = (torch.cuda.stream(compute_stream)
                   if compute_stream is not None else _nullctx())

    with torch.inference_mode(), _stream_ctx:
        if gate_threshold > 0.0:
            gate_prob, _ = model(
                tb.gate_bands_dev,
                tb.gate_doy_dev,
                tb.gate_mask_dev,
                tb.gate_n_obs_dev,
                global_feats=tb.global_feats_dev,
                is_s1=tb.gate_is_s1_dev,
            )
            survivor_idx_dev = torch.where(gate_prob.float() >= gate_threshold)[0]
            n_survivors = len(survivor_idx_dev)   # one unavoidable sync
            n_gate_cut  = B - n_survivors

            probs_out = np.zeros(B, dtype=np.float32)
            if n_survivors > 0:
                if use_varlen:
                    mask_surv  = tb.mask_dev[survivor_idx_dev]
                    seq_lens   = (~mask_surv).sum(dim=1).to(torch.int32)
                    cu_seqlens = torch.zeros(n_survivors + 1, dtype=torch.int32,
                                             device=tb.bands_dev.device)
                    cu_seqlens[1:] = seq_lens.cumsum(0)
                    max_seqlen = int(seq_lens.max().item())
                    valid_mask = ~mask_surv
                    prob, _ = model.forward_varlen(
                        bands_flat=tb.bands_dev[survivor_idx_dev][valid_mask],
                        doy_flat=tb.doy_dev[survivor_idx_dev][valid_mask],
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                        n_obs=tb.n_obs_dev[survivor_idx_dev],
                        global_feats=(tb.global_feats_dev[survivor_idx_dev]
                                      if tb.global_feats_dev is not None else None),
                        is_s1_flat=(tb.is_s1_dev[survivor_idx_dev][valid_mask]
                                    if tb.is_s1_dev is not None else None),
                    )
                else:
                    prob, _ = model(
                        tb.bands_dev[survivor_idx_dev],
                        tb.doy_dev[survivor_idx_dev],
                        tb.mask_dev[survivor_idx_dev],
                        tb.n_obs_dev[survivor_idx_dev],
                        global_feats=(tb.global_feats_dev[survivor_idx_dev]
                                      if tb.global_feats_dev is not None else None),
                        is_s1=(tb.is_s1_dev[survivor_idx_dev]
                               if tb.is_s1_dev is not None else None),
                    )
                survivor_idx_cpu = survivor_idx_dev.cpu().numpy()
                probs_out[survivor_idx_cpu] = prob.cpu().float().numpy()
            return tb.pids, tb.years, probs_out, n_gate_cut

        else:
            if use_varlen:
                seq_lens   = (~tb.mask_dev).sum(dim=1).to(torch.int32)
                cu_seqlens = torch.zeros(B + 1, dtype=torch.int32,
                                         device=tb.bands_dev.device)
                cu_seqlens[1:] = seq_lens.cumsum(0)
                max_seqlen = int(seq_lens.max().item())
                valid_mask = ~tb.mask_dev
                prob, _ = model.forward_varlen(
                    bands_flat=tb.bands_dev[valid_mask],
                    doy_flat=tb.doy_dev[valid_mask],
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    n_obs=tb.n_obs_dev,
                    global_feats=tb.global_feats_dev,
                    is_s1_flat=(tb.is_s1_dev[valid_mask]
                                if tb.is_s1_dev is not None else None),
                )
            else:
                prob, _ = model(
                    tb.bands_dev, tb.doy_dev, tb.mask_dev, tb.n_obs_dev,
                    global_feats=tb.global_feats_dev,
                    is_s1=tb.is_s1_dev,
                )
            return tb.pids, tb.years, prob.cpu().float().numpy(), 0


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


def _flush_to_writer(writer: "pq.ParquetWriter", pids_list: list, years_list: list, probs_list: list) -> None:
    """Write one row group to a streaming ParquetWriter and clear the buffers."""
    import pyarrow as pa
    tbl = pa.table({
        "point_id":     pa.array(np.concatenate(pids_list),                    type=pa.large_utf8()),
        "year":         pa.array(np.concatenate(years_list).astype("int16"),   type=pa.int16()),
        "prob_tam_raw": pa.array(np.concatenate(probs_list).astype("float32"), type=pa.float32()),
    })
    writer.write_table(tbl)


def score_pixels_chunked(
    parquet: Path,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 10,
    batch_size: int = 4096,
    buffer_row_groups: int = 4,
    target_chunk_rows: int = 393_216,
    n_prep_workers: int = 5,
    device: str | None = None,
    tile_id: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    n_total_pixels: int | None = None,
    s1_only: bool = False,
    mixed: bool = True,
    pixel_zscore: bool = False,
    s1_despeckle_window: int = 0,
    feature_cols: list[str] | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    global_feat_mean: np.ndarray | None = None,
    global_feat_std: np.ndarray | None = None,
    pixel_zscore_stats: _ZscoreArrays | tuple[dict, dict] | None = None,
    s1_zscore_stats: _ZscoreArrays | tuple[dict, dict] | None = None,
    # accumulators — pass across years to merge results before final aggregation
    _all_pids:  list | None = None,
    _all_years: list | None = None,
    _all_probs: list | None = None,
    # streaming output — when set, scores are written to this writer instead of
    # accumulated in memory; caller owns the writer lifecycle
    out_writer: "pq.ParquetWriter | None" = None,
    write_flush_rows: int = 500_000,
    progress_log_interval: int = 50_000,
    gate_threshold: float = 0.0,
    T_gate: int = 8,
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
    pin = False  # pin_memory cost (325ms first alloc) far exceeds H2D transfer savings
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32: ~2× attention throughput on Ampere+
    model.to(device)
    model.eval()

    logger.info("Warming up numba kernels ...")
    _numba_warmup()

    # Warm up GPU: pre-pin memory (CUDA allocator caches pinned pages after first call)
    # and trigger cuDNN autotuning at the shapes we'll use during inference.
    # Without this, the first real batch pays ~2s in pin_memory page faults and
    # ~700ms in cuDNN algorithm selection.
    if device.startswith("cuda") and torch.cuda.is_available():
        _n_feat  = model.n_bands if hasattr(model, "n_bands") else 14
        _T_full  = getattr(model, "_max_seq_len", MAX_SEQ_LEN)
        # 1. Pre-pin: allocate and immediately free pinned memory at batch shape.
        #    PyTorch caches pinned allocations so subsequent calls are O(1).
        for _T_w in (T_gate, _T_full):
            _ = torch.zeros(batch_size, _T_w, _n_feat).pin_memory()
            _ = torch.zeros(batch_size, _T_w, dtype=torch.int64).pin_memory()
            _ = torch.zeros(batch_size, _T_w, dtype=torch.bool).pin_memory()
        del _
        # 2. Trigger cuDNN algorithm selection at the exact shapes used in inference.
        with torch.inference_mode():
            for _T_w in (T_gate, _T_full):
                _dm_b = torch.zeros(batch_size, _T_w, _n_feat, device=device)
                _dm_d = torch.zeros(batch_size, _T_w, dtype=torch.int64, device=device)
                _dm_m = torch.zeros(batch_size, _T_w, dtype=torch.bool, device=device)
                _dm_n = torch.full((batch_size,), _T_w / _T_full, device=device)
                try:
                    model(_dm_b, _dm_d, _dm_m, _dm_n)
                except Exception:
                    pass
        torch.cuda.synchronize(device)

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

    # Two-stage read pipeline:
    #   io_q:  IO thread → parser thread.  Holds raw PyArrow tables (one per rg).
    #          Depth=n_io_prefetch lets the IO thread read ahead while the parser
    #          runs date extraction + sub-division on the previous rg.
    #   raw_q: parser thread → prep workers.  Holds target_chunk_rows-sized Polars
    #          slices with year/doy columns already attached.
    #   prep_q: prep workers → GPU.
    n_io_prefetch = 4  # row groups to buffer in IO thread ahead of parser
    io_q:   Queue = Queue(maxsize=n_io_prefetch)
    raw_q:  Queue = Queue(maxsize=n_prep_workers * 4)
    prep_q: Queue = Queue(maxsize=n_prep_workers * 8)
    # Stage 3→4: scored chunks (only used when out_writer is set)
    out_q: Queue = Queue(maxsize=8)

    # --- Stage 1a: IO thread — reads raw row groups, no processing ---
    def _io_reader() -> None:
        try:
            for rg in range(n_rg):
                tbl = pf.read_row_group(rg, columns=read_cols)
                io_q.put(tbl)
        except Exception:
            logger.exception("IO reader thread crashed")
        finally:
            io_q.put(_SENTINEL)

    # --- Stage 1b: parser thread — date extraction, year filter, sub-division ---
    def _reader() -> None:
        # In mixed mode the accumulator holds raw PyArrow tables and workers do
        # _extract_mixed_pa() (GIL-releasing) so N workers run truly in parallel.
        # In s1_only/s2-only mode we keep the Polars path (less common, single worker).
        import pyarrow as _pa
        import pyarrow.compute as _pac

        _eff_s2_cols = s2_feature_cols or list(ALL_FEATURE_COLS)
        _eff_s1_cols = s1_feature_cols or ["s1_vh", "s1_vv"]

        def _emit_raw_pa(tbl: "pa.Table") -> None:
            """Slice PA table on point_id boundaries and put _PASlice on raw_q.

            The parquet is pixel-sorted so tbl is at most 2 Arrow chunks (a small
            leftover from the previous rg prepended to the current rg).  combine_chunks
            is therefore cheap — at most one copy of ~80 leftover rows is needed.
            """
            if tbl.num_rows == 0:
                return
            pid_col = tbl.column("point_id")
            pid_np  = np.asarray(pid_col.combine_chunks()) if pid_col.num_chunks > 1 \
                      else np.asarray(pid_col.chunks[0])
            n = len(pid_np)
            start = 0
            while start < n:
                end = min(start + target_chunk_rows, n)
                if end < n:
                    pivot = pid_np[end - 1]
                    while end > start and pid_np[end - 1] == pivot:
                        end -= 1
                    if end == start:
                        while end < n and pid_np[end] == pivot:
                            end += 1
                raw_q.put(_PASlice(tbl.slice(start, end - start), _eff_s2_cols, _eff_s1_cols, scl_purity_min))
                start = end

        def _emit_raw_pl(chunk: pl.DataFrame) -> None:
            """Sub-divide Polars chunk on point_id boundaries and put _RawChunk on raw_q."""
            if chunk.is_empty():
                return
            pid_np = chunk["point_id"].to_numpy()
            n = len(pid_np)
            start = 0
            while start < n:
                end = min(start + target_chunk_rows, n)
                if end < n:
                    pivot = pid_np[end - 1]
                    while end > start and pid_np[end - 1] == pivot:
                        end -= 1
                    if end == start:
                        while end < n and pid_np[end] == pivot:
                            end += 1
                raw = _extract_raw_arrays(
                    chunk.slice(start, end - start),
                    scl_purity_min=scl_purity_min,
                    s1_only=s1_only,
                    mixed=False,
                    feature_cols=feature_cols,
                    s2_feature_cols=s2_feature_cols,
                    s1_feature_cols=s1_feature_cols,
                    pixel_zscore=pixel_zscore,
                    vh_mean=_vh_mean, vh_std=_vh_std,
                    vv_mean=_vv_mean, vv_std=_vv_std,
                    despeckle_lookup=_despeckle_lookup,
                )
                if raw is not None:
                    raw_q.put(raw)
                start = end

        # Leftover: at most ~80 rows (one pixel's observations) carried from the
        # previous rg to handle pixels that straddle an rg boundary.  Never grows
        # beyond one pixel's worth because the parquet is pixel-sorted.
        pa_leftover:  "pa.Table | None" = None
        pl_leftover:  "pl.DataFrame | None" = None

        try:
            while True:
                tbl = io_q.get()
                if tbl is _SENTINEL:
                    break
                # tile_id filter in PyArrow before conversion
                if tile_prefix and "item_id" in tbl.schema.names:
                    item_ids = tbl.column("item_id")
                    null_mask  = _pac.is_null(item_ids)
                    match_mask = _pac.match_substring(item_ids.fill_null(""), tile_prefix)
                    tbl = tbl.filter(_pac.or_(null_mask, match_mask))
                    if tbl.num_rows == 0:
                        continue
                # Fast date extraction in PyArrow — 80× faster than Polars cast(Datetime("us"))
                date_col = tbl.column("date")
                year_col = _pac.year(date_col).cast("int32")
                doy_col  = _pac.day_of_year(date_col).cast("int32")
                tbl = tbl.append_column(_pa.field("year", _pa.int32()), year_col)
                tbl = tbl.append_column(_pa.field("doy",  _pa.int32()), doy_col)
                if end_year:
                    tbl = tbl.filter(_pac.less_equal(year_col, end_year))
                if tbl.num_rows == 0:
                    continue
                if mixed:
                    # Prepend the tiny leftover (≤1 pixel) from the previous rg.
                    # concat_tables with a near-empty table is cheap; the result has
                    # at most 2 chunks so combine_chunks in _emit_raw_pa is fast.
                    if pa_leftover is not None:
                        tbl = _pa.concat_tables([pa_leftover, tbl])
                        pa_leftover = None
                    # Peel off the trailing pixel — it may continue in the next rg.
                    pid_last = tbl.column("point_id")[-1].as_py()
                    # Find where the last pixel starts (scan from end — usually <80 rows)
                    pid_arr = np.asarray(tbl.column("point_id").chunks[-1])
                    tail_start_in_chunk = int(np.searchsorted(pid_arr, pid_last, side="left"))
                    n_chunks = tbl.column("point_id").num_chunks
                    if n_chunks > 1:
                        # leftover straddles the concat boundary; recompute on combined
                        pid_arr_full = np.asarray(tbl.column("point_id").combine_chunks())
                        tail_start = int(np.searchsorted(pid_arr_full, pid_last, side="left"))
                    else:
                        tail_start = tail_start_in_chunk
                    if tail_start > 0:
                        pa_leftover = tbl.slice(tail_start)
                        tbl = tbl.slice(0, tail_start)
                        _emit_raw_pa(tbl)
                    else:
                        # Entire rg is one pixel (extremely rare) — carry it all forward
                        pa_leftover = tbl
                else:
                    chunk = pl.from_arrow(tbl)
                    del tbl
                    if not s1_only and "scl_purity" in chunk.columns:
                        chunk = chunk.filter(pl.col("scl_purity") >= scl_purity_min)
                    if chunk.is_empty():
                        continue
                    if pl_leftover is not None:
                        chunk = pl.concat([pl_leftover, chunk])
                        pl_leftover = None
                    boundary_pid = chunk["point_id"][-1]
                    tail = chunk.filter(pl.col("point_id") == boundary_pid)
                    chunk = chunk.filter(pl.col("point_id") != boundary_pid)
                    pl_leftover = tail if not tail.is_empty() else None
                    _emit_raw_pl(chunk)

            # Flush final leftovers (no next rg to continue into)
            if pa_leftover is not None:
                _emit_raw_pa(pa_leftover)
            if pl_leftover is not None:
                _emit_raw_pl(pl_leftover)
        except Exception:
            logger.exception("Parser thread crashed")
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
                if isinstance(item, _PASlice):
                    raw = _extract_mixed_pa(item)
                else:
                    raw = item
                if raw is None:
                    continue
                batches = _preprocess(raw, band_mean, band_std, min_obs_per_year, pin=pin,
                                      mixed=mixed,
                                      pixel_zscore=pixel_zscore,
                                      pixel_zscore_stats=pixel_zscore_stats,
                                      s1_zscore_stats=s1_zscore_stats,
                                      max_seq_len=getattr(model, "_max_seq_len", MAX_SEQ_LEN),
                                      batch_size=batch_size,
                                      global_feat_mean=global_feat_mean,
                                      global_feat_std=global_feat_std,
                                      n_global_features=model.n_global_features)
                for b in batches:
                    prep_q.put(b)
        except Exception:
            logger.exception("Preprocessor worker crashed")
        finally:
            prep_q.put(_SENTINEL)

    # Convert stats dicts → vectorised _ZscoreArrays for O(1) batch lookups in preprocessors.
    if isinstance(pixel_zscore_stats, tuple):
        _s2_cols_n = len(next(iter(pixel_zscore_stats[0].values()))) if pixel_zscore_stats[0] else 1
        pixel_zscore_stats = _ZscoreArrays(*pixel_zscore_stats, n_feat=_s2_cols_n)
    if isinstance(s1_zscore_stats, tuple):
        _s1_cols_n = len(next(iter(s1_zscore_stats[0].values()))) if s1_zscore_stats[0] else 1
        s1_zscore_stats = _ZscoreArrays(*s1_zscore_stats, n_feat=_s1_cols_n)

    # --- Stage 4: writer thread (streaming path only) ---
    def _writer() -> None:
        buf_pids:  list[np.ndarray] = []
        buf_years: list[np.ndarray] = []
        buf_probs: list[np.ndarray] = []
        buf_rows = 0
        try:
            while True:
                item = out_q.get()
                if item is _SENTINEL:
                    break
                p, y, pr, _ = item
                buf_pids.append(p)
                buf_years.append(y)
                buf_probs.append(pr)
                buf_rows += len(p)
                if buf_rows >= write_flush_rows:
                    _flush_to_writer(out_writer, buf_pids, buf_years, buf_probs)
                    buf_pids, buf_years, buf_probs, buf_rows = [], [], [], 0
            if buf_rows:
                _flush_to_writer(out_writer, buf_pids, buf_years, buf_probs)
        except Exception:
            logger.exception("Writer thread crashed")

    io_thread     = Thread(target=_io_reader, daemon=True)
    reader_thread = Thread(target=_reader,    daemon=True)
    io_thread.start()
    reader_thread.start()

    prep_pool = ThreadPoolExecutor(max_workers=n_prep_workers)
    prep_futures = [prep_pool.submit(_preprocessor) for _ in range(n_prep_workers)]

    writer_thread: Thread | None = None
    if out_writer is not None:
        writer_thread = Thread(target=_writer, daemon=True)
        writer_thread.start()

    all_pids:  list[np.ndarray] = _all_pids  if _all_pids  is not None else []
    all_years: list[np.ndarray] = _all_years if _all_years is not None else []
    all_probs: list[np.ndarray] = _all_probs if _all_probs is not None else []
    sentinels_seen = 0
    n_scored = 0
    n_last_logged = 0
    n_total_gate_cut = 0
    import time as _time
    _t0 = _time.monotonic()
    _t_last_logged = _t0

    # -------------------------------------------------------------------------
    # GPU pipeline — 3 concurrent stages after prep_q:
    #
    #   Stage A (main thread):     drain prep_q, accumulate to batch_size,
    #                              merge tensors, put _PreparedBatch on xfer_q.
    #   Stage B (xfer thread):     build_gate_tensors (Numba CPU) + pin_memory +
    #                              non-blocking H2D transfer → _TransferredBatch
    #                              on gpu_ready_q.  Runs concurrently with Stage A
    #                              and Stage C.
    #   Stage C (gpu thread):      pull _TransferredBatch from gpu_ready_q, run
    #                              _gpu_forward (pure GPU, zero H2D stalls), put
    #                              result on result_q.
    #
    # While the GPU is running the forward on batch N, Stage B is transferring
    # batch N+1 and Stage A is building batch N+2.
    # -------------------------------------------------------------------------

    # xfer_q depth=2: main thread can pre-build one batch while xfer is running.
    # gpu_ready_q depth=2: xfer thread can pre-transfer one batch while GPU runs.
    xfer_q:      Queue = Queue(maxsize=2)
    gpu_ready_q: Queue = Queue(maxsize=2)

    # Two CUDA streams: the copy engine (transfer_stream) and compute engine
    # (compute_stream) run in parallel on the GPU.  An event recorded after the
    # H2D DMAs lets the compute stream wait for the transfer without a CPU sync.
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    transfer_stream: "torch.cuda.Stream | None" = (
        torch.cuda.Stream(device=device) if use_cuda else None)
    compute_stream:  "torch.cuda.Stream | None" = (
        torch.cuda.Stream(device=device) if use_cuda else None)

    # --- Stage B: H2D transfer thread ---
    _t_xfer_bgt_total  = 0.0   # transfer thread: Numba build_gate_tensors
    _t_xfer_h2d_total  = 0.0   # transfer thread: pin+H2D enqueue
    _t_xfer_push_total = 0.0   # transfer thread: blocked on gpu_ready_q.put()

    def _transfer_worker() -> None:
        nonlocal _t_xfer_bgt_total, _t_xfer_h2d_total, _t_xfer_push_total
        from tam.core._preprocess_numba import build_gate_tensors as _bgt
        while True:
            item = xfer_q.get()
            if item is _SENTINEL:
                gpu_ready_q.put(_SENTINEL)
                break
            merged: _PreparedBatch = item
            B_ = len(merged.pids)
            T_full = merged.bands.shape[1]
            n_feat = merged.bands.shape[2]
            gate_t = min(T_gate, T_full)

            # --- Gate tensor prep (CPU Numba, GIL-releasing) ---
            bands_np  = merged.bands.numpy()
            doy_np    = merged.doy.numpy()
            mask_np   = merged.mask.numpy()
            is_s1_np  = (merged.is_s1.numpy().astype(bool) if merged.is_s1 is not None
                         else np.zeros((B_, T_full), dtype=bool))

            gate_bands_np   = np.zeros((B_, gate_t, n_feat), dtype=np.float32)
            gate_doy_np     = np.zeros((B_, gate_t),         dtype=np.int64)
            gate_is_s1_np   = np.zeros((B_, gate_t),         dtype=bool)
            gate_mask_np    = np.ones( (B_, gate_t),         dtype=bool)
            _tb = _time.monotonic()
            _bgt(bands_np, doy_np, mask_np, is_s1_np, np.int64(gate_t),
                 gate_bands_np, gate_doy_np, gate_mask_np, gate_is_s1_np)
            _t_xfer_bgt_total += _time.monotonic() - _tb
            gate_n_obs_np = (np.clip((~gate_mask_np).sum(axis=1), 1, None)
                             .astype(np.float32) / T_full)

            # H2D on transfer_stream so the copy engine runs in parallel with
            # whatever the compute engine is doing.  pin_memory() enables async DMA.
            _ts_ctx = (torch.cuda.stream(transfer_stream)
                       if transfer_stream is not None else _nullctx())
            _th = _time.monotonic()
            with _ts_ctx:
                def _to_dev(arr) -> torch.Tensor:
                    t = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
                    return (t.pin_memory().to(device, non_blocking=True)
                            if use_cuda else t.to(device))
                gate_bands_dev  = _to_dev(gate_bands_np)
                gate_doy_dev    = _to_dev(gate_doy_np)
                gate_mask_dev   = _to_dev(gate_mask_np)
                gate_n_obs_dev  = _to_dev(gate_n_obs_np)
                gate_is_s1_dev  = (_to_dev(gate_is_s1_np)
                                   if merged.is_s1 is not None else None)
                bands_dev  = _to_dev(merged.bands)
                doy_dev    = _to_dev(merged.doy)
                mask_dev   = _to_dev(merged.mask)
                n_obs_dev  = _to_dev(merged.n_obs)
                is_s1_dev  = (_to_dev(merged.is_s1) if merged.is_s1 is not None else None)
                gf_dev     = (_to_dev(merged.global_feats)
                              if merged.global_feats is not None else None)
                # Record event after all non_blocking transfers are enqueued.
                xfer_event: "torch.cuda.Event | None" = None
                if use_cuda:
                    xfer_event = torch.cuda.Event()
                    xfer_event.record(transfer_stream)
            _t_xfer_h2d_total += _time.monotonic() - _th

            tb = _TransferredBatch(
                gate_bands_dev=gate_bands_dev,
                gate_doy_dev=gate_doy_dev,
                gate_mask_dev=gate_mask_dev,
                gate_n_obs_dev=gate_n_obs_dev,
                gate_is_s1_dev=gate_is_s1_dev,
                bands_dev=bands_dev,
                doy_dev=doy_dev,
                mask_dev=mask_dev,
                n_obs_dev=n_obs_dev,
                is_s1_dev=is_s1_dev,
                global_feats_dev=gf_dev,
                pids=merged.pids,
                years=merged.years,
                B=B_,
                xfer_event=xfer_event,
            )
            _tp = _time.monotonic()
            gpu_ready_q.put(tb)
            _t_xfer_push_total += _time.monotonic() - _tp

    # result_q: gpu thread → main thread for bookkeeping/logging
    result_q: Queue = Queue(maxsize=8)

    # --- Stage C: dedicated GPU forward thread ---
    # Model forward confirmed safe to call from a background thread (tested).
    _n_gpu_calls   = 0
    _t_score_total = 0.0
    _t_gpu_wait_total = 0.0   # time GPU worker spent blocked on gpu_ready_q

    _gpu_batch_times: list[float] = []   # per-batch forward time for variance analysis

    def _gpu_worker() -> None:
        nonlocal _n_gpu_calls, _t_score_total, _t_gpu_wait_total
        while True:
            _tw = _time.monotonic()
            tb = gpu_ready_q.get()
            _t_gpu_wait_total += _time.monotonic() - _tw
            if tb is _SENTINEL:
                result_q.put(_SENTINEL)
                break
            _ts = _time.monotonic()
            chunk = _gpu_forward(tb, model, gate_threshold, compute_stream)
            _dt = _time.monotonic() - _ts
            _t_score_total += _dt
            _n_gpu_calls   += 1
            _gpu_batch_times.append(_dt)
            if _dt > 0.5:   # log any batch taking >500ms — should be ~100ms normally
                logger.debug("GPU batch %d: %.0fms  B=%d  n_gate_cut=%d",
                             _n_gpu_calls, _dt * 1000, tb.B, chunk[3])
            result_q.put((chunk, tb.B))

    xfer_thread = Thread(target=_transfer_worker, daemon=True)
    gpu_thread  = Thread(target=_gpu_worker,      daemon=True)
    xfer_thread.start()
    gpu_thread.start()

    # --- Stage A: main thread — accumulate prep_q → xfer_q,
    #              and drain result_q for bookkeeping/logging ---
    _t_idle_total  = 0.0
    _t_merge_total = 0.0
    _t_xfer_wait_total = 0.0  # time main thread blocked on xfer_q.put()
    _acc: list[_PreparedBatch] = []
    _acc_px = 0
    import queue as _queue

    def _flush_acc() -> None:
        nonlocal _t_merge_total, _t_xfer_wait_total
        _tm = _time.monotonic()
        if len(_acc) == 1:
            merged = _acc[0]
        else:
            merged = _PreparedBatch(
                bands=torch.cat([b.bands for b in _acc], dim=0),
                doy=torch.cat([b.doy   for b in _acc], dim=0),
                mask=torch.cat([b.mask  for b in _acc], dim=0),
                n_obs=torch.cat([b.n_obs for b in _acc], dim=0),
                pids=np.concatenate([b.pids  for b in _acc]),
                years=np.concatenate([b.years for b in _acc]),
                is_s1=(torch.cat([b.is_s1 for b in _acc], dim=0)
                       if _acc[0].is_s1 is not None else None),
                global_feats=(np.concatenate([b.global_feats for b in _acc], axis=0)
                              if _acc[0].global_feats is not None else None),
            )
        _t_merge_total += _time.monotonic() - _tm
        _tx = _time.monotonic()
        xfer_q.put(merged)   # blocks if xfer_q is full — natural backpressure
        _t_xfer_wait_total += _time.monotonic() - _tx

    def _collect_results() -> None:
        """Drain result_q non-blocking; update n_scored and log progress."""
        nonlocal n_scored, n_total_gate_cut, n_last_logged, _t_last_logged
        while True:
            try:
                res = result_q.get_nowait()
            except _queue.Empty:
                break
            if res is _SENTINEL:
                break
            chunk, px_count = res
            n_total_gate_cut += chunk[3]
            if out_writer is not None:
                out_q.put(chunk)
            else:
                all_pids.append(chunk[0]); all_years.append(chunk[1]); all_probs.append(chunk[2])
            n_scored += px_count
            if n_scored - n_last_logged >= progress_log_interval:
                _t_now = _time.monotonic()
                dt_interval = _t_now - _t_last_logged
                rate = (n_scored - n_last_logged) / dt_interval if dt_interval > 0 else 0.0
                if n_total_pixels:
                    logger.info(
                        "[%s yr=%s] %.1f%% (%d/%d px) @ %.0f px/s",
                        tile_id or "?", end_year or "?",
                        100.0 * n_scored / n_total_pixels, n_scored, n_total_pixels, rate,
                    )
                else:
                    logger.info(
                        "[%s yr=%s] %d px scored @ %.0f px/s",
                        tile_id or "?", end_year or "?", n_scored, rate,
                    )
                n_last_logged = n_scored
                _t_last_logged = _t_now

    while sentinels_seen < n_prep_workers:
        _t_get_start = _time.monotonic()
        item = prep_q.get()
        _t_idle_total += _time.monotonic() - _t_get_start
        if item is _SENTINEL:
            sentinels_seen += 1
            if sentinels_seen == n_prep_workers and _acc:
                _flush_acc()
                _acc.clear(); _acc_px = 0
            continue
        _acc.append(item)
        _acc_px += len(item.pids)
        if _acc_px < batch_size:
            continue
        _flush_acc()
        _acc.clear(); _acc_px = 0
        _collect_results()

    xfer_q.put(_SENTINEL)

    # Drain result_q until GPU thread signals done.
    while True:
        res = result_q.get()
        if res is _SENTINEL:
            break
        chunk, px_count = res
        n_total_gate_cut += chunk[3]
        if out_writer is not None:
            out_q.put(chunk)
        else:
            all_pids.append(chunk[0]); all_years.append(chunk[1]); all_probs.append(chunk[2])
        n_scored += px_count
        if n_scored - n_last_logged >= progress_log_interval:
            _t_now = _time.monotonic()
            dt_interval = _t_now - _t_last_logged
            rate = (n_scored - n_last_logged) / dt_interval if dt_interval > 0 else 0.0
            if n_total_pixels:
                logger.info(
                    "[%s yr=%s] %.1f%% (%d/%d px) @ %.0f px/s",
                    tile_id or "?", end_year or "?",
                    100.0 * n_scored / n_total_pixels, n_scored, n_total_pixels, rate,
                )
            else:
                logger.info(
                    "[%s yr=%s] %d px scored @ %.0f px/s",
                    tile_id or "?", end_year or "?", n_scored, rate,
                )
            n_last_logged = n_scored
            _t_last_logged = _t_now

    xfer_thread.join()
    gpu_thread.join()

    io_thread.join()
    reader_thread.join()
    for f in prep_futures:
        f.result()
    prep_pool.shutdown(wait=False)

    _t_total = _time.monotonic() - _t0
    _t_other  = max(0.0, _t_total - _t_idle_total - _t_merge_total - _t_score_total)
    logger.info(
        "GPU pipeline breakdown — total %.1fs: score %.1fs (%.0f%%), "
        "merge %.1fs (%.0f%%), prep_wait %.1fs (%.0f%%), xfer_backpressure %.1fs (%.0f%%), "
        "gpu_starvation %.1fs (%.0f%%), other %.1fs (%.0f%%) | %d GPU calls",
        _t_total,
        _t_score_total,    100.0 * _t_score_total    / _t_total if _t_total else 0,
        _t_merge_total,    100.0 * _t_merge_total     / _t_total if _t_total else 0,
        _t_idle_total,     100.0 * _t_idle_total      / _t_total if _t_total else 0,
        _t_xfer_wait_total,100.0 * _t_xfer_wait_total / _t_total if _t_total else 0,
        _t_gpu_wait_total, 100.0 * _t_gpu_wait_total  / _t_total if _t_total else 0,
        _t_other,          100.0 * _t_other            / _t_total if _t_total else 0,
        _n_gpu_calls,
    )
    logger.info(
        "Transfer thread breakdown — bgt %.1fs (%.0f%%), h2d %.1fs (%.0f%%), "
        "push_wait %.1fs (%.0f%%)",
        _t_xfer_bgt_total,  100.0 * _t_xfer_bgt_total  / _t_total if _t_total else 0,
        _t_xfer_h2d_total,  100.0 * _t_xfer_h2d_total  / _t_total if _t_total else 0,
        _t_xfer_push_total, 100.0 * _t_xfer_push_total / _t_total if _t_total else 0,
    )
    if _gpu_batch_times:
        _bt = np.array(_gpu_batch_times)
        logger.info(
            "GPU forward per-batch (ms): mean=%.1f  median=%.1f  p95=%.1f  max=%.1f  "
            "min=%.1f  n=%d",
            _bt.mean() * 1000, np.median(_bt) * 1000,
            np.percentile(_bt, 95) * 1000, _bt.max() * 1000,
            _bt.min() * 1000, len(_bt),
        )

    if writer_thread is not None:
        out_q.put(_SENTINEL)
        writer_thread.join()
        return pl.DataFrame()   # caller owns the writer; return value is not used

    if gate_threshold > 0.0 and n_scored > 0:
        pct = 100.0 * n_total_gate_cut / n_scored
        logger.info(
            "Gate (T=%d, thresh=%.2f): %d/%d px cut (%.1f%%) — %.1f%% scored at full T",
            T_gate, gate_threshold, n_total_gate_cut, n_scored, pct, 100.0 - pct,
        )

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
    min_obs_per_year: int = 10,
    batch_size: int = 4096,
    n_prep_workers: int = 5,
    device: str | None = None,
    tile_id: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    n_total_pixels: int | None = None,
    s1_only: bool = False,
    mixed: bool = True,
    pixel_zscore: bool = False,
    s1_despeckle_window: int = 0,
    feature_cols: list[str] | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    summary_feature_cols: list[str] | None = None,
    global_feat_mean: np.ndarray | None = None,
    global_feat_std: np.ndarray | None = None,
    gate_threshold: float = 0.0,
    T_gate: int = 8,
) -> pl.DataFrame:
    """Score a location across multiple annual parquets and aggregate.

    year_parquets: [(year, path), ...] — must be pixel-sorted parquets.

    Accumulates (pid, year, prob) triples across all years then calls
    aggregate_year_probs() once, so decay weighting is globally consistent.
    """
    all_pids:  list[np.ndarray] = []
    all_years: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    # Resolve effective column lists:
    # - _s2_cols: S2 spectral features used in _preprocess
    # - _s1_cols: S1 dB features used in _preprocess (mixed mode only)
    # - _summary_cols: band-summary global features (must match global_feat_mean shape)
    _s2_cols = s2_feature_cols or feature_cols or list(ALL_FEATURE_COLS)
    _s1_cols = s1_feature_cols or ["s1_vh", "s1_vv"]

    # All normalisation (pixel z-score + band summaries) is now computed per-chunk
    # inside _preprocess from the assembled pixel windows — no pre-pass needed.

    _eff_end_year = end_year or max(y for y, _ in year_parquets)

    for year, path in sorted(year_parquets):
        if end_year and year > end_year:
            logger.info("Skipping %d (past end_year=%d)", year, end_year)
            continue
        logger.info("Scoring year %d — %s", year, path.name)
        _tile_id = tile_id or path.stem.split("-by-pixel")[0].split(".s1")[0].split(".s2")[0]
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
            tile_id=_tile_id,
            end_year=_eff_end_year,
            decay=decay,
            n_total_pixels=n_total_pixels,
            s1_only=s1_only,
            mixed=mixed,
            pixel_zscore=pixel_zscore,
            s1_despeckle_window=s1_despeckle_window,
            feature_cols=None,
            s2_feature_cols=_s2_cols,
            s1_feature_cols=_s1_cols,
            global_feat_mean=global_feat_mean,
            global_feat_std=global_feat_std,
            pixel_zscore_stats=None,
            s1_zscore_stats=None,
            _all_pids=all_pids,
            _all_years=all_years,
            _all_probs=all_probs,
            gate_threshold=gate_threshold,
            T_gate=T_gate,
        )

    logger.info("Aggregating scores across %d years ...", len(year_parquets))
    return aggregate_year_probs(all_pids, all_years, all_probs, end_year=_eff_end_year, decay=decay)


_STAGING_WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["point_id"],
    write_statistics=False,
)


_STAGING_SCHEMA = None  # initialised lazily to avoid top-level pyarrow import


def _get_staging_schema() -> "pa.Schema":
    import pyarrow as pa
    return pa.schema([
        pa.field("point_id",     pa.large_utf8()),
        pa.field("year",         pa.int16()),
        pa.field("prob_tam_raw", pa.float32()),
    ])


def score_tile_year(
    parquet: Path,
    tile_id: str,
    year: int,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    staging_dir: Path,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 10,
    batch_size: int = 4096,
    n_prep_workers: int = 5,
    device: str | None = None,
    s1_only: bool = False,
    mixed: bool = True,
    s1_despeckle_window: int = 0,
    n_total_pixels: int | None = None,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    global_feat_mean: np.ndarray | None = None,
    global_feat_std: np.ndarray | None = None,
) -> Path:
    """Score a single (tile_id, year) parquet and write a staging parquet.

    Runs parallel pre-passes (S2 z-score + S1 z-score) before opening the
    ParquetWriter, then streams inference results directly to disk so memory
    is bounded regardless of tile size (~100M pixels on full catchment tiles).

    Pre-passes read the parquet concurrently (I/O-bound); scoring starts only
    after both complete. The writer thread flushes to disk every 500K rows
    while inference runs, so the output is partially written throughout.

    The staging parquet has columns: point_id (str), year (int16), prob_tam_raw (float32).
    If the staging file already exists it is returned immediately (idempotent / crash-safe).

    Returns the staging parquet path.
    """
    import pyarrow.parquet as pq
    from tam.core.dataset import ALL_FEATURE_COLS, V10_S1_FEATURE_COLS

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{tile_id}_{year}.parquet"
    if out_path.exists():
        logger.info("Staging file exists, skipping %s year=%d", tile_id, year)
        return out_path

    tmp_path = out_path.with_suffix(".tmp.parquet")
    tmp_path.unlink(missing_ok=True)   # clean up any stale crash remnant

    _s2_cols = s2_feature_cols or list(ALL_FEATURE_COLS)
    _s1_cols = s1_feature_cols or list(V10_S1_FEATURE_COLS)

    # Pixel z-score stats are computed per-chunk inside _preprocess — no pre-pass needed.
    logger.info("[%s yr=%d] Scoring ...", tile_id, year)
    with pq.ParquetWriter(tmp_path, schema=_get_staging_schema(), **_STAGING_WRITE_OPTS) as staging_writer:
        score_pixels_chunked(
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
            mixed=mixed,
            pixel_zscore=not s1_only,
            global_feat_mean=global_feat_mean,
            global_feat_std=global_feat_std,
            pixel_zscore_stats=None,
            s1_zscore_stats=None,
            s2_feature_cols=_s2_cols,
            s1_feature_cols=_s1_cols,
            s1_despeckle_window=s1_despeckle_window,
            n_total_pixels=n_total_pixels,
            out_writer=staging_writer,
        )

    tmp_path.rename(out_path)   # atomic on POSIX; only visible once complete
    logger.info("[%s yr=%d] Wrote staging %s", tile_id, year, out_path.name)
    return out_path


def _score_tile_worker(args: tuple) -> tuple[str, list[Path]]:
    """Worker target for torch.multiprocessing: score one tile across all years.

    Returns (tile_id, staging_paths).
    """
    (
        tile_id, year_parquets, model, band_mean, band_std,
        staging_dir, scl_purity_min, min_obs_per_year,
        batch_size, n_prep_workers, device, n_tile_workers, s1_only, mixed,
        s1_despeckle_window, s2_feature_cols, s1_feature_cols,
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
            mixed=mixed,
            s1_despeckle_window=s1_despeckle_window,
            s2_feature_cols=s2_feature_cols,
            s1_feature_cols=s1_feature_cols,
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
    min_obs_per_year: int = 10,
    batch_size: int = 4096,
    n_prep_workers: int = 5,
    n_tile_workers: int = 1,
    device: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
    s1_only: bool = False,
    mixed: bool = True,
    s1_despeckle_window: int = 0,
    s2_feature_cols: list[str] | None = None,
    s1_feature_cols: list[str] | None = None,
    gate_threshold: float = 0.0,
    T_gate: int = 8,
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

    already_staged = sum(
        1 for tid, yps in tile_year_parquets.items()
        for y, _ in yps
        if (staging_dir / f"{tid}_{y}.parquet").exists()
    )
    total_pairs = sum(len(yps) for yps in tile_year_parquets.values())
    logger.info(
        "Catchment score: %d tiles, %d tile-year pairs (%d already staged, %d to run)",
        len(tile_year_parquets), total_pairs, already_staged, total_pairs - already_staged,
    )

    # --- Phase 1: score each (tile, year) ---
    worker_args = [
        (
            tile_id, year_parquets, model, band_mean, band_std,
            staging_dir, scl_purity_min, min_obs_per_year,
            batch_size, n_prep_workers, device, n_tile_workers, s1_only, mixed,
            s1_despeckle_window, s2_feature_cols, s1_feature_cols,
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

        # Read all staging parquets for this tile (lazy scan avoids materialising all
        # years simultaneously before the group-by)
        raw = pl.concat([
            pl.scan_parquet(p, hive_partitioning=False)
            for p in s_paths
        ]).select(["point_id", "year", "prob_tam_raw"]).collect()

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
