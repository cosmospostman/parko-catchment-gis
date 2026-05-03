"""tam/core/score.py — Chunked inference for TAMClassifier."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, MAX_SEQ_LEN, S1_FEATURE_COLS
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
        df = pf.read_row_group(rg, columns=read_cols).to_pandas()
        if "source" in df.columns:
            df = df[df["source"] == "S1"]
        df = df.dropna(subset=["vh", "vv"])
        if not df.empty:
            chunks.append(df[["point_id", "vh", "vv"]])
    if not chunks:
        return {}, {}, {}, {}
    all_s1 = pd.concat(chunks, ignore_index=True)
    all_s1["vh_db"] = np.where(all_s1["vh"] > 0, 10 * np.log10(all_s1["vh"]), np.nan)
    all_s1["vv_db"] = np.where(all_s1["vv"] > 0, 10 * np.log10(all_s1["vv"]), np.nan)
    grp = all_s1.groupby("point_id")
    vh_mean = grp["vh_db"].mean()
    vh_std  = grp["vh_db"].std().clip(lower=0.1)
    vv_mean = grp["vv_db"].mean()
    vv_std  = grp["vv_db"].std().clip(lower=0.1)
    return (vh_mean.to_dict(), vh_std.to_dict(),
            vv_mean.to_dict(), vv_std.to_dict())


def _extract_s1_features(
    chunk: pd.DataFrame,
    pixel_zscore: bool = False,
    vh_mean: dict | None = None,
    vh_std:  dict | None = None,
    vv_mean: dict | None = None,
    vv_std:  dict | None = None,
) -> np.ndarray:
    """Extract 4 S1 features (vh_db, vv_db, vh_vv, rvi) from linear vh/vv columns.

    When pixel_zscore=True, VH and VV are z-scored by each pixel's own multi-year
    mean/std (pre-computed via _compute_pixel_s1_stats). VH-VV and RVI are left
    in their natural units — they are self-normalising ratios that carry absolute
    canopy structure signal.
    """
    vh_lin = chunk["vh"].values.astype(np.float32)
    vv_lin = chunk["vv"].values.astype(np.float32)
    vh_db  = np.where(vh_lin > 0, 10 * np.log10(vh_lin), np.nan)
    vv_db  = np.where(vv_lin > 0, 10 * np.log10(vv_lin), np.nan)
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
    bands: torch.Tensor   # (W, MAX_SEQ_LEN, N_FEATURES) float32
    doy:   torch.Tensor   # (W, MAX_SEQ_LEN) int64
    mask:  torch.Tensor   # (W, MAX_SEQ_LEN) bool
    n_obs: torch.Tensor   # (W,) float32, n / MAX_SEQ_LEN
    pids:  np.ndarray     # (W,) object
    years: np.ndarray     # (W,) int32


def _preprocess(
    chunk: pd.DataFrame,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float,
    min_obs_per_year: int,
    pin: bool = False,
    s1_only: bool = False,
    pixel_zscore: bool = False,
    vh_mean: dict | None = None,
    vh_std:  dict | None = None,
    vv_mean: dict | None = None,
    vv_std:  dict | None = None,
) -> _PreparedBatch | None:
    """CPU-side preprocessing using numba kernels for maximum throughput.

    Pipeline:
      1. SCL filter (already applied by reader, but guard here too)
      2. Extract features into C-contiguous float32 via numba parallel kernel
      3. Find pixel-year window boundaries with numpy
      4. Fill padded (W, SEQ, F) arrays + normalise via numba parallel kernel
    """
    from tam.core._preprocess_numba import extract_features, fill_windows

    if s1_only:
        chunk = chunk[chunk["source"] == "S1"].copy() if "source" in chunk.columns else chunk
        chunk = chunk.dropna(subset=["vh", "vv"])
    elif "scl_purity" in chunk.columns:
        chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
    if chunk.empty:
        return None

    N = len(chunk)

    if s1_only:
        feat = _extract_s1_features(chunk, pixel_zscore=pixel_zscore,
                                    vh_mean=vh_mean, vh_std=vh_std,
                                    vv_mean=vv_mean, vv_std=vv_std)
        n_feat = _N_FEATURES_S1
    else:
        # Step 1: extract feature matrix — bands are already float32, zero-copy .values
        n_feat = _N_FEATURES
        feat = np.empty((N, n_feat), dtype=np.float32)
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
    bands_np = np.zeros((W, MAX_SEQ_LEN, n_feat), dtype=np.float32)
    doy_np   = np.zeros((W, MAX_SEQ_LEN), dtype=np.int64)
    mask_np  = np.ones( (W, MAX_SEQ_LEN), dtype=np.bool_)

    fill_windows(feat, doy_arr, valid_starts, capped, band_mean, band_std,
                 bands_np, doy_np, mask_np)

    pids  = pid_arr[valid_starts]
    years = year_arr[valid_starts]

    n_obs_np = (capped / MAX_SEQ_LEN).astype(np.float32)

    bands_th = torch.from_numpy(bands_np)
    doy_th   = torch.from_numpy(doy_np)
    mask_th  = torch.from_numpy(mask_np)
    n_obs_th = torch.from_numpy(n_obs_np)
    if pin:
        bands_th = bands_th.pin_memory()
        doy_th   = doy_th.pin_memory()
        mask_th  = mask_th.pin_memory()
        n_obs_th = n_obs_th.pin_memory()

    return _PreparedBatch(bands_th, doy_th, mask_th, n_obs_th, pids, years)


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
    bands_th, doy_th, mask_th, n_obs_th, pids, years = prepared
    W = len(pids)
    with torch.inference_mode():
        for start in range(0, W, batch_size):
            end = min(start + batch_size, W)
            prob, _ = model(
                bands_th[start:end].to(device, non_blocking=True),
                doy_th[start:end].to(device, non_blocking=True),
                mask_th[start:end].to(device, non_blocking=True),
                n_obs_th[start:end].to(device, non_blocking=True),
                global_feats=None,  # global features not yet supported in chunked scoring
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
    s1_only: bool = False,
    pixel_zscore: bool = False,
    # accumulators — pass across years to merge results before final aggregation
    _all_pids:  list | None = None,
    _all_years: list | None = None,
    _all_probs: list | None = None,
) -> pd.DataFrame:
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

    pf = pq.ParquetFile(parquet)
    n_rg = pf.metadata.num_row_groups
    tile_prefix = f"_{tile_id}_" if tile_id else None

    if s1_only:
        read_cols = (
            ["point_id", "date", "source"]
            + (["item_id"] if tile_id else [])
            + ["vh", "vv"]
        )
    else:
        read_cols = (
            ["point_id", "date", "scl_purity"]
            + (["item_id"] if tile_id else [])
            + BAND_COLS
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
            if not s1_only and "scl_purity" in chunk.columns:
                chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
            if chunk.empty:
                continue
            ts_us = pd.to_datetime(chunk["date"]).values.astype("datetime64[us]").astype("int64")
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
            prepared = _preprocess(item, band_mean, band_std, scl_purity_min, min_obs_per_year, pin=pin, s1_only=s1_only,
                                   pixel_zscore=pixel_zscore, vh_mean=_vh_mean, vh_std=_vh_std,
                                   vv_mean=_vv_mean, vv_std=_vv_std)
            if prepared is not None:
                prep_q.put(prepared)
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
    pixel_zscore: bool = False,
) -> pd.DataFrame:
    """Score a location across multiple annual parquets and aggregate.

    year_parquets: [(year, path), ...] — must be pixel-sorted parquets.

    Accumulates (pid, year, prob) triples across all years then calls
    aggregate_year_probs() once, so decay weighting is globally consistent.
    """
    all_pids:  list[np.ndarray] = []
    all_years: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

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
            pixel_zscore=pixel_zscore,
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
    )

    scores["year"] = np.int16(year)
    tbl = pa.Table.from_pandas(
        scores[["point_id", "year", "prob_tam"]].rename(columns={"prob_tam": "prob_tam_raw"}),
        preserve_index=False,
    )
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
        chunks = [pd.read_parquet(p, columns=["point_id", "year", "prob_tam_raw"]) for p in s_paths]
        raw = pd.concat(chunks, ignore_index=True)

        all_pids  = [raw["point_id"].values]
        all_years = [raw["year"].values.astype(np.int32)]
        all_probs = [raw["prob_tam_raw"].values.astype(np.float32)]

        agg = aggregate_year_probs(all_pids, all_years, all_probs, end_year=_eff_end_year, decay=decay)

        # Convert float [0,1] → uint8 [0,100]
        agg["prob_tam"] = (agg["prob_tam"].clip(0.0, 1.0) * 100).round().astype(np.uint8)

        tbl = pa.Table.from_pandas(agg[["point_id", "prob_tam"]], preserve_index=False)
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
