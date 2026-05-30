"""tam/core/train.py — Training loop for TAMClassifier."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import multiprocessing
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tam.core.config import TAMConfig
from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX
from tam.core.dataset import BAND_COLS, MAX_SEQ_LEN, S1_FEATURE_COLS, TAMDataset, V9_FEATURE_COLS, collate_fn, lin_to_db, ForcedGateDataset, GateAugDataset
from tam.core.global_features import GLOBAL_FEATURE_NAMES, compute_global_features
from tam.core.model import TAMClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial train/val split
# ---------------------------------------------------------------------------

def spatial_split(
    labels: dict[str, float],
    pixel_coords: pl.DataFrame,
    val_frac: float = 0.2,
) -> tuple[dict[str, float], dict[str, float]]:
    """Hold out a spatially contiguous subset of pixels for validation.

    Splits by latitude within each class: the southernmost val_frac of
    presence pixels and the southernmost val_frac of absence pixels become
    the validation set. This avoids spatial autocorrelation leakage from
    random splits.

    Parameters
    ----------
    labels:
        dict mapping point_id → label value in {0.0, 1.0}.
    pixel_coords:
        DataFrame with point_id, lat columns (one row per pixel).
    val_frac:
        Fraction of each class to reserve for validation.

    Returns
    -------
    (train_labels, val_labels) — both dicts mapping point_id → float.
    """
    # Build {point_id: lat} lookup
    coord_df = pixel_coords.select(["point_id", "lat"])
    pid_to_lat: dict[str, float] = dict(zip(coord_df["point_id"].to_list(), coord_df["lat"].to_list()))

    val_ids: list[str] = []
    for cls_val in [0.0, 1.0]:
        cls_pids = sorted(
            [pid for pid, v in labels.items() if v == cls_val],
            key=lambda pid: pid_to_lat.get(pid, 0.0),
        )
        n_val = max(1, int(len(cls_pids) * val_frac))
        val_ids.extend(cls_pids[:n_val])

    val_set = set(val_ids)
    train_labels = {k: v for k, v in labels.items() if k not in val_set}
    val_labels   = {k: v for k, v in labels.items() if k in val_set}
    return train_labels, val_labels


def region_holdout_split(
    labels: dict[str, float],
    val_region_ids: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Hold out pixels whose region ID is in val_region_ids.

    Point IDs have the form <region_id>_<row>_<col>. The region is recovered
    by stripping the trailing _<row>_<col> numeric suffix.
    """
    _suffix = re.compile(r"_\d+_\d+$")
    val_set = set(val_region_ids)
    train_labels = {k: v for k, v in labels.items() if _suffix.sub("", k) not in val_set}
    val_labels   = {k: v for k, v in labels.items() if _suffix.sub("", k) in val_set}
    return train_labels, val_labels


def site_holdout_split(
    labels: dict[str, float],
    val_sites: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Hold out all pixels whose region ID starts with any of val_sites.

    Point IDs have the form <region_id>_<row>_<col>, and region IDs have the
    form <site>_presence_N or <site>_absence_N, so the site is the prefix
    before the first '_presence' or '_absence'.
    """
    def point_site(pid: str) -> str:
        m = re.match(r"^(.+?)_(presence|absence)", pid)
        return m.group(1) if m else pid

    val_site_set = set(val_sites)
    train_labels = {k: v for k, v in labels.items() if point_site(k) not in val_site_set}
    val_labels   = {k: v for k, v in labels.items() if point_site(k) in val_site_set}
    return train_labels, val_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_band_summaries(pixel_df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Per-pixel [p5, p95, std] for each feature column, computed from S2 rows.

    Returns a DataFrame with columns point_id, <col>_p5, <col>_p95, <col>_std
    for each col in feature_cols.
    Used as global features when cfg.use_band_summaries=True.

    Index columns (NDVI, NDWI, etc.) are computed inline as Polars expressions
    inside the lazy plan so they are never materialised as full-frame columns.
    """
    # Polars expressions for index columns not already present in pixel_df.
    # quality guard: scl_purity >= 0.5 (mirrors Signal.quality_mask for S2 rows).
    _qm = pl.lit(True)
    if "scl_purity" in pixel_df.columns:
        _qm = pl.col("scl_purity") >= 0.5

    def _safe_ratio(num: pl.Expr, denom: pl.Expr) -> pl.Expr:
        return pl.when(_qm & (denom != 0)).then(num / denom).otherwise(pl.lit(None))

    _index_exprs: dict[str, pl.Expr] = {
        "NDVI":  _safe_ratio(pl.col("B08") - pl.col("B04"), pl.col("B08") + pl.col("B04")),
        "NDWI":  _safe_ratio(pl.col("B03") - pl.col("B08"), pl.col("B03") + pl.col("B08")),
        "EVI":   _safe_ratio(
            2.5 * (pl.col("B08") - pl.col("B04")),
            pl.col("B08") + 6 * pl.col("B04") - 7.5 * pl.col("B02") + 1,
        ),
        "MAVI":  _safe_ratio(pl.col("B08") - pl.col("B04"), pl.col("B08") + pl.col("B04") + pl.col("B11")),
        "NDRE":  _safe_ratio(pl.col("B8A") - pl.col("B05"), pl.col("B8A") + pl.col("B05")),
        "CI_RE": pl.when(_qm & (pl.col("B05") != 0)).then(pl.col("B07") / pl.col("B05") - 1).otherwise(pl.lit(None)),
    }

    lf = pixel_df.lazy()
    if "source" in pixel_df.columns:
        lf = lf.filter(pl.col("source") == "S2")

    # Add index columns that are missing from the stored data.
    extra = {name: expr for name, expr in _index_exprs.items()
             if name not in pixel_df.columns}
    if extra:
        lf = lf.with_columns([expr.alias(name) for name, expr in extra.items()])

    cols = [c for c in feature_cols if c in pixel_df.columns or c in extra]
    aggs = (
        [pl.col(c).quantile(0.05).alias(f"{c}_p5")  for c in cols] +
        [pl.col(c).quantile(0.95).alias(f"{c}_p95") for c in cols] +
        [pl.col(c).std().alias(f"{c}_std")           for c in cols]
    )
    grp = lf.group_by("point_id").agg(aggs).collect()
    ordered = ["point_id"] + [f"{c}{s}" for c in cols for s in ("_p5", "_p95", "_std")]
    return grp.select([c for c in ordered if c in grp.columns])


def _global_features_cache_key(pixel_df: pl.DataFrame) -> str:
    """Stable hash of the inputs to compute_global_features."""
    cols = [c for c in ("point_id", "date", "B08", "B04", "vh", "vv", "source")
            if c in pixel_df.columns]
    key_df = pixel_df.select(cols).sort(cols)
    return hashlib.md5(key_df.write_csv().encode()).hexdigest()


def _load_or_compute_global_features(
    pixel_df: pl.DataFrame,
    out_dir: Path,
    feature_names: list[str],
) -> pl.DataFrame:
    cache_path = out_dir / "global_features_cache.parquet"
    key_path   = out_dir / "global_features_cache.key"
    cache_key  = _global_features_cache_key(pixel_df)

    if cache_path.exists() and key_path.exists() and key_path.read_text().strip() == cache_key:
        logger.info("Loading cached global features from %s", cache_path)
        return pl.read_parquet(cache_path)

    logger.info("Computing global features: %s", feature_names)
    global_feat_df = compute_global_features(pixel_df)
    logger.info(
        "Global feature means — %s",
        "  ".join(f"{k}={global_feat_df[k].mean():.4f}" for k in feature_names),
    )
    global_feat_df.write_parquet(cache_path)
    key_path.write_text(cache_key)
    logger.info("Cached global features to %s", cache_path)
    return global_feat_df


def _site_class(pid: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    if not m:
        return (pid, "unknown")
    site = re.sub(r"_val$", "", m.group(1))
    return (site, m.group(2))


def _cvar_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    pids: list[str],
    finite: np.ndarray,
    alpha: float,
) -> tuple[float, list[dict]]:
    """CVaR AUC over per-site scores."""
    site_ids = np.array([_site_class(p)[0] for p in pids])
    site_records: list[dict] = []
    for site in np.unique(site_ids):
        mask = finite & (site_ids == site)
        n = int(mask.sum())
        if n == 0 or len(set(labels[mask])) < 2:
            continue
        try:
            auc = roc_auc_score(labels[mask], probs[mask])
        except ValueError:
            continue
        site_records.append({"site": site, "n_pixels": n, "auc": auc, "in_tail": False})

    if not site_records:
        return float("nan"), site_records

    site_aucs = np.array([r["auc"] for r in site_records])
    order = np.argsort(site_aucs)
    n_tail = max(1, int(np.ceil(alpha * len(site_aucs))))
    for i, idx in enumerate(order):
        site_records[idx]["in_tail"] = i < n_tail

    tail_aucs = np.sort(site_aucs)[:n_tail]
    return float(tail_aucs.mean()), site_records


def _apply_presence_filter(
    lbl_py: dict[tuple[str, int], float],
    mean_vh_dry_py: dict[tuple[str, int], float],
    cfg: TAMConfig,
    pid_to_sc: dict[str, tuple[str, str]],
    noise_removed: dict,
    mean_ndvi_dry_py: dict[tuple[str, int], float] | None = None,
) -> dict[tuple[str, int], float]:
    drop_keys: set[tuple[str, int]] = set()
    for key, lbl in lbl_py.items():
        if lbl != 1:
            continue
        vh = mean_vh_dry_py.get(key)
        if vh is None:
            continue
        fails_strict = vh < cfg.presence_min_vh_dry_db
        if not fails_strict:
            continue
        if mean_ndvi_dry_py is not None:
            ndvi = mean_ndvi_dry_py.get(key)
            rescued = (
                vh >= cfg.presence_ndvi_rescue_vh_db
                and ndvi is not None
                and ndvi >= cfg.presence_ndvi_rescue_min
            )
            if rescued:
                continue
        drop_keys.add(key)

    for key in drop_keys:
        sc = pid_to_sc.get(key[0])
        if sc:
            noise_removed[sc] = noise_removed.get(sc, 0) + 1

    return {k: v for k, v in lbl_py.items() if k not in drop_keys}


def _compute_band_stats_worker(
    pixel_df_path: str,
    out_path: str,
    use_s1: "bool | str",
    feature_cols: list,
    s1_feature_cols: list,
    scl_purity_min: float,
    s1_despeckle_window: int,
    global_features_df_path: str | None,
) -> None:
    """Subprocess worker: compute band normalisation stats and write to *out_path*.npz.

    Reads the parquet, applies the same pre-processing as TAMDataset (despeckle,
    dB conversion, scl_purity filter) but only materialises the columns needed
    for mean/std — no sort, no group-by, no window construction.  Peak RSS is
    roughly the size of the parquet in memory (~2–4× compression ratio).
    """
    import gc as _gc
    import numpy as _np
    import polars as _pl
    import pyarrow.parquet as _pq
    from analysis.constants import ensure_float32_bands
    from tam.core.dataset import S1_FEATURE_COLS as _S1_COLS, despeckle_s1 as _despeckle, prepare_s1_frame as _prep_s1, prepare_s2_frame as _prep_s2

    # Only load columns needed for stats — skip point_id, date, year, doy.
    _schema = _pq.read_schema(pixel_df_path)
    _keep = set(feature_cols) | set(s1_feature_cols) | {"source", "scl_purity", "vh", "vv"}
    _read_cols = [f.name for f in _schema if f.name in _keep]
    df = ensure_float32_bands(_pl.read_parquet(pixel_df_path, columns=_read_cols))
    _gc.collect()

    _s1_col_set = set(s1_feature_cols)

    if use_s1 == "mixed" or use_s1 is True:
        s2 = _prep_s2(df.filter(_pl.col("source") == "S2"), scl_purity_min, feature_cols)
        s1 = _prep_s1(_despeckle(df.filter(_pl.col("source") == "S1"), s1_despeckle_window))
        # Add missing feature cols as nulls so we can stack
        for c in feature_cols:
            if c not in s2.columns: s2 = s2.with_columns(_pl.lit(None).cast(_pl.Float32).alias(c))
            if c not in s1.columns: s1 = s1.with_columns(_pl.lit(None).cast(_pl.Float32).alias(c))
        df = _pl.concat([s2.select(feature_cols + ["source"]), s1.select(feature_cols + ["source"])], how="diagonal_relaxed")
        del s2, s1
        _gc.collect()

        # Per-source stats: S2 cols from S2 rows, S1 cols from S1 rows
        _src_stats = (
            df.lazy()
            .group_by("source")
            .agg(
                [_pl.col(c).mean().alias(f"{c}__mean") for c in feature_cols] +
                [_pl.col(c).std().alias(f"{c}__std")  for c in feature_cols]
            )
            .collect()
        )
        _s2_st = _src_stats.filter(_pl.col("source") == "S2")
        _s1_st = _src_stats.filter(_pl.col("source") == "S1")
        band_mean = _np.array([
            (_s1_st[f"{c}__mean"][0] if c in _s1_col_set else _s2_st[f"{c}__mean"][0])
            for c in feature_cols
        ], dtype=_np.float32)
        band_std = _np.array([
            (_s1_st[f"{c}__std"][0] if c in _s1_col_set else _s2_st[f"{c}__std"][0])
            for c in feature_cols
        ], dtype=_np.float32)
    else:
        if "source" in df.columns:
            df = df.filter(_pl.col("source") == "S2")
        df = _prep_s2(df, scl_purity_min, feature_cols)
        _col_stats = (
            df.lazy()
            .select(
                [_pl.col(c).mean().alias(f"{c}__mean") for c in feature_cols] +
                [_pl.col(c).std().alias(f"{c}__std")  for c in feature_cols]
            )
            .collect()
        )
        band_mean = _np.array([_col_stats[f"{c}__mean"][0] for c in feature_cols], dtype=_np.float32)
        band_std  = _np.array([_col_stats[f"{c}__std"][0]  for c in feature_cols], dtype=_np.float32)

    band_mean = _np.where(_np.isnan(band_mean), 0.0, band_mean)
    band_std  = _np.where(band_std < 1e-6, 1.0, band_std)

    # Global feature stats
    global_feat_mean = _np.zeros(0, dtype=_np.float32)
    global_feat_std  = _np.ones(0,  dtype=_np.float32)
    if global_features_df_path is not None:
        gf = _pl.from_arrow(_pq.read_table(global_features_df_path))
        feat_cols_gf = [c for c in gf.columns if c != "point_id"]
        gf_arr = gf.select(feat_cols_gf).to_numpy().astype(_np.float32)
        global_feat_mean = _np.nanmean(gf_arr, axis=0)
        global_feat_std  = _np.nanstd(gf_arr,  axis=0)
        global_feat_mean = _np.where(_np.isnan(global_feat_mean), 0.0, global_feat_mean)
        global_feat_std  = _np.where(global_feat_std < 1e-6, 1.0, global_feat_std)

    _np.savez(out_path, band_mean=band_mean, band_std=band_std,
              global_feat_mean=global_feat_mean, global_feat_std=global_feat_std)


def _compute_band_stats_subprocess(
    parquet_path: Path,
    out_npz: Path,
    use_s1: "bool | str",
    feature_cols: list,
    s1_feature_cols: list,
    scl_purity_min: float,
    s1_despeckle_window: int,
    global_features_df_path: "Path | None",
) -> dict:
    """Run _compute_band_stats_worker in a subprocess; return stats as dict of arrays."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_compute_band_stats_worker,
        args=(
            str(parquet_path), str(out_npz),
            use_s1, feature_cols, s1_feature_cols,
            scl_purity_min, s1_despeckle_window,
            str(global_features_df_path) if global_features_df_path else None,
        ),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Band stats subprocess exited with code {p.exitcode}")
    data = np.load(str(out_npz) + ".npz")
    return {k: data[k] for k in data.files}


def _build_dataset_worker(
    pixel_df_path: str,
    labels: dict,
    out_path: str,
    kwargs: dict,
) -> None:
    """Multiprocessing worker: build TAMDataset, write to files, exit.

    Runs in a child process so all jemalloc arenas allocated during Polars
    operations (sort, concat, filter) are unconditionally reclaimed when this
    process exits — the OS reclaims the arenas regardless of jemalloc's pool.
    """
    import gc as _gc
    import polars as _pl
    from analysis.constants import ensure_float32_bands
    from tam.core.dataset import TAMDataset

    pixel_df = ensure_float32_bands(_pl.read_parquet(pixel_df_path))
    _gc.collect()

    ds = TAMDataset(pixel_df, labels, **kwargs)
    del pixel_df
    _gc.collect()

    ds.to_files(Path(out_path))


def _write_dataset_parquet(
    pixel_df: pl.DataFrame,
    tmp_dir: Path,
    name: str,
) -> Path:
    """Write pixel_df to a parquet file and return its path.

    Call this BEFORE freeing pixel_df in the caller, then del the frame, then
    call _build_dataset_subprocess_from_parquet.  This two-phase split lets the
    caller drop the large frame before the child process starts, preventing
    parent+child from holding duplicate copies simultaneously.
    """
    parquet_path = tmp_dir / f"{name}_pixel_df.parquet"
    _str_cols = [c for c in pixel_df.columns if pixel_df[c].dtype == pl.Categorical]
    if _str_cols:
        pixel_df = pixel_df.with_columns([pl.col(c).cast(pl.String) for c in _str_cols])
    pixel_df.write_parquet(parquet_path, compression="uncompressed")
    return parquet_path


def _build_dataset_subprocess(
    parquet_path: Path,
    labels: dict,
    tmp_dir: Path,
    name: str,
    kwargs: dict,
) -> "TAMDataset":
    """Build a TAMDataset in a subprocess to reclaim jemalloc phantom arenas.

    Caller must have already freed the pixel_df before calling this (use
    _write_dataset_parquet first, then del the frame, then call this).
    The child reads the parquet, builds TAMDataset, writes numpy arrays to
    *tmp_dir/name/*, then exits — unconditionally reclaiming all jemalloc arenas.
    The parent loads the arrays via mmap (no second allocation of the payload).
    """
    from tam.core.dataset import TAMDataset

    ds_dir = tmp_dir / name
    ds_dir.mkdir(exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_build_dataset_worker,
        args=(str(parquet_path), labels, str(ds_dir), kwargs),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"TAMDataset subprocess for '{name}' exited with code {p.exitcode}")

    train_augment_kwargs = {
        k: kwargs[k] for k in ("doy_jitter", "doy_phase_shift", "band_noise_std", "obs_dropout_min", "p_gate", "T_gate")
        if k in kwargs
    }
    return TAMDataset.from_files(ds_dir, labels, **train_augment_kwargs)


def _build_dataset_sharded(
    parquet_path: Path,
    labels: dict,
    tmp_dir: Path,
    name: str,
    kwargs: dict,
    n_shards: int,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    global_feat_mean: np.ndarray,
    global_feat_std: np.ndarray,
) -> "TAMDataset":
    """Build a TAMDataset by splitting pixels across N sequential subprocesses.

    Each shard covers a disjoint subset of point_ids.  Subprocesses run one at
    a time so peak RSS = parent_baseline + one_shard_peak rather than
    parent_baseline + full_dataset_peak.

    Band normalisation stats are pre-computed (by _compute_band_stats_subprocess)
    and passed in; every shard uses the same stats so the merged arrays are
    in a consistent normalised space.
    """
    from tam.core.dataset import TAMDataset

    import pyarrow as _pa
    import pyarrow.parquet as _pq

    import numpy as _np

    # Unique pids stay in PyArrow — no Python list materialisation over all rows.
    pid_col = _pq.read_table(str(parquet_path), columns=["point_id"])["point_id"]
    unique_pids_pa = _pa.chunked_array([pid_col.dictionary_encode().combine_chunks().dictionary])
    n_unique = len(unique_pids_pa)
    del pid_col

    # Split into n_shards disjoint slices (PyArrow slice = zero-copy view).
    shard_size = max(1, n_unique // n_shards)
    pid_groups_pa: list[_pa.Array] = []
    for i in range(n_shards):
        start = i * shard_size
        end   = n_unique if i == n_shards - 1 else (i + 1) * shard_size
        pid_groups_pa.append(unique_pids_pa.slice(start, end - start).combine_chunks())

    train_augment_kwargs = {
        k: kwargs[k] for k in ("doy_jitter", "doy_phase_shift", "band_noise_std", "obs_dropout_min", "p_gate", "T_gate")
        if k in kwargs
    }

    shard_parquets: list[Path] = []
    shard_dirs: list[Path] = []
    for i in range(n_shards):
        shard_name = f"{name}_shard{i}"
        shard_dir  = tmp_dir / shard_name
        shard_dir.mkdir(exist_ok=True)
        shard_dirs.append(shard_dir)
        shard_parquets.append(tmp_dir / f"{shard_name}_pixel_df.parquet")

    # Route rows to shards using a lazy scan + join — avoids materialising the full
    # frame in the parent process.  Each shard parquet is written via sink_parquet
    # (streaming), keeping peak RSS to roughly one shard at a time rather than the
    # full dataset.
    def _rss_gb() -> float:
        with open("/proc/self/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS:"):
                    return int(_l.split()[1]) / 1e6
        return float("nan")

    logger.info("_build_dataset_sharded: RSS before shard split: %.1f GB", _rss_gb())
    _shard_map = pl.DataFrame({
        "point_id": [p for g in pid_groups_pa for p in g.to_pylist()],
        "_shard":   pl.Series(
            [i for i, g in enumerate(pid_groups_pa) for _ in range(len(g))],
            dtype=pl.Int8,
        ),
    })
    _lazy_full = pl.scan_parquet(str(parquet_path)).join(
        _shard_map.lazy(), on="point_id", how="left"
    )
    del _shard_map
    # Write each shard parquet by streaming — never materialises the full frame.
    for i, _path in enumerate(shard_parquets):
        (
            _lazy_full
            .filter(pl.col("_shard") == i)
            .drop("_shard")
            .sink_parquet(str(_path), compression="uncompressed")
        )
        logger.info("_build_dataset_sharded: shard %d written, RSS=%.1f GB", i, _rss_gb())
    del _lazy_full
    gc.collect()

    # Launch one subprocess per shard sequentially; each reads its pre-written parquet.
    shard_kwargs = dict(kwargs,
                        band_mean=band_mean, band_std=band_std,
                        global_feat_mean=global_feat_mean,
                        global_feat_std=global_feat_std)

    shard_label_sets: list[dict] = []
    for pid_group_pa in pid_groups_pa:
        pid_set_s = set(pid_group_pa.to_pylist())
        shard_label_sets.append({k: v for k, v in labels.items()
                                  if (k[0] if isinstance(k, tuple) else k) in pid_set_s})

    ctx = multiprocessing.get_context("spawn")
    shards: list[TAMDataset] = []
    for i, shard_labels in enumerate(shard_label_sets):
        shard_name = f"{name}_shard{i}"
        logger.info("Building TAMDataset(%s shard %d/%d, %d pixels) in subprocess",
                    name, i + 1, n_shards, len(pid_groups_pa[i]))
        p = ctx.Process(
            target=_build_dataset_worker,
            args=(str(shard_parquets[i]), shard_labels, str(shard_dirs[i]), shard_kwargs),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"TAMDataset shard subprocess '{shard_name}' exited with code {p.exitcode}"
            )
        shards.append(TAMDataset.from_files(shard_dirs[i], shard_labels, **train_augment_kwargs))
        gc.collect()

    return TAMDataset.merge_shards(
        shards, labels,
        band_mean=band_mean, band_std=band_std,
        global_feat_mean=global_feat_mean, global_feat_std=global_feat_std,
        **train_augment_kwargs,
    )


def train_tam(
    pixel_df: pl.DataFrame,
    labels: dict[str, float],
    pixel_coords: pl.DataFrame,
    out_dir: Path,
    cfg: TAMConfig | None = None,
    device: str | None = None,
    precomputed_band_summaries: pl.DataFrame | None = None,
    precomputed_split: dict | None = None,
) -> tuple[TAMClassifier, float]:
    """Train a TAMClassifier and save checkpoint to out_dir.

    Parameters
    ----------
    pixel_df:
        Raw observations for labeled pixels (all years). Must contain year column
        (add via signals._shared.load_and_filter). Pass via [frame].pop() at the
        call site so the caller's reference is dropped before Python binds the
        local name here — otherwise del pixel_df only goes refcount 2→1.
    labels:
        dict mapping point_id → label in {0.0, 1.0}.
    pixel_coords:
        DataFrame with point_id, lon, lat (one row per unique pixel) for spatial split.
    out_dir:
        Directory to write tam_model.pt, tam_band_stats.npz, tam_config.json.
    cfg:
        TAMConfig instance. Defaults to TAMConfig() if not provided.
    precomputed_split:
        When provided (by _train_worker.py after _prep_worker.py has run), skip all
        preprocessing and go straight to dataset construction. Must contain keys:
        train_pixel_df, val_pixel_df, train_py_labels, val_py_labels, global_feat_df.

    Returns
    -------
    Tuple of (best-val-AUC TAMClassifier with weights loaded from checkpoint, best_val_auc float).
    """

    if cfg is None:
        cfg = TAMConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    _feature_cols_override = list(cfg.feature_cols_override) if cfg.feature_cols_override else None
    _s1_feature_cols_override = list(cfg.s1_feature_cols) if cfg.s1_feature_cols else None

    def _log_rss(tag: str) -> None:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        logger.info("RSS %s: %.1f GB", tag, int(line.split()[1]) / 1e6)
                        break
        except Exception:
            pass

    # --- Fast path: preprocessing already done by _prep_worker ---------------
    _presorted = precomputed_split is not None
    # precomputed_split values may be Path objects (parquet paths) rather than
    # loaded DataFrames — the train worker passes paths to avoid loading them
    # in the parent process before the dataset subprocess runs.
    _precomputed_paths: dict[str, Path] | None = None
    if precomputed_split is not None:
        _raw_train = precomputed_split["train_pixel_df"]
        _raw_val   = precomputed_split["val_pixel_df"]
        if isinstance(_raw_train, Path):
            # Path mode: defer loading entirely to the dataset subprocess.
            _precomputed_paths = {
                "train": _raw_train,
                "val":   _raw_val,
            }
            train_pixel_df  = None  # type: ignore[assignment]
            val_pixel_df    = None  # type: ignore[assignment]
        else:
            train_pixel_df  = _raw_train
            val_pixel_df    = _raw_val
        train_py_labels = precomputed_split["train_py_labels"]
        val_py_labels   = precomputed_split["val_py_labels"]
        global_feat_df  = precomputed_split["global_feat_df"]
        _log_rss("entry (precomputed_split)")
    else:
        # --- Split labels -----------------------------------------------------
        if cfg.val_region_ids:
            train_labels, val_labels = region_holdout_split(labels, cfg.val_region_ids)
        elif cfg.val_sites:
            train_labels, val_labels = site_holdout_split(labels, cfg.val_sites)
        else:
            train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)

        _log_rss("after load")
        logger.info(
            "pixel_df on entry: %d rows × %d cols  estimated_size=%.1f GB  cols=%s",
            len(pixel_df), pixel_df.width,
            pixel_df.estimated_size() / 1e9,
            pixel_df.columns,
        )

        # --- Early column trim -----------------------------------------------
        _active_s1_cols = _s1_feature_cols_override or S1_FEATURE_COLS
        _s1_raw = ["vh", "vv"] if cfg.use_s1 not in (False, None) else []
        _feature_cols_base = set(_feature_cols_override) if _feature_cols_override else set(BAND_COLS)
        _keep_cols = {"point_id", "date", "year", "doy", "scl_purity", "scl", "source"} | \
                     _feature_cols_base | set(_active_s1_cols) | set(_s1_raw)
        _trim_cols = [c for c in pixel_df.columns if c in _keep_cols]
        if len(_trim_cols) < pixel_df.width:
            pixel_df = pixel_df.select(_trim_cols)
        _str_cols = [c for c in ("point_id", "source") if c in pixel_df.columns
                     and pixel_df[c].dtype != pl.Categorical]
        if _str_cols:
            pixel_df = pixel_df.with_columns([pl.col(c).cast(pl.Categorical) for c in _str_cols])
        gc.collect()
        logger.info("pixel_df after column trim: estimated_size=%.1f GB  cols=%s", pixel_df.estimated_size()/1e9, pixel_df.columns)
        _log_rss("after column trim")

        # --- SCL=6 exclusion -------------------------------------------------
        if "scl" in pixel_df.columns and "source" in pixel_df.columns:
            n_before = len(pixel_df)
            pixel_df = pixel_df.filter(
                ~((pl.col("source") == "S2") & (pl.col("scl") == 6))
            )
            logger.info("SCL=6 exclusion: removed %d observations", n_before - len(pixel_df))
        if "scl" in pixel_df.columns:
            pixel_df = pixel_df.drop("scl")
        _log_rss("after SCL exclusion")

        # --- Global features + presence-filter slim extracts -----------------
        global_feat_df: pl.DataFrame | None = None
        _presence_filter_s1_slim: pl.DataFrame | None = None
        _presence_filter_s2_slim: pl.DataFrame | None = None
        _has_source = "source" in pixel_df.columns

        if cfg.use_band_summaries:
            if precomputed_band_summaries is not None:
                global_feat_df = precomputed_band_summaries
                logger.info("Using precomputed band summaries: %d pixels, %d features", len(global_feat_df), global_feat_df.width - 1)
            else:
                _s2_n = int((pixel_df["source"] == "S2").sum()) if _has_source else len(pixel_df)
                logger.info("Computing band summaries (%d rows) ...", _s2_n)
                global_feat_df = _compute_band_summaries(pixel_df, V9_FEATURE_COLS)
                logger.info("Band summaries computed: %d pixels, %d features", len(global_feat_df), global_feat_df.width - 1)
        elif cfg.n_global_features > 0:
            global_feat_df = _load_or_compute_global_features(pixel_df, out_dir, GLOBAL_FEATURE_NAMES)
        _log_rss("after band summaries")

        if (cfg.presence_min_vh_dry_db > -99 and _has_source and "vh" in pixel_df.columns):
            _doy_col = "doy" if "doy" in pixel_df.columns else None
            _date_col = "doy" if _doy_col else "date"
            _presence_pids = {pid for pid, lbl in labels.items() if lbl == 1.0}
            _presence_filter_s1_slim = (
                pixel_df.lazy()
                .filter(
                    (pl.col("source") == "S1") &
                    pl.col("point_id").is_in(_presence_pids) &
                    pl.col(_date_col).is_between(_DRY_DOY_MIN, _DRY_DOY_MAX)
                )
                .select(["point_id", "year", "vh", _date_col])
                .collect()
            )
            _has_scl = "scl" in pixel_df.columns
            _s2_ndvi_cols = ["point_id", "year", _date_col] + (["scl"] if _has_scl else [])
            _ndvi_expr = (pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))
            _presence_filter_s2_slim = (
                pixel_df.lazy()
                .filter(
                    (pl.col("source") == "S2") &
                    pl.col("point_id").is_in(_presence_pids) &
                    pl.col(_date_col).is_between(_DRY_DOY_MIN, _DRY_DOY_MAX)
                )
                .with_columns(_ndvi_expr.alias("NDVI"))
                .select(_s2_ndvi_cols + ["NDVI"])
                .collect()
            )

        if "source" in pixel_df.columns and cfg.use_s1 not in (True, "mixed", "s1_only"):
            pixel_df = pixel_df.filter(pl.col("source") == "S2")
            gc.collect()
        logger.info("pixel_df after S1 drop: estimated_size=%.1f GB  rows=%d", pixel_df.estimated_size()/1e9, len(pixel_df))
        _log_rss("after S1 drop")

        # --- Spatial stride --------------------------------------------------
        px_counts: dict[tuple[str, str], int] = Counter(_site_class(pid) for pid in labels)
        raw_counts: dict[tuple[str, str], int] = {}
        noise_removed: dict[tuple[str, str], int] = {}
        stride_removed: dict[tuple[str, str], int] = {}

        if cfg.spatial_stride > 1:
            candidate_pids = set(train_labels.keys())
            coord_subset = pixel_coords.filter(pl.col("point_id").is_in(candidate_pids))
            if cfg.stride_exclude_sites:
                exc_sites = set(cfg.stride_exclude_sites)
                _site_map = {pid: _site_class(pid)[0] for pid in coord_subset["point_id"].unique().to_list()}
                coord_subset = coord_subset.with_columns(
                    pl.col("point_id").replace(_site_map).alias("_site")
                )
                excluded_pids = set(coord_subset.filter(pl.col("_site").is_in(exc_sites))["point_id"].to_list())
                to_stride = coord_subset.filter(~pl.col("_site").is_in(exc_sites))
            else:
                excluded_pids: set[str] = set()
                to_stride = coord_subset
            strided_pids = set(
                to_stride.sort(["lat", "lon"])["point_id"].to_list()[::cfg.spatial_stride]
            ) | excluded_pids
            removed_by_stride = set(train_labels.keys()) - strided_pids
            for pid in removed_by_stride:
                key = _site_class(pid)
                stride_removed[key] = stride_removed.get(key, 0) + 1
            train_labels = {k: v for k, v in train_labels.items() if k in strided_pids}

        # --- Broadcast pixel labels → pixel-year labels ----------------------
        labeled_pids = set(labels.keys())
        pixel_years = (
            pixel_df.filter(pl.col("point_id").is_in(labeled_pids))
            .select(["point_id", "year"])
            .unique()
        )

        def _broadcast_to_pixel_years(lbl: dict[str, float]) -> dict[tuple[str, int], float]:
            lbl_df = pl.DataFrame({"point_id": list(lbl.keys()), "_label": list(lbl.values())}).with_columns(pl.col("point_id").cast(pl.Categorical))
            joined = pixel_years.join(lbl_df, on="point_id", how="inner")
            return {
                (pid, yr): label
                for pid, yr, label in zip(
                    joined["point_id"].to_list(),
                    joined["year"].to_list(),
                    joined["_label"].to_list(),
                )
            }

        train_py_labels = _broadcast_to_pixel_years(train_labels)
        val_py_labels   = _broadcast_to_pixel_years(val_labels)

        all_py = {**train_py_labels, **val_py_labels}
        pid_to_sc: dict[str, tuple[str, str]] = {pid: _site_class(pid) for pid in labeled_pids}
        raw_counts = Counter(pid_to_sc[k[0]] for k in all_py)

        # --- Presence filter -------------------------------------------------
        if _presence_filter_s1_slim is not None:
            s1_slim = _presence_filter_s1_slim
            s2_slim = _presence_filter_s2_slim
            _doy_col = "doy" if "doy" in s1_slim.columns else None
            _date_col = "doy" if _doy_col else "date"
            _has_scl = s2_slim is not None and "scl" in s2_slim.columns
            if len(s1_slim) > 0:
                if _doy_col:
                    doy_vals = s1_slim[_date_col].to_numpy()
                else:
                    doy_vals = s1_slim["date"].cast(pl.Utf8).str.to_date().dt.ordinal_day().to_numpy()
                vh_lin = s1_slim["vh"].cast(pl.Float32).to_numpy()
                vh_db  = lin_to_db(vh_lin)
                dry_mask = (doy_vals >= _DRY_DOY_MIN) & (doy_vals <= _DRY_DOY_MAX) & np.isfinite(vh_db)
                dry_s1 = pl.DataFrame({
                    "point_id": s1_slim["point_id"].to_numpy()[dry_mask],
                    "year":     s1_slim["year"].to_numpy()[dry_mask],
                    "_vh_db":   vh_db[dry_mask].astype(np.float32),
                })
                _mean_vh = dry_s1.group_by(["point_id", "year"]).agg(pl.col("_vh_db").mean().alias("mean_vh"))
                mean_vh_dry_py: dict[tuple[str, int], float] = {
                    (pid, yr): val
                    for pid, yr, val in zip(
                        _mean_vh["point_id"].to_list(),
                        _mean_vh["year"].to_list(),
                        _mean_vh["mean_vh"].to_list(),
                    )
                }
                mean_ndvi_dry_py: dict[tuple[str, int], float] | None = None
                if s2_slim is not None and len(s2_slim) > 0:
                    if _doy_col:
                        s2_doy = s2_slim[_date_col].to_numpy()
                    else:
                        s2_doy = s2_slim["date"].cast(pl.Utf8).str.to_date().dt.ordinal_day().to_numpy()
                    s2_dry_mask = (s2_doy >= _DRY_DOY_MIN) & (s2_doy <= _DRY_DOY_MAX)
                    if _has_scl:
                        scl_vals = s2_slim["scl"].to_numpy()
                        s2_dry_mask &= np.isin(scl_vals, [4.0, 5.0])
                    dry_s2_df = s2_slim.filter(
                        pl.Series(s2_dry_mask) & pl.col("NDVI").is_not_null()
                    )
                    if len(dry_s2_df) > 0:
                        _mean_ndvi = dry_s2_df.group_by(["point_id", "year"]).agg(
                            pl.col("NDVI").mean().alias("mean_ndvi")
                        )
                        mean_ndvi_dry_py = {
                            (pid, yr): val
                            for pid, yr, val in zip(
                                _mean_ndvi["point_id"].to_list(),
                                _mean_ndvi["year"].to_list(),
                                _mean_ndvi["mean_ndvi"].to_list(),
                            )
                        }
                train_py_labels = _apply_presence_filter(train_py_labels, mean_vh_dry_py, cfg, pid_to_sc, noise_removed, mean_ndvi_dry_py)
                val_py_labels   = _apply_presence_filter(val_py_labels,   mean_vh_dry_py, cfg, pid_to_sc, noise_removed, mean_ndvi_dry_py)
                ndvi_note = (f", NDVI rescue: VH>={cfg.presence_ndvi_rescue_vh_db:.1f} dB"
                             f" & NDVI>={cfg.presence_ndvi_rescue_min:.2f}"
                             if mean_ndvi_dry_py is not None else "")
                logger.info(
                    "Presence filter (mean_vh_dry < %.1f dB%s): removed %d presence pixel-years",
                    cfg.presence_min_vh_dry_db, ndvi_note, sum(noise_removed.values()),
                )
            del s1_slim, s2_slim, _presence_filter_s1_slim, _presence_filter_s2_slim
            gc.collect()
        _log_rss("after presence filter")

        # --- Pixel-year summary table ----------------------------------------
        train_pids: set[str] = {k[0] for k in train_py_labels}
        val_pids:   set[str] = {k[0] for k in val_py_labels}
        train_px_counts:    dict[tuple[str, str], int] = Counter(pid_to_sc[p] for p in train_pids if p in pid_to_sc)
        val_px_counts:      dict[tuple[str, str], int] = Counter(pid_to_sc[p] for p in val_pids   if p in pid_to_sc)
        train_raw_counts:   dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in train_py_labels)
        val_raw_counts:     dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in val_py_labels)
        train_final_counts: dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in train_py_labels)
        val_final_counts:   dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in val_py_labels)

        all_sc_keys = set(train_px_counts) | set(val_px_counts)
        train_sites    = sorted({s for s, _ in all_sc_keys if (s, "presence") in train_px_counts or (s, "absence") in train_px_counts})
        val_sites_only = sorted({s for s, _ in all_sc_keys if s not in set(train_sites)
                                  and ((s, "presence") in val_px_counts or (s, "absence") in val_px_counts)})
        val_sites_also = sorted({s for s, _ in all_sc_keys if s in set(train_sites)
                                  and ((s, "presence") in val_px_counts or (s, "absence") in val_px_counts)})
        val_sites_sorted = val_sites_only + val_sites_also

        col_w = max(
            (len(f"HOLDOUT: {s} {c}") for s in val_sites_sorted for c in ("presence", "absence")),
            default=20
        ) + 2

        def neg(n: int, w: int) -> str:
            return f"-{n:>{w - 2},}" if n else f"{'':>{w}}"

        def row(label: str, px: int, raw: int, noise: int, stride: int, final: int) -> str:
            return f"{label:>{col_w}}  {px:>8,}  {raw:>8,}  {neg(noise, 8)}  {neg(stride, 10)}  {final:>8,}"

        header = f"{'':>{col_w}}  {'Raw px':>8}  {'Raw py':>8}  {'Noise py':>8}  {'Stride px':>10}  {'Total py':>8}"
        sep    = "-" * len(header)

        def site_rows_split(
            sites: list[str],
            spx: dict, sraw: dict, snoise: dict, sstride: dict, sfinal: dict,
            prefix: str = "",
        ) -> tuple[list[str], int, int, int, int, int]:
            block_lines = []
            b_px = b_raw = b_noise = b_stride = b_final = 0
            for site in sites:
                for cls in ("presence", "absence"):
                    key = (site, cls)
                    if key not in sraw:
                        continue
                    px     = spx.get(key, 0)
                    raw    = sraw.get(key, 0)
                    noise  = snoise.get(key, 0)
                    stride = sstride.get(key, 0)
                    final  = sfinal.get(key, 0)
                    block_lines.append(row(f"{prefix}{site} {cls}", px, raw, noise, stride, final))
                    b_px += px; b_raw += raw; b_noise += noise; b_stride += stride; b_final += final
            return block_lines, b_px, b_raw, b_noise, b_stride, b_final

        lines = [sep, header, sep]
        train_rows, train_px, train_raw, train_noise, train_stride, train_final = site_rows_split(
            train_sites, train_px_counts, train_raw_counts, noise_removed, stride_removed, train_final_counts)
        lines.extend(train_rows)
        lines.append("")
        lines.append(row("PRESENCE",
            sum(train_px_counts.get((s, "presence"), 0) for s in train_sites),
            sum(train_raw_counts.get((s, "presence"), 0) for s in train_sites),
            sum(noise_removed.get((s, "presence"), 0) for s in train_sites),
            sum(stride_removed.get((s, "presence"), 0) for s in train_sites),
            sum(train_final_counts.get((s, "presence"), 0) for s in train_sites),
        ))
        lines.append(row("ABSENCE",
            sum(train_px_counts.get((s, "absence"), 0) for s in train_sites),
            sum(train_raw_counts.get((s, "absence"), 0) for s in train_sites),
            sum(noise_removed.get((s, "absence"), 0) for s in train_sites),
            sum(stride_removed.get((s, "absence"), 0) for s in train_sites),
            sum(train_final_counts.get((s, "absence"), 0) for s in train_sites),
        ))
        lines.append(row("TRAIN TOTAL", train_px, train_raw, train_noise, train_stride, train_final))
        lines.append(sep)
        val_rows, val_px, val_raw, val_noise, val_stride, val_final = site_rows_split(
            val_sites_sorted, val_px_counts, val_raw_counts, noise_removed, stride_removed, val_final_counts,
            prefix="HOLDOUT: ")
        lines.extend(val_rows)
        lines.append("")
        lines.append(row("PRESENCE",
            sum(val_px_counts.get((s, "presence"), 0) for s in val_sites_sorted),
            sum(val_raw_counts.get((s, "presence"), 0) for s in val_sites_sorted),
            sum(noise_removed.get((s, "presence"), 0) for s in val_sites_sorted),
            0,
            sum(val_final_counts.get((s, "presence"), 0) for s in val_sites_sorted),
        ))
        lines.append(row("ABSENCE",
            sum(val_px_counts.get((s, "absence"), 0) for s in val_sites_sorted),
            sum(val_raw_counts.get((s, "absence"), 0) for s in val_sites_sorted),
            sum(noise_removed.get((s, "absence"), 0) for s in val_sites_sorted),
            0,
            sum(val_final_counts.get((s, "absence"), 0) for s in val_sites_sorted),
        ))
        lines.append(row("VAL TOTAL", val_px, val_raw, val_noise, val_stride, val_final))
        lines.append(sep)
        total_px       = train_px    + val_px
        total_stride_n = train_stride + val_stride
        total_raw      = train_raw   + val_raw
        total_noise    = train_noise + val_noise
        total_final    = train_final + val_final
        lines.append(row("TOTAL", total_px, total_raw, total_noise, total_stride_n, total_final))
        lines.append(sep)
        logger.info("Pixel summary (stride=%d):\n%s", cfg.spatial_stride, "\n".join(lines))

        # Slice global features to exactly the columns the model head expects.
        if global_feat_df is not None and not cfg.use_band_summaries:
            global_feat_df = global_feat_df.select(
                ["point_id"] + global_feat_df.columns[1:cfg.n_global_features + 1]
            )

        # Slice pixel_df into train/val and free the original.
        if "source" in pixel_df.columns and cfg.use_s1 not in (True, "mixed", "s1_only"):
            pixel_df = pixel_df.drop("source")
            gc.collect()
        train_pids_ds = set(k[0] for k in train_py_labels)
        val_pids_ds   = set(k[0] for k in val_py_labels)
        train_pixel_df = pixel_df.filter(pl.col("point_id").is_in(train_pids_ds))
        val_pixel_df   = pixel_df.filter(pl.col("point_id").is_in(val_pids_ds))
        del pixel_df
        gc.collect()
        _log_rss("after split + free pixel_df")

    # --- Train dataset -------------------------------------------------------
    _log_rss("before train_ds")

    _doy_s2_counts: np.ndarray | None = None
    _doy_s1_counts: np.ndarray | None = None
    if cfg.doy_density_norm and train_pixel_df is not None:
        if "source" in train_pixel_df.columns:
            s2_doys = train_pixel_df.filter(pl.col("source") == "S2")["doy"].to_numpy().astype(int)
            s1_doys = train_pixel_df.filter(pl.col("source") == "S1")["doy"].to_numpy().astype(int)
        else:
            s2_doys = train_pixel_df["doy"].to_numpy().astype(int)
            s1_doys = s2_doys
        _doy_s2_counts = np.bincount(s2_doys, minlength=366)
        _doy_s1_counts = np.bincount(s1_doys, minlength=366)
        del s2_doys, s1_doys

    _ds_common_kwargs: dict = dict(
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        global_features_df=global_feat_df,
        use_s1=cfg.use_s1,
        pixel_zscore=cfg.pixel_zscore,
        s1_despeckle_window=cfg.s1_despeckle_window,
        feature_cols_override=_feature_cols_override,
        s1_feature_cols_override=_s1_feature_cols_override,
        max_seq_len=cfg.max_seq_len,
        presorted=_presorted,
    )

    _ds_tmp_dir: tempfile.TemporaryDirectory | None = None
    try:
        if cfg.dataset_subprocess:
            # Prefer /dev/shm (guaranteed tmpfs on Linux) over /tmp which may be
            # disk-backed on some systems.  Fall back to /tmp if /dev/shm is absent.
            _shm_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
            _ds_tmp_dir = tempfile.TemporaryDirectory(prefix="tam_ds_", dir=_shm_base)
            _ds_tmp_path = Path(_ds_tmp_dir.name)
    
            if _precomputed_paths is not None:
                _train_parquet = _precomputed_paths["train"]
            else:
                _train_parquet = _write_dataset_parquet(train_pixel_df, _ds_tmp_path, "train")
                del train_pixel_df
                gc.collect()
    
            _train_kwargs = dict(**_ds_common_kwargs,
                                 doy_jitter=cfg.doy_jitter,
                                 doy_phase_shift=cfg.doy_phase_shift,
                                 band_noise_std=cfg.band_noise_std,
                                 obs_dropout_min=cfg.obs_dropout_min,
                                 p_gate=cfg.p_gate,
                                 T_gate=cfg.T_gate)
    
            if cfg.n_dataset_shards > 1:
                # Step 1: compute normalisation stats in a lightweight subprocess.
                # Write global_feat_df to a temp parquet so the worker can read it.
                # Resolve feature cols the same way TAMDataset.__init__ would.
                from tam.core.dataset import ALL_FEATURE_COLS as _ALL_FC
                _s2_base_cols    = list(_feature_cols_override) if _feature_cols_override else list(_ALL_FC)
                _stats_s1_cols   = list(_s1_feature_cols_override) if _s1_feature_cols_override else list(S1_FEATURE_COLS)
                if cfg.use_s1 in (True, "mixed"):
                    _stats_feat_cols = _s2_base_cols + _stats_s1_cols
                else:
                    _stats_feat_cols = _s2_base_cols
                _gf_parquet_path: Path | None = None
                if global_feat_df is not None:
                    _gf_parquet_path = _ds_tmp_path / "global_feat_for_stats.parquet"
                    global_feat_df.write_parquet(str(_gf_parquet_path))
                elif _precomputed_paths is not None and "global_feat" in _precomputed_paths:
                    _gf_parquet_path = _precomputed_paths["global_feat"]
    
                logger.info("Computing band stats from train parquet in subprocess")
                _stats = _compute_band_stats_subprocess(
                    parquet_path=_train_parquet,
                    out_npz=_ds_tmp_path / "band_stats",
                    use_s1=cfg.use_s1,
                    feature_cols=_stats_feat_cols or [],
                    s1_feature_cols=_stats_s1_cols,
                    scl_purity_min=cfg.scl_purity_min,
                    s1_despeckle_window=cfg.s1_despeckle_window,
                    global_features_df_path=_gf_parquet_path,
                )
                _band_mean_pre   = _stats["band_mean"]
                _band_std_pre    = _stats["band_std"]
                _gf_mean_pre     = _stats["global_feat_mean"]
                _gf_std_pre      = _stats["global_feat_std"]
                _log_rss("after band stats subprocess")
    
                # Step 2: build train shards, each using the shared stats.
                logger.info("Building TAMDataset(train) in %d shards", cfg.n_dataset_shards)
                train_ds = _build_dataset_sharded(
                    parquet_path=_train_parquet,
                    labels=train_py_labels,
                    tmp_dir=_ds_tmp_path,
                    name="train",
                    kwargs=_train_kwargs,
                    n_shards=cfg.n_dataset_shards,
                    band_mean=_band_mean_pre,
                    band_std=_band_std_pre,
                    global_feat_mean=_gf_mean_pre,
                    global_feat_std=_gf_std_pre,
                )
            else:
                logger.info("Building TAMDataset(train) in subprocess → %s", _ds_tmp_path / "train")
                train_ds = _build_dataset_subprocess(
                    _train_parquet, train_py_labels, _ds_tmp_path, "train",
                    kwargs=_train_kwargs,
                )
        else:
            train_ds = TAMDataset(
                train_pixel_df, train_py_labels,
                doy_jitter=cfg.doy_jitter,
                doy_phase_shift=cfg.doy_phase_shift,
                band_noise_std=cfg.band_noise_std,
                obs_dropout_min=cfg.obs_dropout_min,
                p_gate=cfg.p_gate,
                T_gate=cfg.T_gate,
                _log_rss=_log_rss,
                **_ds_common_kwargs,
            )
    
        band_mean, band_std = train_ds.band_stats
        if not cfg.dataset_subprocess:
            del train_pixel_df
        gc.collect()
        _log_rss("after train_ds, before val_ds")
    
        # --- Val dataset ---------------------------------------------------------
        if cfg.dataset_subprocess:
            if _precomputed_paths is not None:
                _val_parquet = _precomputed_paths["val"]
            else:
                _val_parquet = _write_dataset_parquet(val_pixel_df, _ds_tmp_path, "val")
                del val_pixel_df
                gc.collect()
    
            logger.info("Building TAMDataset(val) in subprocess → %s", _ds_tmp_path / "val")
            val_ds = _build_dataset_subprocess(
                _val_parquet, val_py_labels, _ds_tmp_path, "val",
                kwargs=dict(**_ds_common_kwargs,
                            band_mean=band_mean, band_std=band_std,
                            global_feat_mean=train_ds.global_feat_mean,
                            global_feat_std=train_ds.global_feat_std,
                            doy_jitter=0),
            )
        else:
            val_ds = TAMDataset(
                val_pixel_df, val_py_labels,
                band_mean=band_mean, band_std=band_std,
                global_feat_mean=train_ds.global_feat_mean,
                global_feat_std=train_ds.global_feat_std,
                doy_jitter=0,
                _log_rss=_log_rss,
                **_ds_common_kwargs,
            )
            del val_pixel_df
        gc.collect()
        logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))
        _log_rss("after val_ds")
    
        n_cpu = os.cpu_count() or 4
    
        if cfg.dataloader_workers >= 0:
            n_workers = cfg.dataloader_workers
            _avail_gb = float("nan")
        else:
            n_workers = min(max(2, n_cpu - 2), 4)  # GPU-bound; cap keeps GPU fed without forking excess RAM
    
            # Scale workers down when RSS is high: each worker spawns a new process that
            # receives a pickled copy of the dataset over a pipe. At >40 GB RSS the OOM
            # killer hits the worker before it finishes unpickling (exit code -9,
            # UnpicklingError: pickle data was truncated). Fall back to 0 workers
            # (in-process, no fork) when available memory is tight.
            try:
                with open("/proc/meminfo") as _mf:
                    _avail_kb = next(
                        int(l.split()[1]) for l in _mf if l.startswith("MemAvailable")
                    )
                _avail_gb = _avail_kb / 1e6
            except Exception:
                _avail_gb = float("inf")
    
            try:
                with open("/proc/self/status") as _sf:
                    for _sl in _sf:
                        if _sl.startswith("VmRSS:"):
                            _rss_gb = int(_sl.split()[1]) / 1e6
                            break
                    else:
                        _rss_gb = float("nan")
            except Exception:
                _rss_gb = float("nan")
    
            if _avail_gb < 20:
                n_workers = 0
    
        _pin = n_workers > 0  # pin_memory requires worker processes; useless at 0
        _persist = n_workers > 0
        _prefetch = 4 if n_workers > 0 else None
        # DataLoader is created before model.to(device) so no CUDA context exists yet.
        # fork workers share dataset memory via copy-on-write — negligible extra RSS
        # vs spawn which re-pickles the full dataset into each worker.
        _mp_ctx = "fork" if n_workers > 0 else None
    
        torch.set_num_threads(1)
    
        # Wrap train_ds with gate augmentation if p_gate > 0.
        # GateAugDataset expands __len__ by round(p_gate*N) extra slots, each a
        # T_gate-truncated view of a randomly selected pixel. reshuffle_gate() is
        # called at the start of every epoch so coverage rotates without replacement.
        _gate_aug_ds: GateAugDataset | None = None
        if cfg.p_gate > 0.0:
            _gate_aug_ds = GateAugDataset(train_ds, p_gate=cfg.p_gate, T_gate=cfg.T_gate)
            _train_source = _gate_aug_ds
            logger.info(
                "GateAugDataset: %d base + %d gate slots/epoch  (p_gate=%.2f T_gate=%d)",
                len(train_ds), _gate_aug_ds._n_gate, cfg.p_gate, cfg.T_gate,
            )
        else:
            _train_source = train_ds
    
        train_loader = DataLoader(
            _train_source, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=n_workers, persistent_workers=_persist,
            pin_memory=_pin, prefetch_factor=_prefetch, multiprocessing_context=_mp_ctx,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=n_workers, persistent_workers=_persist,
            pin_memory=_pin, prefetch_factor=_prefetch, multiprocessing_context=_mp_ctx,
        )
        # Gate val loader: forces T_gate truncation on every val item; used to
        # measure cascade TNR (fraction of absence pixels scoring < 0.5 at T=T_gate).
        _gate_val_loader = None
        if cfg.p_gate > 0.0:
            _gate_val_ds = ForcedGateDataset(val_ds, T_gate=cfg.T_gate)
            _gate_val_loader = DataLoader(
                _gate_val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )
        _log_rss("after DataLoader creation")
        logger.info(
            "DataLoader workers: %d  PyTorch threads: %d  avail_mem=%.1f GB",
            n_workers, torch.get_num_threads(), _avail_gb,
        )
    
        # --- Class imbalance weight ---
        n_pos = float(sum(v == 1 for v in train_py_labels.values()))
        n_neg = float(sum(v == 0 for v in train_py_labels.values()))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device) if n_pos > 0 and n_neg > 0 else None
    
        # --- Model + optimiser --------------------------------------------------
        if cfg.use_band_summaries and global_feat_df is not None:
            cfg.n_global_features = global_feat_df.width - 1  # exclude point_id column
        del global_feat_df
        gc.collect()
        model = TAMClassifier.from_config(cfg)
        model._use_s1 = cfg.use_s1
        model._pixel_zscore = cfg.pixel_zscore
        model._max_seq_len = cfg.max_seq_len
        model._feature_cols    = _feature_cols_override
        model._s1_feature_cols = _s1_feature_cols_override
        model.to(device)
        # Keep a reference to the unwrapped model for attribute access (.config(),
        # .band_proj, etc.) which are not forwarded through the compiled wrapper.
        _raw_model = model
        if torch.cuda.is_available():
            model = torch.compile(model, dynamic=True)
        logger.info(
            "Model: d_model=%d n_heads=%d n_layers=%d d_ff=%d  params=%d",
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    
        if cfg.doy_density_norm and _doy_s2_counts is not None:
            _raw_model.set_doy_frequencies(_doy_s2_counts, _doy_s1_counts)
            logger.info(
                "DOY density normalisation enabled — S2: %d obs, S1: %d obs",
                int(_doy_s2_counts.sum()), int(_doy_s1_counts.sum()),
            )
        _log_rss("after model init")
    
        np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
        if train_ds.global_feat_mean is not None and len(train_ds.global_feat_mean) > 0:
            np.savez(out_dir / "tam_global_feat_stats.npz",
                     mean=train_ds.global_feat_mean, std=train_ds.global_feat_std)
        with open(out_dir / "tam_config.json", "w") as fh:
            json.dump(_raw_model.config(), fh, indent=2)
    
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    
        _temporal_modules = [_raw_model.band_proj, _raw_model.encoder]
    
        def _set_temporal_frozen(frozen: bool) -> None:
            for m in _temporal_modules:
                for p in m.parameters():
                    p.requires_grad = not frozen
            if frozen:
                logger.info("Temporal stream frozen — training head only (globals warmup)")
            else:
                logger.info("Temporal stream unfrozen — full model training")
    
        if cfg.warmup_freeze_epochs > 0:
            _set_temporal_frozen(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 3, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.warmup_freeze_epochs)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    
        # --- Training loop ------------------------------------------------------
        best_val_auc = 0.0
        epochs_without_improvement = 0
        checkpoint_path = out_dir / "tam_model.pt"
    
        for epoch in range(1, cfg.n_epochs + 1):
            if cfg.warmup_freeze_epochs > 0 and epoch == cfg.warmup_freeze_epochs + 1:
                _set_temporal_frozen(False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.n_epochs - cfg.warmup_freeze_epochs
                )
            if _gate_aug_ds is not None:
                _gate_aug_ds.reshuffle_gate()
            model.train()
            epoch_loss = 0.0
            train_probs, train_labels_list = [], []
            for batch in train_loader:
                bands         = batch["bands"].to(device)
                doy           = batch["doy"].to(device)
                mask          = batch["mask"].to(device)
                n_obs         = batch["n_obs"].to(device)
                global_feats  = batch["global_feats"].to(device)
                label         = batch["label"].to(device)
                weight        = batch["weight"].to(device)
    
                is_s1 = batch["is_s1"].to(device)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    prob, logit = model(bands, doy, mask, n_obs, global_feats, is_s1=is_s1)
    
                    if torch.isnan(prob).any():
                        nan_mask = torch.isnan(prob)
                        logger.warning(
                            "NaN in model output: %d/%d samples. "
                            "bands NaN=%s, doy NaN=%s, mask all-pad=%s, global_feats NaN=%s",
                            nan_mask.sum().item(), len(prob),
                            torch.isnan(bands).any().item(),
                            torch.isnan(doy.float()).any().item(),
                            mask.all(dim=1).any().item(),
                            torch.isnan(global_feats).any().item(),
                        )
                        prob = torch.nan_to_num(prob, nan=0.5)
                        logit = torch.nan_to_num(logit, nan=0.0)
    
                    loss = (criterion(logit, label) * weight).mean()
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                train_probs.extend(prob.detach().cpu().float().numpy())
                train_labels_list.extend(label.cpu().float().numpy())
    
            scheduler.step()
    
            train_probs_arr = np.array(train_probs)
            train_labels_arr = np.array(train_labels_list)
            if len(set(train_labels_arr)) > 1:
                train_auc = roc_auc_score(train_labels_arr, train_probs_arr)
            else:
                train_auc = float("nan")
    
            # --- Validation -------------------------------------------------------
            model.eval()
            val_probs, val_labels_list, val_pids = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    prob, _ = model(
                        batch["bands"].to(device),
                        batch["doy"].to(device),
                        batch["mask"].to(device),
                        batch["n_obs"].to(device),
                        batch["global_feats"].to(device),
                        is_s1=batch["is_s1"].to(device),
                    )
                    val_probs.extend(prob.cpu().float().numpy())
                    val_labels_list.extend(batch["label"].float().numpy())
                    val_pids.extend(batch["point_id"])
    
            val_probs_arr = np.array(val_probs)
            val_labels_arr = np.array(val_labels_list)
            finite = np.isfinite(val_probs_arr)
            n_nan = (~finite).sum()
            if n_nan:
                logger.warning("epoch %d: %d NaN predictions in validation set", epoch, n_nan)
    
            val_cvar, site_records = _cvar_auc(
                val_probs_arr, val_labels_arr, val_pids, finite,
                alpha=cfg.cvar_alpha,
            )
            val_auc_macro = float(np.mean([r["auc"] for r in site_records])) if site_records else float("nan")
    
            # Gate val pass: T_gate-truncated sequences, measure TNR on absence pixels.
            # Logged as supplementary info; does not affect early stopping.
            gate_tnr = float("nan")
            gate_fnr = float("nan")
            if _gate_val_loader is not None:
                model.eval()
                gate_probs, gate_labels = [], []
                with torch.no_grad():
                    for batch in _gate_val_loader:
                        prob, _ = model(
                            batch["bands"].to(device),
                            batch["doy"].to(device),
                            batch["mask"].to(device),
                            batch["n_obs"].to(device),
                            batch["global_feats"].to(device),
                            is_s1=batch["is_s1"].to(device),
                        )
                        gate_probs.extend(prob.cpu().float().numpy())
                        gate_labels.extend(batch["label"].float().numpy())
                gate_probs_arr  = np.array(gate_probs)
                gate_labels_arr = np.array(gate_labels)
                absence_mask  = gate_labels_arr == 0
                presence_mask = gate_labels_arr == 1
                if absence_mask.any():
                    gate_tnr = float((gate_probs_arr[absence_mask] < 0.5).mean())
                if presence_mask.any():
                    gate_fnr = float((gate_probs_arr[presence_mask] < 0.5).mean())
    
            gate_suffix = ""
            if not np.isnan(gate_tnr):
                gate_suffix += f"  gate_tnr={gate_tnr:.3f}"
            if not np.isnan(gate_fnr):
                gate_suffix += f"  gate_fnr={gate_fnr:.3f}"
    
            logger.info(
                "epoch %3d/%d  loss=%.4f  train_auc=%.3f  val_cvar%.0f=%.3f%s%s",
                epoch, cfg.n_epochs,
                epoch_loss / max(len(train_loader), 1),
                train_auc,
                cfg.cvar_alpha * 100, val_cvar,
                "  *" if (not np.isnan(val_cvar) and val_cvar >= best_val_auc) else "",
                gate_suffix,
            )
    
            if logger.isEnabledFor(logging.DEBUG) and site_records:
                for r in sorted(site_records, key=lambda r: r["auc"]):
                    logger.debug(
                        "  %s  n=%6d  auc=%.3f%s",
                        r["site"], r["n_pixels"], r["auc"],
                        "  <-- tail" if r["in_tail"] else "",
                    )
    
            if not np.isnan(val_cvar) and val_cvar > best_val_auc + cfg.min_delta:
                best_val_auc = val_cvar
                epochs_without_improvement = 0
                _sd = model.state_dict()
                _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
                torch.save(_sd, checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, cfg.patience)
                    break
    
        logger.info(
            "Best val CVaR%.0f AUC: %.3f  macro: %.3f — checkpoint: %s",
            cfg.cvar_alpha * 100, best_val_auc, val_auc_macro, checkpoint_path,
        )
    
        np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
        if train_ds.global_feat_mean is not None and len(train_ds.global_feat_mean) > 0:
            np.savez(out_dir / "tam_global_feat_stats.npz",
                     mean=train_ds.global_feat_mean, std=train_ds.global_feat_std)
        cfg_dict = _raw_model.config()
        cfg_dict["best_val_auc"] = round(best_val_auc, 6)
        with open(out_dir / "tam_config.json", "w") as fh:
            json.dump(cfg_dict, fh, indent=2)
    
        if not checkpoint_path.exists():
            _sd = model.state_dict()
            _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
            torch.save(_sd, checkpoint_path)
    
        _raw_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        _raw_model.eval()
    
    finally:
        if _ds_tmp_dir is not None:
            _ds_tmp_dir.cleanup()
        # Explicitly delete DataLoader references so worker processes and their
        # multiprocessing.Connection objects are closed before GC runs.
        # Without this, a session-scoped test fixture triggers
        # PytestUnraisableExceptionWarning (OSError: Bad file descriptor).
        try:
            del train_loader
        except NameError:
            pass
        try:
            del val_loader
        except NameError:
            pass
        try:
            del _gate_val_loader
        except NameError:
            pass

    return _raw_model, best_val_auc


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_tam(out_dir: Path, device: str | None = None) -> tuple[TAMClassifier, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load a saved TAMClassifier checkpoint.

    Returns
    -------
    (model, band_mean, band_std, global_feat_mean, global_feat_std)
    global_feat_mean/std are None when the model has no global features.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(out_dir / "tam_config.json") as fh:
        cfg_dict = json.load(fh)
    cfg = TAMConfig.from_dict(cfg_dict)

    state = torch.load(out_dir / "tam_model.pt", map_location=device, weights_only=True)

    # Derive n_bands from the actual weight shape — config value may be stale
    # if feature_cols_override was set but n_bands was not updated at save time.
    cfg.n_bands = state["band_proj.weight"].shape[1]

    head_in = state["head.weight"].shape[1]
    extra = head_in - cfg.d_model
    cfg.use_n_obs = extra > 0
    cfg.n_global_features = extra - (1 if cfg.use_n_obs else 0)

    model = TAMClassifier.from_config(cfg)
    if "doy_inv_freq" in state:
        ckpt_freq = state["doy_inv_freq"]
        expected = model.doy_inv_freq.shape[0]
        if ckpt_freq.shape[0] != expected:
            state["doy_inv_freq"] = ckpt_freq[:expected] if ckpt_freq.shape[0] > expected else torch.nn.functional.pad(ckpt_freq, (0, expected - ckpt_freq.shape[0]), value=1.0)
    model.load_state_dict(state, strict=False)
    model._use_s1 = cfg_dict.get("use_s1", None)
    model._pixel_zscore = cfg_dict.get("pixel_zscore", None)
    model._max_seq_len = cfg_dict.get("max_seq_len", MAX_SEQ_LEN)
    model.to(device)
    model.eval()

    stats = np.load(out_dir / "tam_band_stats.npz")
    # Columns excluded by feature_cols_override were never computed and saved as NaN.
    # Replace with identity (mean=0, std=1) so fill_windows passes them through as zero.
    band_mean_arr = np.where(np.isnan(stats["mean"]), 0.0, stats["mean"]).astype(np.float32)
    band_std_arr  = np.where(np.isnan(stats["std"]),  1.0, stats["std"]).astype(np.float32)
    gf_stats_path = out_dir / "tam_global_feat_stats.npz"
    if gf_stats_path.exists():
        gf = np.load(gf_stats_path)
        global_feat_mean: np.ndarray | None = gf["mean"]
        global_feat_std:  np.ndarray | None = gf["std"]
    else:
        global_feat_mean = None
        global_feat_std  = None
    return model, band_mean_arr, band_std_arr, global_feat_mean, global_feat_std
