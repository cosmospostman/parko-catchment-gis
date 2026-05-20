"""tam/core/train.py — Training loop for TAMClassifier."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tam.core.config import TAMConfig
from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX
from tam.core.dataset import MAX_SEQ_LEN, TAMDataset, V9_FEATURE_COLS, collate_fn, lin_to_db
from tam.core.global_features import GLOBAL_FEATURE_NAMES, compute_global_features
from tam.core.model import TAMClassifier
from analysis.constants import add_spectral_indices

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
    """
    s2 = pixel_df.filter(pl.col("source") == "S2") if "source" in pixel_df.columns else pixel_df
    cols = [c for c in feature_cols if c in s2.columns]

    aggs = (
        [pl.col(c).quantile(0.05).alias(f"{c}_p5")  for c in cols] +
        [pl.col(c).quantile(0.95).alias(f"{c}_p95") for c in cols] +
        [pl.col(c).std().alias(f"{c}_std")           for c in cols]
    )
    grp = s2.group_by("point_id").agg(aggs)
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
    return (m.group(1), m.group(2)) if m else (pid, "unknown")


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


def train_tam(
    pixel_df: pl.DataFrame,
    labels: dict[str, float],
    pixel_coords: pl.DataFrame,
    out_dir: Path,
    cfg: TAMConfig | None = None,
    device: str | None = None,
) -> tuple[TAMClassifier, float]:
    """Train a TAMClassifier and save checkpoint to out_dir.

    Parameters
    ----------
    pixel_df:
        Raw observations for labeled pixels (all years). Must contain year column
        (add via signals._shared.load_and_filter).
    labels:
        dict mapping point_id → label in {0.0, 1.0}.
    pixel_coords:
        DataFrame with point_id, lon, lat (one row per unique pixel) for spatial split.
    out_dir:
        Directory to write tam_model.pt, tam_band_stats.npz, tam_config.json.
    cfg:
        TAMConfig instance. Defaults to TAMConfig() if not provided.

    Returns
    -------
    Tuple of (best-val-AUC TAMClassifier with weights loaded from checkpoint, best_val_auc float).
    """
    if cfg is None:
        cfg = TAMConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # --- Split labels ---------------------------------------------------------
    if cfg.val_region_ids:
        train_labels, val_labels = region_holdout_split(labels, cfg.val_region_ids)
    elif cfg.val_sites:
        train_labels, val_labels = site_holdout_split(labels, cfg.val_sites)
    else:
        train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)

    def _log_rss(tag: str) -> None:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_gb = int(line.split()[1]) / 1e6
                        logger.info("RSS %s: %.1f GB", tag, rss_gb)
                        break
        except Exception:
            pass

    _log_rss("after load")

    # --- SCL=6 exclusion — strip dark-area misclassifications from S2 rows only ---
    if "scl" in pixel_df.columns and "source" in pixel_df.columns:
        n_before = len(pixel_df)
        pixel_df = pixel_df.filter(
            ~((pl.col("source") == "S2") & (pl.col("scl") == 6))
        )
        logger.info("SCL=6 exclusion: removed %d observations", n_before - len(pixel_df))
    _log_rss("after SCL exclusion")

    # --- Global features -----------------------------------------------------
    global_feat_df: pl.DataFrame | None = None
    if cfg.use_band_summaries:
        logger.info("Computing band summaries (%d rows) ...", len(pixel_df))
        if "NDVI" not in pixel_df.columns or "NDWI" not in pixel_df.columns:
            pixel_df = add_spectral_indices(pixel_df)
        global_feat_df = _compute_band_summaries(pixel_df, V9_FEATURE_COLS)
        logger.info("Band summaries computed: %d pixels, %d features", len(global_feat_df), global_feat_df.width - 1)
    elif cfg.n_global_features > 0:
        global_feat_df = _load_or_compute_global_features(pixel_df, out_dir, GLOBAL_FEATURE_NAMES)
    _log_rss("after band summaries")

    # Pre-extract S1 slim for presence filter, then drop S1 rows to free RAM.
    _presence_filter_s1_slim: pl.DataFrame | None = None
    _presence_filter_s2_slim: pl.DataFrame | None = None
    if (cfg.presence_min_vh_dry_db > -99
            and "source" in pixel_df.columns
            and "vh" in pixel_df.columns):
        _doy_col = "doy" if "doy" in pixel_df.columns else None
        _date_col = "doy" if _doy_col else "date"
        _s1_cols = ["point_id", "year", "vh", _date_col]
        _presence_filter_s1_slim = pixel_df.filter(pl.col("source") == "S1").select(_s1_cols)
        if "NDVI" in pixel_df.columns:
            _has_scl = "scl" in pixel_df.columns
            _s2_cols = ["point_id", "year", _date_col, "NDVI"] + (["scl"] if _has_scl else [])
            _presence_filter_s2_slim = pixel_df.filter(pl.col("source") == "S2").select(_s2_cols)

    # Drop S1 rows — no longer needed after band summaries and slim extraction.
    if "source" in pixel_df.columns:
        pixel_df = pixel_df.filter(pl.col("source") == "S2")
        gc.collect()
    _log_rss("after S1 drop")

    # px_counts: unique pixel count per (site, class)
    px_counts: dict[tuple[str, str], int] = Counter(_site_class(pid) for pid in labels)

    raw_counts: dict[tuple[str, str], int] = {}
    noise_removed: dict[tuple[str, str], int] = {}
    stride_removed: dict[tuple[str, str], int] = {}

    # --- Spatial stride — thin training pixels to reduce within-site redundancy.
    if cfg.spatial_stride > 1:
        coord_df = pixel_coords
        candidate_pids = set(train_labels.keys())
        coord_subset = coord_df.filter(pl.col("point_id").is_in(candidate_pids))

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

    # --- Broadcast pixel labels → (point_id, year) labels ------------------
    labeled_pids = set(labels.keys())
    pixel_years = (
        pixel_df.filter(pl.col("point_id").is_in(labeled_pids))
        .select(["point_id", "year"])
        .unique()
    )

    def _broadcast_to_pixel_years(lbl: dict[str, float]) -> dict[tuple[str, int], float]:
        """Expand pixel-level labels to a (point_id, year) dict."""
        lbl_pids = set(lbl)
        return {
            (row[0], row[1]): lbl[row[0]]
            for row in pixel_years.filter(pl.col("point_id").is_in(lbl_pids)).iter_rows()
        }

    train_py_labels = _broadcast_to_pixel_years(train_labels)
    val_py_labels   = _broadcast_to_pixel_years(val_labels)

    # Pixel-year counts before heuristic filter
    all_py = {**train_py_labels, **val_py_labels}
    pid_to_sc: dict[str, tuple[str, str]] = {pid: _site_class(pid) for pid in labeled_pids}
    raw_counts = Counter(pid_to_sc[k[0]] for k in all_py)

    # --- Presence filter: drop presence pixel-years with low dry-season VH ---
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
            _mean_vh = (
                dry_s1.group_by(["point_id", "year"])
                .agg(pl.col("_vh_db").mean().alias("mean_vh"))
            )
            mean_vh_dry_py: dict[tuple[str, int], float] = {
                (r[0], r[1]): r[2] for r in _mean_vh.iter_rows()
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
                    _mean_ndvi = (
                        dry_s2_df.group_by(["point_id", "year"])
                        .agg(pl.col("NDVI").mean().alias("mean_ndvi"))
                    )
                    mean_ndvi_dry_py = {
                        (r[0], r[1]): r[2] for r in _mean_ndvi.iter_rows()
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

    # --- Pixel-year summary table --------------------------------------------
    val_site_set = set(cfg.val_sites) if cfg.val_sites else set()
    if cfg.val_region_ids:
        val_site_set |= {_site_class(rid)[0] for rid in cfg.val_region_ids}

    final_all_py = {**train_py_labels, **val_py_labels}
    final_counts: dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in final_all_py)

    all_keys = sorted(raw_counts.keys(), key=lambda k: (k[0] in val_site_set, k[0], k[1]))

    col_w = max((len(f"{s} {c}") for s, c in all_keys), default=20) + 2

    def neg(n: int, w: int) -> str:
        return f"-{n:>{w - 2},}" if n else f"{'':>{w}}"

    def row(label: str, px: int, raw: int, noise: int, stride: int, final: int) -> str:
        return f"{label:>{col_w}}  {px:>8,}  {raw:>8,}  {neg(noise, 8)}  {neg(stride, 10)}  {final:>8,}"

    header = f"{'':>{col_w}}  {'Raw px':>8}  {'Raw py':>8}  {'Noise py':>8}  {'Stride px':>10}  {'Total py':>8}"
    sep    = "-" * len(header)

    train_sites = sorted({s for s, _ in all_keys if s not in val_site_set})
    val_sites_sorted = sorted({s for s, _ in all_keys if s in val_site_set})

    def site_rows(sites: list[str], prefix: str = "") -> tuple[list[str], int, int, int, int, int]:
        block_lines = []
        b_px = b_raw = b_noise = b_stride = b_final = 0
        for site in sites:
            for cls in ("presence", "absence"):
                key = (site, cls)
                if key not in raw_counts:
                    continue
                px     = px_counts.get(key, 0)
                raw    = raw_counts.get(key, 0)
                noise  = noise_removed.get(key, 0)
                stride = stride_removed.get(key, 0)
                final  = final_counts.get(key, 0)
                block_lines.append(row(f"{prefix}{site} {cls}", px, raw, noise, stride, final))
                b_px += px; b_raw += raw; b_noise += noise; b_stride += stride; b_final += final
        return block_lines, b_px, b_raw, b_noise, b_stride, b_final

    lines = [sep, header, sep]

    train_rows, train_px, train_raw, train_noise, train_stride, train_final = site_rows(train_sites)
    lines.extend(train_rows)
    lines.append("")
    lines.append(row("PRESENCE",
        sum(px_counts.get((s, "presence"), 0) for s in train_sites),
        sum(raw_counts.get((s, "presence"), 0) for s in train_sites),
        sum(noise_removed.get((s, "presence"), 0) for s in train_sites),
        sum(stride_removed.get((s, "presence"), 0) for s in train_sites),
        sum(final_counts.get((s, "presence"), 0) for s in train_sites),
    ))
    lines.append(row("ABSENCE",
        sum(px_counts.get((s, "absence"), 0) for s in train_sites),
        sum(raw_counts.get((s, "absence"), 0) for s in train_sites),
        sum(noise_removed.get((s, "absence"), 0) for s in train_sites),
        sum(stride_removed.get((s, "absence"), 0) for s in train_sites),
        sum(final_counts.get((s, "absence"), 0) for s in train_sites),
    ))
    lines.append(row("TRAIN TOTAL", train_px, train_raw, train_noise, train_stride, train_final))
    lines.append(sep)

    val_rows, val_px, val_raw, val_noise, val_stride, val_final = site_rows(val_sites_sorted, prefix="HOLDOUT: ")
    lines.extend(val_rows)
    lines.append("")
    lines.append(row("PRESENCE",
        sum(px_counts.get((s, "presence"), 0) for s in val_sites_sorted),
        sum(raw_counts.get((s, "presence"), 0) for s in val_sites_sorted),
        sum(noise_removed.get((s, "presence"), 0) for s in val_sites_sorted),
        0,
        sum(final_counts.get((s, "presence"), 0) for s in val_sites_sorted),
    ))
    lines.append(row("ABSENCE",
        sum(px_counts.get((s, "absence"), 0) for s in val_sites_sorted),
        sum(raw_counts.get((s, "absence"), 0) for s in val_sites_sorted),
        sum(noise_removed.get((s, "absence"), 0) for s in val_sites_sorted),
        0,
        sum(final_counts.get((s, "absence"), 0) for s in val_sites_sorted),
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

    # --- Datasets -----------------------------------------------------------
    if "source" in pixel_df.columns:
        pixel_df = pixel_df.drop("source")

    _log_rss("before train_ds")
    _feature_cols_override = list(cfg.feature_cols_override) if cfg.feature_cols_override else None
    train_ds = TAMDataset(
        pixel_df, train_py_labels,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=cfg.doy_jitter,
        doy_phase_shift=cfg.doy_phase_shift,
        band_noise_std=cfg.band_noise_std,
        obs_dropout_min=cfg.obs_dropout_min,
        global_features_df=global_feat_df,
        use_s1=cfg.use_s1,
        pixel_zscore=cfg.pixel_zscore,
        s1_despeckle_window=cfg.s1_despeckle_window,
        feature_cols_override=_feature_cols_override,
        max_seq_len=cfg.max_seq_len,
    )
    band_mean, band_std = train_ds.band_stats
    _log_rss("after train_ds, before val_ds")
    val_ds = TAMDataset(
        pixel_df, val_py_labels,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=0,
        global_features_df=global_feat_df,
        global_feat_mean=train_ds.global_feat_mean,
        global_feat_std=train_ds.global_feat_std,
        use_s1=cfg.use_s1,
        pixel_zscore=cfg.pixel_zscore,
        s1_despeckle_window=cfg.s1_despeckle_window,
        feature_cols_override=_feature_cols_override,
        max_seq_len=cfg.max_seq_len,
    )
    logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))
    _log_rss("after val_ds")

    n_cpu = os.cpu_count() or 4
    n_workers = max(2, n_cpu - 2)
    torch.set_num_threads(2)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers, persistent_workers=True,
        pin_memory=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers, persistent_workers=True,
        pin_memory=True, prefetch_factor=4,
    )
    _log_rss("after DataLoader creation")
    logger.info("DataLoader workers: %d  PyTorch threads: %d", n_workers, torch.get_num_threads())

    # --- Class imbalance weight ---
    n_pos = float(sum(v == 1 for v in train_py_labels.values()))
    n_neg = float(sum(v == 0 for v in train_py_labels.values()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device) if n_pos > 0 and n_neg > 0 else None

    # --- Model + optimiser --------------------------------------------------
    if cfg.use_band_summaries and global_feat_df is not None:
        cfg.n_global_features = global_feat_df.width - 1  # exclude point_id column
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = cfg.use_s1
    model._pixel_zscore = cfg.pixel_zscore
    model._max_seq_len = cfg.max_seq_len
    model._feature_cols = _feature_cols_override
    model.to(device)
    logger.info(
        "Model: d_model=%d n_heads=%d n_layers=%d d_ff=%d  params=%d",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff,
        sum(p.numel() for p in model.parameters()),
    )

    if cfg.doy_density_norm:
        train_pids = {k[0] for k in train_py_labels}
        train_doys = (
            pixel_df.filter(pl.col("point_id").is_in(train_pids))["doy"]
            .to_numpy().astype(int)
        )
        doy_counts = np.bincount(train_doys, minlength=366)
        model.set_doy_frequencies(doy_counts)
        logger.info("DOY density normalisation enabled — inverse-freq weights computed from %d training observations", len(train_doys))

    del pixel_df
    gc.collect()
    _log_rss("after pixel_df free")

    np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
    if train_ds.global_feat_mean is not None and len(train_ds.global_feat_mean) > 0:
        np.savez(out_dir / "tam_global_feat_stats.npz",
                 mean=train_ds.global_feat_mean, std=train_ds.global_feat_std)
    with open(out_dir / "tam_config.json", "w") as fh:
        json.dump(model.config(), fh, indent=2)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    _temporal_modules = [model.band_proj, model.encoder]

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

            prob, logit = model(bands, doy, mask, n_obs, global_feats)

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
            train_probs.extend(prob.detach().cpu().numpy())
            train_labels_list.extend(label.cpu().numpy())

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
                )
                val_probs.extend(prob.cpu().numpy())
                val_labels_list.extend(batch["label"].numpy())
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

        logger.info(
            "epoch %3d/%d  loss=%.4f  train_auc=%.3f  val_cvar%.0f=%.3f%s",
            epoch, cfg.n_epochs,
            epoch_loss / max(len(train_loader), 1),
            train_auc,
            cfg.cvar_alpha * 100, val_cvar,
            "  *" if (not np.isnan(val_cvar) and val_cvar >= best_val_auc) else "",
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
            torch.save(model.state_dict(), checkpoint_path)
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
    cfg_dict = model.config()
    cfg_dict["best_val_auc"] = round(best_val_auc, 6)
    with open(out_dir / "tam_config.json", "w") as fh:
        json.dump(cfg_dict, fh, indent=2)

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, best_val_auc


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
    gf_stats_path = out_dir / "tam_global_feat_stats.npz"
    if gf_stats_path.exists():
        gf = np.load(gf_stats_path)
        global_feat_mean: np.ndarray | None = gf["mean"]
        global_feat_std:  np.ndarray | None = gf["std"]
    else:
        global_feat_mean = None
        global_feat_std  = None
    return model, stats["mean"], stats["std"], global_feat_mean, global_feat_std
