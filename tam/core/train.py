"""tam/core/train.py — Training loop for TAMClassifier."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
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
    labels: pd.Series,
    pixel_coords: pd.DataFrame,
    val_frac: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """Hold out a spatially contiguous subset of pixels for validation.

    Splits by latitude within each class: the southernmost val_frac of
    presence pixels and the southernmost val_frac of absence pixels become
    the validation set. This avoids spatial autocorrelation leakage from
    random splits.

    Parameters
    ----------
    labels:
        Series indexed by point_id, values in {0.0, 1.0}.
    pixel_coords:
        DataFrame with point_id, lat columns (one row per pixel).
    val_frac:
        Fraction of each class to reserve for validation.

    Returns
    -------
    (train_labels, val_labels) — both Series indexed by point_id.
    """
    coords = pixel_coords.set_index("point_id")[["lat"]]
    labelled = labels.to_frame("label").join(coords)

    val_ids: list[str] = []
    for cls_val in [0.0, 1.0]:
        cls = labelled[labelled["label"] == cls_val].sort_values("lat")
        n_val = max(1, int(len(cls) * val_frac))
        val_ids.extend(cls.index[:n_val].tolist())

    val_set  = set(val_ids)
    train_labels = labels[~labels.index.isin(val_set)]
    val_labels   = labels[labels.index.isin(val_set)]
    return train_labels, val_labels


def region_holdout_split(
    labels: pd.Series,
    val_region_ids: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    """Hold out pixels whose region ID is in val_region_ids.

    Point IDs have the form <region_id>_<row>_<col>. The region is recovered
    by stripping the trailing _<row>_<col> numeric suffix.
    """
    _suffix = re.compile(r"_\d+_\d+$")
    val_set = set(val_region_ids)
    is_val = labels.index.map(lambda pid: _suffix.sub("", pid) in val_set)
    return labels[~is_val], labels[is_val]


def site_holdout_split(
    labels: pd.Series,
    val_sites: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    """Hold out all pixels whose region ID starts with any of val_sites.

    Point IDs have the form <region_id>_<row>_<col>, and region IDs have the
    form <site>_presence_N or <site>_absence_N, so the site is the prefix
    before the first '_presence' or '_absence'.
    """
    def point_site(pid: str) -> str:
        m = re.match(r"^(.+?)_(presence|absence)", pid)
        return m.group(1) if m else pid

    in_val = labels.index.map(lambda pid: point_site(pid) in val_sites)
    return labels[~in_val], labels[in_val]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_band_summaries(pixel_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-pixel [p5, p95, std] for each feature column, computed from S2 rows.

    Returns a DataFrame indexed by point_id with columns
    <col>_p5, <col>_p95, <col>_std for each col in feature_cols.
    Used as global features when cfg.use_band_summaries=True.
    """
    # Work on a minimal S2-only slice — avoid copying the full pixel_df.
    s2_mask = pixel_df["source"] == "S2" if "source" in pixel_df.columns else pd.Series(True, index=pixel_df.index)
    cols = [c for c in feature_cols if c in pixel_df.columns]
    keep_cols = ["point_id"] + cols
    s2 = pixel_df.loc[s2_mask, keep_cols]  # view, no copy

    # Compute all three stats sequentially, deleting each intermediate after concat,
    # to avoid three full (n_pixels × n_cols) result tables live simultaneously.
    g = s2.groupby("point_id")[cols]
    p5  = g.quantile(0.05).rename(columns={c: f"{c}_p5"  for c in cols})
    p95 = g.quantile(0.95).rename(columns={c: f"{c}_p95" for c in cols})
    result = pd.concat([p5, p95], axis=1)
    del p5, p95
    std = g.std().rename(columns={c: f"{c}_std" for c in cols})
    result = pd.concat([result, std], axis=1)
    del std
    ordered = [f"{c}{s}" for c in cols for s in ("_p5", "_p95", "_std")]
    return result[ordered]


def _global_features_cache_key(pixel_df: pd.DataFrame) -> str:
    """Stable hash of the inputs to compute_global_features."""
    cols = [c for c in ("point_id", "date", "B08", "B04", "vh", "vv", "source")
            if c in pixel_df.columns]
    key_df = pixel_df[cols].sort_values(cols).reset_index(drop=True)
    return hashlib.md5(key_df.to_csv(index=False).encode()).hexdigest()


def _load_or_compute_global_features(
    pixel_df: pd.DataFrame,
    out_dir: Path,
    feature_names: list[str],
) -> pd.DataFrame:
    cache_path = out_dir / "global_features_cache.parquet"
    key_path   = out_dir / "global_features_cache.key"
    cache_key  = _global_features_cache_key(pixel_df)

    if cache_path.exists() and key_path.exists() and key_path.read_text().strip() == cache_key:
        logger.info("Loading cached global features from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Computing global features: %s", feature_names)
    global_feat_df = compute_global_features(pixel_df)
    logger.info(
        "Global feature means — %s",
        "  ".join(f"{k}={global_feat_df[k].mean():.4f}" for k in feature_names),
    )
    global_feat_df.to_parquet(cache_path)
    key_path.write_text(cache_key)
    logger.info("Cached global features to %s", cache_path)
    return global_feat_df


def _site_class(pid: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return (m.group(1), m.group(2)) if m else (pid, "unknown")


def _region_from_pid(pid: str) -> str:
    """Return the region ID from a point_id of the form <region_id>_<row>_<col>."""
    return "_".join(pid.split("_")[:-2])


def _cvar_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    pids: list[str],
    finite: np.ndarray,
    alpha: float,
    min_pair_pixels: int,
) -> float:
    """Site-weighted CVaR AUC across all presence/absence bbox pairs.

    Each site contributes equal weight regardless of how many pairs it contains.
    Sites with no valid pairs are excluded and the remaining weights renormalise
    to 1.0.  Returns nan if no valid pairs exist.
    """
    # Group pixel indices by region, then split regions into presence/absence per site.
    region_ids = np.array([_region_from_pid(p) for p in pids])
    site_ids = np.array([_site_class(p)[0] for p in pids])

    # Build per-site lists of presence and absence region names.
    sites = np.unique(site_ids)
    site_pairs: dict[str, list[tuple[str, str]]] = {}
    for site in sites:
        site_mask = site_ids == site
        regions_in_site = np.unique(region_ids[site_mask])
        presence_regions = [r for r in regions_in_site if "_presence" in r]
        absence_regions = [r for r in regions_in_site if "_absence" in r]
        pairs = [(p, a) for p in presence_regions for a in absence_regions]
        if pairs:
            site_pairs[site] = pairs

    if not site_pairs:
        return float("nan")

    # Compute AUC for each pair; filter pairs below min_pair_pixels.
    pair_aucs: list[float] = []
    pair_site: list[str] = []
    for site, pairs in site_pairs.items():
        for pres_r, abs_r in pairs:
            pres_mask = finite & (region_ids == pres_r)
            abs_mask = finite & (region_ids == abs_r)
            if pres_mask.sum() < min_pair_pixels or abs_mask.sum() < min_pair_pixels:
                continue
            pair_probs = np.concatenate([probs[pres_mask], probs[abs_mask]])
            pair_labels = np.concatenate([labels[pres_mask], labels[abs_mask]])
            if len(set(pair_labels)) < 2:
                continue
            try:
                auc = roc_auc_score(pair_labels, pair_probs)
            except ValueError:
                continue
            pair_aucs.append(auc)
            pair_site.append(site)

    if not pair_aucs:
        return float("nan")

    # Site-weighted: each active site's pairs share a budget of 1/n_active_sites.
    active_sites = list(dict.fromkeys(pair_site))  # preserves order, dedups
    n_active = len(active_sites)
    site_budget = 1.0 / n_active
    site_pair_counts = {s: pair_site.count(s) for s in active_sites}
    weights = np.array([site_budget / site_pair_counts[s] for s in pair_site])
    # Weights already sum to 1.0 by construction.

    # Weighted CVaR: sort pairs ascending by AUC, accumulate weight until alpha.
    order = np.argsort(pair_aucs)
    sorted_aucs = np.array(pair_aucs)[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    tail_mask = cumulative <= alpha
    # Always include at least the bottom pair.
    if not tail_mask.any():
        tail_mask[0] = True
    return float(np.average(sorted_aucs[tail_mask], weights=sorted_weights[tail_mask]))


def _apply_presence_filter(
    lbl_py: pd.Series,
    mean_vh_dry_py: pd.Series,
    cfg: TAMConfig,
    pid_to_sc: pd.Series,
    noise_removed: dict,
    mean_ndvi_dry_py: pd.Series | None = None,
) -> pd.Series:
    presence_py = lbl_py[lbl_py == 1]
    vh = mean_vh_dry_py.reindex(presence_py.index)
    fails_strict = vh < cfg.presence_min_vh_dry_db
    if mean_ndvi_dry_py is not None:
        ndvi = mean_ndvi_dry_py.reindex(presence_py.index)
        rescued = (
            (vh >= cfg.presence_ndvi_rescue_vh_db)
            & (ndvi >= cfg.presence_ndvi_rescue_min)
        )
        drop_mask = fails_strict & ~rescued
    else:
        drop_mask = fails_strict
    drop_idx  = presence_py.index[drop_mask.fillna(False)]
    drop_pids = drop_idx.get_level_values("point_id")
    sc = pd.Series(drop_pids).map(pid_to_sc).value_counts()
    for key, cnt in sc.items():
        noise_removed[key] = noise_removed.get(key, 0) + cnt
    return lbl_py[~lbl_py.index.isin(drop_idx)]


def train_tam(
    pixel_df: pd.DataFrame,
    labels: pd.Series,
    pixel_coords: pd.DataFrame,
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
        Series indexed by point_id, values in {0.0, 1.0}.
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
        # Identify only the bad S2 rows and drop by index — avoids materialising a
        # full-length boolean array over all 63M rows while the filtered copy is live.
        bad_idx = pixel_df.index[(pixel_df["source"] == "S2") & (pixel_df["scl"] == 6)]
        pixel_df.drop(index=bad_idx, inplace=True)
        pixel_df.reset_index(drop=True, inplace=True)
        del bad_idx
        gc.collect()
        logger.info("SCL=6 exclusion: removed %d observations", n_before - len(pixel_df))
    _log_rss("after SCL exclusion")

    # --- Global features -----------------------------------------------------
    global_feat_df = None
    if cfg.use_band_summaries:
        logger.info("Computing band summaries (%d rows) ...", len(pixel_df))
        if "NDVI" not in pixel_df.columns or "NDWI" not in pixel_df.columns:
            pixel_df = add_spectral_indices(pixel_df)
        global_feat_df = _compute_band_summaries(pixel_df, V9_FEATURE_COLS)
        logger.info("Band summaries computed: %d pixels, %d features", len(global_feat_df), global_feat_df.shape[1])
    elif cfg.n_global_features > 0:
        global_feat_df = _load_or_compute_global_features(pixel_df, out_dir, GLOBAL_FEATURE_NAMES)
    _log_rss("after band summaries")

    # Pre-extract S1 slim for presence filter, then drop S1 rows to free RAM.
    # The presence filter only needs the slim slices, not the full S1 rows.
    _presence_filter_s1_slim: pd.DataFrame | None = None
    _presence_filter_s2_slim: pd.DataFrame | None = None
    if (cfg.presence_min_vh_dry_db > -99
            and "source" in pixel_df.columns
            and "vh" in pixel_df.columns):
        _doy_col = "doy" if "doy" in pixel_df.columns else None
        _date_col = "doy" if _doy_col else "date"
        _s1_cols = ["point_id", "year", "vh", _date_col]
        _presence_filter_s1_slim = pixel_df.loc[pixel_df["source"] == "S1", _s1_cols].copy()
        if "NDVI" in pixel_df.columns:
            _has_scl = "scl" in pixel_df.columns
            _s2_cols = ["point_id", "year", _date_col, "NDVI"] + (["scl"] if _has_scl else [])
            _presence_filter_s2_slim = pixel_df.loc[pixel_df["source"] == "S2", _s2_cols].copy()

    # Drop S1 rows — no longer needed after band summaries and slim extraction.
    if "source" in pixel_df.columns:
        pixel_df = pixel_df[pixel_df["source"] == "S2"].reset_index(drop=True)
        gc.collect()
    _log_rss("after S1 drop")

    # px_counts: unique pixel count per (site, class) — area indicator, computed once before broadcast.
    px_counts: dict[tuple[str, str], int] = pd.Series(
        list(map(_site_class, labels.index)), index=labels.index
    ).value_counts().to_dict()

    # raw_counts populated in pixel-year units after the broadcast step below.
    raw_counts: dict[tuple[str, str], int] = {}

    noise_removed: dict[tuple[str, str], int] = {}

    stride_removed: dict[tuple[str, str], int] = {}

    # --- Spatial stride — thin training pixels to reduce within-site redundancy.
    # Runs at pixel granularity before broadcasting to pixel-years.
    if cfg.spatial_stride > 1:
        coords = pixel_coords.set_index("point_id")
        candidate = coords.loc[coords.index.intersection(train_labels.index)]

        if cfg.stride_exclude_sites:
            excluded = candidate.index[candidate.index.map(
                lambda pid: _site_class(pid)[0] in cfg.stride_exclude_sites
            )]
            to_stride = candidate[~candidate.index.isin(excluded)]
        else:
            excluded = candidate.index[[False] * len(candidate)]
            to_stride = candidate

        strided_ids = set(
            to_stride.sort_values(["lat", "lon"]).iloc[::cfg.spatial_stride].index
        ) | set(excluded)

        removed_by_stride = set(train_labels.index) - strided_ids
        for pid in removed_by_stride:
            key = _site_class(pid)
            stride_removed[key] = stride_removed.get(key, 0) + 1

        train_labels = train_labels[train_labels.index.isin(strided_ids)]

    # --- Broadcast pixel labels → (point_id, year) labels ------------------
    # Each surviving pixel contributes one entry per year it has observations in.
    # Compute the (point_id, year) universe from labeled pixels only — cheaper than
    # drop_duplicates on the full pixel_df which may have tens of millions of rows.
    labeled_pids = set(labels.index)
    pixel_years = (
        pixel_df.loc[pixel_df["point_id"].isin(labeled_pids), ["point_id", "year"]]
        .drop_duplicates()
    )

    def _broadcast_to_pixel_years(lbl: pd.Series) -> pd.Series:
        """Expand pixel-level labels to a MultiIndex (point_id, year) Series."""
        py = pixel_years[pixel_years["point_id"].isin(lbl.index)]
        py = py.merge(lbl.rename("label"), left_on="point_id", right_index=True)
        return py.set_index(["point_id", "year"])["label"]

    train_py_labels = _broadcast_to_pixel_years(train_labels)
    val_py_labels   = _broadcast_to_pixel_years(val_labels)

    # Pixel-year counts before heuristic filter — used as "Raw py" in summary table.
    all_py = pd.concat([train_py_labels, val_py_labels])
    all_pids = all_py.index.get_level_values("point_id")
    # Map _site_class once per unique point_id then broadcast — avoids regex per row.
    unique_pids = pd.Index(np.unique(all_pids))
    pid_to_sc = pd.Series(list(map(_site_class, unique_pids)), index=unique_pids)
    raw_counts = pd.Series(all_pids).map(pid_to_sc).value_counts().to_dict()

    # --- Presence filter: drop presence pixel-years with low dry-season VH unless rescued by NDVI.
    # Drop logic per (point_id, year):
    #   drop if mean_vh_dry < presence_min_vh_dry_db
    #           AND NOT (mean_vh_dry >= presence_ndvi_rescue_vh_db AND mean_ndvi_dry >= presence_ndvi_rescue_min)
    if _presence_filter_s1_slim is not None:
        s1_slim = _presence_filter_s1_slim
        s2_slim = _presence_filter_s2_slim
        _doy_col = "doy" if "doy" in s1_slim.columns else None
        _date_col = "doy" if _doy_col else "date"
        _has_scl = s2_slim is not None and "scl" in s2_slim.columns

        if not s1_slim.empty:
            doy_vals = (s1_slim[_date_col].values if _doy_col
                        else pd.to_datetime(s1_slim["date"]).dt.day_of_year.values)
            vh_lin = s1_slim["vh"].values.astype(np.float32)
            vh_db  = lin_to_db(vh_lin)
            dry_mask = (doy_vals >= _DRY_DOY_MIN) & (doy_vals <= _DRY_DOY_MAX) & np.isfinite(vh_db)

            dry_s1 = pd.DataFrame({
                "point_id": s1_slim["point_id"].values[dry_mask],
                "year":     s1_slim["year"].values[dry_mask],
                "_vh_db":   vh_db[dry_mask].astype(np.float32),
            })
            mean_vh_dry_py = dry_s1.groupby(["point_id", "year"])["_vh_db"].mean()

            # S2 NDVI per (point_id, year) — clear observations only
            mean_ndvi_dry_py: pd.Series | None = None
            if s2_slim is not None and not s2_slim.empty:
                s2_doy = (s2_slim[_date_col].values if _doy_col
                          else pd.to_datetime(s2_slim["date"]).dt.day_of_year.values)
                s2_dry_mask = (s2_doy >= _DRY_DOY_MIN) & (s2_doy <= _DRY_DOY_MAX)
                if _has_scl:
                    s2_dry_mask &= s2_slim["scl"].isin([4.0, 5.0]).values
                dry_s2 = s2_slim[s2_dry_mask & s2_slim["NDVI"].notna()]
                if not dry_s2.empty:
                    mean_ndvi_dry_py = dry_s2.groupby(["point_id", "year"])["NDVI"].mean()

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
    # All counts (Raw, Noise, Total) are in pixel-year units.
    # Stride is pixel-level (applied before broadcast) and shown separately.
    val_site_set = set(cfg.val_sites) if cfg.val_sites else set()
    if cfg.val_region_ids:
        val_site_set |= {_site_class(rid)[0] for rid in cfg.val_region_ids}

    # final_counts: surviving pixel-years per (site, class) after all filters.
    final_all_py = pd.concat([train_py_labels, val_py_labels])
    final_pids = final_all_py.index.get_level_values("point_id")
    final_counts = pd.Series(final_pids).map(pid_to_sc).value_counts().to_dict()

    all_keys = sorted(raw_counts.keys(), key=lambda k: (k[0] in val_site_set, k[0], k[1]))

    col_w = max((len(f"{s} {c}") for s, c in all_keys), default=20) + 2

    def neg(n: int, w: int) -> str:
        return f"-{n:>{w - 1},}" if n else f"{'':>{w}}"

    def row(label: str, px: int, raw: int, noise: int, stride: int, final: int) -> str:
        return f"{label:>{col_w}}  {px:>8,}  {raw:>8,}  {neg(noise, 7)}  {neg(stride, 9)}  {final:>8,}"

    header = f"{'':>{col_w}}  {'Raw px':>8}  {'Raw py':>8}  {'Noise py':>7}  {'Stride px':>9}  {'Total py':>8}"
    sep    = "-" * len(header)
    thin   = " " * len(header)

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
    lines.append(thin)
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
    lines.append(thin)
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

    total_px    = train_px    + val_px
    total_stride_n = train_stride + val_stride
    total_raw   = train_raw   + val_raw
    total_noise = train_noise + val_noise
    total_final = train_final + val_final
    lines.append(row("TOTAL", total_px, total_raw, total_noise, total_stride_n, total_final))
    lines.append(sep)

    logger.info("Pixel summary (stride=%d):\n%s", cfg.spatial_stride, "\n".join(lines))

    # Slice global features to exactly the columns the model head expects.
    # Band-summary mode supplies all columns; named-global mode slices to n_global_features.
    if global_feat_df is not None and not cfg.use_band_summaries:
        global_feat_df = global_feat_df.iloc[:, :cfg.n_global_features]

    # --- Datasets -----------------------------------------------------------
    # Drop source column if present — pixel_df is already S2-only at this point,
    # so the column only causes TAMDataset to do a redundant boolean scan + copy.
    if "source" in pixel_df.columns:
        pixel_df = pixel_df.drop(columns=["source"])

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
        doy_jitter=0,  # no augmentation at eval time
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
    # Model is small and GPU-bound; DataLoader throughput is the bottleneck.
    # Give most cores to workers, leave 2 for PyTorch matmuls and the main thread.
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

    # --- Class imbalance weight — computed from pixel-year labels after heuristic filter ---
    n_pos = float((train_py_labels == 1).sum())
    n_neg = float((train_py_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device) if n_pos > 0 and n_neg > 0 else None

    # --- Model + optimiser --------------------------------------------------
    # Band-summary mode: override n_global_features to match actual summary width.
    if cfg.use_band_summaries and global_feat_df is not None:
        cfg.n_global_features = global_feat_df.shape[1]
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = cfg.use_s1          # persisted to tam_config.json for score pipeline
    model._pixel_zscore = cfg.pixel_zscore
    model._max_seq_len = cfg.max_seq_len
    model._feature_cols = _feature_cols_override  # None = default ALL_FEATURE_COLS
    model.to(device)
    logger.info(
        "Model: d_model=%d n_heads=%d n_layers=%d d_ff=%d  params=%d",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff,
        sum(p.numel() for p in model.parameters()),
    )

    if cfg.doy_density_norm:
        # Compute DOY observation counts from training pixels and set inverse-freq weights
        train_pids = set(train_py_labels.index.get_level_values("point_id"))
        train_doys = pixel_df.loc[pixel_df["point_id"].isin(train_pids), "doy"].values
        doy_counts = np.bincount(train_doys.astype(int), minlength=366)
        model.set_doy_frequencies(doy_counts)
        logger.info("DOY density normalisation enabled — inverse-freq weights computed from %d training observations", len(train_doys))

    # pixel_df no longer needed — free before DataLoader workers fork.
    del pixel_df
    gc.collect()
    _log_rss("after pixel_df free")

    # Save config + band stats immediately so inference can run even if training is interrupted
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
        # Head-only warmup: use 3× lr so the head orients correctly before unfreeze
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
            # Rebuild optimizer at normal lr for full-model phase
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

        # Macro-site AUC: unweighted mean of per-site AUCs (summary only).
        site_aucs = []
        val_sites_arr = np.array([_site_class(pid)[0] for pid in val_pids])
        for site in np.unique(val_sites_arr):
            mask = finite & (val_sites_arr == site)
            if mask.sum() > 0 and len(set(val_labels_arr[mask])) > 1:
                site_aucs.append(roc_auc_score(val_labels_arr[mask], val_probs_arr[mask]))
        val_auc_macro = float(np.mean(site_aucs)) if site_aucs else float("nan")

        # Primary checkpoint metric: site-weighted CVaR AUC across bbox pairs.
        val_cvar = _cvar_auc(
            val_probs_arr, val_labels_arr, val_pids, finite,
            alpha=cfg.cvar_alpha, min_pair_pixels=cfg.min_pair_pixels,
        )

        logger.info(
            "epoch %3d/%d  loss=%.4f  train_auc=%.3f  val_cvar%.0f=%.3f%s",
            epoch, cfg.n_epochs,
            epoch_loss / max(len(train_loader), 1),
            train_auc,
            cfg.cvar_alpha * 100, val_cvar,
            "  *" if (not np.isnan(val_cvar) and val_cvar >= best_val_auc) else "",
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

    # Save final band stats + config (may differ from mid-training save if interrupted)
    np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
    if train_ds.global_feat_mean is not None and len(train_ds.global_feat_mean) > 0:
        np.savez(out_dir / "tam_global_feat_stats.npz",
                 mean=train_ds.global_feat_mean, std=train_ds.global_feat_std)
    cfg_dict = model.config()
    cfg_dict["best_val_auc"] = round(best_val_auc, 6)
    with open(out_dir / "tam_config.json", "w") as fh:
        json.dump(cfg_dict, fh, indent=2)

    # If val AUC never improved (e.g. tiny smoke datasets), save current weights so
    # the checkpoint always exists for the load below.
    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)

    # Load best weights before returning
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

    # Infer use_n_obs and n_global_features from saved head weight shape
    head_in = state["head.weight"].shape[1]
    extra = head_in - cfg.d_model          # total scalars appended after pooling
    cfg.use_n_obs = extra > 0              # n_obs always first if present
    cfg.n_global_features = extra - (1 if cfg.use_n_obs else 0)

    model = TAMClassifier.from_config(cfg)
    # Allow loading checkpoints saved before doy_inv_freq buffer was added,
    # or with a mismatched size (old checkpoints used minlength=367 → shape (367,);
    # current model registers (366,)). Truncate or pad to match.
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
