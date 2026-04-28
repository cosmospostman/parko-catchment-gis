"""tam/core/train.py — Training loop for TAMClassifier."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tam.core.config import TAMConfig
from tam.core.dataset import TAMDataset, collate_fn
from tam.core.model import TAMClassifier

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


def site_holdout_split(
    labels: pd.Series,
    val_sites: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    """Hold out all pixels whose region ID starts with any of val_sites.

    Point IDs have the form <region_id>_<row>_<col>, and region IDs have the
    form <site>_presence_N or <site>_absence_N, so the site is the prefix
    before the first '_presence' or '_absence'.
    """
    import re
    def point_site(pid: str) -> str:
        m = re.match(r"^(.+?)_(presence|absence)", pid)
        return m.group(1) if m else pid

    in_val = labels.index.map(lambda pid: point_site(pid) in val_sites)
    return labels[~in_val], labels[in_val]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _global_features_cache_key(pixel_df: pd.DataFrame) -> str:
    """Stable hash of the inputs to compute_global_features (point_id, date, B08, B04)."""
    cols = [c for c in ("point_id", "date", "B08", "B04") if c in pixel_df.columns]
    key_df = pixel_df[cols].sort_values(cols).reset_index(drop=True)
    return hashlib.md5(key_df.to_csv(index=False).encode()).hexdigest()


def _load_or_compute_global_features(
    pixel_df: pd.DataFrame,
    out_dir: Path,
    feature_names: list[str],
) -> pd.DataFrame:
    from tam.core.global_features import compute_global_features
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


def train_tam(
    pixel_df: pd.DataFrame,
    labels: pd.Series,
    pixel_coords: pd.DataFrame,
    out_dir: Path,
    cfg: TAMConfig | None = None,
    device: str | None = None,
) -> TAMClassifier:
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
    The best-val-AUC TAMClassifier (weights loaded from checkpoint).
    """
    if cfg is None:
        cfg = TAMConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    import re as _re

    def _site_class(pid: str) -> tuple[str, str]:
        m = _re.match(r"^(.+?)_(presence|absence)", pid)
        return (m.group(1), m.group(2)) if m else (pid, "unknown")

    # --- Split labels ---------------------------------------------------------
    if cfg.val_sites:
        train_labels, val_labels = site_holdout_split(labels, cfg.val_sites)
    else:
        train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)

    # --- SCL=6 exclusion — strip dark-area misclassifications from all pixels ---
    if "scl" in pixel_df.columns:
        n_before = len(pixel_df)
        pixel_df = pixel_df[pixel_df["scl"] != 6]
        logger.info("SCL=6 exclusion: removed %d observations", n_before - len(pixel_df))

    # --- Global features -----------------------------------------------------
    global_feat_df = None
    if cfg.n_global_features > 0:
        from tam.core.global_features import GLOBAL_FEATURE_NAMES
        global_feat_df = _load_or_compute_global_features(pixel_df, out_dir, GLOBAL_FEATURE_NAMES)

    # Track per-(site, class) pixel counts at each stage for the summary table.
    # raw_counts: counts from `labels` before any filter; keyed by (site, class).
    raw_counts: dict[tuple[str, str], int] = {}
    for pid in labels.index:
        key = _site_class(pid)
        raw_counts[key] = raw_counts.get(key, 0) + 1

    noise_removed: dict[tuple[str, str], int] = {}

    # --- Presence noise filter — remove obvious non-Parkinsonia presence pixels
    if global_feat_df is not None:
        def _apply_noise_filter(lbl: pd.Series) -> pd.Series:
            presence_pids = set(lbl[lbl == 1].index)
            gf = global_feat_df.reindex(list(presence_pids))
            noise_mask = (
                (gf["dry_ndvi"] < cfg.presence_min_dry_ndvi) |
                (gf["rec_p"]    < cfg.presence_min_rec_p) |
                (gf["nir_cv"]   > cfg.presence_grass_nir_cv)
            )
            noisy_pids = presence_pids & set(gf[noise_mask].index)
            for pid in noisy_pids:
                key = _site_class(pid)
                noise_removed[key] = noise_removed.get(key, 0) + 1
            return lbl[~lbl.index.isin(noisy_pids)]

        train_labels = _apply_noise_filter(train_labels)
        val_labels   = _apply_noise_filter(val_labels)

    stride_removed: dict[tuple[str, str], int] = {}

    # --- Spatial stride — thin training pixels to reduce within-site redundancy
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

    # --- Pixel summary table -------------------------------------------------
    val_site_set = set(cfg.val_sites) if cfg.val_sites else set()

    final_counts: dict[tuple[str, str], int] = {}
    for pid in train_labels.index:
        key = _site_class(pid)
        final_counts[key] = final_counts.get(key, 0) + 1
    for pid in val_labels.index:
        key = _site_class(pid)
        final_counts[key] = final_counts.get(key, 0) + 1

    all_keys = sorted(raw_counts.keys(), key=lambda k: (k[0] in val_site_set, k[0], k[1]))

    col_w = max((len(f"{s} {c}") for s, c in all_keys), default=20) + 2

    def neg(n: int, w: int) -> str:
        return f"-{n:>{w - 1},}" if n else f"{'':>{w}}"

    def row(label: str, raw: int, noise: int, stride: int, final: int) -> str:
        return f"{label:>{col_w}}  {raw:>8,}  {neg(noise, 7)}  {neg(stride, 7)}  {final:>8,}"

    header = f"{'':>{col_w}}  {'Raw px':>8}  {'Noise':>7}  {'Stride':>7}  {'Total':>8}"
    sep    = "-" * len(header)
    thin   = " " * len(header)

    train_sites = sorted({s for s, _ in all_keys if s not in val_site_set})
    val_sites_sorted = sorted({s for s, _ in all_keys if s in val_site_set})

    def site_rows(sites: list[str], prefix: str = "") -> tuple[list[str], int, int, int, int]:
        block_lines = []
        b_raw = b_noise = b_stride = b_final = 0
        for site in sites:
            for cls in ("presence", "absence"):
                key = (site, cls)
                if key not in raw_counts:
                    continue
                raw    = raw_counts.get(key, 0)
                noise  = noise_removed.get(key, 0)
                stride = stride_removed.get(key, 0)
                final  = final_counts.get(key, 0)
                block_lines.append(row(f"{prefix}{site} {cls}", raw, noise, stride, final))
                b_raw += raw; b_noise += noise; b_stride += stride; b_final += final
        return block_lines, b_raw, b_noise, b_stride, b_final

    lines = [sep, header, sep]

    train_rows, train_raw, train_noise, train_stride, train_final = site_rows(train_sites)
    lines.extend(train_rows)
    lines.append(thin)
    lines.append(row("PRESENCE", *[
        sum(raw_counts.get((s, "presence"), 0) for s in train_sites),
        sum(noise_removed.get((s, "presence"), 0) for s in train_sites),
        sum(stride_removed.get((s, "presence"), 0) for s in train_sites),
        sum(final_counts.get((s, "presence"), 0) for s in train_sites),
    ]))
    lines.append(row("ABSENCE", *[
        sum(raw_counts.get((s, "absence"), 0) for s in train_sites),
        sum(noise_removed.get((s, "absence"), 0) for s in train_sites),
        sum(stride_removed.get((s, "absence"), 0) for s in train_sites),
        sum(final_counts.get((s, "absence"), 0) for s in train_sites),
    ]))
    lines.append(row("TRAIN TOTAL", train_raw, train_noise, train_stride, train_final))
    lines.append(sep)

    val_rows, val_raw, val_noise, val_stride, val_final = site_rows(val_sites_sorted, prefix="HOLDOUT: ")
    lines.extend(val_rows)
    lines.append(thin)
    lines.append(row("PRESENCE", *[
        sum(raw_counts.get((s, "presence"), 0) for s in val_sites_sorted),
        sum(noise_removed.get((s, "presence"), 0) for s in val_sites_sorted),
        0,
        sum(final_counts.get((s, "presence"), 0) for s in val_sites_sorted),
    ]))
    lines.append(row("ABSENCE", *[
        sum(raw_counts.get((s, "absence"), 0) for s in val_sites_sorted),
        sum(noise_removed.get((s, "absence"), 0) for s in val_sites_sorted),
        0,
        sum(final_counts.get((s, "absence"), 0) for s in val_sites_sorted),
    ]))
    lines.append(row("VAL TOTAL", val_raw, val_noise, val_stride, val_final))
    lines.append(sep)

    total_stride_n = train_stride + val_stride
    total_raw   = train_raw   + val_raw
    total_noise = train_noise + val_noise
    total_final = train_final + val_final
    lines.append(row("TOTAL", total_raw, total_noise, total_stride_n, total_final))
    lines.append(sep)

    logger.info("Pixel summary (stride=%d):\n%s", cfg.spatial_stride, "\n".join(lines))

    # --- Datasets -----------------------------------------------------------
    train_ds = TAMDataset(
        pixel_df, train_labels,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=cfg.doy_jitter,
        band_noise_std=cfg.band_noise_std,
        obs_dropout_min=cfg.obs_dropout_min,
        global_features_df=global_feat_df,
    )
    band_mean, band_std = train_ds.band_stats
    val_ds = TAMDataset(
        pixel_df, val_labels,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=0,  # no augmentation at eval time
        global_features_df=global_feat_df,
        global_feat_mean=train_ds.global_feat_mean,
        global_feat_std=train_ds.global_feat_std,
    )
    logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))

    n_cpu = os.cpu_count() or 4
    # Reserve cores for DataLoader workers; give the rest to PyTorch matmuls.
    n_workers = max(2, n_cpu // 4)
    torch.set_num_threads(max(1, n_cpu - n_workers))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers, persistent_workers=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers, persistent_workers=True, pin_memory=True,
    )
    logger.info("DataLoader workers: %d  PyTorch threads: %d", n_workers, torch.get_num_threads())

    # --- Class imbalance weight -------------------------------------------
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device) if n_pos > 0 and n_neg > 0 else None

    # --- Model + optimiser --------------------------------------------------
    model = TAMClassifier.from_config(cfg)
    model.to(device)
    logger.info(
        "Model: d_model=%d n_heads=%d n_layers=%d d_ff=%d  params=%d",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff,
        sum(p.numel() for p in model.parameters()),
    )

    if cfg.doy_density_norm:
        # Compute DOY observation counts from training pixels and set inverse-freq weights
        train_pids = set(train_labels.index)
        train_doys = pixel_df.loc[pixel_df["point_id"].isin(train_pids), "doy"].values
        doy_counts = np.bincount(train_doys.astype(int), minlength=366)
        model.set_doy_frequencies(doy_counts)
        logger.info("DOY density normalisation enabled — inverse-freq weights computed from %d training observations", len(train_doys))

    # Save config + band stats immediately so inference can run even if training is interrupted
    np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
    with open(out_dir / "tam_config.json", "w") as fh:
        json.dump(model.config(), fh, indent=2)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    # --- Training loop ------------------------------------------------------
    best_val_auc = 0.0
    epochs_without_improvement = 0
    checkpoint_path = out_dir / "tam_model.pt"

    for epoch in range(1, cfg.n_epochs + 1):
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
        val_probs, val_labels_list = [], []
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

        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        finite = np.isfinite(val_probs_arr)
        n_nan = (~finite).sum()
        if n_nan:
            logger.warning("epoch %d: %d NaN predictions in validation set", epoch, n_nan)
        if finite.any() and len(set(val_labels_arr[finite])) > 1:
            val_auc = roc_auc_score(val_labels_arr[finite], val_probs_arr[finite])
        else:
            val_auc = float("nan")

        logger.info(
            "epoch %3d/%d  loss=%.4f  train_auc=%.3f  val_auc=%.3f%s",
            epoch, cfg.n_epochs,
            epoch_loss / max(len(train_loader), 1),
            train_auc,
            val_auc,
            "  *" if (not np.isnan(val_auc) and val_auc >= best_val_auc) else "",
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc + cfg.min_delta:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, cfg.patience)
                break

    logger.info("Best val AUC: %.3f — checkpoint: %s", best_val_auc, checkpoint_path)

    # Save final band stats + config (may differ from mid-training save if interrupted)
    np.savez(out_dir / "tam_band_stats.npz", mean=band_mean, std=band_std)
    with open(out_dir / "tam_config.json", "w") as fh:
        json.dump(model.config(), fh, indent=2)

    # If val AUC never improved (e.g. tiny smoke datasets), save current weights so
    # the checkpoint always exists for the load below.
    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)

    # Load best weights before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_tam(out_dir: Path, device: str | None = None) -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Load a saved TAMClassifier checkpoint.

    Returns
    -------
    (model, band_mean, band_std)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(out_dir / "tam_config.json") as fh:
        cfg = TAMConfig.from_dict(json.load(fh))

    state = torch.load(out_dir / "tam_model.pt", map_location=device, weights_only=True)

    # Infer use_n_obs and n_global_features from saved head weight shape
    head_in = state["head.weight"].shape[1]
    extra = head_in - cfg.d_model          # total scalars appended after pooling
    cfg.use_n_obs = extra > 0              # n_obs always first if present
    cfg.n_global_features = extra - (1 if cfg.use_n_obs else 0)

    model = TAMClassifier.from_config(cfg)
    # Allow loading checkpoints saved before doy_inv_freq buffer was added
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    stats = np.load(out_dir / "tam_band_stats.npz")
    return model, stats["mean"], stats["std"]
