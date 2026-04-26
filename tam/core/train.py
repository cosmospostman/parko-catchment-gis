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

    # --- Split labels ---------------------------------------------------------
    if cfg.val_sites:
        train_labels, val_labels = site_holdout_split(labels, cfg.val_sites)
        logger.info(
            "Site holdout split %s — train: %d presence / %d absence | val: %d presence / %d absence",
            cfg.val_sites,
            (train_labels == 1).sum(), (train_labels == 0).sum(),
            (val_labels   == 1).sum(), (val_labels   == 0).sum(),
        )
    else:
        train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)
        logger.info(
            "Spatial split — train: %d presence / %d absence | val: %d presence / %d absence",
            (train_labels == 1).sum(), (train_labels == 0).sum(),
            (val_labels   == 1).sum(), (val_labels   == 0).sum(),
        )

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

    # --- Presence noise filter — remove obvious non-Parkinsonia presence pixels
    if global_feat_df is not None:
        def _apply_noise_filter(labels: pd.Series, split: str) -> pd.Series:
            presence_pids = set(labels[labels == 1].index)
            gf = global_feat_df.reindex(list(presence_pids))
            water_mask = gf["dry_ndvi"] < cfg.presence_min_dry_ndvi
            amp_mask   = gf["rec_p"]    < cfg.presence_min_rec_p
            grass_mask = gf["nir_cv"]   > cfg.presence_grass_nir_cv
            noise_mask = water_mask | amp_mask | grass_mask
            noisy_pids = presence_pids & set(gf[noise_mask].index)
            remaining = len(presence_pids) - len(noisy_pids)
            logger.info(
                "Presence noise filter (%s): removed %d pixels, %d remaining "
                "(dry_ndvi<%.2f: %d  rec_p<%.2f: %d  nir_cv>%.2f: %d)",
                split, len(noisy_pids), remaining,
                cfg.presence_min_dry_ndvi, water_mask.sum(),
                cfg.presence_min_rec_p,    amp_mask.sum(),
                cfg.presence_grass_nir_cv, grass_mask.sum(),
            )
            return labels[~labels.index.isin(noisy_pids)]

        train_labels = _apply_noise_filter(train_labels, "train")
        val_labels   = _apply_noise_filter(val_labels,   "val")

    # --- Spatial stride — thin training pixels to reduce within-site redundancy
    if cfg.spatial_stride > 1:
        import re
        coords = pixel_coords.set_index("point_id")
        candidate = coords.loc[coords.index.intersection(train_labels.index)]

        if cfg.stride_exclude_sites:
            def _site(pid: str) -> str:
                m = re.match(r"^(.+?)_(presence|absence)", pid)
                return m.group(1) if m else pid
            excluded = candidate.index[candidate.index.map(
                lambda pid: _site(pid) in cfg.stride_exclude_sites
            )]
            to_stride = candidate[~candidate.index.isin(excluded)]
        else:
            excluded = candidate.index[[False] * len(candidate)]
            to_stride = candidate

        strided_ids = set(
            to_stride.sort_values(["lat", "lon"]).iloc[::cfg.spatial_stride].index
        ) | set(excluded)

        train_labels = train_labels[train_labels.index.isin(strided_ids)]
        logger.info(
            "Spatial stride %d — train pixels reduced to: %d presence / %d absence"
            " (excluded from stride: %s)",
            cfg.spatial_stride,
            (train_labels == 1).sum(), (train_labels == 0).sum(),
            cfg.stride_exclude_sites or "none",
        )

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
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    stats = np.load(out_dir / "tam_band_stats.npz")
    return model, stats["mean"], stats["std"]
