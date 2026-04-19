"""tam/train.py — Training loop for TAMClassifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tam.config import TAMConfig
from tam.dataset import TAMDataset, collate_fn
from tam.model import TAMClassifier

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    # --- Split labels spatially -------------------------------------------
    train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)
    logger.info(
        "Spatial split — train: %d presence / %d absence | val: %d presence / %d absence",
        (train_labels == 1).sum(), (train_labels == 0).sum(),
        (val_labels   == 1).sum(), (val_labels   == 0).sum(),
    )

    # --- Datasets -----------------------------------------------------------
    train_ds = TAMDataset(
        pixel_df, train_labels,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=cfg.doy_jitter,
    )
    band_mean, band_std = train_ds.band_stats
    val_ds = TAMDataset(
        pixel_df, val_labels,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=0,  # no augmentation at eval time
    )
    logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))

    n_workers = 4
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
        for batch in train_loader:
            bands  = batch["bands"].to(device)
            doy    = batch["doy"].to(device)
            mask   = batch["mask"].to(device)
            label  = batch["label"].to(device)
            weight = batch["weight"].to(device)

            _, logit = model(bands, doy, mask)
            loss = (criterion(logit, label) * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # --- Validation -------------------------------------------------------
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                prob, _ = model(
                    batch["bands"].to(device),
                    batch["doy"].to(device),
                    batch["mask"].to(device),
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
            "epoch %3d/%d  loss=%.4f  val_auc=%.3f%s",
            epoch, cfg.n_epochs,
            epoch_loss / max(len(train_loader), 1),
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

    model = TAMClassifier.from_config(cfg)
    model.load_state_dict(torch.load(out_dir / "tam_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    stats = np.load(out_dir / "tam_band_stats.npz")
    return model, stats["mean"], stats["std"]
