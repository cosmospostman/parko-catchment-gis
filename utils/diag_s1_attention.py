"""
Diagnostic: how much attention does the V10 model place on S1 vs S2 tokens?

Samples a random subset of training presence and absence pixels, runs the
model's get_attention_weights() on each, and summarises the fraction of
attention weight falling on S1 tokens (last-layer, mean over heads).

Also runs an ablation: score each pixel with S1 bands zeroed out and
measures the score delta to estimate how much S1 contributes per pixel.

Usage:
    python utils/diag_s1_attention.py --model outputs/models/tam-v10-0.875
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tam.core.model import TAMClassifier
from tam.core.dataset import (
    TAMDataset,
    collate_fn,
    V9_FEATURE_COLS,
    S1_FEATURE_COLS,
    MAX_SEQ_LEN,
    lin_to_db,
    V10_FEATURE_COLS,
    V10_S1_FEATURE_COLS,
)
from tam.core.train import load_tam


def load_model(model_dir: Path, device: str) -> tuple:
    model, band_mean, band_std, gf_mean, gf_std = load_tam(model_dir, device)
    model.eval()
    return model, band_mean, band_std, gf_mean, gf_std


def build_pixel_windows(
    df: pl.DataFrame,
    point_ids: list[str],
    feature_cols: list[str],
    s1_feature_cols: list[str],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    max_seq_len: int,
    cfg_dict: dict,
) -> list[dict]:
    """Build normalised TAMDataset windows for a list of point_ids."""
    import json

    pixel_df = df.filter(pl.col("point_id").is_in(point_ids))
    if len(pixel_df) == 0:
        return []

    # Build minimal labels dict (label doesn't matter for this diagnostic)
    labels = {pid: 0.0 for pid in point_ids}
    coords = (
        pixel_df.select(["point_id", "lon", "lat"])
        .unique("point_id")
    )

    from tam.core.config import TAMConfig
    cfg = TAMConfig.from_dict(cfg_dict)

    dataset = TAMDataset(
        pixel_df=pixel_df,
        labels=labels,
        band_mean=band_mean,
        band_std=band_std,
        max_seq_len=max_seq_len,
        cfg=cfg,
        feature_cols_override=feature_cols,
        s1_feature_cols_override=s1_feature_cols,
        training=False,
    )
    return dataset


def attention_s1_fraction(
    model: TAMClassifier,
    batch: dict,
    device: str,
) -> np.ndarray:
    """Return per-sample fraction of last-layer mean attention on S1 tokens."""
    bands = batch["bands"].to(device)
    doy   = batch["doy"].to(device)
    mask  = batch["mask"].to(device)
    is_s1 = batch.get("is_s1")
    if is_s1 is not None:
        is_s1 = is_s1.to(device)

    B, T = doy.shape
    fractions = []
    with torch.no_grad():
        # Run through all encoder layers manually to get last-layer attn
        from tam.core.model import _doy_encoding
        x = model.band_proj(bands) + _doy_encoding(doy, model.d_model)

        all_masked = mask.all(dim=1, keepdim=True)
        safe_mask = mask & ~all_masked

        for i, layer in enumerate(model.encoder.layers):
            attn_out, w = layer.self_attn(
                x, x, x,
                key_padding_mask=safe_mask,
                need_weights=True,
                average_attn_weights=True,  # mean over heads → (B, T, T)
            )
            # Only use last layer
            if i == len(model.encoder.layers) - 1:
                # w: (B, T, T) — attn[b, q, k]: query q attends to key k
                # Sum over queries to get total attention received per token
                attn_received = w.sum(dim=1)  # (B, T)
                valid = (~mask.to(device)).float()  # (B, T)

                if is_s1 is not None:
                    s1_mask = is_s1.float() * valid  # S1 non-pad tokens
                    s2_mask = (~is_s1).float() * valid
                    s1_attn = (attn_received * s1_mask).sum(dim=1)  # (B,)
                    total_attn = (attn_received * valid).sum(dim=1)  # (B,)
                    frac = (s1_attn / total_attn.clamp(min=1e-6)).cpu().numpy()
                else:
                    frac = np.zeros(B)
                fractions = frac

            x2 = layer.norm1(x + layer.dropout1(attn_out))
            x2 = layer.norm2(x2 + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x2)))
            )))
            x = x2

    return fractions


def score_with_ablation(
    model: TAMClassifier,
    batch: dict,
    device: str,
    n_s2_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score_full, score_s1_zeroed)."""
    bands = batch["bands"].to(device)
    doy   = batch["doy"].to(device)
    mask  = batch["mask"].to(device)
    is_s1 = batch.get("is_s1")
    n_obs = batch["n_obs"].to(device)

    if is_s1 is not None:
        is_s1_dev = is_s1.to(device)
    else:
        is_s1_dev = None

    with torch.no_grad():
        prob_full, _ = model(bands, doy, mask, n_obs, is_s1=is_s1_dev)

        # Zero out S1 band positions (last n_s1_bands columns)
        bands_abl = bands.clone()
        n_s1_bands = bands.shape[-1] - n_s2_bands
        if n_s1_bands > 0 and is_s1_dev is not None:
            # Zero S1 band slots in S1 rows
            s1_rows = is_s1_dev.unsqueeze(-1).expand_as(bands_abl)
            bands_abl[s1_rows & (torch.arange(bands.shape[-1], device=device) >= n_s2_bands).unsqueeze(0).unsqueeze(0).expand_as(bands_abl)] = 0.0

        prob_abl, _ = model(bands_abl, doy, mask, n_obs, is_s1=is_s1_dev)

    return prob_full.cpu().numpy(), prob_abl.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "outputs/models/tam-v10-0.875",
                        help="Path to model directory")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Number of pixels to sample per class")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model} on {device} ...")
    model, band_mean, band_std, gf_mean, gf_std = load_tam(args.model, device)
    model.eval()

    import json
    with open(args.model / "tam_config.json") as f:
        cfg_dict = json.load(f)

    feature_cols   = cfg_dict.get("feature_cols") or V10_FEATURE_COLS
    s1_feature_cols = cfg_dict.get("s1_feature_cols") or V10_S1_FEATURE_COLS
    use_s1 = cfg_dict.get("use_s1", True)
    max_seq_len = cfg_dict.get("max_seq_len", MAX_SEQ_LEN)
    n_s2_bands = len(feature_cols)

    print(f"  feature_cols: {feature_cols}")
    print(f"  s1_feature_cols: {s1_feature_cols}")
    print(f"  use_s1: {use_s1}  max_seq_len: {max_seq_len}")

    if not use_s1:
        print("Model does not use S1 — attention diagnostic not applicable.")
        return

    # Load training tile parquets
    print("Loading training pixels ...")
    tiles_dir = ROOT / "data/training/tiles"
    frames = []
    for tp in sorted(tiles_dir.glob("*.parquet")):
        df = pl.read_parquet(tp)
        df = df.with_columns(
            pl.col("point_id").str.extract(r"^(.+)_\d{4}_\d{4}$", 1).alias("region_id")
        )
        frames.append(df)
    all_df = pl.concat(frames, how="diagonal_relaxed")
    print(f"  {len(all_df):,} rows from {all_df['region_id'].n_unique()} regions")

    # Sample pixel IDs
    rng = np.random.default_rng(42)
    pixel_region = (
        all_df.select(["point_id", "region_id"])
        .unique("point_id")
        .with_columns(
            pl.when(pl.col("region_id").str.contains("presence"))
            .then(pl.lit("presence"))
            .when(pl.col("region_id").str.contains("absence"))
            .then(pl.lit("absence"))
            .otherwise(pl.lit("unknown"))
            .alias("label")
        )
    )
    pres_pids = pixel_region.filter(pl.col("label") == "presence")["point_id"].to_list()
    abs_pids  = pixel_region.filter(pl.col("label") == "absence")["point_id"].to_list()

    n = min(args.n_samples, len(pres_pids), len(abs_pids))
    pres_sample = rng.choice(pres_pids, n, replace=False).tolist()
    abs_sample  = rng.choice(abs_pids,  n, replace=False).tolist()
    all_sample  = pres_sample + abs_sample
    sample_labels = {pid: 1.0 for pid in pres_sample} | {pid: 0.0 for pid in abs_sample}

    print(f"  Sampled {n} presence + {n} absence pixels")

    # Build TAMDataset — add year/doy if absent (tile parquets don't pre-compute them)
    pixel_df = all_df.filter(pl.col("point_id").is_in(all_sample))
    if "year" not in pixel_df.columns or "doy" not in pixel_df.columns:
        pixel_df = pixel_df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.ordinal_day().alias("doy"),
        ])
    from tam.core.config import TAMConfig
    cfg = TAMConfig.from_dict(cfg_dict)

    dataset = TAMDataset(
        pixel_df=pixel_df,
        labels=sample_labels,
        band_mean=band_mean,
        band_std=band_std,
        max_seq_len=max_seq_len,
        use_s1=True,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        pixel_zscore=cfg.pixel_zscore,
        feature_cols_override=feature_cols,
        s1_feature_cols_override=s1_feature_cols,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_scores_full = []
    all_scores_abl  = []
    all_s1_fracs    = []
    all_labels      = []
    all_n_s1_tokens = []

    print("Running inference ...")
    for batch in loader:
        labels_b = batch["label"].numpy()
        is_s1_b  = batch.get("is_s1")

        # Count S1 tokens per sample
        if is_s1_b is not None:
            mask_b = batch["mask"]
            valid  = (~mask_b).float()
            s1_tok = (is_s1_b.float() * valid).sum(dim=1).numpy()
            all_n_s1_tokens.extend(s1_tok.tolist())

        # Attention fractions
        fracs = attention_s1_fraction(model, batch, device)
        all_s1_fracs.extend(fracs.tolist())

        # Score full vs ablated
        sf, sa = score_with_ablation(model, batch, device, n_s2_bands)
        all_scores_full.extend(sf.tolist())
        all_scores_abl.extend(sa.tolist())
        all_labels.extend(labels_b.tolist())

    scores_full = np.array(all_scores_full)
    scores_abl  = np.array(all_scores_abl)
    s1_fracs    = np.array(all_s1_fracs)
    labels_arr  = np.array(all_labels)
    n_s1_tokens = np.array(all_n_s1_tokens) if all_n_s1_tokens else None

    delta = scores_full - scores_abl  # positive = S1 lowers the score

    print("\n--- S1 Attention fraction (last layer, mean over heads) ---")
    for lbl, name in [(1, "presence"), (0, "absence")]:
        m = labels_arr == lbl
        print(f"  {name:8s}: mean={s1_fracs[m].mean():.3f}  "
              f"p25={np.percentile(s1_fracs[m],25):.3f}  "
              f"p75={np.percentile(s1_fracs[m],75):.3f}")

    if n_s1_tokens is not None:
        print("\n--- S1 token count per pixel ---")
        for lbl, name in [(1, "presence"), (0, "absence")]:
            m = labels_arr == lbl
            print(f"  {name:8s}: mean={n_s1_tokens[m].mean():.1f}  "
                  f"p5={np.percentile(n_s1_tokens[m],5):.0f}  "
                  f"p95={np.percentile(n_s1_tokens[m],95):.0f}")

    print("\n--- Score delta: full - S1-zeroed (positive = S1 pushes score UP) ---")
    for lbl, name in [(1, "presence"), (0, "absence")]:
        m = labels_arr == lbl
        d = delta[m]
        print(f"  {name:8s}: mean={d.mean():+.3f}  "
              f"p5={np.percentile(d,5):+.3f}  p25={np.percentile(d,25):+.3f}  "
              f"p75={np.percentile(d,75):+.3f}  p95={np.percentile(d,95):+.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. S1 attention fraction
    ax = axes[0]
    for lbl, name, col in [(1,"presence","darkorange"), (0,"absence","steelblue")]:
        m = labels_arr == lbl
        ax.hist(s1_fracs[m], bins=30, alpha=0.6, color=col, density=True, label=name)
    ax.set_xlabel("S1 attention fraction (last layer)")
    ax.set_ylabel("Density")
    ax.set_title("How much attention goes to S1 tokens?")
    ax.legend()

    # 2. Score delta distribution
    ax = axes[1]
    for lbl, name, col in [(1,"presence","darkorange"), (0,"absence","steelblue")]:
        m = labels_arr == lbl
        ax.hist(delta[m], bins=40, alpha=0.6, color=col, density=True, label=name)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Score(full) − Score(S1 zeroed)")
    ax.set_ylabel("Density")
    ax.set_title("Score change when S1 bands are zeroed")
    ax.legend()

    # 3. Score full vs ablated scatter (presence only)
    ax = axes[2]
    m = labels_arr == 1
    ax.scatter(scores_abl[m], scores_full[m], alpha=0.3, s=8, c="darkorange", label="presence")
    m = labels_arr == 0
    ax.scatter(scores_abl[m], scores_full[m], alpha=0.3, s=8, c="steelblue", label="absence")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("Score (S1 zeroed)")
    ax.set_ylabel("Score (full)")
    ax.set_title("Full vs S1-ablated scores")
    ax.legend(markerscale=3)

    plt.tight_layout()
    out_path = args.out or ROOT / "outputs/diag_s1_attention.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
