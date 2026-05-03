"""Attention visualisation for TAMClassifier.

Outputs all files into a single subdirectory alongside the checkpoint:
  <checkpoint>/attention/
    summary.txt          — text summary of peak attention by site/class/head
    mean_<site>.png      — mean attention across all heads and layers
    perhead_<site>_<cls>.png — per-head attention for last layer

Usage:
    python -m tam.viz_attention --checkpoint outputs/models/tam-v7_norman_road_only \
        --experiment v7_norman_road_only

    python -m tam.viz_attention --checkpoint outputs/models/tam-v7_frenchs_only \
        --experiment v7_frenchs_only --sites frenchs --n-pixels 50
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.core.config import TAMConfig
from tam.core.train import load_tam
from tam.core.dataset import TAMDataset
from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

DOY_BINS   = np.arange(1, 366, 7)  # weekly bins
MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def attention_by_doy(
    model,
    ds: TAMDataset,
    indices: list[int],
    device: str,
    per_head: bool = False,
    layer: int = -1,
) -> np.ndarray:
    """Return mean attention weight per DOY bin.

    Returns
    -------
    per_head=False : (n_bins,)            — mean across all heads and layers
    per_head=True  : (n_heads, n_bins)    — per head, last layer only
    """
    n_bins = len(DOY_BINS) - 1
    sample0 = ds[indices[0]]
    attn0   = model.get_attention_weights(
        sample0.bands.unsqueeze(0).to(device),
        sample0.doy.unsqueeze(0).to(device),
        sample0.mask.unsqueeze(0).to(device),
    )
    n_heads = attn0[0].shape[0]

    shape      = (n_heads, n_bins) if per_head else (n_bins,)
    bin_sums   = np.zeros(shape)
    bin_counts = np.zeros(shape)

    for idx in indices:
        sample    = ds[idx]
        bands     = sample.bands.unsqueeze(0).to(device)
        doy       = sample.doy.unsqueeze(0).to(device)
        mask      = sample.mask.unsqueeze(0).to(device)
        attn_list = model.get_attention_weights(bands, doy, mask)
        doy_np    = sample.doy.numpy()
        valid     = doy_np > 0

        if per_head:
            key_attn = attn_list[layer].cpu().numpy().mean(axis=1)  # (n_heads, T)
            for h in range(n_heads):
                for val, d in zip(key_attn[h][valid], doy_np[valid]):
                    b = np.searchsorted(DOY_BINS, d, side="right") - 1
                    if 0 <= b < n_bins:
                        bin_sums[h, b]   += val
                        bin_counts[h, b] += 1
        else:
            for attn in attn_list:
                key_attn = attn.cpu().numpy().mean(axis=(0, 1))
                for val, d in zip(key_attn[valid], doy_np[valid]):
                    b = np.searchsorted(DOY_BINS, d, side="right") - 1
                    if 0 <= b < n_bins:
                        bin_sums[b]   += val
                        bin_counts[b] += 1

    with np.errstate(invalid="ignore"):
        return np.where(bin_counts > 0, bin_sums / bin_counts, 0.0)


def bar_axes(ax, profile, color, title):
    bin_centres = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2
    ax.bar(bin_centres, profile, width=6, color=color, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(MONTH_DOYS)
    ax.set_xticklabels(MONTH_LBLS, fontsize=7, rotation=45)
    ax.set_ylabel("mean attn", fontsize=8)
    ax.set_xlabel("Day of year", fontsize=8)


def peak_months(profile: np.ndarray, top_n: int = 3) -> str:
    """Return top_n month names by mean attention."""
    bin_centres = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2
    top_idx = np.argsort(profile)[::-1][:top_n]
    months = []
    for i in top_idx:
        doy = bin_centres[i]
        m = np.searchsorted(MONTH_DOYS, doy, side="right") - 1
        months.append(MONTH_LBLS[max(0, m)])
    return ", ".join(months)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sites",     nargs="+", default=None)
    parser.add_argument("--n-pixels",  type=int, default=30)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    out_dir  = ckpt_dir / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT

    # --- Load pixels ----------------------------------------------------------
    regions  = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        s1_cols = [c for c in ("source", "vh", "vv") if c in available]
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=read_cols)
            chunks.append(tbl.to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year
    pixel_df["doy"]  = pd.to_datetime(pixel_df["date"]).dt.day_of_year

    pixel_coords = pixel_df[["point_id","lon","lat"]].drop_duplicates("point_id").reset_index(drop=True)
    labelled     = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    all_labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    # Filter pixel_df to labelled pixels before snap_s1_to_s2 — snap is O(n²)
    # over rows so running it on the full tile parquet is very slow.
    pixel_df = pixel_df[pixel_df["point_id"].isin(all_labels.index)]

    # --- Load checkpoint ------------------------------------------------------
    model, band_mean, band_std = load_tam(ckpt_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(ckpt_dir / "tam_config.json") as fh:
        cfg = TAMConfig.from_dict(json.load(fh))

    global_feat_df: pd.DataFrame | None = None
    cache_path = ckpt_dir / "global_features_cache.parquet"
    if cache_path.exists():
        global_feat_df = pd.read_parquet(cache_path)

    sites = args.sites or sorted({point_site(p) for p in all_labels.index})

    ds = TAMDataset(
        pixel_df, all_labels,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=cfg.scl_purity_min,
        min_obs_per_year=cfg.min_obs_per_year,
        doy_jitter=0,
        global_features_df=global_feat_df,
        use_s1={4: "s1_only", 17: True}.get(cfg.n_bands, False),
    )

    pid_to_indices: dict[str, list[int]] = {}
    for i, (pid, *_) in enumerate(ds._windows):
        pid_to_indices.setdefault(pid, []).append(i)

    rng     = np.random.default_rng(42)
    n_heads = model.n_heads
    summary_lines: list[str] = [
        f"Attention summary — {args.experiment}",
        f"Checkpoint: {ckpt_dir}",
        f"n_heads={n_heads}  n_layers={model.n_layers}",
        "",
        f"{'site':<20} {'class':<10} {'n_px':>6}  {'peak_months (mean)':30}  {'peak_months per head (last layer)'}",
        "-" * 110,
    ]

    for site in sites:
        site_labels = all_labels[all_labels.index.map(point_site) == site]

        for cls in [1.0, 0.0]:
            cls_name = "presence" if cls == 1.0 else "absence"
            color    = "steelblue" if cls == 1.0 else "coral"
            cls_pids = site_labels[site_labels == cls].index.tolist()
            if not cls_pids:
                continue

            sample_pids = rng.choice(cls_pids, size=min(args.n_pixels, len(cls_pids)), replace=False)
            indices     = [i for p in sample_pids for i in pid_to_indices.get(p, [])]
            if not indices:
                continue

            # --- Mean plot ----------------------------------------------------
            mean_profile = attention_by_doy(model, ds, indices, device, per_head=False)
            fig, ax = plt.subplots(figsize=(10, 3))
            bar_axes(ax, mean_profile, color, f"{site} — {cls_name} — mean attention (all heads/layers)  n={len(indices)}")
            plt.tight_layout()
            plt.savefig(out_dir / f"mean_{site}_{cls_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Per-head plot ------------------------------------------------
            head_profiles = attention_by_doy(model, ds, indices, device, per_head=True)
            fig2, axs = plt.subplots(1, n_heads, figsize=(5 * n_heads, 3.5), sharey=True)
            if n_heads == 1:
                axs = [axs]
            fig2.suptitle(f"{site} — {cls_name} — per-head attention (last layer)  n={len(indices)}", fontsize=10)
            for h, ax in enumerate(axs):
                bar_axes(ax, head_profiles[h], color, f"head {h}")
            plt.tight_layout()
            plt.savefig(out_dir / f"perhead_{site}_{cls_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)

            # --- Text summary -------------------------------------------------
            mean_peaks = peak_months(mean_profile)
            head_peaks = "  |  ".join(f"h{h}: {peak_months(head_profiles[h])}" for h in range(n_heads))
            summary_lines.append(
                f"{site:<20} {cls_name:<10} {len(cls_pids):>6}  {mean_peaks:<30}  {head_peaks}"
            )

        summary_lines.append("")  # blank line between sites

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)

    print(summary_text)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
