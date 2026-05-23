"""Feature importance visualisation for TAMClassifier.

Uses gradient saliency to measure which input bands the model relies on.
For each sampled pixel window, computes |grad| w.r.t. the logit, pools over
valid timesteps, and accumulates a mean relative importance per band.

Note: input×gradient is not used because normalised S1/S2 cross-source features
are zero by construction (nan std → zeroed by nan_to_num), which would suppress
those bands regardless of their actual gradient.

Outputs all files into a single subdirectory alongside the checkpoint:
  <checkpoint>/features/
    summary.txt                    — top-3 features per site/class
    importance_<site>_<cls>.png   — horizontal bar chart per site

Usage:
    python -m tam.viz_features --checkpoint outputs/models/tam-v10 \
        --experiment v10

    python -m tam.viz_features --checkpoint outputs/models/tam-v10 \
        --experiment v10 --sites frenchs --n-pixels 50
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.core.config import TAMConfig
from tam.core.train import load_tam
from tam.core.dataset import TAMDataset, S1_FEATURE_COLS
from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path


def point_site(pid: str) -> str:
    m = re.match(r"^(.+?)_(presence|absence)", pid)
    return m.group(1) if m else pid


def feature_importance(
    model,
    ds: TAMDataset,
    indices: list[int],
    device: str,
    n_s2: int,
    n_s1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean relative gradient saliency per band, split by sensor source.

    S2 and S1 timesteps are evaluated separately: gradients on S2 rows are
    pooled over S2-only bands, and gradients on S1 rows over S1-only bands.
    Each vector is normalised independently so they are comparable within source.

    Returns
    -------
    s2_imp : (n_s2,) float, sums to 1.0 (or all-zero if no S2 rows)
    s1_imp : (n_s1,) float, sums to 1.0 (or all-zero if no S1 rows)
    """
    n_bands  = n_s2 + n_s1
    s2_sums  = np.zeros(n_s2, dtype=np.float64)
    s1_sums  = np.zeros(n_s1, dtype=np.float64)
    s2_count = 0
    s1_count = 0

    for idx in indices:
        sample = ds[idx]
        bands  = sample.bands.unsqueeze(0).to(device).float().requires_grad_(True)
        doy    = sample.doy.unsqueeze(0).to(device)
        mask   = sample.mask.unsqueeze(0).to(device)
        n_obs  = sample.n_obs.unsqueeze(0).to(device)
        is_s1  = sample.is_s1.unsqueeze(0).to(device)

        _prob, logit = model(bands, doy, mask, n_obs, is_s1=is_s1)
        logit.backward()

        grad   = bands.grad.detach().cpu().numpy()[0]  # (T, n_bands)
        sal    = np.abs(grad)                          # (T, n_bands)
        valid  = ~sample.mask.numpy()                  # (T,) bool
        is_s1_np = sample.is_s1.numpy().astype(bool)  # (T,) bool

        s2_rows = valid & ~is_s1_np
        s1_rows = valid &  is_s1_np

        if s2_rows.sum() > 0 and n_s2 > 0:
            s2_sums  += sal[s2_rows, :n_s2].mean(axis=0)
            s2_count += 1
        if s1_rows.sum() > 0 and n_s1 > 0:
            s1_sums  += sal[s1_rows, n_s2:].mean(axis=0)
            s1_count += 1

    def _normalise(arr, count):
        if count == 0:
            return np.zeros_like(arr)
        result = arr / count
        total  = result.sum()
        return result / total if total > 0 else result

    return _normalise(s2_sums, s2_count), _normalise(s1_sums, s1_count)


def plot_importance(ax, importance: np.ndarray, feature_cols: list[str], color: str, title: str) -> None:
    order = np.argsort(importance)  # ascending so largest is at top of barh
    labels = [feature_cols[i] for i in order]
    values = importance[order]
    ax.barh(labels, values, color=color, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("relative importance", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


def top_features(importance: np.ndarray, feature_cols: list[str], top_n: int = 3) -> str:
    if importance.sum() == 0:
        return "—"
    top_idx = np.argsort(importance)[::-1][:top_n]
    return ", ".join(feature_cols[i] for i in top_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sites",     nargs="+", default=None)
    parser.add_argument("--n-pixels",  type=int, default=30)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    out_dir  = ckpt_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = importlib.import_module(f"tam.experiments.{args.experiment}").EXPERIMENT

    all_regions = select_regions(exp.region_ids)

    all_site_region_ids: dict[str, list[str]] = {}
    for rid in exp.region_ids:
        site = point_site(rid)
        all_site_region_ids.setdefault(site, []).append(rid)

    sites = args.sites or sorted(all_site_region_ids.keys())

    model, band_mean, band_std, *_ = load_tam(ckpt_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(ckpt_dir / "tam_config.json") as fh:
        cfg_dict = json.load(fh)
    cfg = TAMConfig.from_dict(cfg_dict)
    saved_feature_cols = cfg_dict.get("feature_cols") or (list(cfg.feature_cols_override) if cfg.feature_cols_override else None)
    saved_s1_feature_cols = (
        list(cfg.s1_feature_cols) if cfg.s1_feature_cols
        else list(exp.train_kwargs["s1_feature_cols"]) if "s1_feature_cols" in exp.train_kwargs
        else None
    )

    global_feat_df: pl.DataFrame | None = None
    cache_path = ckpt_dir / "global_features_cache.parquet"
    if cache_path.exists():
        global_feat_df = pl.read_parquet(cache_path)

    feature_cols = saved_feature_cols or exp.feature_cols
    rng = np.random.default_rng(42)

    summary_lines: list[str] = [
        f"Feature importance summary — {args.experiment}",
        f"Checkpoint: {ckpt_dir}",
        f"Method: gradient saliency (|grad|)  n_pixels={args.n_pixels}",
        "",
        f"{'site':<20} {'class':<10} {'n_px':>6}  top-3 features",
        "-" * 90,
    ]

    for site in sites:
        site_region_ids = all_site_region_ids.get(site)
        if not site_region_ids:
            print(f"  skipping {site} — no regions found in experiment")
            continue

        print(f"  processing {site} ...")

        site_regions = select_regions(site_region_ids)
        tile_ids     = tile_ids_for_regions(site_region_ids)

        chunks: list[pl.DataFrame] = []
        for tid in tile_ids:
            path = tile_parquet_path(tid)
            if not path.exists():
                continue
            pf        = pq.ParquetFile(path)
            available = set(pf.schema_arrow.names)
            s1_cols   = [c for c in ("source", "vh", "vv") if c in available]
            base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
            read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols
            for rg in range(pf.metadata.num_row_groups):
                chunks.append(pl.from_arrow(pf.read_row_group(rg, columns=read_cols)))

        pixel_df = pl.concat(chunks).with_columns([
            pl.col("date").cast(pl.Date).dt.year().alias("year"),
            pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
        ])
        del chunks

        pixel_coords = pixel_df.select(["point_id", "lon", "lat"]).unique("point_id")
        labelled     = label_pixels(pixel_coords, site_regions).filter(pl.col("is_presence").is_not_null())
        site_labels  = {
            row["point_id"]: 1.0 if row["is_presence"] else 0.0
            for row in labelled.iter_rows(named=True)
        }

        pixel_df = pixel_df.filter(pl.col("point_id").is_in(set(site_labels)))

        ds = TAMDataset(
            pixel_df, site_labels,
            band_mean=band_mean, band_std=band_std,
            scl_purity_min=cfg.scl_purity_min,
            min_obs_per_year=cfg.min_obs_per_year,
            doy_jitter=0,
            global_features_df=global_feat_df,
            use_s1=cfg_dict.get("use_s1", False),
            feature_cols_override=saved_feature_cols,
            s1_feature_cols_override=saved_s1_feature_cols,
        )
        del pixel_df

        pid_to_indices: dict[str, list[int]] = {}
        for i, pid in enumerate(ds._pids):
            pid_to_indices.setdefault(pid, []).append(i)

        # Determine actual band counts from dataset
        sample0   = ds[0]
        s1_labels = saved_s1_feature_cols or S1_FEATURE_COLS
        s2_labels = list(feature_cols or [])
        n_s2      = len(s2_labels)
        n_s1      = len(s1_labels)

        use_s1_mixed = cfg_dict.get("use_s1", False) not in (False, None, "s1_only")
        classes = [(1.0, "presence", "steelblue"), (0.0, "absence", "coral")]

        n_rows = 2 if (use_s1_mixed and n_s1 > 0) else 1
        fig, axs = plt.subplots(
            n_rows, 2,
            figsize=(12, max(4, 0.35 * n_s2 + 1.5) * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"{site} — feature importance (gradient saliency)", fontsize=10)

        for col_idx, (cls, cls_name, color) in enumerate(classes):
            cls_pids = [pid for pid, v in site_labels.items() if v == cls]
            if not cls_pids:
                for row in range(n_rows):
                    axs[row][col_idx].set_visible(False)
                continue

            sample_pids = rng.choice(cls_pids, size=min(args.n_pixels, len(cls_pids)), replace=False)
            indices     = [i for p in sample_pids for i in pid_to_indices.get(p, [])]
            if not indices:
                for row in range(n_rows):
                    axs[row][col_idx].set_visible(False)
                continue

            s2_imp, s1_imp = feature_importance(model, ds, indices, device, n_s2, n_s1)

            plot_importance(axs[0][col_idx], s2_imp, s2_labels, color,
                            f"S2 — {cls_name}  n={len(indices)}")
            if n_rows == 2:
                plot_importance(axs[1][col_idx], s1_imp, s1_labels, color,
                                f"S1 — {cls_name}  n={len(indices)}")

            summary_lines.append(
                f"{site:<20} {cls_name:<10} {len(cls_pids):>6}"
                f"  S2: {top_features(s2_imp, s2_labels)}"
                + (f"  |  S1: {top_features(s1_imp, s1_labels)}" if n_s1 > 0 else "")
            )

        plt.tight_layout()
        plt.savefig(out_dir / f"importance_{site}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        del ds, pid_to_indices
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)

    print(summary_text)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
