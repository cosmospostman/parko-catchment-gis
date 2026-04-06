"""utils/heatmap.py — Parkinsonia probability heatmap plots.

Produces two figures from a scored DataFrame (output of ParkoClassifier.score()):
  1. Satellite underlay + per-pixel probability overlay
  2. Black-background probability scatter

Both functions accept an optional WMS image array; if None is passed the
satellite panel is rendered on a dark background instead (graceful degradation).

Usage
-----
from utils.heatmap import plot_prob_heatmaps

paths = plot_prob_heatmaps(scored_df, loc, out_dir, stem="longreach")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HALF_DEG = 0.000045   # ~5 m half-width of a pixel square at 10 m resolution


def _load_qglobe():
    spec = importlib.util.spec_from_file_location(
        "qglobe_plot", PROJECT_ROOT / "utils" / "qglobe-plot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene_bbox(scored_df: pd.DataFrame, margin: float = 0.00005) -> list[float]:
    return [
        scored_df["lon"].min() - margin,
        scored_df["lat"].min() - margin,
        scored_df["lon"].max() + margin,
        scored_df["lat"].max() + margin,
    ]


def plot_prob_heatmaps(
    scored_df: pd.DataFrame,
    loc,
    out_dir: Path,
    stem: str,
    wms_width: int = 512,
) -> list[Path]:
    """Render satellite+overlay and black-background probability heatmaps.

    Parameters
    ----------
    scored_df : DataFrame with columns lon, lat, prob_lr, is_presence
    loc       : Location object (used for title only)
    out_dir   : directory to write PNGs into
    stem      : filename prefix, e.g. "longreach" → longreach_prob_vs_imagery.png
    wms_width : pixel width for WMS tile fetch (default 512)

    Returns
    -------
    List of Paths written.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    bbox = _scene_bbox(scored_df)
    try:
        mod = _load_qglobe()
        img = mod.fetch_wms_image(bbox, width_px=wms_width)
    except Exception as exc:
        print(f"  WARNING: WMS fetch failed ({exc}) — plots will render without imagery")
        img = None

    lon_min, lat_min, lon_max, lat_max = bbox
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)
    half = _HALF_DEG
    n_px = len(scored_df)

    # ------------------------------------------------------------------
    # Figure 1: satellite underlay + probability overlay (left)
    #           black-background probability scatter (right)
    # ------------------------------------------------------------------
    fig, (ax_img, ax_score) = plt.subplots(1, 2, figsize=(14, 18))
    fig.suptitle(
        f"Parkinsonia probability vs Queensland Globe 20cm imagery\n"
        f"{loc.name}  ({n_px:,} pixels)",
        fontsize=12,
    )

    if img is not None:
        ax_img.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                      origin="upper", aspect="auto")
    else:
        ax_img.set_facecolor("#222222")
    ax_img.set_title("Queensland Globe 20cm + probability overlay", fontsize=10)

    for _, row in scored_df.iterrows():
        if pd.isna(row["prob_lr"]):
            continue
        ax_img.add_patch(mpatches.Rectangle(
            (row["lon"] - half, row["lat"] - half),
            width=half * 2, height=half * 2,
            linewidth=0, facecolor=cmap(norm(row["prob_lr"])), alpha=0.55,
        ))

    ax_score.set_facecolor("#222222")
    ax_score.set_title("Parkinsonia probability score (logistic regression)", fontsize=10)
    sc = ax_score.scatter(
        scored_df["lon"], scored_df["lat"],
        c=scored_df["prob_lr"], cmap=cmap, norm=norm,
        s=60, marker="s", linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax_score, fraction=0.03, pad=0.04)
    cb.set_label("Probability (Parkinsonia)", fontsize=9)
    for thresh in np.percentile(scored_df["prob_lr"].dropna(), np.arange(10, 100, 10)):
        cb.ax.axhline(thresh, color="white", linewidth=0.8, linestyle="--")

    for ax in (ax_img, ax_score):
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p1 = out_dir / f"{stem}_prob_vs_imagery.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ------------------------------------------------------------------
    # Figure 2: top / bottom decile on satellite
    # ------------------------------------------------------------------
    p90 = scored_df["prob_lr"].quantile(0.90)
    p10 = scored_df["prob_lr"].quantile(0.10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 18))
    fig.suptitle(f"Top vs bottom decile Parkinsonia probability\n{loc.name}", fontsize=12)

    for ax, subset, colour, title in [
        (axes[0], scored_df[scored_df["prob_lr"] >= p90], "#e74c3c",
         f"Top decile (prob ≥ {p90:.2f}, n={(scored_df['prob_lr'] >= p90).sum():,})"),
        (axes[1], scored_df[scored_df["prob_lr"] <= p10], "#3498db",
         f"Bottom decile (prob ≤ {p10:.2f}, n={(scored_df['prob_lr'] <= p10).sum():,})"),
    ]:
        if img is not None:
            ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                      origin="upper", aspect="auto")
        else:
            ax.set_facecolor("#222222")
        for _, row in subset.iterrows():
            ax.add_patch(mpatches.Rectangle(
                (row["lon"] - half, row["lat"] - half),
                width=half * 2, height=half * 2,
                linewidth=0.5, edgecolor="white", facecolor=colour, alpha=0.65,
            ))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p2 = out_dir / f"{stem}_prob_deciles.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    return [p1, p2]
