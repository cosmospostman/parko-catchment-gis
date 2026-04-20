"""utils/heatmap.py — Parkinsonia probability heatmap plots.

Produces two separate figures from a scored DataFrame:
  1. <stem>_prob_vs_imagery.png  — Queensland Globe satellite underlay + per-pixel overlay
  2. <stem>_prob_black.png       — Black-background probability grid

Usage
-----
from utils.heatmap import plot_prob_heatmaps

paths = plot_prob_heatmaps(scored_df, loc, out_dir, stem="longreach")

# With annotation overlays (e.g. training bbox):
paths = plot_prob_heatmaps(
    scored_df, loc, out_dir, stem="longreach",
    annotations=[dict(
        xy=(lon_min, lat_min), width=lon_span, height=lat_span,
        edgecolor="white", linewidth=1.2, linestyle="--",
        label="Training: infestation patch",
    )],
)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PIXEL_DEG = 0.000100   # Sentinel-2 pixel spacing in degrees (measured from data)


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


def _prob_grid(
    valid: pd.DataFrame,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    cmap,
    alpha: float = 0.85,
    prob_col: str = "prob_tam",
) -> np.ndarray:
    """Rasterise probability values onto a regular lon/lat grid.

    Returns an RGBA array of shape (H, W, 4) aligned to the bbox extent,
    with transparent cells where no pixel falls.  Each source point maps to
    exactly one grid cell; if multiple points fall in the same cell (should
    not happen with 10 m Sentinel-2 data) the mean is used.
    """
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    W = max(int(round(lon_span / _PIXEL_DEG)), 1)
    H = max(int(round(lat_span / _PIXEL_DEG)), 1)

    col_idx = np.clip(
        ((valid["lon"].values - lon_min) / lon_span * W).astype(int), 0, W - 1
    )
    # imshow origin="upper" → row 0 = lat_max
    row_idx = np.clip(
        ((lat_max - valid["lat"].values) / lat_span * H).astype(int), 0, H - 1
    )

    prob = valid[prob_col].values

    # Accumulate sum and count to compute mean per cell
    grid_sum   = np.zeros((H, W), dtype=np.float32)
    grid_count = np.zeros((H, W), dtype=np.int32)
    np.add.at(grid_sum,   (row_idx, col_idx), prob)
    np.add.at(grid_count, (row_idx, col_idx), 1)

    filled = grid_count > 0
    grid_mean = np.where(filled, grid_sum / np.maximum(grid_count, 1), np.nan)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[filled] = cmap(grid_mean[filled])
    rgba[filled, 3] = alpha   # set alpha only for populated cells

    return rgba


def _apply_annotations(ax, annotations: list[dict]) -> None:
    """Draw Rectangle annotations and add a legend if any have labels."""
    import matplotlib.patches as mpatches

    has_label = False
    for ann in annotations:
        kw = dict(ann)
        xy = kw.pop("xy")
        width = kw.pop("width")
        height = kw.pop("height")
        kw.setdefault("fill", False)
        kw.setdefault("zorder", 4)
        ax.add_patch(mpatches.Rectangle(xy, width, height, **kw))
        if "label" in kw:
            has_label = True

    if has_label:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
                  facecolor="black", labelcolor="white", edgecolor="none")


def plot_prob_heatmaps(
    scored_df: pd.DataFrame,
    loc,
    out_dir: Path,
    stem: str,
    annotations: list[dict] | None = None,
    prob_col: str = "prob_tam",
) -> list[Path]:
    """Render two separate probability heatmap figures.

    Figure 1 (<stem>_prob_vs_imagery.png):
        Queensland Globe 20cm satellite underlay with per-pixel probability
        colour overlay (small circular dots, one per pixel).

    Figure 2 (<stem>_prob_black.png):
        Black-background scatter, one dot per pixel, coloured by probability.

    Parameters
    ----------
    scored_df  : DataFrame with columns lon, lat, <prob_col>, is_presence
    loc        : Location object (used for title)
    out_dir    : directory to write PNGs into
    stem       : filename prefix
    annotations: optional list of Rectangle annotation dicts. Each dict must
                 contain ``xy``, ``width``, ``height`` and any valid
                 ``mpatches.Rectangle`` kwargs (e.g. edgecolor, linestyle, label).
    prob_col   : name of the probability column (default: "prob_tam")

    Returns
    -------
    List of Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if annotations is None:
        annotations = []

    bbox = _scene_bbox(scored_df)
    lon_min, lat_min, lon_max, lat_max = bbox
    n_px = len(scored_df)

    try:
        mod = _load_qglobe()
        cache = mod.SceneTileCache()
        cache.expand(bbox)
        cache.prefetch()
        img = cache._img
    except Exception as exc:
        print(f"  WARNING: WMS fetch failed ({exc}) — imagery panel will use dark background")
        img = None

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    aspect = (lon_max - lon_min) / (lat_max - lat_min)
    fig_h = 10
    fig_w = max(fig_h * aspect, 4)

    valid = scored_df.dropna(subset=[prob_col])
    extent = [lon_min, lon_max, lat_min, lat_max]

    # ------------------------------------------------------------------
    # Figure 1: satellite underlay + pixel grid overlay
    # ------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig1.suptitle(
        f"{loc.name} — Parkinsonia probability\n"
        f"Queensland Globe 20cm imagery  ({n_px:,} pixels)",
        fontsize=11,
    )

    if img is not None:
        ax.imshow(img, extent=extent, origin="upper", aspect="auto")
    else:
        ax.set_facecolor("#111111")

    rgba = _prob_grid(valid, lon_min, lon_max, lat_min, lat_max, cmap, alpha=0.75, prob_col=prob_col)
    ax.imshow(rgba, extent=extent, origin="upper", aspect="auto", zorder=2,
              interpolation="nearest")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("P(Parkinsonia)", fontsize=9)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    _apply_annotations(ax, annotations)

    fig1.tight_layout()
    p1 = out_dir / f"{stem}_prob_vs_imagery.png"
    fig1.savefig(p1, dpi=800, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")

    # ------------------------------------------------------------------
    # Figure 2: black background pixel grid
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    fig2.suptitle(
        f"{loc.name} — Parkinsonia probability\n({n_px:,} pixels)",
        fontsize=11, color="white",
    )

    rgba2 = _prob_grid(valid, lon_min, lon_max, lat_min, lat_max, cmap, alpha=1.0, prob_col=prob_col)
    ax2.imshow(rgba2, extent=extent, origin="upper", aspect="auto", zorder=2,
               interpolation="nearest")

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cb2 = fig2.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.02)
    cb2.set_label("P(Parkinsonia)", fontsize=9, color="white")
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")

    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_xlabel("Longitude", fontsize=9, color="white")
    ax2.set_ylabel("Latitude", fontsize=9, color="white")
    ax2.tick_params(labelsize=8, colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")
    _apply_annotations(ax2, annotations)

    fig2.tight_layout()
    p2 = out_dir / f"{stem}_prob_black.png"
    fig2.savefig(p2, dpi=800, bbox_inches="tight", facecolor="black")
    plt.close(fig2)
    print(f"Saved: {p2}")

    return [p1, p2]
