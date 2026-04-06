"""utils/heatmap.py — Parkinsonia probability heatmap plots.

Produces two separate figures from a scored DataFrame (output of ParkoClassifier.score()):
  1. <stem>_prob_vs_imagery.png  — Queensland Globe satellite underlay + per-pixel overlay
  2. <stem>_prob_black.png       — Black-background probability scatter

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
_HALF_DEG = 0.000045   # ~5 m half-width of a 10 m pixel in degrees


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


def _marker_size(ax, lon_min: float, lon_max: float) -> float:
    """Compute scatter marker size (points²) so one marker ≈ one 10 m pixel."""
    fig = ax.get_figure()
    ax_width_in = ax.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    lon_span = lon_max - lon_min
    frac = (2 * _HALF_DEG) / lon_span
    px_width_pts = frac * ax_width_in * 72
    return max(px_width_pts ** 2 * 0.5, 1.0)


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
    wms_width: int = 512,
    annotations: list[dict] | None = None,
) -> list[Path]:
    """Render two separate probability heatmap figures.

    Figure 1 (<stem>_prob_vs_imagery.png):
        Queensland Globe 20cm satellite underlay with per-pixel probability
        colour overlay (small circular dots, one per pixel).

    Figure 2 (<stem>_prob_black.png):
        Black-background scatter, one dot per pixel, coloured by probability.

    Parameters
    ----------
    scored_df  : DataFrame with columns lon, lat, prob_lr, is_presence
    loc        : Location object (used for title)
    out_dir    : directory to write PNGs into
    stem       : filename prefix
    wms_width  : pixel width for WMS tile fetch
    annotations: optional list of Rectangle annotation dicts. Each dict must
                 contain ``xy``, ``width``, ``height`` and any valid
                 ``mpatches.Rectangle`` kwargs (e.g. edgecolor, linestyle, label).

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
        img = mod.fetch_wms_image(bbox, width_px=wms_width)
    except Exception as exc:
        print(f"  WARNING: WMS fetch failed ({exc}) — imagery panel will use dark background")
        img = None

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    aspect = (lon_max - lon_min) / (lat_max - lat_min)
    fig_h = 10
    fig_w = max(fig_h * aspect, 4)

    valid = scored_df.dropna(subset=["prob_lr"])

    # ------------------------------------------------------------------
    # Figure 1: satellite underlay + dot overlay
    # ------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig1.suptitle(
        f"{loc.name} — Parkinsonia probability\n"
        f"Queensland Globe 20cm imagery  ({n_px:,} pixels)",
        fontsize=11,
    )

    if img is not None:
        ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin="upper", aspect="auto")
    else:
        ax.set_facecolor("#111111")

    fig1.canvas.draw()
    s = _marker_size(ax, lon_min, lon_max)

    ax.scatter(
        valid["lon"], valid["lat"],
        c=valid["prob_lr"], cmap=cmap, norm=norm,
        s=s, marker="o", linewidths=0, alpha=0.7, zorder=2,
    )

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
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")

    # ------------------------------------------------------------------
    # Figure 2: black background scatter
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    fig2.suptitle(
        f"{loc.name} — Parkinsonia probability\n({n_px:,} pixels)",
        fontsize=11, color="white",
    )

    fig2.canvas.draw()
    s2 = _marker_size(ax2, lon_min, lon_max)

    sc = ax2.scatter(
        valid["lon"], valid["lat"],
        c=valid["prob_lr"], cmap=cmap, norm=norm,
        s=s2, marker="o", linewidths=0, zorder=2,
    )

    cb2 = fig2.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
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
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig2)
    print(f"Saved: {p2}")

    return [p1, p2]
