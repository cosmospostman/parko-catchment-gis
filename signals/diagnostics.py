"""signals/diagnostics.py — Shared diagnostic plotting for all signals.

Functions
---------
plot_signal_map       — per-pixel scatter overlaid on WMS imagery
plot_distributions    — histogram split by presence/absence class
separability_score    — (presence_median - absence_median) / pooled_std
_resolve_classes      — derive presence/absence pixel IDs from a Location's sub_bboxes
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def _resolve_classes(
    stats: pd.DataFrame,
    loc: object,
) -> tuple[pd.Index | None, pd.Index | None]:
    """Return (presence_ids, absence_ids) point_id arrays from loc.sub_bboxes.

    Looks for sub_bboxes with role 'presence' and 'absence'. If none are
    defined, returns (None, None) and diagnostics run without class splits.
    """
    presence_bbox = None
    absence_bbox = None
    for sub in loc.sub_bboxes.values():
        if sub.role == "presence":
            presence_bbox = sub.bbox
        elif sub.role == "absence":
            absence_bbox = sub.bbox

    presence_ids = None
    absence_ids = None

    if presence_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = presence_bbox
        presence_ids = stats.loc[
            stats["lon"].between(lon_min, lon_max) &
            stats["lat"].between(lat_min, lat_max),
            "point_id",
        ]

    if absence_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = absence_bbox
        absence_ids = stats.loc[
            stats["lon"].between(lon_min, lon_max) &
            stats["lat"].between(lat_min, lat_max),
            "point_id",
        ]

    return presence_ids, absence_ids


# ---------------------------------------------------------------------------
# Separability score
# ---------------------------------------------------------------------------

def separability_score(
    stats: pd.DataFrame,
    value_col: str,
    presence_ids: pd.Index | None,
    absence_ids: pd.Index | None,
) -> float | None:
    """(presence_median - absence_median) / pooled_std.

    Returns None if either class is missing or has fewer than 2 pixels.
    Positive values mean presence is higher; negative means lower.
    """
    if presence_ids is None or absence_ids is None:
        return None

    pres = stats.loc[stats["point_id"].isin(presence_ids), value_col].dropna()
    abs_ = stats.loc[stats["point_id"].isin(absence_ids), value_col].dropna()

    if len(pres) < 2 or len(abs_) < 2:
        return None

    pooled_std = math.sqrt(
        (pres.std() ** 2 * (len(pres) - 1) + abs_.std() ** 2 * (len(abs_) - 1))
        / (len(pres) + len(abs_) - 2)
    )
    if pooled_std == 0:
        return None
    return (pres.median() - abs_.median()) / pooled_std


# ---------------------------------------------------------------------------
# Spatial map
# ---------------------------------------------------------------------------

def plot_signal_map(
    stats: pd.DataFrame,
    value_col: str,
    loc: object,
    title: str,
    out_path: Path | None = None,
    colormap: str = "viridis",
) -> plt.Figure | None:
    """Scatterplot of per-pixel signal value overlaid on WMS imagery.

    Parameters
    ----------
    stats:
        Per-pixel DataFrame with ``lon``, ``lat``, and ``value_col`` columns.
    value_col:
        Column to colour-map.
    loc:
        ``utils.location.Location`` — bbox used for WMS fetch and map extent.
    title:
        Figure title.
    out_path:
        If given, figure is saved here.
    colormap:
        Matplotlib colormap name.

    Returns
    -------
    The Figure, or None if plotting failed.
    """
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parent.parent
    _spec = _ilu.spec_from_file_location("qglobe_plot", _root / "utils" / "qglobe-plot.py")
    _mod = _ilu.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        fetch_wms_image = _mod.fetch_wms_image
        bbox = loc.bbox
        try:
            bg_img = fetch_wms_image(bbox, width_px=2048)
        except Exception:
            bg_img = None
    except Exception:
        fetch_wms_image = None
        bg_img = None

    lon_min, lat_min, lon_max, lat_max = loc.bbox
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_centre = (lat_min + lat_max) / 2
    lon_m_per_deg = 111_320 * math.cos(math.radians(lat_centre))
    lat_m_per_deg = 111_320

    fig_w = 7
    fig_h = max(4.0, fig_w * (lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg))

    try:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

        if bg_img is not None:
            ax.imshow(
                bg_img,
                extent=[lon_min, lon_max, lat_min, lat_max],
                origin="upper",
                aspect="auto",
                interpolation="bilinear",
                zorder=0,
            )

        pt_per_deg = fig_w * 72 / lon_span
        marker_pt = (10 / lon_m_per_deg) * pt_per_deg
        marker_s = max(0.6, marker_pt ** 2 / 10)

        sc = ax.scatter(
            stats["lon"], stats["lat"],
            c=stats[value_col], cmap=colormap,
            s=marker_s,
            linewidths=0.0,
            alpha=0.55,
            zorder=2,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(value_col, fontsize=8)

        # Overlay presence sub-bbox if defined
        for sub in loc.sub_bboxes.values():
            if sub.role == "presence":
                slon_min, slat_min, slon_max, slat_max = sub.bbox
                ax.add_patch(mpatches.Rectangle(
                    (slon_min, slat_min),
                    slon_max - slon_min,
                    slat_max - slat_min,
                    fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
                    label=sub.label, zorder=4,
                ))
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
                  facecolor="black", labelcolor="white", edgecolor="none")

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        ax.set_title(title, fontsize=9)

        fig.tight_layout()

        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    except Exception as exc:
        print(f"  WARNING: plot_signal_map failed ({exc})", flush=True)
        return None


# ---------------------------------------------------------------------------
# Distribution histogram
# ---------------------------------------------------------------------------

def plot_distributions(
    stats: pd.DataFrame,
    value_col: str,
    loc: object,
    presence_ids: pd.Index | None = None,
    absence_ids: pd.Index | None = None,
    out_path: Path | None = None,
) -> plt.Figure | None:
    """Histogram of signal value, optionally split by presence/absence class.

    Parameters
    ----------
    stats:
        Per-pixel DataFrame with ``value_col`` column.
    value_col:
        Column to histogram.
    loc:
        ``utils.location.Location`` — used for figure title.
    presence_ids:
        point_id values for the presence class.
    absence_ids:
        point_id values for the absence class.
    out_path:
        If given, figure is saved here.

    Returns
    -------
    The Figure, or None if plotting failed.
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 4))

        if presence_ids is not None and absence_ids is not None:
            pres = stats.loc[stats["point_id"].isin(presence_ids), value_col].dropna()
            abs_ = stats.loc[stats["point_id"].isin(absence_ids), value_col].dropna()
            all_vals = pd.concat([pres, abs_])
            bins = np.linspace(all_vals.min(), all_vals.max(), 31)

            ax.hist(pres, bins=bins, color="darkorange", alpha=0.7, label=f"Presence (n={len(pres)})", edgecolor="white", linewidth=0.4)
            ax.hist(abs_, bins=bins, color="steelblue", alpha=0.7, label=f"Absence (n={len(abs_)})", edgecolor="white", linewidth=0.4)
            ax.axvline(pres.median(), color="darkorange", linestyle="--", linewidth=1.5,
                       label=f"Presence median = {pres.median():.4f}")
            ax.axvline(abs_.median(), color="steelblue", linestyle="--", linewidth=1.5,
                       label=f"Absence median = {abs_.median():.4f}")
        else:
            ax.hist(stats[value_col].dropna(), bins=30, color="steelblue",
                    edgecolor="white", linewidth=0.5, label="All pixels")

        ax.set_xlabel(value_col, fontsize=9)
        ax.set_ylabel("Pixel count", fontsize=9)
        ax.set_title(f"{loc.name} — {value_col} distribution", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()

        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    except Exception as exc:
        print(f"  WARNING: plot_distributions failed ({exc})", flush=True)
        return None
