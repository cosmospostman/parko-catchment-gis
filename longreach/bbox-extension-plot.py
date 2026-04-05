"""Plot current HD bbox vs proposed southern extension over Queensland Globe imagery.

Outputs: outputs/longreach-dry-nir/bbox_extension.png
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_spec = _ilu.spec_from_file_location("qglobe_plot", PROJECT_ROOT / "scripts" / "qglobe-plot.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_wms_image = _mod.fetch_wms_image

# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

# Current pixel grid extents (from longreach_pixels.parquet)
HD_LON_MIN, HD_LON_MAX =  145.423948,  145.424956
HD_LAT_MIN, HD_LAT_MAX = -22.764033,  -22.761054

# Southern extension: 34 additional S2 pixels (≈342 m) at 10.05 m/pixel spacing
LAT_STEP   = 0.00009030   # deg per pixel, northward
EXT_LAT_MIN = HD_LAT_MIN - 34 * LAT_STEP   # -22.767104

MARGIN = 0.0005
WMS_BBOX = [
    HD_LON_MIN - MARGIN,
    EXT_LAT_MIN - MARGIN,
    HD_LON_MAX + MARGIN,
    HD_LAT_MAX + MARGIN,
]

OUT_PATH = PROJECT_ROOT / "outputs" / "longreach-dry-nir" / "bbox_extension.png"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Fetching WMS tile for bbox {WMS_BBOX} ...")
    bg = fetch_wms_image(WMS_BBOX, width_px=2048)
    print(f"  Tile: {bg.shape[1]}×{bg.shape[0]} px")

    lon_min, lat_min, lon_max, lat_max = WMS_BBOX
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    lat_centre = (lat_min + lat_max) / 2
    lon_m_per_deg = 111320 * np.cos(np.radians(lat_centre))
    lat_m_per_deg = 111320

    fig_w = 5
    fig_h = fig_w * (lat_span * lat_m_per_deg) / (lon_span * lon_m_per_deg)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    ax.imshow(
        bg,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )

    # Extended bbox (orange, behind)
    ext_rect = mpatches.Rectangle(
        (HD_LON_MIN, EXT_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN,
        HD_LAT_MAX - EXT_LAT_MIN,
        fill=True, facecolor="orange", alpha=0.15,
        edgecolor="orange", linewidth=1.5, linestyle="--",
        label=f"Extended bbox (proposed)", zorder=2,
    )
    ax.add_patch(ext_rect)

    # Current HD bbox (white, on top)
    hd_rect = mpatches.Rectangle(
        (HD_LON_MIN, HD_LAT_MIN),
        HD_LON_MAX - HD_LON_MIN,
        HD_LAT_MAX - HD_LAT_MIN,
        fill=True, facecolor="white", alpha=0.15,
        edgecolor="white", linewidth=1.5,
        label="Current HD bbox", zorder=3,
    )
    ax.add_patch(hd_rect)

    # Labels inside each zone
    ax.text(
        (HD_LON_MIN + HD_LON_MAX) / 2, (HD_LAT_MIN + HD_LAT_MAX) / 2,
        "current grid\n(374 pixels)",
        ha="center", va="center", fontsize=7, color="white", fontweight="bold",
        zorder=4,
    )
    ax.text(
        (HD_LON_MIN + HD_LON_MAX) / 2, (EXT_LAT_MIN + HD_LAT_MIN) / 2,
        "southern extension\n(+34 cols, ≈342 m)",
        ha="center", va="center", fontsize=7, color="orange", fontweight="bold",
        zorder=4,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.set_title(
        "Longreach — current pixel grid vs proposed southern extension\n"
        f"Current: lat [{HD_LAT_MIN}, {HD_LAT_MAX}]   "
        f"Extension: +34 pixels (≈342 m) → lat [{EXT_LAT_MIN:.6f}, {HD_LAT_MIN}]",
        fontsize=8,
    )

    legend = ax.legend(loc="lower right", fontsize=7, framealpha=0.7,
                       facecolor="black", labelcolor="white", edgecolor="none")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
