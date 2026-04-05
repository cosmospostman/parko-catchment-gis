#!/usr/bin/env python3
"""
scripts/qglobe-plot — plot bounding boxes and points over Queensland Globe satellite imagery.

Fetches a WMS tile from the Queensland Government SISP aerial ortho service
(LatestStateProgram_AllUsers) and overlays bounding boxes and/or points.

Usage:
    python scripts/qglobe-plot [--bbox LON_MIN,LAT_MIN,LON_MAX,LAT_MAX]
                               [--points-csv PATH]
                               [--boxes-csv PATH]
                               [--width PX]
                               [--out PATH]

--points-csv columns: point_id,lon,lat[,label]   (label: 1=presence, 0=absence)
--boxes-csv  columns: id,lon_min,lat_min,lon_max,lat_max
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults — Mitchell River infestation site
# ---------------------------------------------------------------------------

DEFAULT_BBOX = [141.550, -14.690, 141.620, -14.630]  # [lon_min, lat_min, lon_max, lat_max]
DEFAULT_WIDTH = 2048
DEFAULT_OUT = "input-img/qld_globe_overlay.png"

WMS_URL = (
    "https://spatial-img.information.qld.gov.au/arcgis/services/"
    "Basemaps/LatestStateProgram_AllUsers/ImageServer/WMSServer"
)
WMS_LAYER = "LatestStateProgram_AllUsers"
MAX_TILE_PX = 4096

# Overlay colours
COLOUR_PRESENCE = "#e74c3c"   # red
COLOUR_ABSENCE  = "#3498db"   # blue
COLOUR_UNLABELLED = "#f39c12" # orange
COLOUR_BOX      = "#f39c12"   # orange

logger = logging.getLogger(__name__)


def fetch_wms_image(bbox: list[float], width_px: int) -> np.ndarray:
    """Fetch a JPEG tile from the Queensland Globe WMS and return an (H, W, 3) uint8 array.

    bbox is [lon_min, lat_min, lon_max, lat_max] in WGS84 degrees.

    WMS 1.3.0 with CRS=EPSG:4326 uses axis order lat/lon, so BBOX is
    lat_min,lon_min,lat_max,lon_max. This is a common silent failure mode —
    if axis order is wrong you get a valid JPEG of the wrong place.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    height_px = max(1, int(width_px * lat_span / lon_span))

    if width_px > MAX_TILE_PX or height_px > MAX_TILE_PX:
        scale = MAX_TILE_PX / max(width_px, height_px)
        width_px  = max(1, int(width_px  * scale))
        height_px = max(1, int(height_px * scale))
        logger.warning("Tile dimensions clamped to %dx%d (Esri max %d px)", width_px, height_px, MAX_TILE_PX)

    # WMS 1.3.0 + EPSG:4326 → BBOX is lat_min,lon_min,lat_max,lon_max
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "CRS": "EPSG:4326",
        "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "WIDTH": width_px,
        "HEIGHT": height_px,
        "FORMAT": "image/jpeg",
        "STYLES": "",
    }

    logger.info("Fetching WMS tile %dx%d for bbox %s ...", width_px, height_px, bbox)
    resp = requests.get(WMS_URL, params=params, timeout=60)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(
            f"WMS returned non-image response (content-type: {content_type!r}). "
            "Server may have returned an error — check bbox and layer name.\n"
            f"Body: {resp.text[:500]}"
        )

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(img)

    # Sanity check: a uniform-colour image usually means the bbox axis order is wrong
    if arr.std() < 1.0:
        logger.warning(
            "WMS image is nearly uniform (std=%.2f) — possible axis-order error in BBOX", arr.std()
        )

    logger.info("Tile received: %dx%d px", arr.shape[1], arr.shape[0])
    return arr


def load_points(path: str) -> list[dict]:
    """Load a CSV with columns point_id,lon,lat[,label]. label is optional."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "point_id": row["point_id"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "label": int(row["label"]) if "label" in row and row["label"] != "" else None,
            })
    logger.info("Loaded %d points from %s", len(rows), path)
    return rows


def load_boxes(path: str) -> list[dict]:
    """Load a CSV with columns id,lon_min,lat_min,lon_max,lat_max."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "id": row["id"],
                "lon_min": float(row["lon_min"]),
                "lat_min": float(row["lat_min"]),
                "lon_max": float(row["lon_max"]),
                "lat_max": float(row["lat_max"]),
            })
    logger.info("Loaded %d boxes from %s", len(rows), path)
    return rows


def _nice_grid_spacing(span: float, target_lines: int = 8) -> float:
    """Return a round grid spacing (in degrees) that gives ~target_lines divisions."""
    import math
    raw = span / target_lines
    mag = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        step = factor * mag
        if span / step <= target_lines:
            return step
    return 10 * mag


def render(
    bbox: list[float],
    img_array: np.ndarray,
    points: list[dict],
    boxes: list[dict],
    out_path: str,
    grid: bool = False,
    grid_metres: float | None = None,
) -> None:
    """Render the satellite image with overlays and save to out_path."""
    lon_min, lat_min, lon_max, lat_max = bbox
    aspect = (lat_max - lat_min) / (lon_max - lon_min)
    fig_w = 12
    fig_h = max(4.0, fig_w * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    # imshow with geographic extent: subsequent patches/scatter use lon/lat directly.
    # Note: aspect="auto" avoids distortion in figure coords; the image itself is
    # north-up so origin="upper" is correct. At sub-regional scale the lon/lat
    # degree aspect ratio is close enough for visual inspection.
    ax.imshow(
        img_array,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )

    # Overlay bounding boxes
    for box in boxes:
        w = box["lon_max"] - box["lon_min"]
        h = box["lat_max"] - box["lat_min"]
        rect = mpatches.Rectangle(
            (box["lon_min"], box["lat_min"]), w, h,
            linewidth=1.5, edgecolor=COLOUR_BOX, facecolor="none",
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(
            box["lon_min"], box["lat_max"],
            box["id"],
            color=COLOUR_BOX, fontsize=6, va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, edgecolor="none"),
        )

    # Overlay points
    if points:
        presence  = [p for p in points if p["label"] == 1]
        absence   = [p for p in points if p["label"] == 0]
        unlabelled = [p for p in points if p["label"] is None]

        for group, colour, marker, zorder in [
            (presence,   COLOUR_PRESENCE,   "o", 4),
            (absence,    COLOUR_ABSENCE,    "o", 4),
            (unlabelled, COLOUR_UNLABELLED, "^", 4),
        ]:
            if not group:
                continue
            lons = [p["lon"] for p in group]
            lats = [p["lat"] for p in group]
            ax.scatter(lons, lats, s=12, color=colour, marker=marker,
                       linewidths=0.4, edgecolors="white", zorder=zorder)

        # Labels — only if <= 80 points to keep the plot readable
        if len(points) <= 80:
            for p in points:
                ax.annotate(
                    p["point_id"],
                    (p["lon"], p["lat"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=4, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"),
                )

        # Legend
        legend_handles = []
        if presence:
            legend_handles.append(mpatches.Patch(color=COLOUR_PRESENCE, label=f"Presence ({len(presence)})"))
        if absence:
            legend_handles.append(mpatches.Patch(color=COLOUR_ABSENCE,  label=f"Absence ({len(absence)})"))
        if unlabelled:
            legend_handles.append(mpatches.Patch(color=COLOUR_UNLABELLED, label=f"Unlabelled ({len(unlabelled)})"))
        if boxes:
            legend_handles.append(mpatches.Patch(edgecolor=COLOUR_BOX, facecolor="none", label=f"Boxes ({len(boxes)})"))
        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
                      framealpha=0.7, facecolor="black", labelcolor="white", edgecolor="none")

    # Fixed-metre grid (e.g. 10m Sentinel-2 pixel overlay)
    if grid_metres is not None:
        import math
        lat_centre = (lat_min + lat_max) / 2
        lon_step_m = grid_metres / (111320 * math.cos(math.radians(lat_centre)))
        lat_step_m = grid_metres / 111320

        lon_start = math.ceil(lon_min / lon_step_m) * lon_step_m
        lat_start = math.ceil(lat_min / lat_step_m) * lat_step_m

        grid_kw = dict(color="cyan", linewidth=0.4, alpha=0.5, linestyle="-")
        for x in np.arange(lon_start, lon_max, lon_step_m):
            ax.axvline(x, **grid_kw)
        for y in np.arange(lat_start, lat_max, lat_step_m):
            ax.axhline(y, **grid_kw)

    # Lon/lat reference grid
    if grid:
        import math
        lon_step = _nice_grid_spacing(lon_max - lon_min)
        lat_step = _nice_grid_spacing(lat_max - lat_min)
        lon_start = math.ceil(lon_min / lon_step) * lon_step
        lat_start = math.ceil(lat_min / lat_step) * lat_step
        grid_kw = dict(color="white", linewidth=0.6, alpha=0.55, linestyle="--")
        for x in np.arange(lon_start, lon_max, lon_step):
            ax.axvline(x, **grid_kw)
            ax.text(x, lat_max - (lat_max - lat_min) * 0.01, f"{x:.4f}",
                    ha="center", va="top", fontsize=5.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))
        for y in np.arange(lat_start, lat_max, lat_step):
            ax.axhline(y, **grid_kw)
            ax.text(lon_min + (lon_max - lon_min) * 0.005, y, f"{y:.4f}",
                    ha="left", va="center", fontsize=5.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5, edgecolor="none"))

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.set_title(
        f"Queensland Globe — {lon_min:.4f}–{lon_max:.4f}E, {abs(lat_min):.4f}–{abs(lat_max):.4f}S",
        fontsize=9,
    )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def parse_bbox(s: str) -> list[float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return parts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Plot bounding boxes and points over Queensland Globe satellite imagery."
    )
    parser.add_argument(
        "--bbox", type=parse_bbox, default=DEFAULT_BBOX,
        metavar="LON_MIN,LAT_MIN,LON_MAX,LAT_MAX",
        help=f"Scene extent in WGS84 (default: Mitchell River site {DEFAULT_BBOX})",
    )
    parser.add_argument(
        "--points-csv", metavar="PATH",
        help="CSV with columns point_id,lon,lat[,label] (label: 1=presence, 0=absence)",
    )
    parser.add_argument(
        "--boxes-csv", metavar="PATH",
        help="CSV with columns id,lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH, metavar="PX",
        help=f"Tile width in pixels (default: {DEFAULT_WIDTH}; max {MAX_TILE_PX})",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, metavar="PATH",
        help=f"Output PNG path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Overlay a labelled lon/lat reference grid",
    )
    parser.add_argument(
        "--grid-metres", type=float, metavar="M",
        help="Overlay a fixed-spacing grid at M metres (e.g. 10 for Sentinel-2 pixels)",
    )
    args = parser.parse_args()

    points = load_points(args.points_csv) if args.points_csv else []
    boxes  = load_boxes(args.boxes_csv)   if args.boxes_csv  else []

    img_array = fetch_wms_image(args.bbox, args.width)
    render(args.bbox, img_array, points, boxes, args.out,
           grid=args.grid, grid_metres=args.grid_metres)
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
