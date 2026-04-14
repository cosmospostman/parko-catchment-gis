#!/usr/bin/env python3
"""
utils/pixel-inspect — inspect individual S2 pixels against Queensland Globe imagery.

For each target pixel, produces a 30×30 m plot (3×3 S2 pixel grid, target plus
one-pixel surround) with QG aerial imagery underlay and the S2 pixel grid overlaid.

Pipeline API
------------
from utils import pixel_inspect  # via importlib — see heatmap.py pattern

    inspect_pixel(lon, lat, out_path, score=0.97)
    inspect_top_n(score_da, out_dir, n=20)
    inspect_threshold(score_da, out_dir, percentile=95.0, max_plots=50)

CLI
---
    # Single pixel
    python utils/pixel-inspect.py --lon 141.571 --lat -14.652 --out out/pixel.png

    # Top-N from a raster
    python utils/pixel-inspect.py --raster scores.tif --top-n 20 --out-dir out/inspect/

    # Percentile threshold from a raster
    python utils/pixel-inspect.py --raster scores.tif --percentile 95 --max-plots 50 --out-dir out/inspect/
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# S2 pixel size in metres
S2_PX_M = 10.0

# Half-width of a 3×3 block in metres: 1.5 pixels each side
HALF_BLOCK_M = 1.5 * S2_PX_M  # 15 m

# WMS tile width for a ~30 m scene at 20 cm GSD → 150 QG pixels; request 4× for sharpness
WMS_WIDTH_DEFAULT = 600


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_qglobe():
    """Load qglobe-plot module (hyphenated filename requires importlib).

    Cached at module level so repeated calls within a session reuse the same
    module instance — and therefore the same _WMS_CACHE dict.
    """
    if _load_qglobe._mod is None:
        spec = importlib.util.spec_from_file_location(
            "qglobe_plot", PROJECT_ROOT / "utils" / "qglobe-plot.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_qglobe._mod = mod
    return _load_qglobe._mod

_load_qglobe._mod = None


def _utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code for the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _snap_to_s2_grid(easting: float, northing: float) -> tuple[float, float]:
    """Snap a UTM coordinate to the nearest S2 10 m pixel centre.

    S2 pixels are aligned to 10 m intervals from the UTM zone origin, with
    pixel *centres* at multiples of 10 m (i.e. 5, 15, 25, … m from origin).
    Snapping: round to nearest 10, then the centre is at that value + 0 m
    (S2 actually tiles from the origin at 0, 10, 20, … so pixel centres are
    at 5, 15, 25, … but for display purposes snapping to the nearest 10 m
    boundary is sufficient).
    """
    snapped_e = round(easting / S2_PX_M) * S2_PX_M
    snapped_n = round(northing / S2_PX_M) * S2_PX_M
    return snapped_e, snapped_n


def _utm_to_wgs84(easting: float, northing: float, epsg: int) -> tuple[float, float]:
    """Convert UTM easting/northing to WGS84 lon/lat."""
    from pyproj import Transformer
    tf = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(easting, northing)
    return lon, lat


def _wgs84_to_utm(lon: float, lat: float, epsg: int) -> tuple[float, float]:
    """Convert WGS84 lon/lat to UTM easting/northing."""
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = tf.transform(lon, lat)
    return easting, northing


def _pixel_grid_wgs84(
    centre_e: float, centre_n: float, epsg: int
) -> dict:
    """Return the 3×3 pixel grid corners and centres in WGS84.

    Returns a dict with:
      bbox       : [lon_min, lat_min, lon_max, lat_max] of the 30×30 m block
      cell_edges : list of (lon, lat) corner pairs for drawing grid lines
      centres    : list of (lon, lat, row, col) for the 9 pixel centres
                   row/col in 0..2, (1,1) is the target pixel
    """
    from pyproj import Transformer
    tf = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # Block extent in UTM
    blk_min_e = centre_e - HALF_BLOCK_M
    blk_max_e = centre_e + HALF_BLOCK_M
    blk_min_n = centre_n - HALF_BLOCK_M
    blk_max_n = centre_n + HALF_BLOCK_M

    corners = [
        tf.transform(blk_min_e, blk_min_n),
        tf.transform(blk_max_e, blk_min_n),
        tf.transform(blk_max_e, blk_max_n),
        tf.transform(blk_min_e, blk_max_n),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    # Grid lines: vertical and horizontal edges at every 10 m step
    h_lines = []
    for dn in np.arange(-HALF_BLOCK_M, HALF_BLOCK_M + 1, S2_PX_M):
        n = centre_n + dn
        p0 = tf.transform(blk_min_e, n)
        p1 = tf.transform(blk_max_e, n)
        h_lines.append((p0, p1))

    v_lines = []
    for de in np.arange(-HALF_BLOCK_M, HALF_BLOCK_M + 1, S2_PX_M):
        e = centre_e + de
        p0 = tf.transform(e, blk_min_n)
        p1 = tf.transform(e, blk_max_n)
        v_lines.append((p0, p1))

    # Pixel centres for the 3×3 block
    centres = []
    for row, dn in enumerate([-S2_PX_M, 0, S2_PX_M]):
        for col, de in enumerate([-S2_PX_M, 0, S2_PX_M]):
            lon, lat = tf.transform(centre_e + de, centre_n + dn)
            centres.append((lon, lat, row, col))

    return dict(bbox=bbox, h_lines=h_lines, v_lines=v_lines, centres=centres)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_pixel(
    lon: float,
    lat: float,
    out_path: Path | str,
    score: float | None = None,
    score_label: str = "score",
    wms_width: int = WMS_WIDTH_DEFAULT,
    title: str | None = None,
    scene_cache=None,
) -> Path:
    """Render a 30×30 m QG imagery plot centred on the S2 pixel containing (lon, lat).

    The target pixel is highlighted with a yellow border; surrounding pixels
    have a cyan grid. If ``score`` is provided it is annotated in the centre
    cell.

    Parameters
    ----------
    lon, lat     : WGS84 target coordinate
    out_path     : output PNG path
    score        : optional scalar value to annotate in the target cell
    score_label  : label for the score annotation (default "score")
    wms_width    : WMS tile width in pixels (ignored when scene_cache is provided)
    title        : optional plot title override
    scene_cache  : optional SceneTileCache — if provided, imagery is cropped
                   from the pre-fetched scene tile instead of making a WMS call

    Returns
    -------
    Path to the saved PNG.
    """
    out_path = Path(out_path)

    epsg = _utm_epsg(lon, lat)
    easting, northing = _wgs84_to_utm(lon, lat, epsg)
    centre_e, centre_n = _snap_to_s2_grid(easting, northing)

    grid = _pixel_grid_wgs84(centre_e, centre_n, epsg)
    bbox = grid["bbox"]

    qg = _load_qglobe()
    try:
        if scene_cache is not None:
            img = scene_cache.crop(bbox)
        else:
            img = qg.fetch_wms_image(bbox, width_px=wms_width)
    except Exception as exc:
        logger.warning("WMS fetch failed (%s) — using dark background", exc)
        img = None

    lon_min, lat_min, lon_max, lat_max = bbox

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    if img is not None:
        ax.imshow(
            img,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            aspect="auto",
            interpolation="bilinear",
        )
    else:
        ax.set_facecolor("#111111")

    # Draw S2 grid lines
    grid_kw = dict(color="cyan", linewidth=0.8, alpha=0.8, linestyle="-")
    for (lon0, lat0), (lon1, lat1) in grid["h_lines"]:
        ax.plot([lon0, lon1], [lat0, lat1], **grid_kw)
    for (lon0, lat0), (lon1, lat1) in grid["v_lines"]:
        ax.plot([lon0, lon1], [lat0, lat1], **grid_kw)

    # Highlight the target pixel with a yellow border
    # Find target pixel corners: centre ± half S2 pixel in UTM → WGS84
    from pyproj import Transformer
    tf = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    half = S2_PX_M / 2
    tgt_corners = [
        tf.transform(centre_e - half, centre_n - half),
        tf.transform(centre_e + half, centre_n - half),
        tf.transform(centre_e + half, centre_n + half),
        tf.transform(centre_e - half, centre_n + half),
    ]
    tgt_lons = [c[0] for c in tgt_corners]
    tgt_lats = [c[1] for c in tgt_corners]
    tgt_lons.append(tgt_lons[0])
    tgt_lats.append(tgt_lats[0])
    ax.plot(tgt_lons, tgt_lats, color="yellow", linewidth=2.0, zorder=5)

    # Annotate score in the target cell
    if score is not None:
        tgt_lon, tgt_lat = tf.transform(centre_e, centre_n)
        ax.text(
            tgt_lon, tgt_lat,
            f"{score_label}\n{score:.3f}",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"),
            zorder=6,
        )

    # Axes formatting
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    _title = title or (
        f"S2 pixel @ {lon:.5f}E, {abs(lat):.5f}S\n"
        f"UTM {epsg} ({centre_e:.0f}E, {centre_n:.0f}N)  |  3×3 pixel context"
    )
    ax.set_title(_title, fontsize=8)

    # Legend
    legend_handles = [
        mpatches.Patch(edgecolor="yellow", facecolor="none", linewidth=1.5, label="Target pixel"),
        mpatches.Patch(edgecolor="cyan",   facecolor="none", linewidth=0.8, label="S2 10 m grid"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6,
              framealpha=0.7, facecolor="black", labelcolor="white", edgecolor="none")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def inspect_top_n(
    score_da,
    out_dir: Path | str,
    n: int = 20,
    score_label: str = "score",
    stem: str = "pixel",
    wms_width: int = WMS_WIDTH_DEFAULT,
) -> list[Path]:
    """Render pixel-inspect plots for the top-N scoring pixels in an xarray DataArray.

    The DataArray must have spatial coordinates. Supports:
      - 2-D (y, x) arrays in a projected CRS (UTM) with ``crs`` in ``attrs``
      - 2-D arrays with ``lon``/``lat`` coordinate variables

    Parameters
    ----------
    score_da    : xarray.DataArray of scores (higher = more interesting)
    out_dir     : directory for output PNGs
    n           : number of top pixels to plot
    score_label : label for score annotations
    stem        : filename prefix (files named <stem>_rank001.png etc.)
    wms_width   : WMS tile width per plot

    Returns
    -------
    List of Paths written (one per pixel, length ≤ n).
    """
    lons, lats, scores = _extract_pixel_coords(score_da)
    order = np.argsort(scores)[::-1]
    top_idx = order[:n]
    return _render_pixels(lons, lats, scores, top_idx, out_dir, stem, score_label, wms_width)


def inspect_threshold(
    score_da,
    out_dir: Path | str,
    percentile: float = 95.0,
    max_plots: int = 50,
    score_label: str = "score",
    stem: str = "pixel",
    wms_width: int = WMS_WIDTH_DEFAULT,
) -> list[Path]:
    """Render pixel-inspect plots for pixels above a percentile threshold.

    Parameters
    ----------
    score_da    : xarray.DataArray of scores
    out_dir     : directory for output PNGs
    percentile  : score percentile threshold (default 95 → top 5%)
    max_plots   : cap on number of plots (ranked by score, best first)
    score_label : label for score annotations
    stem        : filename prefix
    wms_width   : WMS tile width per plot

    Returns
    -------
    List of Paths written.
    """
    lons, lats, scores = _extract_pixel_coords(score_da)
    threshold = np.nanpercentile(scores, percentile)
    above = np.where(scores >= threshold)[0]
    order = above[np.argsort(scores[above])[::-1]]
    top_idx = order[:max_plots]
    logger.info(
        "Threshold p%.0f = %.4f — %d pixels above threshold, plotting %d",
        percentile, threshold, len(above), len(top_idx),
    )
    return _render_pixels(lons, lats, scores, top_idx, out_dir, stem, score_label, wms_width)


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------

def _extract_pixel_coords(score_da) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (lons, lats, scores) flat arrays from an xarray DataArray.

    Handles both:
      - Projected arrays (UTM CRS in attrs) with x/y dimension coordinates
      - Arrays with lon/lat coordinate variables directly
    """
    import xarray as xr

    da = score_da
    vals = da.values.ravel()

    # Try explicit lon/lat coords first
    if "lon" in da.coords and "lat" in da.coords:
        lons = da.coords["lon"].values.ravel()
        lats = da.coords["lat"].values.ravel()
    elif hasattr(da, "rio") and da.rio.crs is not None:
        # rioxarray projected array — reproject pixel centres to WGS84
        from pyproj import Transformer
        crs = da.rio.crs
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = np.meshgrid(da.x.values, da.y.values)
        lons_2d, lats_2d = tf.transform(xs, ys)
        lons = lons_2d.ravel()
        lats = lats_2d.ravel()
    else:
        raise ValueError(
            "Cannot determine pixel coordinates from DataArray. "
            "Provide an array with 'lon'/'lat' coords or a rioxarray CRS."
        )

    mask = np.isfinite(vals)
    return lons[mask], lats[mask], vals[mask]


def _render_pixels(
    lons: np.ndarray,
    lats: np.ndarray,
    scores: np.ndarray,
    idx: np.ndarray,
    out_dir: Path | str,
    stem: str,
    score_label: str,
    wms_width: int,
) -> list[Path]:
    """Shared rendering loop used by inspect_top_n and inspect_threshold.

    Prefetches a single WMS scene tile covering all target pixels at MAX_TILE_PX,
    then crops per-pixel from that tile — one WMS request regardless of pixel count.
    """
    out_dir = Path(out_dir)

    # Build scene cache: accumulate all pixel bboxes then fetch once
    qg = _load_qglobe()
    cache = qg.SceneTileCache()
    for i in idx:
        epsg = _utm_epsg(lons[i], lats[i])
        e, n = _wgs84_to_utm(lons[i], lats[i], epsg)
        ce, cn = _snap_to_s2_grid(e, n)
        grid = _pixel_grid_wgs84(ce, cn, epsg)
        cache.expand(grid["bbox"])

    try:
        cache.prefetch()
    except Exception as exc:
        logger.warning("Scene prefetch failed (%s) — falling back to per-pixel fetches", exc)
        cache = None

    paths = []
    for rank, i in enumerate(idx, start=1):
        out_path = out_dir / f"{stem}_rank{rank:03d}.png"
        title = f"Rank {rank}  |  {score_label}={scores[i]:.4f}  |  {lons[i]:.5f}E {abs(lats[i]):.5f}S"
        p = inspect_pixel(
            lon=lons[i],
            lat=lats[i],
            out_path=out_path,
            score=scores[i],
            score_label=score_label,
            wms_width=wms_width,
            title=title,
            scene_cache=cache,
        )
        paths.append(p)
        print(f"  [{rank:>3}/{len(idx)}] {p.name}  ({score_label}={scores[i]:.4f})")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect S2 pixels against Queensland Globe imagery."
    )

    coord_grp = parser.add_argument_group("single pixel")
    coord_grp.add_argument("--lon", type=float, metavar="DEG", help="Target longitude (WGS84)")
    coord_grp.add_argument("--lat", type=float, metavar="DEG", help="Target latitude (WGS84)")
    coord_grp.add_argument("--out", metavar="PATH", help="Output PNG path (single pixel mode)")

    raster_grp = parser.add_argument_group("raster mode")
    raster_grp.add_argument("--raster", metavar="PATH", help="Input raster (GeoTIFF) of scores")
    raster_grp.add_argument("--band", type=int, default=1, metavar="N", help="Band index (default: 1)")
    raster_grp.add_argument("--top-n", type=int, metavar="N", help="Plot the top-N pixels by score")
    raster_grp.add_argument(
        "--percentile", type=float, metavar="P",
        help="Plot pixels above the P-th percentile (e.g. 95 for top 5%%)",
    )
    raster_grp.add_argument(
        "--max-plots", type=int, default=50, metavar="N",
        help="Cap on number of plots in percentile mode (default: 50)",
    )
    raster_grp.add_argument("--out-dir", metavar="PATH", help="Output directory for raster mode")
    raster_grp.add_argument(
        "--stem", default="pixel", metavar="STR",
        help="Filename prefix for raster mode (default: pixel)",
    )

    parser.add_argument(
        "--score-label", default="score", metavar="STR",
        help="Label shown in score annotation (default: score)",
    )
    parser.add_argument(
        "--wms-width", type=int, default=WMS_WIDTH_DEFAULT, metavar="PX",
        help=f"WMS tile width per plot (default: {WMS_WIDTH_DEFAULT})",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    args = _parse_args(argv)

    # ---- Single pixel mode ------------------------------------------------
    if args.lon is not None and args.lat is not None:
        if not args.out:
            args.out = f"pixel_{args.lon:.5f}_{args.lat:.5f}.png"
        p = inspect_pixel(
            lon=args.lon,
            lat=args.lat,
            out_path=Path(args.out),
            score_label=args.score_label,
            wms_width=args.wms_width,
        )
        print(f"Done → {p}")
        return

    # ---- Raster mode ------------------------------------------------------
    if args.raster is None:
        print("ERROR: provide either --lon/--lat or --raster", file=sys.stderr)
        sys.exit(1)
    if args.out_dir is None:
        print("ERROR: --out-dir required in raster mode", file=sys.stderr)
        sys.exit(1)
    if args.top_n is None and args.percentile is None:
        print("ERROR: provide --top-n or --percentile in raster mode", file=sys.stderr)
        sys.exit(1)

    import rioxarray  # noqa: F401 — registers .rio accessor
    import xarray as xr

    da = xr.open_dataarray(args.raster, engine="rasterio").sel(band=args.band).drop_vars("band")

    out_dir = Path(args.out_dir)
    kw = dict(
        score_label=args.score_label,
        stem=args.stem,
        wms_width=args.wms_width,
    )

    if args.top_n is not None:
        paths = inspect_top_n(da, out_dir, n=args.top_n, **kw)
    else:
        paths = inspect_threshold(da, out_dir, percentile=args.percentile, max_plots=args.max_plots, **kw)

    print(f"Done — {len(paths)} plot(s) written to {out_dir}")


if __name__ == "__main__":
    main()
