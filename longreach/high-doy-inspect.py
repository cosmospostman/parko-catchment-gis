#!/usr/bin/env python3
"""
longreach/high-doy-inspect.py

Produce pixel-inspect imagery for all pixels with mean annual peak DOY > 180,
bucketed into 10-percentile bands of Parkinsonia probability (prob_lr).

Outputs
-------
outputs/high-doy/p10/   — bottom 0–10th percentile of prob_lr within DOY>180
outputs/high-doy/p20/   — 10–20th percentile
...
outputs/high-doy/p100/  — top 90–100th percentile

Within each directory, files are named:
    doy{DOY:.0f}_prob{prob:.3f}_{point_id}.png
sorted by descending peak_doy so the most extreme late-peakers appear first.

Options
-------
--sample N    Randomly sample up to N pixels per bucket (default: all pixels).
              Sampling is reproducible via a fixed seed.
--workers N   Parallel render workers (default: 4). Each worker receives a
              pre-fetched full-resolution tile for its pixel, so WMS fetches
              are deduplicated and rendering is parallelised independently.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GREEN_STATS_CACHE = (
    PROJECT_ROOT
    / "research" / "recession-and-greenup"
    / "longreach-recession-greenup"
    / "_cache_green_stats.parquet"
)
RANKING_CSV = (
    PROJECT_ROOT
    / "outputs" / "longreach-8x8"
    / "longreach_8x8km_pixel_ranking.csv"
)
OUT_ROOT = PROJECT_ROOT / "outputs" / "high-doy"

DOY_THRESHOLD = 180


# ---------------------------------------------------------------------------
# Load pixel-inspect module (hyphenated filename)
# ---------------------------------------------------------------------------

def _load_pixel_inspect():
    spec = importlib.util.spec_from_file_location(
        "pixel_inspect", PROJECT_ROOT / "utils" / "pixel-inspect.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Worker: module loaded once per process via initializer
# ---------------------------------------------------------------------------

_worker_pixel_inspect = None


def _worker_init() -> None:
    global _worker_pixel_inspect
    _worker_pixel_inspect = _load_pixel_inspect()


def _render_one(r: dict) -> str:
    """Render a single pixel-inspect plot in a worker process.

    r["tile"] is the pre-fetched (H, W, 3) uint8 image array for this pixel's
    bbox, fetched at full resolution in the main process. We wrap it in a
    minimal SceneTileCache so inspect_pixel can crop without a WMS call.
    """
    pi = _worker_pixel_inspect
    qg = pi._load_qglobe()

    tile: np.ndarray | None = r.get("tile")
    bbox: list[float] | None = r.get("bbox")

    scene_cache = None
    if tile is not None and bbox is not None:
        scene_cache = qg.SceneTileCache()
        scene_cache._img = tile
        lon_min, lat_min, lon_max, lat_max = bbox
        scene_cache._lon_min = lon_min
        scene_cache._lat_min = lat_min
        scene_cache._lon_max = lon_max
        scene_cache._lat_max = lat_max

    title = (
        f"{r['pid']}  |  peak_doy={r['doy']:.1f}  |  prob_lr={r['prob_val']:.3f}\n"
        f"{r['lon']:.5f}E  {abs(r['lat']):.5f}S  [{r['bucket']}]"
    )
    pi.inspect_pixel(
        lon=r["lon"],
        lat=r["lat"],
        out_path=Path(r["out_path"]),
        score=r["prob_val"],
        score_label="prob_lr",
        title=title,
        scene_cache=scene_cache,
    )
    return r["out_path"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Max pixels per bucket (random sample, reproducible). Default: all."
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel render workers (default: 4)."
    )
    args = parser.parse_args()

    # -- Load data -----------------------------------------------------------
    print("Loading green_stats cache …")
    green = pl.read_parquet(GREEN_STATS_CACHE).select(
        ["point_id", "lon", "lat", "peak_doy"]
    )

    print("Loading pixel ranking …")
    ranking = pl.read_csv(RANKING_CSV).select(["point_id", "prob_lr"])

    # -- Filter and join -----------------------------------------------------
    high_doy = (
        green
        .filter(pl.col("peak_doy") > DOY_THRESHOLD)
        .join(ranking, on="point_id", how="inner")
        .drop_nulls("prob_lr")
        .sort("peak_doy", descending=True)
    )

    n_unfiltered = len(high_doy)
    print(f"Pixels with peak_doy > {DOY_THRESHOLD}: {n_unfiltered}")

    if n_unfiltered == 0:
        print("No pixels found — check cache path or DOY threshold.")
        sys.exit(1)

    # -- Assign probability buckets (10 bands based on absolute prob_lr) -----
    # p010 = prob_lr in [0.0, 0.1), p020 = [0.1, 0.2), … p100 = [0.9, 1.0]
    high_doy = high_doy.with_columns(
        (
            (pl.col("prob_lr") * 10).floor().cast(pl.Int32).clip(0, 9) * 10 + 10
        )
        .map_elements(lambda b: f"p{b:03d}", return_dtype=pl.Utf8)
        .alias("bucket")
    )

    # -- Optional per-bucket sampling ----------------------------------------
    if args.sample is not None:
        rng = np.random.default_rng(42)
        sampled_frames = []
        for bucket_name, group in high_doy.group_by("bucket"):
            if len(group) <= args.sample:
                sampled_frames.append(group)
            else:
                idx = rng.choice(len(group), size=args.sample, replace=False)
                sampled_frames.append(group[idx])
        high_doy = pl.concat(sampled_frames).sort("bucket")
        print(f"Sampled up to {args.sample} pixels per bucket.")

    # Summary
    counts = high_doy.group_by("bucket").len().sort("bucket")
    print("\nPixels per percentile bucket:")
    for row in counts.iter_rows():
        print(f"  {row[0]}: {row[1]}")
    print()

    n_total = len(high_doy)

    # -- Load pixel-inspect in main process (for bbox computation) -----------
    pixel_inspect = _load_pixel_inspect()
    qg = pixel_inspect._load_qglobe()

    # -- Build render list, skipping already-done outputs --------------------
    rows_to_render = []
    for row in high_doy.iter_rows(named=True):
        bucket   = row["bucket"]
        doy      = row["peak_doy"]
        prob_val = row["prob_lr"]
        pid      = row["point_id"]
        lon      = row["lon"]
        lat      = row["lat"]

        out_dir = OUT_ROOT / bucket
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_pid = pid.replace("/", "_").replace("\\", "_")
        fname = f"doy{doy:.0f}_prob{prob_val:.3f}_{safe_pid}.png"
        out_path = out_dir / fname

        if out_path.exists():
            continue  # resumable

        rows_to_render.append(dict(
            bucket=bucket, doy=doy, prob_val=prob_val,
            pid=pid, lon=lon, lat=lat,
            out_path=str(out_path),
        ))

    n_skip = n_total - len(rows_to_render)
    if n_skip:
        print(f"Skipping {n_skip} already-rendered pixels.")

    if not rows_to_render:
        print("Nothing to render.")
        return

    # -- Pre-fetch WMS tiles at full resolution, deduplicated by snapped centre
    #
    # Each pixel's plot covers a fixed 30×30 m bbox (3×3 S2 pixels).
    # WMS_WIDTH_DEFAULT (600 px over ~30 m) gives ~5 cm/px — full GSD.
    # Many input points may snap to the same S2 pixel centre; we fetch each
    # unique bbox only once and reuse the array across tasks.
    #
    # This is fundamentally different from SceneTileCache.prefetch(), which
    # fetches one giant tile covering the entire study area at MAX_TILE_PX —
    # that dilutes resolution severely when pixels are spread across km.
    # --------------------------------------------------------------------------
    wms_width = pixel_inspect.WMS_WIDTH_DEFAULT

    print(f"Pre-fetching WMS tiles for {len(rows_to_render)} pixels "
          f"(deduplicating by snapped S2 centre) …")

    # centre_key → (bbox, tile array)
    tile_cache: dict[tuple[int, int, int], tuple[list[float], np.ndarray]] = {}

    for i, r in enumerate(rows_to_render):
        epsg = pixel_inspect._utm_epsg(r["lon"], r["lat"])
        e, n = pixel_inspect._wgs84_to_utm(r["lon"], r["lat"], epsg)
        ce, cn = pixel_inspect._snap_to_s2_grid(e, n)
        key = (epsg, int(ce), int(cn))

        if key not in tile_cache:
            grid = pixel_inspect._pixel_grid_wgs84(ce, cn, epsg)
            bbox = grid["bbox"]
            try:
                tile = qg.fetch_wms_image(bbox, width_px=wms_width)
            except Exception as exc:
                print(f"  Warning: WMS fetch failed for {r['pid']} ({exc}) — will render dark background.")
                tile = None
            tile_cache[key] = (bbox, tile)

        bbox, tile = tile_cache[key]
        r["bbox"] = bbox
        r["tile"] = tile  # None on fetch failure → worker falls back to dark bg

        fetched = len(tile_cache)
        if fetched % 50 == 0 or i == len(rows_to_render) - 1:
            print(f"  … {fetched} unique tiles fetched ({i + 1}/{len(rows_to_render)} pixels)")

    unique_fetches = len(tile_cache)
    deduped = len(rows_to_render) - unique_fetches
    print(f"WMS fetches: {unique_fetches} unique tiles "
          f"({deduped} pixels reused a tile from the same S2 centre).\n")

    # -- Parallel render with ProcessPoolExecutor ----------------------------
    n_work = len(rows_to_render)
    total_rendered = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_render_one, r): r for r in rows_to_render}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                r = futures[fut]
                print(f"  ERROR {Path(r['out_path']).name}: {exc}")
                errors += 1
                continue
            total_rendered += 1
            if total_rendered % 50 == 0:
                print(f"  … {total_rendered}/{n_work} rendered")

    print(f"\nDone — {total_rendered} new plots written to {OUT_ROOT}", end="")
    if errors:
        print(f"  ({errors} errors)", end="")
    print()


if __name__ == "__main__":
    main()
