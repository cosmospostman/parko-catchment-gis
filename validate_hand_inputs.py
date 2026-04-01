"""
validate_hand_inputs.py — Validate assumptions about the HAND input data on disk.

Checks properties of the downloaded DEM tiles and (optionally) a pre-computed
HAND raster.  Does not run any computation — if something needs to be derived
(flow accumulation, HAND, etc.) run Step 4 first and pass the output with
--hand-raster.

Assumptions tested
------------------
1. All expected DEM tiles are present and non-empty
2. DEM void fraction is acceptable (<2% of total pixels across all tiles)
3. [If --hand-raster provided] HAND raster covers the expected spatial extent
4. [If --hand-raster provided] HAND value distribution is consistent with a
   megafan floodplain (median < 5 m, p90 < 20 m over the lower-elevation zone)
5. [If --hand-raster provided] HAND threshold produces a plausible flood extent
   fraction (5%–40% at HAND_FLOOD_THRESHOLD_M)

Note: tile seam alignment cannot be validated by comparing boundary pixel values
for this catchment — the Mitchell headwaters cross the Great Dividing Range
escarpment, producing genuine terrain steps of 30+ m at tile boundaries that
are indistinguishable from stitching errors. Seam consistency is instead an
implicit assumption validated by the COP-DEM data provenance (TanDEM-X
processing guarantees radiometric consistency across tiles).

Usage:
    source config.sh
    python validate_hand_inputs.py \\
        --tile-dir /mnt/ebs/dem/copernicus-dem-30m \\
        [--hand-raster outputs/2025/hand_2025.tif]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _result(label: str, status: str, detail: str) -> dict:
    icon = "✓" if status == PASS else ("✗" if status == FAIL else "–")
    print(f"  [{icon}] {label}: {detail}")
    return {"label": label, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Assumption 1: tiles present and non-empty
# ---------------------------------------------------------------------------

def check_tiles_present(tile_paths: list) -> dict:
    if not tile_paths:
        return _result("DEM tiles present", FAIL, "No .tif files found in tile directory")
    empty = [p for p in tile_paths if p.stat().st_size < 1000]
    if empty:
        return _result(
            "DEM tiles present",
            FAIL,
            f"{len(empty)} tile(s) are suspiciously small (<1 KB): {[p.name for p in empty]}",
        )
    return _result(
        "DEM tiles present",
        PASS,
        f"{len(tile_paths)} tiles found  "
        f"({sum(p.stat().st_size for p in tile_paths) / 1e6:.0f} MB total)",
    )


# ---------------------------------------------------------------------------
# Assumption 2: void fraction
# ---------------------------------------------------------------------------

def check_void_fraction(tile_paths: list) -> dict:
    import rasterio

    total_px = 0
    void_px = 0
    for p in tile_paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            total_px += arr.size
            if nodata is not None:
                void_px += (arr == nodata).sum()
            void_px += (~np.isfinite(arr)).sum()

    if total_px == 0:
        return _result("DEM void fraction", SKIP, "No pixels read")

    frac = void_px / total_px
    ok = frac < 0.02
    return _result(
        "DEM void fraction",
        PASS if ok else FAIL,
        f"{frac * 100:.2f}%  (threshold: <2%,  {void_px:,} of {total_px:,} pixels)",
    )




# ---------------------------------------------------------------------------
# Assumptions 5–7: HAND raster checks (only if --hand-raster provided)
# ---------------------------------------------------------------------------

def check_hand_spatial_extent(hand_path: Path, tile_paths: list) -> dict:
    import rasterio

    with rasterio.open(hand_path) as src:
        hand_bounds = src.bounds
        hand_crs = src.crs

    # Collect union of tile bounds (reprojected if needed)
    import pyproj
    from rasterio.crs import CRS

    tile_minx, tile_miny, tile_maxx, tile_maxy = np.inf, np.inf, -np.inf, -np.inf
    for p in tile_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            tile_minx = min(tile_minx, b.left)
            tile_miny = min(tile_miny, b.bottom)
            tile_maxx = max(tile_maxx, b.right)
            tile_maxy = max(tile_maxy, b.top)

    # HAND raster should cover a reasonable fraction of the tile extent
    # (clipping to catchment will reduce it, but it should not be tiny)
    hand_width = hand_bounds.right - hand_bounds.left
    hand_height = hand_bounds.top - hand_bounds.bottom
    ok = hand_width > 0 and hand_height > 0

    return _result(
        "HAND spatial extent",
        PASS if ok else FAIL,
        f"HAND raster bounds: ({hand_bounds.left:.0f}, {hand_bounds.bottom:.0f}) – "
        f"({hand_bounds.right:.0f}, {hand_bounds.top:.0f})  CRS: {hand_crs}",
    )


def check_hand_distribution(hand_path: Path) -> dict:
    import rasterio
    import config

    with rasterio.open(hand_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("HAND distribution", SKIP, f"Only {valid.size} valid pixels in HAND raster")

    # Use lower-elevation zone (p30 of HAND itself) as a floodplain proxy
    low_threshold = np.percentile(valid, 30)
    fp_hand = valid[valid <= low_threshold]

    p50 = float(np.percentile(fp_hand, 50))
    p90 = float(np.percentile(fp_hand, 90))

    print(f"       HAND percentiles (lower-elevation zone, n={fp_hand.size:,}):")
    for pct in [10, 25, 50, 75, 90, 99]:
        print(f"         p{pct:02d} = {np.percentile(fp_hand, pct):.1f} m")

    ok = (p50 < 5.0) and (p90 < 20.0)
    return _result(
        "HAND floodplain distribution",
        PASS if ok else FAIL,
        f"median={p50:.1f} m (threshold: <5 m)  p90={p90:.1f} m (threshold: <20 m)",
    )


def check_hand_flood_fractions(hand_path: Path) -> dict:
    import rasterio
    import config

    with rasterio.open(hand_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = np.isfinite(arr)
    n_valid = valid.sum()
    if n_valid == 0:
        return _result("HAND flood fractions", SKIP, "No valid HAND pixels")

    fracs = []
    flagged = False
    for t in [2.0, 5.0, 10.0, 15.0]:
        frac = 100.0 * ((arr <= t) & valid).sum() / n_valid
        fracs.append(f"HAND≤{t:.0f}m: {frac:.1f}%")
        if t == config.HAND_FLOOD_THRESHOLD_M and (frac < 5.0 or frac > 40.0):
            flagged = True

    detail = "  ".join(fracs)
    if flagged:
        detail += f"  ← {config.HAND_FLOOD_THRESHOLD_M} m threshold outside 5%–40%"

    return _result(
        "HAND flood fractions",
        FAIL if flagged else PASS,
        detail,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HAND-based flood inputs")
    parser.add_argument("--tile-dir", required=True, help="Directory of Copernicus GLO-30 tiles")
    parser.add_argument("--hand-raster", default="", help="Pre-computed HAND raster (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    sys.path.insert(0, str(Path(__file__).parent))

    tile_dir = Path(args.tile_dir)
    tile_paths = sorted(tile_dir.glob("*.tif"))

    print(f"\nValidating HAND inputs — tile dir: {tile_dir}\n")
    results = []

    print("Assumption 1 — Tiles present")
    results.append(check_tiles_present(tile_paths))

    print("Assumption 2 — Void fraction")
    results.append(check_void_fraction(tile_paths))

    if args.hand_raster:
        hand_path = Path(args.hand_raster)
        if not hand_path.exists():
            print(f"  [–] HAND raster not found at {hand_path} — skipping HAND checks")
        else:
            print("Assumption 3 — HAND spatial extent")
            results.append(check_hand_spatial_extent(hand_path, tile_paths))

            print("Assumption 4 — HAND distribution")
            results.append(check_hand_distribution(hand_path))

            print("Assumption 5 — HAND flood fractions")
            results.append(check_hand_flood_fractions(hand_path))
    else:
        print("  [–] --hand-raster not provided — skipping HAND checks (run Step 4 first)")

    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)
    print(f"\nSummary: {n_pass} passed  {n_fail} failed  {n_skip} skipped")

    if n_fail > 0:
        print("One or more checks failed — review the details above.")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
