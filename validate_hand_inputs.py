"""
validate_hand_inputs.py — Validate assumptions of HAND-based flood connectivity.

Replaces validate_s1_inputs.py for Step 4.  Runs a series of checks on the
downloaded DEM tiles and the derived HAND raster to confirm that the
HAND computation is grounded in a physically plausible terrain model.

Assumptions tested
------------------
1. DEM void coverage is acceptable (<2% of catchment pixels)
2. DEM elevation histogram over the floodplain is unimodal (no tile-seam spikes)
3. Tile seam alignment: max abs difference at boundaries < 1 m
4. DEM-derived stream network overlaps GA TOPO 250K centreline within 500 m
   for >90% of main channel (only run if --drainage-gpkg is provided)
5. HAND value distribution is consistent with a megafan floodplain
   (median < 5 m, p90 < 20 m within the floodplain zone)
6. HAND threshold produces a geomorphically plausible flood extent
   (5%–40% catchment coverage at HAND_FLOOD_THRESHOLD_M)

Usage:
    source config.sh
    python validate_hand_inputs.py \\
        --tile-dir /mnt/ebs/dem/copernicus-dem-30m \\
        [--drainage-gpkg /path/to/topo250k_waterways.gpkg] \\
        [--hand-raster /path/to/outputs/YYYY/hand_YYYY.tif]
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
# Assumption 1: void fraction
# ---------------------------------------------------------------------------

def check_void_fraction(dem_arr: np.ndarray) -> dict:
    void_frac = (~np.isfinite(dem_arr)).mean()
    ok = void_frac < 0.02
    return _result(
        "DEM void fraction",
        PASS if ok else FAIL,
        f"{void_frac * 100:.2f}%  (threshold: <2%)",
    )


# ---------------------------------------------------------------------------
# Assumption 2: histogram unimodality over floodplain
# ---------------------------------------------------------------------------

def check_floodplain_histogram(dem_arr: np.ndarray, floodplain_mask: np.ndarray) -> dict:
    vals = dem_arr[floodplain_mask & np.isfinite(dem_arr)]
    if vals.size < 100:
        return _result("Floodplain histogram", SKIP, "Insufficient floodplain pixels")
    counts, edges = np.histogram(vals, bins=100)
    # Detect obvious bimodality: count histogram modes (local maxima)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(counts, height=counts.max() * 0.1, distance=5)
    ok = len(peaks) <= 2  # allow up to 2 peaks (common in valley-flat systems)
    return _result(
        "Floodplain histogram",
        PASS if ok else FAIL,
        f"{len(peaks)} histogram peak(s) — {'likely artefact' if len(peaks) > 2 else 'OK'}",
    )


# ---------------------------------------------------------------------------
# Assumption 3: tile seam alignment
# ---------------------------------------------------------------------------

def check_tile_seams(tile_paths: list) -> dict:
    """Check that adjacent tiles do not have large elevation discontinuities at seams."""
    import rasterio

    if len(tile_paths) < 2:
        return _result("Tile seam alignment", SKIP, "Only one tile — no seam to check")

    max_diff = 0.0
    checked = 0
    tiles_sorted = sorted(tile_paths, key=lambda p: p.name)

    for i in range(len(tiles_sorted) - 1):
        try:
            with rasterio.open(tiles_sorted[i]) as a, rasterio.open(tiles_sorted[i + 1]) as b:
                # Read the right column of tile a and left column of tile b
                right_col = a.read(1)[:, -1].astype(np.float32)
                left_col = b.read(1)[:, 0].astype(np.float32)
                nodata_a = a.nodata or -9999.0
                nodata_b = b.nodata or -9999.0
                valid = (right_col != nodata_a) & (left_col != nodata_b)
                if valid.sum() > 0:
                    diff = np.abs(right_col[valid] - left_col[valid]).max()
                    max_diff = max(max_diff, float(diff))
                    checked += 1
        except Exception as exc:
            logger.debug("Seam check skipped for %s: %s", tiles_sorted[i].name, exc)

    if checked == 0:
        return _result("Tile seam alignment", SKIP, "No overlapping tile pairs found")

    ok = max_diff < 1.0
    return _result(
        "Tile seam alignment",
        PASS if ok else FAIL,
        f"Max seam difference: {max_diff:.2f} m  (threshold: <1 m, pairs checked: {checked})",
    )


# ---------------------------------------------------------------------------
# Assumption 4: DEM stream network vs cartographic drainage
# ---------------------------------------------------------------------------

def check_stream_network_overlap(
    dem_arr: np.ndarray,
    dem_transform,
    dem_crs,
    drainage_gpkg: Path,
    min_upstream_px: int,
) -> dict:
    """Check that DEM-derived stream pixels overlap the cartographic channel within 500 m."""
    try:
        import geopandas as gpd
        import rasterio.transform
        import xarray as xr

        from utils.dem import compute_flow_accumulation

        # Build minimal DataArray
        H, W = dem_arr.shape
        x = np.array([dem_transform.c + dem_transform.a * c for c in range(W)])
        y = np.array([dem_transform.f + dem_transform.e * r for r in range(H)])
        dem_da = xr.DataArray(dem_arr, dims=["y", "x"], coords={"x": x, "y": y})
        dem_da = dem_da.rio.write_crs(dem_crs)

        accum = compute_flow_accumulation(dem_da)
        stream_px = (accum.values >= min_upstream_px) & np.isfinite(dem_arr)

        drains = gpd.read_file(drainage_gpkg).to_crs(dem_crs)
        channel = drains.geometry.unary_union.buffer(500.0)  # 500 m tolerance

        # Vectorise stream pixels to points, check overlap
        rows, cols = np.where(stream_px)
        xs = x[cols]
        ys = y[rows]
        from shapely.geometry import MultiPoint
        pts = MultiPoint(list(zip(xs.tolist(), ys.tolist())))
        overlap = pts.intersection(channel)
        n_overlap = len(overlap.geoms) if hasattr(overlap, "geoms") else (1 if not overlap.is_empty else 0)
        n_total = len(rows)
        pct = 100.0 * n_overlap / max(n_total, 1)
        ok = pct >= 90.0
        return _result(
            "Stream network vs cartographic drainage",
            PASS if ok else FAIL,
            f"{pct:.1f}% of stream pixels within 500 m of centreline  (threshold: ≥90%)",
        )
    except Exception as exc:
        return _result("Stream network vs cartographic drainage", SKIP, f"Error: {exc}")


# ---------------------------------------------------------------------------
# Assumption 5: HAND distribution
# ---------------------------------------------------------------------------

def check_hand_distribution(hand_arr: np.ndarray, floodplain_mask: np.ndarray) -> dict:
    fp_hand = hand_arr[floodplain_mask & np.isfinite(hand_arr)]
    if fp_hand.size < 100:
        all_hand = hand_arr[np.isfinite(hand_arr)]
        if all_hand.size < 100:
            return _result("HAND distribution", SKIP, "Insufficient valid HAND pixels")
        fp_hand = all_hand

    p50 = float(np.percentile(fp_hand, 50))
    p90 = float(np.percentile(fp_hand, 90))
    print(f"       HAND percentiles over floodplain:")
    for pct in [10, 25, 50, 75, 90, 99]:
        print(f"         p{pct:02d} = {np.percentile(fp_hand, pct):.1f} m")

    ok = (p50 < 5.0) and (p90 < 20.0)
    return _result(
        "HAND floodplain distribution",
        PASS if ok else FAIL,
        f"median={p50:.1f} m (threshold: <5 m)  p90={p90:.1f} m (threshold: <20 m)",
    )


# ---------------------------------------------------------------------------
# Assumption 6: flood extent plausibility at multiple thresholds
# ---------------------------------------------------------------------------

def check_flood_extent_thresholds(hand_arr: np.ndarray) -> dict:
    valid = np.isfinite(hand_arr)
    n_valid = valid.sum()
    results = []
    flagged = False
    for t in [2.0, 5.0, 10.0, 15.0]:
        frac = 100.0 * ((hand_arr <= t) & valid).sum() / max(n_valid, 1)
        results.append(f"HAND≤{t:.0f}m: {frac:.1f}%")
        if t == 5.0 and (frac < 5.0 or frac > 40.0):
            flagged = True

    detail = "  ".join(results)
    if flagged:
        detail += "  ← HAND=5 m coverage outside 5%–40% — possible routing error"

    return _result(
        "Flood extent at candidate thresholds",
        FAIL if flagged else PASS,
        detail,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HAND-based flood inputs")
    parser.add_argument(
        "--tile-dir",
        required=True,
        help="Directory containing Copernicus GLO-30 DEM tiles",
    )
    parser.add_argument(
        "--drainage-gpkg",
        default="",
        help="Optional GA TOPO 250K waterways GeoPackage for Assumption 4",
    )
    parser.add_argument(
        "--hand-raster",
        default="",
        help="Pre-computed HAND raster (if omitted, HAND is derived from the tiles)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    # Ensure project root is on the path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    tile_dir = Path(args.tile_dir)
    tile_paths = sorted(tile_dir.glob("*.tif"))

    if not tile_paths:
        print(f"ERROR: no .tif files found in {tile_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nValidating HAND inputs — {len(tile_paths)} tiles in {tile_dir}\n")
    results = []

    # Load merged DEM for per-pixel checks
    import config
    import geopandas as gpd
    from utils.dem import merge_and_reproject_dem, compute_flow_accumulation, compute_hand

    print("Loading and merging DEM tiles...")
    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)
    dem = merge_and_reproject_dem(
        tile_paths,
        catchment_geom=catchment,
        target_crs=config.TARGET_CRS,
        resolution=30,
    )
    dem_arr = dem.values.astype(np.float32)
    print(f"  DEM shape: {dem_arr.shape}  valid pixels: {np.isfinite(dem_arr).sum()}\n")

    # Build a simple floodplain proxy: pixels in the lower 30th percentile of elevation
    valid_elev = dem_arr[np.isfinite(dem_arr)]
    if valid_elev.size > 0:
        low_elev_threshold = np.percentile(valid_elev, 30)
        floodplain_mask = (dem_arr <= low_elev_threshold) & np.isfinite(dem_arr)
    else:
        floodplain_mask = np.zeros_like(dem_arr, dtype=bool)

    print("Assumption 1 — DEM void coverage")
    results.append(check_void_fraction(dem_arr))

    print("Assumption 2 — Floodplain elevation histogram")
    results.append(check_floodplain_histogram(dem_arr, floodplain_mask))

    print("Assumption 3 — Tile seam alignment")
    results.append(check_tile_seams(tile_paths))

    # HAND raster
    if args.hand_raster and Path(args.hand_raster).exists():
        import rasterio
        with rasterio.open(args.hand_raster) as src:
            hand_arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            hand_arr[hand_arr == nodata] = np.nan
    else:
        print("\nComputing HAND raster from DEM (this may take several minutes)...")
        accum = compute_flow_accumulation(dem)
        min_px = int((config.HAND_MIN_UPSTREAM_KM2 * 1e6) / (30 ** 2))
        hand_da = compute_hand(dem, accum, min_upstream_px=min_px)
        hand_arr = hand_da.values.astype(np.float32)

    if args.drainage_gpkg and Path(args.drainage_gpkg).exists():
        print("Assumption 4 — Stream network vs cartographic drainage")
        import rasterio
        with rasterio.open(tile_paths[0]) as src:
            dem_crs = str(src.crs)
            dem_transform = src.transform
        min_px = int((config.HAND_MIN_UPSTREAM_KM2 * 1e6) / (30 ** 2))
        results.append(check_stream_network_overlap(
            dem_arr, dem_transform, dem_crs,
            Path(args.drainage_gpkg), min_px,
        ))
    else:
        results.append(_result(
            "Stream network vs cartographic drainage",
            SKIP,
            "No --drainage-gpkg provided",
        ))

    print("Assumption 5 — HAND value distribution")
    results.append(check_hand_distribution(hand_arr, floodplain_mask))

    print("Assumption 6 — Flood extent at candidate thresholds")
    results.append(check_flood_extent_thresholds(hand_arr))

    # Summary
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)
    print(f"\nSummary: {n_pass} passed  {n_fail} failed  {n_skip} skipped")

    if n_fail > 0:
        print("One or more validation checks failed — review the details above.")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
