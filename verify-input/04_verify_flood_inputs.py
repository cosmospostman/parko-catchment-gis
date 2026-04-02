"""
verify-input/04_verify_flood_inputs.py — Input checks for Stage 04 (flood connectivity).

Stage 04 builds a HAND (Height Above Nearest Drainage) raster from the
Copernicus GLO-30 DEM, optionally burns in a drainage network for flow
conditioning, then derives a flood connectivity mask.

On-disk inputs:
  • Copernicus GLO-30 DEM tiles  (required)
  • Catchment boundary GeoJSON   (required)
  • GA TOPO 250K drainage network GeoPackage  (optional — enables stream burning)
  • Pre-computed HAND raster     (optional — pass with --hand-raster to validate)

Sections
--------
1. Input files    — DEM tiles present and non-empty; catchment GeoJSON valid;
                    drainage network valid (if present)
2. Scientific     — DEM void fraction <2%; HAND distribution consistent with a
                    megafan floodplain; flood fraction at threshold in [5%, 40%]

Usage:
    source config.sh
    python verify-input/04_verify_flood_inputs.py \\
        --tile-dir /mnt/ebs/dem/copernicus-dem-30m \\
        [--hand-raster outputs/2025/hand_2025.tif]

The tile directory can also be set via the DEM_TILE_DIR environment variable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()

_RESET  = "\033[0m"  if _USE_COLOUR else ""
_BOLD   = "\033[1m"  if _USE_COLOUR else ""
_GREEN  = "\033[32m" if _USE_COLOUR else ""
_RED    = "\033[31m" if _USE_COLOUR else ""
_GREY   = "\033[90m" if _USE_COLOUR else ""

_ICON = {PASS: f"{_GREEN}✅{_RESET}", FAIL: f"{_RED}❌{_RESET}", SKIP: f"{_GREY}⏭{_RESET}"}


def _section(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")


def _result(label: str, status: str, detail: str) -> dict:
    icon = _ICON[status]
    print(f"  {icon}  {label}: {detail}")
    return {"label": label, "status": status, "detail": detail}


def _summary(file_results: list, sci_results: list) -> bool:
    fp = sum(1 for r in file_results if r["status"] == PASS)
    ff = sum(1 for r in file_results if r["status"] == FAIL)
    sp = sum(1 for r in sci_results  if r["status"] == PASS)
    sf = sum(1 for r in sci_results  if r["status"] == FAIL)
    ft = len(file_results) - sum(1 for r in file_results if r["status"] == SKIP)
    st = len(sci_results)  - sum(1 for r in sci_results  if r["status"] == SKIP)

    any_fail = ff > 0 or sf > 0
    colour = _RED if any_fail else _GREEN
    print(
        f"\n{colour}{_BOLD}Summary:{_RESET} "
        f"{fp}/{ft} file checks passed · {sp}/{st} scientific checks passed"
    )
    return not any_fail


# ---------------------------------------------------------------------------
# Section 1 — Input files
# ---------------------------------------------------------------------------

def check_tiles_present(tile_paths: list) -> dict:
    if not tile_paths:
        return _result("DEM tiles", FAIL, "No .tif files found in tile directory")
    empty = [p for p in tile_paths if p.stat().st_size < 1000]
    if empty:
        return _result(
            "DEM tiles", FAIL,
            f"{len(empty)} tile(s) suspiciously small (<1 KB): {[p.name for p in empty]}",
        )
    total_mb = sum(p.stat().st_size for p in tile_paths) / 1e6
    return _result("DEM tiles", PASS,
                   f"{len(tile_paths)} tiles  ({total_mb:.0f} MB total)")


def check_catchment_exists(catchment_path: Path) -> dict:
    if not catchment_path.exists():
        return _result("Catchment GeoJSON", FAIL, f"File not found: {catchment_path}")
    size_kb = catchment_path.stat().st_size / 1000
    if size_kb < 1:
        return _result("Catchment GeoJSON", FAIL, f"File suspiciously small (<1 KB)")
    return _result("Catchment GeoJSON", PASS, f"{catchment_path.name}  ({size_kb:.0f} KB)")


def check_catchment_valid(catchment_path: Path) -> dict:
    if not catchment_path.exists():
        return _result("Catchment geometry", SKIP, "File not found")
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(catchment_path))
    except Exception as exc:
        return _result("Catchment geometry", FAIL, f"Failed to read: {exc}")
    if len(gdf) == 0:
        return _result("Catchment geometry", FAIL, "No features in GeoJSON")
    n_invalid = int((~gdf.geometry.is_valid).sum())
    n_null    = int(gdf.geometry.isna().sum())
    issues = []
    if n_null:
        issues.append(f"{n_null} null geometries")
    if n_invalid:
        issues.append(f"{n_invalid} invalid geometries")
    if issues:
        return _result("Catchment geometry", FAIL, "; ".join(issues))
    return _result("Catchment geometry", PASS, f"{len(gdf)} feature(s)  CRS={gdf.crs}")


def check_drainage_network(drain_path: Path) -> dict:
    if not drain_path.exists():
        return _result(
            "Drainage network", SKIP,
            f"Not found at {drain_path} — stream burning will be skipped",
        )
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(drain_path))
    except Exception as exc:
        return _result("Drainage network", FAIL, f"Failed to read: {exc}")
    if len(gdf) == 0:
        return _result("Drainage network", FAIL, "File is empty — no drainage features")
    n_invalid = int((~gdf.geometry.is_valid).sum())
    total_km = float(gdf.to_crs("EPSG:7855").geometry.length.sum()) / 1000.0
    detail = f"{len(gdf)} features  CRS={gdf.crs}  ({total_km:,.0f} km total)"
    if n_invalid:
        detail += f"  ← {n_invalid} invalid geometries"
        return _result("Drainage network", FAIL, detail)
    return _result("Drainage network", PASS, detail)


# ---------------------------------------------------------------------------
# Section 2 — Scientific sanity checks
# ---------------------------------------------------------------------------

def check_void_fraction(tile_paths: list) -> dict:
    import rasterio

    total_px = 0
    void_px  = 0
    for p in tile_paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            total_px += arr.size
            if nodata is not None:
                void_px += int((arr == nodata).sum())
            void_px += int((~np.isfinite(arr)).sum())

    if total_px == 0:
        return _result("DEM void fraction", SKIP, "No pixels read")

    frac = void_px / total_px
    ok = frac < 0.02
    return _result(
        "DEM void fraction",
        PASS if ok else FAIL,
        f"{frac * 100:.2f}%  (threshold <2%,  {void_px:,}/{total_px:,} px)",
    )


def check_hand_spatial_extent(hand_path: Path) -> dict:
    import rasterio

    try:
        with rasterio.open(hand_path) as src:
            b = src.bounds
            crs = src.crs
    except Exception as exc:
        return _result("HAND spatial extent", FAIL, f"Cannot read: {exc}")

    ok = (b.right > b.left) and (b.top > b.bottom)
    return _result(
        "HAND spatial extent",
        PASS if ok else FAIL,
        f"({b.left:.0f}, {b.bottom:.0f}) – ({b.right:.0f}, {b.top:.0f})  CRS={crs}",
    )


def check_hand_distribution(hand_path: Path) -> dict:
    """Lower-elevation zone (p30 of all HAND values) should have median <5 m, p90 <20 m.

    On a megafan floodplain like the Mitchell, the majority of the catchment
    sits at low HAND values. If the lower-elevation zone has a high median HAND
    the DEM or flow-routing may be incorrect.
    """
    import rasterio

    try:
        with rasterio.open(hand_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result("HAND floodplain distribution", FAIL, f"Cannot read: {exc}")

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("HAND floodplain distribution", SKIP,
                       f"Only {valid.size} valid pixels")

    low_threshold = float(np.percentile(valid, 30))
    fp_hand = valid[valid <= low_threshold]
    p50 = float(np.percentile(fp_hand, 50))
    p90 = float(np.percentile(fp_hand, 90))

    ok = (p50 < 5.0) and (p90 < 20.0)
    return _result(
        "HAND floodplain distribution",
        PASS if ok else FAIL,
        f"lower-zone median={p50:.1f} m (threshold <5 m)  p90={p90:.1f} m (threshold <20 m)",
    )


def check_hand_flood_fractions(hand_path: Path, flood_threshold_m: float) -> dict:
    """Fraction of pixels at HAND ≤ flood_threshold_m should be 5%–40%.

    Too low → the threshold is too conservative, flood mask will be tiny.
    Too high → nearly everything is classified as flooded, likely a DEM issue.
    """
    import rasterio

    try:
        with rasterio.open(hand_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result("HAND flood fractions", FAIL, f"Cannot read: {exc}")

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid_mask = np.isfinite(arr)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return _result("HAND flood fractions", SKIP, "No valid HAND pixels")

    fracs = []
    threshold_frac = None
    for t in [2.0, 5.0, 10.0, 15.0]:
        frac = 100.0 * float(((arr <= t) & valid_mask).sum()) / n_valid
        fracs.append(f"HAND≤{t:.0f}m: {frac:.1f}%")
        if t == flood_threshold_m:
            threshold_frac = frac

    detail = "  ".join(fracs)
    if threshold_frac is not None and (threshold_frac < 5.0 or threshold_frac > 40.0):
        detail += f"  ← {flood_threshold_m} m threshold outside [5%, 40%]"
        return _result("HAND flood fractions", FAIL, detail)
    return _result("HAND flood fractions", PASS, detail)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Validate Stage 04 flood connectivity inputs"
    )
    default_tile_dir = (
        os.environ.get("LOCAL_DEM_ROOT")
        or os.environ.get("DEM_CACHE_DIR")
        or "/mnt/ebs/dem/copernicus-dem-30m"
    )
    parser.add_argument(
        "--tile-dir",
        default=default_tile_dir,
        help="Directory of Copernicus GLO-30 DEM tiles "
             "(defaults to LOCAL_DEM_ROOT → DEM_CACHE_DIR → /mnt/ebs/dem/copernicus-dem-30m)",
    )
    parser.add_argument(
        "--hand-raster", default="",
        help="Pre-computed HAND raster (defaults to config.hand_raster_path(YEAR))",
    )
    args = parser.parse_args()

    import config

    print(f"{_BOLD}Stage 04 — Flood connectivity input verification  (year: {config.YEAR}){_RESET}")

    tile_dir   = Path(args.tile_dir)
    tile_paths = sorted(tile_dir.glob("*.tif")) if tile_dir.is_dir() else []

    drain_path = Path(config.BASE_DIR) / "data" / "drainage_network.gpkg"

    # ── Section 1: Input files ───────────────────────────────────────────────
    _section("1. Input files")
    file_results = [
        check_tiles_present(tile_paths),
        check_catchment_exists(config.CATCHMENT_GEOJSON),
        check_catchment_valid(config.CATCHMENT_GEOJSON),
        check_drainage_network(drain_path),
    ]

    # ── Section 2: Scientific sanity checks ──────────────────────────────────
    _section("2. Scientific sanity checks")
    sci_results = []

    if tile_paths:
        sci_results.append(check_void_fraction(tile_paths))
    else:
        sci_results.append(_result("DEM void fraction", SKIP, "No tiles to check"))

    hand_path = Path(args.hand_raster) if args.hand_raster else config.hand_raster_path(config.YEAR)
    if not hand_path.exists():
        sci_results.append(_result(
            "HAND checks", SKIP,
            f"HAND raster not found at {hand_path} — run Stage 04 first",
        ))
    else:
        sci_results.append(check_hand_spatial_extent(hand_path))
        sci_results.append(check_hand_distribution(hand_path))
        sci_results.append(check_hand_flood_fractions(hand_path, config.HAND_FLOOD_THRESHOLD_M))

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
