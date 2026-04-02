"""
verify-input/06_verify_classifier_inputs.py — Input checks for Stage 05 (classifier).

Stage 05 trains a Random Forest classifier using raster feature layers from
Stages 01–03 and ALA occurrence records. This script verifies all inputs
before Stage 05 runs.

Note: The drainage network (used to compute dist_to_watercourse) is an input
to Stage 04, not Stage 05 — validate it with 04_verify_flood_inputs.py.

Sections
--------
1. Input files    — rasters (ndvi_median, ndvi_anomaly, flowering_index) and
                    ALA cache present and non-empty
2. Scientific     — CRS, spatial consistency, NaN fraction, value distributions,
                    and independence between feature layers

Usage:
    source config.sh
    python verify-input/06_verify_classifier_inputs.py [--no-ala-check]
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

def check_raster_present(label: str, path: Path) -> dict:
    if not path.exists():
        return _result(label, FAIL, f"File not found: {path}")
    size_mb = path.stat().st_size / 1e6
    if size_mb < 0.001:
        return _result(label, FAIL, f"File suspiciously small (<1 KB): {path}")
    return _result(label, PASS, f"{path.name}  ({size_mb:.1f} MB)")


def check_ala_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return _result(
            "ALA occurrence cache", SKIP,
            f"Cache not found at {cache_path} — will be fetched at runtime",
        )
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(cache_path))
    except Exception as exc:
        return _result("ALA occurrence cache", FAIL, f"Failed to read: {exc}")

    if len(gdf) == 0:
        return _result("ALA occurrence cache", FAIL, "Cache is empty — no occurrence records")

    n_invalid = int((~gdf.geometry.is_valid).sum())
    n_null    = int(gdf.geometry.isna().sum())
    issues = []
    if n_null:
        issues.append(f"{n_null} null geometries")
    if n_invalid:
        issues.append(f"{n_invalid} invalid geometries")

    detail = f"{len(gdf)} records  CRS={gdf.crs}"
    if issues:
        detail += "  ← " + "; ".join(issues)
        return _result("ALA occurrence cache", FAIL, detail)
    return _result("ALA occurrence cache", PASS, detail)


# ---------------------------------------------------------------------------
# Section 2 — Scientific sanity checks
# ---------------------------------------------------------------------------

def check_raster_crs(label: str, path: Path, expected_epsg: int) -> dict:
    if not path.exists():
        return _result(label, SKIP, "File not found")
    try:
        import rasterio
        with rasterio.open(path) as src:
            crs = src.crs
    except Exception as exc:
        return _result(label, FAIL, f"Read error: {exc}")
    ok = crs is not None and crs.to_epsg() == expected_epsg
    return _result(label, PASS if ok else FAIL,
                   f"CRS={crs}  (expected EPSG:{expected_epsg})")


def check_spatial_consistency(paths: list[Path]) -> dict:
    """All rasters must share at least 99% bounding-box overlap."""
    import rasterio

    bounds_list = []
    for p in paths:
        if not p.exists():
            continue
        try:
            with rasterio.open(p) as src:
                b = src.bounds
                bounds_list.append((b.left, b.bottom, b.right, b.top))
        except Exception:
            pass

    if len(bounds_list) < 2:
        return _result("Raster spatial consistency", SKIP,
                       "Fewer than 2 rasters available to compare")

    minx = max(b[0] for b in bounds_list)
    miny = max(b[1] for b in bounds_list)
    maxx = min(b[2] for b in bounds_list)
    maxy = min(b[3] for b in bounds_list)

    if maxx <= minx or maxy <= miny:
        return _result("Raster spatial consistency", FAIL, "Rasters do not overlap at all")

    overlap_area = (maxx - minx) * (maxy - miny)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bounds_list]
    min_area = min(areas)
    frac = overlap_area / min_area if min_area > 0 else 0.0
    ok = frac >= 0.99
    return _result(
        "Raster spatial consistency",
        PASS if ok else FAIL,
        f"Overlap of smallest raster: {frac * 100:.1f}%  (threshold ≥99%)",
    )


def check_nan_fraction(label: str, path: Path, max_frac: float) -> dict:
    if not path.exists():
        return _result(label, SKIP, "File not found")
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result(label, FAIL, f"Read error: {exc}")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    n_nan = int(np.isnan(arr).sum())
    frac  = n_nan / arr.size
    ok = frac < max_frac
    return _result(
        label, PASS if ok else FAIL,
        f"{frac * 100:.1f}% NaN  (threshold <{max_frac * 100:.0f}%,  {n_nan:,}/{arr.size:,} px)",
    )


def check_ndvi_median_range(path: Path, expected_min: float, expected_max: float) -> dict:
    """Catchment-wide median NDVI must be in the expected vegetation range."""
    if not path.exists():
        return _result("NDVI median value range", SKIP, "File not found")
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result("NDVI median value range", FAIL, f"Read error: {exc}")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("NDVI median value range", SKIP, f"Only {valid.size} valid pixels")
    med = float(np.median(valid))
    ok = expected_min <= med <= expected_max
    return _result(
        "NDVI median value range",
        PASS if ok else FAIL,
        f"Catchment median={med:.3f}  (expected [{expected_min}, {expected_max}])",
    )


def check_ndvi_anomaly_distribution(path: Path, config) -> dict:
    """Anomaly mean should be near zero; std should indicate meaningful variability.

    Because the anomaly is normalised against a long-term baseline, the
    catchment-wide mean should be close to zero in a typical year.  Non-trivial
    std (>0.03) confirms the baseline captured real spatial variation.
    """
    if not path.exists():
        return _result("NDVI anomaly distribution", SKIP, "File not found")
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result("NDVI anomaly distribution", FAIL, f"Read error: {exc}")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("NDVI anomaly distribution", SKIP, f"Only {valid.size} valid pixels")

    mean = float(np.mean(valid))
    std  = float(np.std(valid))
    mean_ok = config.NDVI_ANOMALY_MIN_MEAN <= mean <= config.NDVI_ANOMALY_MAX_MEAN
    std_ok  = config.NDVI_ANOMALY_MIN_STD  <= std  <= config.NDVI_ANOMALY_MAX_STD
    ok = mean_ok and std_ok

    issues = []
    if not mean_ok:
        issues.append(
            f"mean={mean:.3f} outside [{config.NDVI_ANOMALY_MIN_MEAN}, {config.NDVI_ANOMALY_MAX_MEAN}]"
        )
    if not std_ok:
        issues.append(
            f"std={std:.3f} outside [{config.NDVI_ANOMALY_MIN_STD}, {config.NDVI_ANOMALY_MAX_STD}]"
        )
    detail = f"mean={mean:.3f}  std={std:.3f}"
    if issues:
        detail += "  ← " + "; ".join(issues)
    return _result("NDVI anomaly distribution", PASS if ok else FAIL, detail)


def check_flowering_correlation(ndvi_anomaly_path: Path, flowering_path: Path,
                                max_corr: float) -> dict:
    """Flowering index must not be a near-duplicate of NDVI anomaly.

    The classifier uses both as separate features. If |r| is too high, the
    two layers carry the same information and the classifier will not benefit
    from having both.
    """
    if not ndvi_anomaly_path.exists() or not flowering_path.exists():
        return _result("Flowering ≠ NDVI anomaly (correlation)", SKIP,
                       "One or both rasters not found")
    try:
        import rasterio
        with rasterio.open(ndvi_anomaly_path) as src:
            ndvi_arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                ndvi_arr[ndvi_arr == src.nodata] = np.nan
        with rasterio.open(flowering_path) as src2:
            fl_arr = src2.read(1).astype(np.float32)
            if src2.nodata is not None:
                fl_arr[fl_arr == src2.nodata] = np.nan
    except Exception as exc:
        return _result("Flowering ≠ NDVI anomaly (correlation)", FAIL, f"Read error: {exc}")

    min_rows = min(ndvi_arr.shape[0], fl_arr.shape[0])
    min_cols = min(ndvi_arr.shape[1], fl_arr.shape[1])
    ndvi_crop = ndvi_arr[:min_rows, :min_cols]
    fl_crop   = fl_arr[:min_rows, :min_cols]

    valid = np.isfinite(ndvi_crop) & np.isfinite(fl_crop)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return _result("Flowering ≠ NDVI anomaly (correlation)", SKIP,
                       f"Only {n_valid} overlapping valid pixels")

    idx = np.argwhere(valid)
    if len(idx) > 100_000:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(len(idx), 100_000, replace=False)]
    a = ndvi_crop[idx[:, 0], idx[:, 1]]
    b = fl_crop[idx[:, 0], idx[:, 1]]
    corr = float(np.corrcoef(a, b)[0, 1])

    ok = abs(corr) < max_corr
    return _result(
        "Flowering ≠ NDVI anomaly (correlation)",
        PASS if ok else FAIL,
        f"|r| = {abs(corr):.3f}  (threshold <{max_corr})",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 05 classifier inputs")
    parser.add_argument("--no-ala-check", action="store_true",
                        help="Skip the ALA occurrence cache check")
    args = parser.parse_args()

    import config

    year = config.YEAR
    print(f"{_BOLD}Stage 05 — Classifier input verification  (year: {year}){_RESET}")

    raster_paths = {
        "NDVI median":     config.ndvi_median_path(year),
        "NDVI anomaly":    config.ndvi_anomaly_path(year),
        "Flowering index": config.flowering_index_path(year),
    }
    ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"

    # ── Section 1: Input files ───────────────────────────────────────────────
    _section("1. Input files")
    file_results = []
    for label, path in raster_paths.items():
        file_results.append(check_raster_present(label, path))
    if not args.no_ala_check:
        file_results.append(check_ala_cache(ala_cache))
    else:
        print(f"  {_ICON[SKIP]}  ALA occurrence cache: skipped (--no-ala-check)")

    # ── Section 2: Scientific sanity checks ──────────────────────────────────
    _section("2. Scientific sanity checks")
    sci_results = []
    expected_epsg = int(config.TARGET_CRS.split(":")[-1])
    existing = {k: v for k, v in raster_paths.items() if v.exists()}

    for label, path in existing.items():
        sci_results.append(check_raster_crs(f"{label} CRS", path, expected_epsg))

    sci_results.append(check_spatial_consistency(list(raster_paths.values())))

    for label, path in existing.items():
        sci_results.append(check_nan_fraction(
            f"{label} NaN fraction", path, config.NAN_FRACTION_MAX
        ))

    ndvi_path = raster_paths["NDVI median"]
    if ndvi_path.exists():
        sci_results.append(check_ndvi_median_range(
            ndvi_path, config.CATCHMENT_MEDIAN_NDVI_MIN, config.CATCHMENT_MEDIAN_NDVI_MAX
        ))

    anomaly_path   = raster_paths["NDVI anomaly"]
    flowering_path = raster_paths["Flowering index"]
    if anomaly_path.exists():
        sci_results.append(check_ndvi_anomaly_distribution(anomaly_path, config))
    if anomaly_path.exists() and flowering_path.exists():
        sci_results.append(check_flowering_correlation(
            anomaly_path, flowering_path, config.FLOWERING_ANOMALY_CORRELATION_MAX
        ))

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
