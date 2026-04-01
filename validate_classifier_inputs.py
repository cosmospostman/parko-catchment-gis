"""
validate_classifier_inputs.py — Validate assumptions about Step 5 input data on disk.

Checks raster inputs from Steps 1–3, the ALA occurrence cache, and the optional
drainage network before running Step 5.  Does not run the classifier.

Assumptions tested
------------------
1. Raster inputs present: ndvi_median, ndvi_anomaly, flowering_index
2. Raster CRS matches TARGET_CRS (EPSG:7855)
3. Raster spatial extents are mutually consistent (overlap within 1%)
4. NaN fraction of each raster is below NAN_FRACTION_MAX
5. NDVI median values are within the expected catchment range [0.15, 0.50]
6. NDVI anomaly mean/std are within expected ranges (near-zero mean, non-trivial std)
7. Flowering index is not identical to NDVI anomaly (correlation < FLOWERING_ANOMALY_CORRELATION_MAX)
8. ALA occurrence cache is present, non-empty, and has valid geometry
9. [Optional] Drainage network is present, readable, and has valid geometries

Usage:
    source config.sh
    python validate_classifier_inputs.py [--no-ala-check]
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
# Assumption 1: raster files present and non-empty
# ---------------------------------------------------------------------------

def check_raster_present(label: str, path: Path) -> dict:
    if not path.exists():
        return _result(label, FAIL, f"File not found: {path}")
    size_mb = path.stat().st_size / 1e6
    if path.stat().st_size < 1000:
        return _result(label, FAIL, f"File suspiciously small (<1 KB): {path}")
    return _result(label, PASS, f"{path.name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Assumption 2: CRS check
# ---------------------------------------------------------------------------

def check_raster_crs(label: str, path: Path, expected_crs: str) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        crs = src.crs

    ok = crs is not None and crs.to_epsg() == int(expected_crs.split(":")[-1])
    detail = f"CRS={crs}  (expected {expected_crs})"
    return _result(label, PASS if ok else FAIL, detail)


# ---------------------------------------------------------------------------
# Assumption 3: spatial extent consistency
# ---------------------------------------------------------------------------

def check_spatial_consistency(paths: list[Path]) -> dict:
    """Check that all rasters share at least 99% bounding-box overlap."""
    import rasterio

    bounds_list = []
    for p in paths:
        with rasterio.open(p) as src:
            b = src.bounds
            bounds_list.append((b.left, b.bottom, b.right, b.top))

    if len(bounds_list) < 2:
        return _result("Raster spatial consistency", SKIP, "Fewer than 2 rasters to compare")

    minx = max(b[0] for b in bounds_list)
    miny = max(b[1] for b in bounds_list)
    maxx = min(b[2] for b in bounds_list)
    maxy = min(b[3] for b in bounds_list)

    if maxx <= minx or maxy <= miny:
        return _result("Raster spatial consistency", FAIL, "Rasters do not overlap at all")

    overlap_area = (maxx - minx) * (maxy - miny)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bounds_list]
    min_area = min(areas)
    overlap_frac = overlap_area / min_area if min_area > 0 else 0.0

    ok = overlap_frac >= 0.99
    return _result(
        "Raster spatial consistency",
        PASS if ok else FAIL,
        f"Overlap fraction of smallest raster: {overlap_frac * 100:.1f}%  (threshold: ≥99%)",
    )


# ---------------------------------------------------------------------------
# Assumption 4: NaN fraction
# ---------------------------------------------------------------------------

def check_nan_fraction(label: str, path: Path, max_frac: float) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    n_nan = int(np.isnan(arr).sum())
    n_total = arr.size
    frac = n_nan / n_total if n_total > 0 else 1.0

    ok = frac < max_frac
    return _result(
        label,
        PASS if ok else FAIL,
        f"{frac * 100:.1f}% NaN  (threshold: <{max_frac * 100:.0f}%,  {n_nan:,} of {n_total:,} pixels)",
    )


# ---------------------------------------------------------------------------
# Assumption 5: NDVI median value range
# ---------------------------------------------------------------------------

def check_ndvi_median_range(path: Path, expected_min: float, expected_max: float) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("NDVI median value range", SKIP, f"Only {valid.size} valid pixels")

    scene_median = float(np.median(valid))
    ok = expected_min <= scene_median <= expected_max
    return _result(
        "NDVI median value range",
        PASS if ok else FAIL,
        f"Catchment median NDVI = {scene_median:.3f}  "
        f"(expected [{expected_min}, {expected_max}])",
    )


# ---------------------------------------------------------------------------
# Assumption 6: NDVI anomaly distribution
# ---------------------------------------------------------------------------

def check_ndvi_anomaly_distribution(path: Path, import_config) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("NDVI anomaly distribution", SKIP, f"Only {valid.size} valid pixels")

    mean = float(np.mean(valid))
    std = float(np.std(valid))

    mean_ok = import_config.NDVI_ANOMALY_MIN_MEAN <= mean <= import_config.NDVI_ANOMALY_MAX_MEAN
    std_ok = import_config.NDVI_ANOMALY_MIN_STD <= std <= import_config.NDVI_ANOMALY_MAX_STD
    ok = mean_ok and std_ok

    issues = []
    if not mean_ok:
        issues.append(
            f"mean={mean:.3f} outside [{import_config.NDVI_ANOMALY_MIN_MEAN}, {import_config.NDVI_ANOMALY_MAX_MEAN}]"
        )
    if not std_ok:
        issues.append(
            f"std={std:.3f} outside [{import_config.NDVI_ANOMALY_MIN_STD}, {import_config.NDVI_ANOMALY_MAX_STD}]"
        )

    detail = f"mean={mean:.3f}  std={std:.3f}"
    if issues:
        detail += "  ← " + "; ".join(issues)

    return _result("NDVI anomaly distribution", PASS if ok else FAIL, detail)


# ---------------------------------------------------------------------------
# Assumption 7: flowering index is not a near-duplicate of NDVI anomaly
# ---------------------------------------------------------------------------

def check_flowering_correlation(ndvi_anomaly_path: Path, flowering_path: Path, max_corr: float) -> dict:
    import rasterio

    with rasterio.open(ndvi_anomaly_path) as src:
        ndvi_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        ndvi_arr[ndvi_arr == nodata] = np.nan

    with rasterio.open(flowering_path) as src2:
        fl_arr = src2.read(1).astype(np.float32)
        nodata2 = src2.nodata
    if nodata2 is not None:
        fl_arr[fl_arr == nodata2] = np.nan

    # Align shapes (take common valid pixels via smallest shape)
    min_rows = min(ndvi_arr.shape[0], fl_arr.shape[0])
    min_cols = min(ndvi_arr.shape[1], fl_arr.shape[1])
    ndvi_crop = ndvi_arr[:min_rows, :min_cols]
    fl_crop = fl_arr[:min_rows, :min_cols]

    valid = np.isfinite(ndvi_crop) & np.isfinite(fl_crop)
    n_valid = valid.sum()
    if n_valid < 100:
        return _result("Flowering ≠ NDVI anomaly", SKIP, f"Only {n_valid} overlapping valid pixels")

    # Subsample for speed on large rasters
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
        f"|r| = {abs(corr):.3f}  (threshold: <{max_corr})",
    )


# ---------------------------------------------------------------------------
# Assumption 8: ALA occurrence cache
# ---------------------------------------------------------------------------

def check_ala_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return _result(
            "ALA occurrence cache",
            SKIP,
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
    n_null = int(gdf.geometry.isna().sum())
    issues = []
    if n_null > 0:
        issues.append(f"{n_null} null geometries")
    if n_invalid > 0:
        issues.append(f"{n_invalid} invalid geometries")

    detail = f"{len(gdf)} records  CRS={gdf.crs}"
    if issues:
        detail += "  ← " + "; ".join(issues)
        return _result("ALA occurrence cache", FAIL, detail)

    return _result("ALA occurrence cache", PASS, detail)


# ---------------------------------------------------------------------------
# Assumption 9: drainage network (optional)
# ---------------------------------------------------------------------------

def check_drainage_network(drain_path: Path) -> dict:
    if not drain_path.exists():
        return _result(
            "Drainage network",
            SKIP,
            f"Not found at {drain_path} — dist_to_watercourse will be set to 0",
        )

    try:
        import geopandas as gpd

        gdf = gpd.read_file(str(drain_path))
    except Exception as exc:
        return _result("Drainage network", FAIL, f"Failed to read: {exc}")

    if len(gdf) == 0:
        return _result("Drainage network", FAIL, "File is empty — no drainage features")

    n_invalid = int((~gdf.geometry.is_valid).sum())
    detail = f"{len(gdf)} features  CRS={gdf.crs}"
    if n_invalid > 0:
        detail += f"  ← {n_invalid} invalid geometries"
        return _result("Drainage network", FAIL, detail)

    total_len_km = float(gdf.to_crs("EPSG:7855").geometry.length.sum()) / 1000.0
    detail += f"  ({total_len_km:,.0f} km total)"
    return _result("Drainage network", PASS, detail)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 5 classifier inputs")
    parser.add_argument(
        "--no-ala-check",
        action="store_true",
        help="Skip the ALA occurrence cache check (e.g. no internet access)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    sys.path.insert(0, str(Path(__file__).parent))
    import config

    year = config.YEAR
    print(f"\nValidating Step 5 classifier inputs — year: {year}\n")

    raster_paths = {
        "ndvi_median": config.ndvi_median_path(year),
        "ndvi_anomaly": config.ndvi_anomaly_path(year),
        "flowering_index": config.flowering_index_path(year),
    }

    results = []

    # ── Assumption 1: files present ──────────────────────────────────────────
    print("Assumption 1 — Raster inputs present")
    all_present = True
    for label, path in raster_paths.items():
        r = check_raster_present(label, path)
        results.append(r)
        if r["status"] == FAIL:
            all_present = False

    # ── Assumptions 2–7: raster quality (only if files exist) ────────────────
    existing = {k: v for k, v in raster_paths.items() if v.exists()}

    if existing:
        print("Assumption 2 — Raster CRS")
        for label, path in existing.items():
            results.append(check_raster_crs(f"{label} CRS", path, config.TARGET_CRS))

        print("Assumption 3 — Spatial consistency")
        results.append(check_spatial_consistency(list(existing.values())))

        print("Assumption 4 — NaN fraction")
        for label, path in existing.items():
            results.append(check_nan_fraction(f"{label} NaN fraction", path, config.NAN_FRACTION_MAX))

        if "ndvi_median" in existing:
            print("Assumption 5 — NDVI median value range")
            results.append(
                check_ndvi_median_range(
                    existing["ndvi_median"],
                    config.CATCHMENT_MEDIAN_NDVI_MIN,
                    config.CATCHMENT_MEDIAN_NDVI_MAX,
                )
            )

        if "ndvi_anomaly" in existing:
            print("Assumption 6 — NDVI anomaly distribution")
            results.append(check_ndvi_anomaly_distribution(existing["ndvi_anomaly"], config))

        if "ndvi_anomaly" in existing and "flowering_index" in existing:
            print("Assumption 7 — Flowering index independence")
            results.append(
                check_flowering_correlation(
                    existing["ndvi_anomaly"],
                    existing["flowering_index"],
                    config.FLOWERING_ANOMALY_CORRELATION_MAX,
                )
            )
    else:
        print("  [–] No raster inputs found — skipping quality checks")

    # ── Assumption 8: ALA cache ───────────────────────────────────────────────
    if not args.no_ala_check:
        print("Assumption 8 — ALA occurrence cache")
        ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
        results.append(check_ala_cache(ala_cache))
    else:
        print("  [–] --no-ala-check set — skipping ALA cache check")

    # ── Assumption 9: drainage network ───────────────────────────────────────
    print("Assumption 9 — Drainage network")
    drain_path = Path(config.BASE_DIR) / "data" / "drainage_network.gpkg"
    results.append(check_drainage_network(drain_path))

    # ── Summary ───────────────────────────────────────────────────────────────
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
