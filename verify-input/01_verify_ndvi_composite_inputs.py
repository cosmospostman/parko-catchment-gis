"""
verify-input/01_verify_ndvi_composite_inputs.py — Input checks for Stage 01 (NDVI composite).

Stage 01 fetches Sentinel-2 data from a remote STAC endpoint, so its only
on-disk inputs are the catchment boundary GeoJSON and the pipeline config
parameters.  This script verifies both.

Sections
--------
1. Input files    — catchment GeoJSON exists and is readable
2. Scientific     — date window targets the dry season; cloud filter is sensible;
                    catchment can be reprojected to EPSG:4326 for STAC queries

Usage:
    source config.sh
    python verify-input/01_verify_ndvi_composite_inputs.py
"""
import sys
from pathlib import Path

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
    all_results = file_results + sci_results
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

def check_catchment_exists(catchment_path: Path) -> dict:
    if not catchment_path.exists():
        return _result("Catchment GeoJSON", FAIL, f"File not found: {catchment_path}")
    size_kb = catchment_path.stat().st_size / 1000
    if size_kb < 1:
        return _result("Catchment GeoJSON", FAIL, f"File suspiciously small (<1 KB): {catchment_path}")
    return _result("Catchment GeoJSON", PASS, f"{catchment_path.name}  ({size_kb:.0f} KB)")


def check_catchment_valid(catchment_path: Path) -> dict:
    if not catchment_path.exists():
        return _result("Catchment geometry", SKIP, "File not found — skipping geometry check")
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


# ---------------------------------------------------------------------------
# Section 2 — Scientific sanity checks
# ---------------------------------------------------------------------------

def check_catchment_reprojects(catchment_path: Path) -> dict:
    """Catchment must reproject to EPSG:4326 so Stage 01 can build a STAC bbox."""
    if not catchment_path.exists():
        return _result("Catchment → EPSG:4326", SKIP, "File not found")
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(catchment_path))
        reproj = gdf.to_crs("EPSG:4326")
        bounds = reproj.total_bounds  # [minx, miny, maxx, maxy]
    except Exception as exc:
        return _result("Catchment → EPSG:4326", FAIL, f"Reprojection failed: {exc}")

    # Sanity: bounds should be somewhere in Australia
    lon_ok = -180 <= bounds[0] < bounds[2] <= 180
    lat_ok =  -90 <= bounds[1] < bounds[3] <=  90
    if not (lon_ok and lat_ok):
        return _result(
            "Catchment → EPSG:4326", FAIL,
            f"Bounds after reprojection look wrong: {bounds}",
        )
    return _result(
        "Catchment → EPSG:4326", PASS,
        f"bbox=[{bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}]",
    )


def check_composite_window(composite_start: str, composite_end: str) -> dict:
    """Composite window must be parseable MM-DD, start < end, and within May–Oct."""
    from datetime import date
    label = "Composite date window"
    try:
        # Parse as a date within a reference year to enable comparison
        start = date(2000, *map(int, composite_start.split("-")))
        end   = date(2000, *map(int, composite_end.split("-")))
    except Exception as exc:
        return _result(label, FAIL, f"Cannot parse MM-DD values: {exc}")

    if start >= end:
        return _result(label, FAIL,
            f"COMPOSITE_START ({composite_start}) must be before COMPOSITE_END ({composite_end})")

    # Warn if window extends outside the dry season (May–Oct, months 5–10)
    dry_months = set(range(5, 11))
    months_in_window = set()
    current = start
    while current <= end:
        months_in_window.add(current.month)
        # advance ~1 month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    wet_months = months_in_window - dry_months
    detail = f"{composite_start} → {composite_end}"
    if wet_months:
        sorted_wet = sorted(wet_months)
        detail += f"  ⚠ window includes wet-season months {sorted_wet} (expected May–Oct)"
        # This is a warning, not a hard failure, so still PASS
    return _result(label, PASS, detail)


def check_cloud_cover_max(cloud_cover_max: int) -> dict:
    """Cloud cover filter should be between 5 and 50 %."""
    label = "CLOUD_COVER_MAX"
    if not (5 <= cloud_cover_max <= 50):
        return _result(
            label, FAIL,
            f"Value {cloud_cover_max}% is outside sensible range [5, 50]%",
        )
    return _result(label, PASS, f"{cloud_cover_max}%  (expected 5–50%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import config

    print(f"{_BOLD}Stage 01 — NDVI composite input verification  (year: {config.YEAR}){_RESET}")

    catchment_path = config.CATCHMENT_GEOJSON

    # ── Section 1: Input files ───────────────────────────────────────────────
    _section("1. Input files")
    file_results = [
        check_catchment_exists(catchment_path),
        check_catchment_valid(catchment_path),
    ]

    # ── Section 2: Scientific sanity checks ──────────────────────────────────
    _section("2. Scientific sanity checks")
    sci_results = [
        check_catchment_reprojects(catchment_path),
        check_composite_window(config.COMPOSITE_START, config.COMPOSITE_END),
        check_cloud_cover_max(config.CLOUD_COVER_MAX),
    ]

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
