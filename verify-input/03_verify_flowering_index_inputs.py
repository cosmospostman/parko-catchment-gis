"""
verify-input/03_verify_flowering_index_inputs.py — Input checks for Stage 03
(Parkinsonia flowering index).

Stage 03 fetches Sentinel-2 data within the flowering window and computes a
green/NIR ratio intended to capture the Parkinsonia flowering signal. Like
Stage 01, its only on-disk input is the catchment boundary GeoJSON; the
scientific checks focus on the flowering window configuration.

Key scientific contract:
  • The flowering window MUST target the Parkinsonia flowering period (Aug–Oct).
  • The flowering window MUST NOT substantially overlap the NDVI composite
    window — the two indices are designed to capture DIFFERENT phenological
    states. If they overlap heavily, the flowering index will correlate with
    the NDVI composite and lose discriminating power in the classifier.

Sections
--------
1. Input files    — catchment GeoJSON exists and is readable
2. Scientific     — flowering window targets Aug–Oct; minimal overlap with
                    composite window; cloud cover filter is sensible

Usage:
    source config.sh
    python verify-input/03_verify_flowering_index_inputs.py
"""
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ---------------------------------------------------------------------------
# Output helpers (shared pattern across all verify-input scripts)
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

def _parse_mmdd(mmdd: str, ref_year: int = 2000) -> date:
    """Parse 'MM-DD' into a date(ref_year, month, day)."""
    month, day = map(int, mmdd.split("-"))
    return date(ref_year, month, day)


def _window_days(start: date, end: date) -> set:
    """Return the set of day-of-year (1–365) covered by a date window."""
    days = set()
    current = start
    while current <= end:
        days.add(current.timetuple().tm_yday)
        current = date(current.year, current.month, current.day + 1) \
                  if current.day < 28 else \
                  (date(current.year, current.month + 1, 1)
                   if current.month < 12 else date(current.year + 1, 1, 1))
    # Simple day count via ordinal difference
    return (end - start).days + 1


def check_flowering_window(flower_start: str, flower_end: str) -> dict:
    """Flowering window must be parseable, ordered, and cover Aug–Oct."""
    label = "Flowering window"
    try:
        start = _parse_mmdd(flower_start)
        end   = _parse_mmdd(flower_end)
    except Exception as exc:
        return _result(label, FAIL, f"Cannot parse MM-DD values: {exc}")

    if start >= end:
        return _result(label, FAIL,
            f"FLOWERING_WINDOW_START ({flower_start}) must be before FLOWERING_WINDOW_END ({flower_end})")

    # Expected range: August–October (months 8–10)
    expected_months = {8, 9, 10}
    months_covered: set = set()
    current = start
    while current <= end:
        months_covered.add(current.month)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    missing = expected_months - months_covered
    outside = months_covered - expected_months
    detail = f"{flower_start} → {flower_end}"

    issues = []
    if missing:
        issues.append(f"window misses expected months {sorted(missing)} (Aug–Oct)")
    if outside:
        issues.append(f"⚠ extends into months {sorted(outside)} outside Aug–Oct")

    if missing:
        return _result(label, FAIL, detail + "  ← " + "; ".join(issues))
    if outside:
        # Only warn, don't fail — slight extension is acceptable
        detail += "  ⚠ " + "; ".join([i for i in issues if "⚠" in i])
    return _result(label, PASS, detail)


def check_window_independence(
    composite_start: str, composite_end: str,
    flower_start: str, flower_end: str,
) -> dict:
    """Flowering window should not overlap the composite window by more than 30 days.

    The NDVI composite and flowering index are meant to capture different
    phenological signals.  Heavy overlap means both will be computed from the
    same scenes, reducing the classifier's discriminating power.
    """
    label = "Window independence (flowering ≠ composite)"
    try:
        cs = _parse_mmdd(composite_start)
        ce = _parse_mmdd(composite_end)
        fs = _parse_mmdd(flower_start)
        fe = _parse_mmdd(flower_end)
    except Exception as exc:
        return _result(label, SKIP, f"Cannot parse window dates: {exc}")

    # Overlap in days = max(0, min(end1, end2) - max(start1, start2) + 1)
    overlap_start = max(cs, fs)
    overlap_end   = min(ce, fe)
    overlap_days  = max(0, (overlap_end - overlap_start).days + 1)

    detail = (
        f"composite {composite_start}→{composite_end}  ·  "
        f"flowering {flower_start}→{flower_end}  ·  "
        f"overlap {overlap_days} days"
    )
    if overlap_days > 30:
        return _result(label, FAIL,
            detail + f"  ← exceeds 30-day tolerance; indices may be correlated")
    return _result(label, PASS, detail)


def check_cloud_cover_max(cloud_cover_max: int) -> dict:
    """Cloud cover filter should be between 5 and 50 %."""
    label = "CLOUD_COVER_MAX"
    if not (5 <= cloud_cover_max <= 50):
        return _result(label, FAIL,
            f"Value {cloud_cover_max}% is outside sensible range [5, 50]%")
    return _result(label, PASS, f"{cloud_cover_max}%  (expected 5–50%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import config

    print(f"{_BOLD}Stage 03 — Flowering index input verification  (year: {config.YEAR}){_RESET}")

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
        check_flowering_window(config.FLOWERING_WINDOW_START, config.FLOWERING_WINDOW_END),
        check_window_independence(
            config.COMPOSITE_START, config.COMPOSITE_END,
            config.FLOWERING_WINDOW_START, config.FLOWERING_WINDOW_END,
        ),
        check_cloud_cover_max(config.CLOUD_COVER_MAX),
    ]

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
