"""
verify-input/02_verify_ndvi_anomaly_inputs.py — Input checks for Stage 02 (NDVI anomaly).

Stage 02 computes:  anomaly = (current_ndvi − baseline_median) / baseline_std

Its on-disk inputs are:
  • The Stage 01 output: ndvi_median_{year}.tif
  • The optional baseline cache: ndvi_baseline_median.tif (built on first run)

This script verifies both, and checks that the year configuration makes the
baseline computation meaningful.

Sections
--------
1. Input files    — Stage 01 raster exists; baseline cache exists or is absent (OK)
2. Scientific     — Stage 01 raster is valid NDVI; baseline cache is valid if present;
                    baseline year span is sufficient for a stable median

Usage:
    source config.sh
    python verify-input/02_verify_ndvi_anomaly_inputs.py
"""
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
# Shared raster helpers
# ---------------------------------------------------------------------------

def _read_valid(path: Path):
    """Return a float32 array with nodata replaced by NaN, or None on error."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 1 — Input files
# ---------------------------------------------------------------------------

def check_raster_present(label: str, path: Path, optional: bool = False) -> dict:
    if not path.exists():
        status = SKIP if optional else FAIL
        msg = f"{'Not found (will be built at runtime)' if optional else 'File not found'}: {path}"
        return _result(label, status, msg)
    size_mb = path.stat().st_size / 1e6
    if size_mb < 0.001:
        return _result(label, FAIL, f"File suspiciously small (<1 KB): {path}")
    return _result(label, PASS, f"{path.name}  ({size_mb:.1f} MB)")


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
        return _result(label, FAIL, f"Cannot read CRS: {exc}")
    epsg = crs.to_epsg() if crs else None
    ok = epsg == expected_epsg
    return _result(label, PASS if ok else FAIL,
                   f"CRS={crs}  (expected EPSG:{expected_epsg})")


def check_ndvi_range(label: str, path: Path) -> dict:
    """NDVI values must be within [-1, 1]."""
    if not path.exists():
        return _result(label, SKIP, "File not found")
    arr = _read_valid(path)
    if arr is None:
        return _result(label, FAIL, "Cannot read raster")
    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result(label, SKIP, f"Only {valid.size} valid pixels")
    vmin, vmax = float(valid.min()), float(valid.max())
    ok = vmin >= -1.0 and vmax <= 1.0
    return _result(label, PASS if ok else FAIL,
                   f"min={vmin:.3f}  max={vmax:.3f}  (expected [-1, 1])")


def check_nan_fraction(label: str, path: Path, max_frac: float) -> dict:
    if not path.exists():
        return _result(label, SKIP, "File not found")
    arr = _read_valid(path)
    if arr is None:
        return _result(label, FAIL, "Cannot read raster")
    n_nan = int(np.isnan(arr).sum())
    frac = n_nan / arr.size
    ok = frac < max_frac
    return _result(
        label, PASS if ok else FAIL,
        f"{frac * 100:.1f}% NaN  (threshold <{max_frac * 100:.0f}%,  {n_nan:,}/{arr.size:,} px)",
    )


def check_baseline_year_span(year: int, baseline_start_year: int) -> dict:
    """Baseline must span at least 5 years and include at least one year before YEAR."""
    label = "Baseline year span"
    n_years = year - baseline_start_year
    if n_years < 1:
        return _result(label, FAIL,
            f"YEAR={year} is not after BASELINE_START_YEAR={baseline_start_year} — no baseline data available")
    if n_years < 5:
        return _result(label, FAIL,
            f"Only {n_years} year(s) in baseline ({baseline_start_year}–{year - 1}) — "
            f"need ≥5 for a stable median")
    return _result(label, PASS,
        f"{baseline_start_year}–{year - 1}  ({n_years} years)")


def check_baseline_nan_fraction(path: Path, max_frac: float = 0.05) -> dict:
    """Baseline NaN fraction must be low — gaps mean DEA tiles failed for those areas.

    Uses a downsampled read to avoid loading the full raster into memory.
    The threshold is tighter than for the NDVI median because the baseline
    is a multi-decade median and should have near-complete coverage everywhere
    the DEM and Landsat orbit overlap.
    """
    label = "Baseline cache NaN fraction"
    if not path.exists():
        return _result(label, SKIP, "Cache absent")
    import rasterio
    from rasterio.enums import Resampling as _R
    try:
        with rasterio.open(path) as src:
            factor = max(1, min(src.width, src.height) // 1000)
            out_h = max(1, src.height // factor)
            out_w = max(1, src.width  // factor)
            arr = src.read(1, out_shape=(out_h, out_w), resampling=_R.nearest).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result(label, FAIL, f"Cannot read: {exc}")
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    nan_frac = float(np.isnan(arr).mean())
    ok = nan_frac < max_frac
    detail = f"{nan_frac * 100:.1f}% NaN  (threshold <{max_frac * 100:.0f}%)"
    if not ok:
        detail += "  ← DEA tiles likely failed for part of the catchment; rebuild with --rebuild-baseline"
    return _result(label, PASS if ok else FAIL, detail)


def check_baseline_covers_ndvi_extent(baseline_path: Path, ndvi_path: Path) -> dict:
    """Baseline bounds must contain the NDVI median bounds.

    A baseline that doesn't cover the full NDVI extent will produce NaN in
    the anomaly wherever the two rasters don't overlap — the root cause of
    the 73% NaN anomaly failure mode.
    """
    label = "Baseline covers NDVI median extent"
    if not baseline_path.exists():
        return _result(label, SKIP, "Baseline cache absent")
    if not ndvi_path.exists():
        return _result(label, SKIP, "NDVI median not found")
    import rasterio
    try:
        with rasterio.open(str(baseline_path)) as src:
            bb = src.bounds
            b_crs = src.crs
        with rasterio.open(str(ndvi_path)) as src:
            nb = src.bounds
            n_crs = src.crs
    except Exception as exc:
        return _result(label, FAIL, f"Cannot read bounds: {exc}")

    if str(b_crs) != str(n_crs):
        return _result(label, FAIL,
            f"CRS mismatch: baseline={b_crs} ndvi={n_crs} — cannot compare bounds")

    # Allow a small margin (one 30 m pixel) for floating-point edge differences
    margin = 30.0
    covers = (
        bb.left   <= nb.left   + margin and
        bb.right  >= nb.right  - margin and
        bb.bottom <= nb.bottom + margin and
        bb.top    >= nb.top    - margin
    )
    detail = (
        f"baseline=({bb.left:.0f},{bb.bottom:.0f},{bb.right:.0f},{bb.top:.0f})  "
        f"ndvi=({nb.left:.0f},{nb.bottom:.0f},{nb.right:.0f},{nb.top:.0f})"
    )
    if not covers:
        gaps = []
        if bb.left   > nb.left   + margin: gaps.append(f"west short by {bb.left - nb.left:.0f} m")
        if bb.right  < nb.right  - margin: gaps.append(f"east short by {nb.right - bb.right:.0f} m")
        if bb.bottom > nb.bottom + margin: gaps.append(f"south short by {bb.bottom - nb.bottom:.0f} m")
        if bb.top    < nb.top    - margin: gaps.append(f"north short by {nb.top - bb.top:.0f} m")
        detail += "  ← " + "; ".join(gaps) + " — rebuild baseline with --rebuild-baseline"
    return _result(label, PASS if covers else FAIL, detail)


def check_baseline_tag(
    path: Path, year: int, baseline_start_year: int,
    composite_start: str, composite_end: str,
) -> dict:
    """IMAGEDESCRIPTION tag must match years and composite window.

    A mismatch means the cache was built for a different YEAR,
    BASELINE_START_YEAR, or seasonal window and will produce a biased anomaly.
    """
    label = "Baseline cache tag"
    if not path.exists():
        return _result(label, SKIP, "Cache absent")
    import rasterio
    try:
        with rasterio.open(path) as src:
            tag = src.tags().get("IMAGEDESCRIPTION", "")
    except Exception as exc:
        return _result(label, FAIL, f"Cannot read tags: {exc}")
    expected = f"NDVI_BASELINE:{baseline_start_year}-{year - 1}:{composite_start}/{composite_end}"
    if tag != expected:
        return _result(label, FAIL,
            f"Tag '{tag}' does not match expected '{expected}' — "
            "cache is stale, set REBUILD_BASELINE=true")
    return _result(label, PASS, tag)


def check_anomaly_preview(
    ndvi_path: Path,
    baseline_path: Path,
    min_std: float,
    max_std: float,
) -> list:
    """Sample both rasters at 1/8 resolution and preview anomaly std.

    Checks that the anomaly has meaningful spread — a std near zero means the
    subtraction silently failed (e.g. misaligned grids producing all-NaN then
    falling back to the raw NDVI).  The mean is not checked: year-to-year climate
    variability (e.g. La Niña wet years) means a large positive or negative mean
    anomaly is ecologically valid and should not be flagged as a pipeline error.

    Returns two result dicts.
    """
    label_mean = "Anomaly preview mean"
    label_std  = "Anomaly preview std"

    if not ndvi_path.exists() or not baseline_path.exists():
        skip_msg = "Requires both Stage 01 NDVI and baseline cache"
        return [
            _result(label_mean, SKIP, skip_msg),
            _result(label_std,  SKIP, skip_msg),
        ]

    import rasterio
    from rasterio.enums import Resampling as _Resampling

    def _read_decimated(p: Path):
        try:
            with rasterio.open(p) as src:
                factor = 8
                out_h = max(1, src.height // factor)
                out_w = max(1, src.width  // factor)
                arr = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=_Resampling.average,
                ).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            return arr
        except Exception:
            return None

    ndvi_arr     = _read_decimated(ndvi_path)
    baseline_arr = _read_decimated(baseline_path)

    if ndvi_arr is None or baseline_arr is None:
        msg = "Cannot read one or both rasters"
        return [_result(label_mean, FAIL, msg), _result(label_std, FAIL, msg)]

    # Resize baseline to match ndvi if shapes differ (different native resolutions).
    # Use scipy zoom rather than PIL: PIL.Image.fromarray silently converts NaN→0,
    # which makes the anomaly preview spuriously large wherever the baseline has voids.
    if ndvi_arr.shape != baseline_arr.shape:
        from scipy.ndimage import zoom
        zy = ndvi_arr.shape[0] / baseline_arr.shape[0]
        zx = ndvi_arr.shape[1] / baseline_arr.shape[1]
        # order=1 (bilinear); NaN propagates correctly through scipy zoom
        baseline_arr = zoom(baseline_arr, (zy, zx), order=1, mode="nearest")

    anomaly = ndvi_arr - baseline_arr
    valid = anomaly[np.isfinite(anomaly)]

    if valid.size < 100:
        skip_msg = f"Only {valid.size} valid pixels after decimation"
        return [
            _result(label_mean, SKIP, skip_msg),
            _result(label_std,  SKIP, skip_msg),
        ]

    a_mean = float(np.mean(valid))
    a_std  = float(np.std(valid))

    std_ok = min_std <= a_std <= max_std

    result_mean = _result(
        label_mean,
        PASS,
        f"mean={a_mean:.4f}  (informational only — climate variability makes this non-diagnostic)",
    )
    result_std = _result(
        label_std,
        PASS if std_ok else FAIL,
        f"std={a_std:.4f}  (expected [{min_std}, {max_std}])"
        + ("" if std_ok else "  ← near-zero std suggests subtraction failed silently"),
    )
    return [result_mean, result_std]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import config

    print(f"{_BOLD}Stage 02 — NDVI anomaly input verification  (year: {config.YEAR}){_RESET}")

    ndvi_path     = config.ndvi_median_path(config.YEAR)
    baseline_path = config.ndvi_baseline_path()

    # ── Section 1: Input files ───────────────────────────────────────────────
    _section("1. Input files")
    file_results = [
        check_raster_present("Stage 01 NDVI median", ndvi_path),
        check_raster_present("Baseline cache (ndvi_baseline_median.tif)",
                             baseline_path, optional=True),
    ]

    # ── Section 2: Scientific sanity checks ──────────────────────────────────
    _section("2. Scientific sanity checks")
    sci_results = []

    # Stage 01 output quality
    sci_results.append(check_raster_crs(
        "NDVI median CRS", ndvi_path, int(config.TARGET_CRS.split(":")[-1])
    ))
    sci_results.append(check_ndvi_range("NDVI median value range", ndvi_path))
    sci_results.append(check_nan_fraction(
        "NDVI median NaN fraction", ndvi_path, config.NAN_FRACTION_MAX
    ))

    # Baseline cache quality (SKIP if absent — it will be built at runtime)
    if baseline_path.exists():
        sci_results.append(check_raster_crs(
            "Baseline cache CRS", baseline_path, int(config.TARGET_CRS.split(":")[-1])
        ))
        sci_results.append(check_ndvi_range("Baseline cache value range", baseline_path))
        sci_results.append(check_baseline_nan_fraction(baseline_path))
        sci_results.append(check_baseline_covers_ndvi_extent(baseline_path, ndvi_path))
        sci_results.append(check_baseline_tag(
            baseline_path, config.YEAR, config.BASELINE_START_YEAR,
            config.COMPOSITE_START, config.COMPOSITE_END,
        ))
    else:
        sci_results.append(_result(
            "Baseline cache quality", SKIP,
            "Cache absent — will be computed from DEA Landsat on first run",
        ))

    # Year configuration
    sci_results.append(check_baseline_year_span(config.YEAR, config.BASELINE_START_YEAR))

    # Anomaly preview — catch mean/std failures before the full run
    sci_results.extend(check_anomaly_preview(
        ndvi_path,
        baseline_path,
        min_std=config.NDVI_ANOMALY_MIN_STD,
        max_std=config.NDVI_ANOMALY_MAX_STD,
    ))

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
