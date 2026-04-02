"""
verify-input/05_verify_plausibility_inputs.py — Input checks for Stage 05 (plausibility map).

Stage 05 combines three upstream rasters:
  • ndvi_anomaly_{year}.tif   (Stage 2 output)
  • flowering_index_{year}.tif (Stage 3 output)
  • hand_{year}.tif            (Stage 4 output)

Sections
--------
1. Input files    — all three rasters exist and are non-empty
2. Scientific     — NDVI anomaly distribution, flowering vs NDVI decorrelation,
                    HAND floodplain coverage, shared CRS/grid

Usage:
    source config.sh
    python verify-input/05_verify_plausibility_inputs.py

No arguments needed — all paths are resolved via config.py.
"""
import sys
from pathlib import Path

import numpy as np

# Maximum number of valid pixels to sample for statistical checks.
# At 37k×45k the full raster is ~6 GB as float32; 500k pixels gives
# < 2 MB per raster and keeps all statistics well within tolerance.
_SAMPLE_N = 500_000
_RNG = np.random.default_rng(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# ---------------------------------------------------------------------------
# Output helpers (shared pattern with Stage 04 verify-input)
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

def check_raster_exists(label: str, path: Path) -> dict:
    if not path.exists():
        return _result(label, FAIL, f"File not found: {path}")
    size_kb = path.stat().st_size / 1000
    if size_kb < 1:
        return _result(label, FAIL, "File suspiciously small (<1 KB)")
    return _result(label, PASS, f"{path.name}  ({size_kb:,.0f} KB)")


def check_grids_consistent(paths: list[Path], labels: list[str]) -> dict:
    """All three rasters must share the same CRS, shape, and transform."""
    import rasterio

    info = []
    for p, lbl in zip(paths, labels):
        if not p.exists():
            return _result("Grid consistency", SKIP, f"{lbl} not found — skipping")
        with rasterio.open(p) as src:
            info.append({
                "label": lbl,
                "crs": str(src.crs),
                "shape": (src.height, src.width),
                "transform": src.transform,
            })

    ref = info[0]
    mismatches = []
    for item in info[1:]:
        if item["crs"] != ref["crs"]:
            mismatches.append(f"{item['label']} CRS {item['crs']} ≠ {ref['crs']}")
        if item["shape"] != ref["shape"]:
            mismatches.append(f"{item['label']} shape {item['shape']} ≠ {ref['shape']}")
        if item["transform"] != ref["transform"]:
            mismatches.append(f"{item['label']} transform differs from {ref['label']}")

    if mismatches:
        return _result("Grid consistency", FAIL, "; ".join(mismatches))
    shape = ref["shape"]
    return _result(
        "Grid consistency", PASS,
        f"All three rasters share CRS={ref['crs']}  shape={shape[0]}×{shape[1]}",
    )


# ---------------------------------------------------------------------------
# Section 2 — Scientific sanity checks
# ---------------------------------------------------------------------------

def _sample_raster(path: Path, n: int = _SAMPLE_N) -> tuple:
    """Return a 1-D float32 sample of valid pixels and the nodata value.

    Reads the raster in a single overview pass when overviews exist, otherwise
    reads a systematic stride of rows to avoid loading the full array.  Peak
    memory is O(n) not O(width × height).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        nodata = src.nodata
        full_pixels = src.width * src.height
        # Downsample to a thumbnail whose pixel count slightly exceeds n so that
        # after removing nodata we still have enough valid samples.
        scale = max(1, int((full_pixels / (n * 2)) ** 0.5))
        out_h = max(1, src.height // scale)
        out_w = max(1, src.width  // scale)
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        ).astype(np.float32)

    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    if valid.size > n:
        valid = _RNG.choice(valid, size=n, replace=False)
    return valid, nodata


def check_ndvi_anomaly_distribution(ndvi_path: Path) -> dict:
    """Mean should be near zero (|mean| < 0.05); std in [0.03, 0.20].

    A large absolute mean suggests the baseline was computed from the wrong years
    or the NDVI composite contains cloud contamination.
    """
    if not ndvi_path.exists():
        return _result("NDVI anomaly distribution", SKIP, "File not found")
    try:
        valid, _ = _sample_raster(ndvi_path)
    except Exception as exc:
        return _result("NDVI anomaly distribution", FAIL, f"Cannot read: {exc}")

    if valid.size < 100:
        return _result("NDVI anomaly distribution", SKIP, f"Only {valid.size} valid pixels")

    mean = float(valid.mean())
    std  = float(valid.std())
    issues = []
    if abs(mean) >= 0.05:
        issues.append(f"mean={mean:.4f} outside (-0.05, 0.05)")
    if not (0.03 <= std <= 0.20):
        issues.append(f"std={std:.4f} outside [0.03, 0.20]")

    detail = f"mean={mean:.4f}  std={std:.4f}"
    if issues:
        detail += "  ← " + "; ".join(issues)
        return _result("NDVI anomaly distribution", FAIL, detail)
    return _result("NDVI anomaly distribution", PASS, detail)


def check_flowering_ndvi_decorrelation(flower_path: Path, ndvi_path: Path) -> dict:
    """Flowering index should be reasonably decorrelated from NDVI anomaly (r < 0.70).

    High correlation means both features are measuring the same signal; in that
    case one should be dropped before Stage 6.
    """
    import rasterio
    from rasterio.enums import Resampling

    for p in (flower_path, ndvi_path):
        if not p.exists():
            return _result("Flowering–NDVI decorrelation", SKIP, f"File not found: {p.name}")

    # Read both rasters at the same downsampled shape so pixels are co-located.
    try:
        with rasterio.open(flower_path) as src:
            full_pixels = src.width * src.height
            scale = max(1, int((full_pixels / (_SAMPLE_N * 2)) ** 0.5))
            out_h = max(1, src.height // scale)
            out_w = max(1, src.width  // scale)
            f_arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            f_nd  = src.nodata
        with rasterio.open(ndvi_path) as src:
            n_arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            n_nd  = src.nodata
    except Exception as exc:
        return _result("Flowering–NDVI decorrelation", FAIL, f"Cannot read: {exc}")

    if f_nd is not None and np.isfinite(f_nd):
        f_arr[f_arr == f_nd] = np.nan
    if n_nd is not None and np.isfinite(n_nd):
        n_arr[n_arr == n_nd] = np.nan

    valid = np.isfinite(f_arr) & np.isfinite(n_arr)
    if valid.sum() < 100:
        return _result("Flowering–NDVI decorrelation", SKIP, "Too few co-valid pixels")

    f_valid = f_arr[valid]
    n_valid = n_arr[valid]
    if f_valid.size > _SAMPLE_N:
        idx = _RNG.choice(f_valid.size, size=_SAMPLE_N, replace=False)
        f_valid = f_valid[idx]
        n_valid = n_valid[idx]

    r = float(np.corrcoef(f_valid, n_valid)[0, 1])
    ok = abs(r) < 0.70
    detail = f"Pearson r={r:.3f} (threshold <0.70)"
    if not ok:
        detail += "  ← high correlation; both features may be measuring the same signal"
    return _result("Flowering–NDVI decorrelation", PASS if ok else FAIL, detail)


def check_hand_floodplain_coverage(hand_path: Path) -> dict:
    """At least 5% of valid HAND pixels should be below 5 m (confirms floodplain terrain)."""
    if not hand_path.exists():
        return _result("HAND floodplain coverage", SKIP, "File not found")
    try:
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(hand_path) as src:
            full_pixels = src.width * src.height
            scale = max(1, int((full_pixels / (_SAMPLE_N * 2)) ** 0.5))
            out_h = max(1, src.height // scale)
            out_w = max(1, src.width  // scale)
            arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            nodata = src.nodata
    except Exception as exc:
        return _result("HAND floodplain coverage", FAIL, f"Cannot read: {exc}")

    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    valid = arr[np.isfinite(arr)]
    if valid.size < 100:
        return _result("HAND floodplain coverage", SKIP, f"Only {valid.size} valid pixels")

    void_frac   = float(np.isnan(arr).mean())
    below5_frac = float((valid < 5.0).mean())
    issues = []
    if below5_frac < 0.05:
        issues.append(f"only {below5_frac:.1%} of pixels <5 m — too few floodplain pixels")
    if void_frac >= 0.70:
        issues.append(f"void fraction {void_frac:.1%} ≥ 70% — unusually high, check DEM coverage")

    detail = f"pixels<5m: {below5_frac:.1%}  void: {void_frac:.1%}"
    if issues:
        detail += "  ← " + "; ".join(issues)
        return _result("HAND floodplain coverage", FAIL, detail)
    return _result("HAND floodplain coverage", PASS, detail)


def check_crs_projected(paths: list[Path], labels: list[str]) -> dict:
    """All three rasters must be in a projected CRS (expected EPSG:7855)."""
    import rasterio

    wrong = []
    for p, lbl in zip(paths, labels):
        if not p.exists():
            continue
        try:
            with rasterio.open(p) as src:
                crs_str = str(src.crs)
        except Exception:
            continue
        if "7855" not in crs_str and "GDA2020" not in crs_str:
            wrong.append(f"{lbl}: {crs_str}")

    if wrong:
        return _result("CRS check", FAIL, "Unexpected CRS — " + "; ".join(wrong))
    return _result("CRS check", PASS, "All rasters in EPSG:7855 (GDA2020 / MGA zone 55)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import config

    print(f"{_BOLD}Stage 05 — Plausibility map input verification  (year: {config.YEAR}){_RESET}")

    ndvi_path   = config.ndvi_anomaly_path(config.YEAR)
    flower_path = config.flowering_index_path(config.YEAR)
    hand_path   = config.hand_raster_path(config.YEAR)
    all_paths   = [ndvi_path, flower_path, hand_path]
    all_labels  = ["NDVI anomaly", "Flowering index", "HAND"]

    # ── Section 1: Input files ───────────────────────────────────────────────
    _section("1. Input files")
    file_results = [
        check_raster_exists("NDVI anomaly raster",   ndvi_path),
        check_raster_exists("Flowering index raster", flower_path),
        check_raster_exists("HAND raster",            hand_path),
        check_grids_consistent(all_paths, all_labels),
    ]

    # ── Section 2: Scientific sanity checks ──────────────────────────────────
    _section("2. Scientific sanity checks")
    sci_results = [
        check_ndvi_anomaly_distribution(ndvi_path),
        check_flowering_ndvi_decorrelation(flower_path, ndvi_path),
        check_hand_floodplain_coverage(hand_path),
        check_crs_projected(all_paths, all_labels),
    ]

    ok = _summary(file_results, sci_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
