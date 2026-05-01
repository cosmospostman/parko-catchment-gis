"""utils/parquet_validator.py — Validate fetched tile parquets for data quality.

Each check returns a CheckResult; validate_tile bundles them into a TileReport.
validate_location iterates all (year, tile_id) combinations for a Location.

Designed to be imported by cli/location.py and by tests independently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SPECTRAL_INDEX_COLS

if TYPE_CHECKING:
    from utils.location import Location

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS: list[str] = (
    ["point_id", "lon", "lat", "date", "tile_id", "source"]
    + BANDS
    + SPECTRAL_INDEX_COLS
    + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
)
REQUIRED_S1_COLS: list[str] = ["vh", "vv", "orbit"]

# Expected Arrow dtype prefixes (permissive — float vs float32 both accepted)
_DTYPE_PREFIXES: dict[str, tuple[str, ...]] = {
    "point_id":   ("string", "large_string", "dictionary"),
    "lon":        ("float",),
    "lat":        ("float",),
    "date":       ("date",),
    "scl":        ("int",),
    "scl_purity": ("float",),
    "source":     ("string", "large_string", "dictionary"),
    "orbit":      ("string", "large_string", "dictionary"),
    **{b: ("float",) for b in BANDS + SPECTRAL_INDEX_COLS + ["vh", "vv"]},
}

BAND_VALID_RANGE   = (0.0, 1.5)
INDEX_VALID_RANGE  = (-1.0, 1.0)
NAN_FAIL_THRESHOLD = 0.20
NAN_WARN_THRESHOLD = 0.01
MIN_UNIQUE_DATES   = 10
MIN_MONTHS_COVERED = 6
PIXEL_RATIO_FAIL   = 0.30
PIXEL_RATIO_WARN   = 0.80
BBOX_BUFFER_DEG    = 0.02
_SAMPLE_RGS        = 10   # row groups to sample for band/index checks
_DUP_SAMPLE_RGS    = 5    # row groups to sample for duplicate check


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


_STATUS_ORDER = {Status.PASS: 0, Status.WARN: 1, Status.FAIL: 2}


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str = ""


@dataclass
class TileReport:
    location_id: str
    year: int
    tile_id: str
    path: Path | None
    checks: list[CheckResult] = field(default_factory=list)
    n_rows: int = 0
    n_pixels: int = 0
    n_dates: int = 0
    has_s1: bool = False
    s1_old_format: bool = False   # True if S1 rows lack orbit

    @property
    def status(self) -> Status:
        if not self.checks:
            return Status.FAIL
        return max(self.checks, key=lambda c: _STATUS_ORDER[c.status]).status

    @property
    def issues(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status != Status.PASS]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evenly_spaced(n_total: int, n_sample: int, exclude_last: int = 0) -> list[int]:
    """Return n_sample evenly-spaced indices in [0, n_total - exclude_last)."""
    upper = max(n_total - exclude_last, 1)
    if n_sample >= upper:
        return list(range(upper))
    step = upper / n_sample
    return sorted({int(i * step) for i in range(n_sample)})


def _count_s1_tail_rgs(pf: pq.ParquetFile) -> int:
    """Count how many trailing row groups contain S1 rows (source=="S1")."""
    n = pf.metadata.num_row_groups
    count = 0
    for i in range(n - 1, -1, -1):
        tbl = pf.read_row_group(i, columns=["source"])
        sources = tbl.column("source")
        vals = set(pc.unique(sources).to_pylist())
        if "S1" in vals:
            count += 1
        else:
            break
    return count


def _read_s2_sample(pf: pq.ParquetFile, n_s1_rgs: int) -> "pa.Table":
    """Read a sample of S2 row groups for band/index/scl_purity checks."""
    n_total = pf.metadata.num_row_groups
    cols = BANDS + SPECTRAL_INDEX_COLS + ["source", "scl_purity"]
    indices = _evenly_spaced(n_total, _SAMPLE_RGS, exclude_last=n_s1_rgs)
    tables = [pf.read_row_group(i, columns=cols) for i in indices]
    combined = pa.concat_tables(tables)
    mask = pc.equal(combined.column("source"), "S2")
    return combined.filter(mask)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_row_count(pf: pq.ParquetFile) -> CheckResult:
    n = pf.metadata.num_rows
    if n == 0:
        return CheckResult("row_count", Status.FAIL, "0 rows")
    return CheckResult("row_count", Status.PASS)


def check_schema(pf: pq.ParquetFile) -> CheckResult:
    schema = pf.schema_arrow
    present = set(schema.names)
    missing = [c for c in REQUIRED_COLS if c not in present]
    if missing:
        return CheckResult("schema", Status.FAIL, f"missing columns: {', '.join(missing)}")
    issues = []
    for col, prefixes in _DTYPE_PREFIXES.items():
        if col not in present:
            continue
        dtype_str = str(schema.field(col).type)
        if not any(dtype_str.startswith(p) for p in prefixes):
            issues.append(f"{col}={dtype_str}")
    if issues:
        return CheckResult("schema", Status.FAIL, f"unexpected dtypes: {', '.join(issues)}")
    return CheckResult("schema", Status.PASS)


def check_source_presence(pf: pq.ParquetFile) -> tuple[CheckResult, bool]:
    """Return (CheckResult, has_s1). Reads only the last row group's source column."""
    n = pf.metadata.num_row_groups
    tbl = pf.read_row_group(n - 1, columns=["source"])
    vals = set(pc.unique(tbl.column("source")).to_pylist())
    has_s1 = "S1" in vals
    if not has_s1:
        return CheckResult("s1_presence", Status.WARN, "no S1 rows found — re-run fetch to append S1"), False
    return CheckResult("s1_presence", Status.PASS), True


def check_pixel_count(pf: pq.ParquetFile, expected: int) -> tuple[CheckResult, int]:
    """Return (CheckResult, n_pixels). Reads point_id + source for all row groups."""
    tbl = pf.read(columns=["point_id", "source"])
    mask = pc.equal(tbl.column("source"), "S2")
    s2_pids = tbl.column("point_id").filter(mask)
    n_pixels = len(pc.unique(s2_pids))
    if expected > 0:
        ratio = n_pixels / expected
        if ratio < PIXEL_RATIO_FAIL:
            return (
                CheckResult("pixel_count", Status.FAIL,
                            f"{n_pixels:,} pixels ({ratio:.0%} of expected {expected:,})"),
                n_pixels,
            )
        if ratio < PIXEL_RATIO_WARN:
            return (
                CheckResult("pixel_count", Status.WARN,
                            f"{n_pixels:,} pixels ({ratio:.0%} of expected {expected:,})"),
                n_pixels,
            )
    return CheckResult("pixel_count", Status.PASS), n_pixels


def check_nan_bands(s2_sample: "pa.Table") -> CheckResult:
    if len(s2_sample) == 0:
        return CheckResult("nan_bands", Status.WARN, "no S2 rows in sample")
    worst_name, worst_frac = "", 0.0
    for band in BANDS:
        col = s2_sample.column(band)
        n_null = pc.sum(pc.is_null(col)).as_py() or 0
        frac = n_null / len(s2_sample)
        if frac > worst_frac:
            worst_frac, worst_name = frac, band
    if worst_frac > NAN_FAIL_THRESHOLD:
        return CheckResult("nan_bands", Status.FAIL,
                           f"{worst_name} has {worst_frac:.1%} NaN (>{NAN_FAIL_THRESHOLD:.0%})")
    if worst_frac > NAN_WARN_THRESHOLD:
        return CheckResult("nan_bands", Status.WARN,
                           f"{worst_name} has {worst_frac:.1%} NaN (>{NAN_WARN_THRESHOLD:.1%})")
    return CheckResult("nan_bands", Status.PASS)


def check_band_range(s2_sample: "pa.Table") -> CheckResult:
    if len(s2_sample) == 0:
        return CheckResult("band_range", Status.PASS)
    lo, hi = BAND_VALID_RANGE
    for band in BANDS:
        col = s2_sample.column(band).drop_null()
        if len(col) == 0:
            continue
        col_min = pc.min(col).as_py()
        col_max = pc.max(col).as_py()
        if col_min < lo or col_max > hi:
            return CheckResult("band_range", Status.FAIL,
                               f"{band} range [{col_min:.3f}, {col_max:.3f}] outside [{lo}, {hi}]")
    return CheckResult("band_range", Status.PASS)


def check_index_range(s2_sample: "pa.Table") -> CheckResult:
    if len(s2_sample) == 0:
        return CheckResult("index_range", Status.PASS)
    lo, hi = INDEX_VALID_RANGE
    for idx_col in SPECTRAL_INDEX_COLS:
        col = s2_sample.column(idx_col).drop_null()
        if len(col) == 0:
            continue
        col_min = pc.min(col).as_py()
        col_max = pc.max(col).as_py()
        if col_min < lo or col_max > hi:
            return CheckResult("index_range", Status.FAIL,
                               f"{idx_col} range [{col_min:.3f}, {col_max:.3f}] outside [{lo}, {hi}]")
    return CheckResult("index_range", Status.PASS)


def check_scl_purity(s2_sample: "pa.Table") -> CheckResult:
    if len(s2_sample) == 0:
        return CheckResult("scl_purity", Status.PASS)
    col = s2_sample.column("scl_purity").drop_null()
    if len(col) == 0:
        return CheckResult("scl_purity", Status.WARN, "scl_purity all null")
    mean_val = pc.mean(col).as_py()
    if mean_val == 0.0:
        return CheckResult("scl_purity", Status.FAIL, "scl_purity all 0.0 — column may be corrupt")
    if mean_val < 0.5:
        return CheckResult("scl_purity", Status.WARN,
                           f"scl_purity mean={mean_val:.3f} < 0.5 (unexpectedly many cloudy rows)")
    return CheckResult("scl_purity", Status.PASS)


def check_duplicates(pf: pq.ParquetFile, n_s1_rgs: int) -> CheckResult:
    n_total = pf.metadata.num_row_groups
    indices = _evenly_spaced(n_total, _DUP_SAMPLE_RGS, exclude_last=n_s1_rgs)
    for i in indices:
        tbl = pf.read_row_group(i, columns=["point_id", "date", "source"])
        mask = pc.equal(tbl.column("source"), "S2")
        s2 = tbl.filter(mask)
        if len(s2) == 0:
            continue
        # check for duplicate (point_id, date) within this row group
        pid_str  = pc.cast(s2.column("point_id"), pa.large_utf8())
        date_str = pc.cast(pc.cast(s2.column("date"), pa.int32()), pa.large_utf8())
        sep = pa.chunked_array([pa.array(["|"] * len(s2), type=pa.large_utf8())])
        combined = pc.binary_join_element_wise(pid_str, date_str, sep)
        n_unique = len(pc.unique(combined))
        if n_unique < len(s2):
            dupes = len(s2) - n_unique
            return CheckResult("duplicates", Status.FAIL,
                               f"{dupes} duplicate (point_id, date) in row group {i}")
    return CheckResult("duplicates", Status.PASS)


def check_s1_integrity(pf: pq.ParquetFile) -> tuple[CheckResult, bool]:
    """Check S1 rows for cross-contamination and old-format detection.

    Returns (CheckResult, s1_old_format). Old format = orbit column absent or all-null.
    """
    schema = pf.schema_arrow
    has_orbit_col = "orbit" in schema.names

    n = pf.metadata.num_row_groups
    tbl = pf.read_row_group(n - 1, columns=["source", "B02"] + (["orbit"] if has_orbit_col else []))
    mask = pc.equal(tbl.column("source"), "S1")
    s1 = tbl.filter(mask)

    if len(s1) == 0:
        return CheckResult("s1_integrity", Status.PASS), False

    # Cross-contamination: S1 rows should have null B02
    b02 = s1.column("B02")
    n_nonnull = pc.sum(pc.invert(pc.is_null(b02))).as_py() or 0
    if n_nonnull > 0:
        return (
            CheckResult("s1_integrity", Status.FAIL,
                        f"{n_nonnull} S1 rows have non-null B02 (schema corruption)"),
            False,
        )

    # Old-format detection: orbit must be present and non-null
    if not has_orbit_col:
        return (
            CheckResult("s1_integrity", Status.FAIL,
                        "S1 rows missing 'orbit' column — old-format pre-Planetary-Computer data; re-fetch"),
            True,
        )
    orbit_col = s1.column("orbit")
    n_null_orbit = pc.sum(pc.is_null(orbit_col)).as_py() or 0
    if n_null_orbit > 0:
        return (
            CheckResult("s1_integrity", Status.FAIL,
                        f"{n_null_orbit} S1 rows have null orbit — old-format data; re-fetch"),
            True,
        )
    return CheckResult("s1_integrity", Status.PASS), False


def check_temporal_coverage(pf: pq.ParquetFile, year: int) -> tuple[CheckResult, int]:
    """Return (CheckResult, n_unique_dates)."""
    tbl = pf.read(columns=["date", "source"])
    mask = pc.equal(tbl.column("source"), "S2")
    s2_dates = tbl.column("date").filter(mask)
    unique_dates = pc.unique(s2_dates)
    n_dates = len(unique_dates)

    if n_dates < MIN_UNIQUE_DATES:
        return (
            CheckResult("temporal_coverage", Status.FAIL,
                        f"only {n_dates} unique S2 dates (expected >= {MIN_UNIQUE_DATES})"),
            n_dates,
        )

    months = pc.unique(pc.month(unique_dates))
    n_months = len(months)
    if n_months < MIN_MONTHS_COVERED:
        return (
            CheckResult("temporal_coverage", Status.WARN,
                        f"only {n_months} calendar months covered (expected >= {MIN_MONTHS_COVERED})"),
            n_dates,
        )
    return CheckResult("temporal_coverage", Status.PASS), n_dates


def check_bbox(pf: pq.ParquetFile, bbox: list[float]) -> CheckResult:
    """bbox = [lon_min, lat_min, lon_max, lat_max]"""
    tbl = pf.read(columns=["lon", "lat", "source"])
    mask = pc.equal(tbl.column("source"), "S2")
    s2 = tbl.filter(mask)
    if len(s2) == 0:
        return CheckResult("bbox", Status.PASS)

    lon_min, lat_min, lon_max, lat_max = bbox
    buf = BBOX_BUFFER_DEG
    lons = s2.column("lon")
    lats = s2.column("lat")

    out_lon = pc.sum(
        pc.or_(pc.less(lons, lon_min - buf), pc.greater(lons, lon_max + buf))
    ).as_py() or 0
    out_lat = pc.sum(
        pc.or_(pc.less(lats, lat_min - buf), pc.greater(lats, lat_max + buf))
    ).as_py() or 0
    n_out = out_lon + out_lat
    if n_out > 0:
        return CheckResult("bbox", Status.FAIL,
                           f"{n_out} pixels outside declared bbox + {buf}° buffer")
    return CheckResult("bbox", Status.PASS)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def validate_tile(
    path: Path,
    location_id: str,
    year: int,
    tile_id: str,
    expected_pixels: int,
    bbox: list[float],
) -> TileReport:
    report = TileReport(location_id=location_id, year=year, tile_id=tile_id, path=path)

    if not path.exists():
        report.checks.append(CheckResult("file_exists", Status.FAIL,
                                         f"missing: {path.relative_to(PROJECT_ROOT)}"))
        return report

    pf = pq.ParquetFile(path)
    report.n_rows = pf.metadata.num_rows

    # 1. Row count (fast — metadata only)
    rc = check_row_count(pf)
    report.checks.append(rc)
    if rc.status == Status.FAIL:
        return report

    # 2. Schema (fast — schema only)
    report.checks.append(check_schema(pf))
    if any(c.status == Status.FAIL for c in report.checks):
        return report

    # 3. S1 presence (reads last row group source col only)
    s1_result, has_s1 = check_source_presence(pf)
    report.checks.append(s1_result)
    report.has_s1 = has_s1

    n_s1_rgs = _count_s1_tail_rgs(pf) if has_s1 else 0

    # 4. Pixel count (full point_id + source column scan — narrow but full)
    px_result, n_pixels = check_pixel_count(pf, expected_pixels)
    report.checks.append(px_result)
    report.n_pixels = n_pixels

    # 5–8. Band/index/scl checks — shared sample read
    s2_sample = _read_s2_sample(pf, n_s1_rgs)
    report.checks.append(check_nan_bands(s2_sample))
    report.checks.append(check_band_range(s2_sample))
    report.checks.append(check_index_range(s2_sample))
    report.checks.append(check_scl_purity(s2_sample))

    # 9. Duplicates (5 sampled S2 row groups)
    report.checks.append(check_duplicates(pf, n_s1_rgs))

    # 10. S1 integrity (old-format detection + cross-contamination)
    if has_s1:
        s1i_result, s1_old = check_s1_integrity(pf)
        report.checks.append(s1i_result)
        report.s1_old_format = s1_old

    # 11. Temporal coverage (full date + source column scan)
    tc_result, n_dates = check_temporal_coverage(pf, year)
    report.checks.append(tc_result)
    report.n_dates = n_dates

    # 12. Bbox (full lon + lat + source column scan)
    report.checks.append(check_bbox(pf, bbox))

    return report


def validate_location(
    loc: "Location",
    years: list[int] | None = None,
) -> list[TileReport]:
    """Validate all tile parquets for a location.

    If years is None, uses all years found on disk. Also emits FAIL reports
    for expected tiles that are missing.
    """
    tile_paths = loc.parquet_tile_paths()
    if years is None:
        years = sorted(tile_paths.keys())
    if not years:
        return []

    expected_pixels = loc.pixel_count
    bbox = loc.bbox

    reports: list[TileReport] = []
    for year in years:
        year_paths = tile_paths.get(year, [])
        found_tile_ids = {p.stem for p in year_paths}

        # Validate files that exist
        for path in year_paths:
            tile_id = path.stem
            reports.append(
                validate_tile(path, loc.id, year, tile_id, expected_pixels, bbox)
            )

        # Emit FAIL for expected tiles that are missing
        for tile_id in loc.tile_ids():
            if tile_id not in found_tile_ids:
                reports.append(
                    validate_tile(
                        loc.parquet_path(year, tile_id),
                        loc.id, year, tile_id, expected_pixels, bbox,
                    )
                )

    return reports
