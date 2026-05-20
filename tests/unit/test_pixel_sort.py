"""Tests for signals._shared.is_pixel_sorted and ensure_pixel_sorted.

Real pipeline point IDs follow the convention "px_<easting_col>_<northing_row>"
(e.g. "px_0042_0031") — the "px" prefix means the northing row is at part index 2,
not index 1.  sort_parquet_by_pixel must group by northing row so that pixels in the
same geographic row are written to the same output row group.  Grouping by easting
column instead causes the score reader's buffer boundaries to bisect easting columns,
splitting pixel histories and producing regularly-spaced null stripes in output rasters.

Covers:
  PS-1  Single-row-group file is trivially sorted.
  PS-2  Two row groups with no overlap → sorted.
  PS-3  Two row groups that share a point_id → not sorted.
  PS-4  is_pixel_sorted only checks n_check=2 pairs by default — overlap at
        pair (2,3) in a 4-row-group file is missed (documents the known gap).
  PS-5  ensure_pixel_sorted returns original path when already sorted.
  PS-6  ensure_pixel_sorted creates a -by-pixel sidecar for unsorted input.
  PS-7  -by-pixel sidecar is actually pixel-sorted (is_pixel_sorted passes).
  PS-8  -by-pixel sidecar is reused on second call (no re-sort).
  PS-9  0-byte sidecar is deleted and re-sorted.
  PS-10 Sidecar stem: ensure_pixel_sorted(<tile>.parquet) → <tile>-by-pixel.parquet,
        so p.stem on the *original* path still yields the correct tile ID.
  PS-11 IDs without underscores cause ZeroDivisionError in sort_parquet_by_pixel
        (documents the known input-format requirement).
  PS-12 sort_parquet_by_pixel groups by northing row, not easting column, for real
        pipeline IDs ("px_<easting>_<northing>").  Pixels sharing the same northing
        row but different easting columns must never be split across output row groups.
  PS-13 sort_parquet_by_pixel does NOT mix different northing rows in a single output
        row group (catches the easting-axis regression in reverse).
"""

from __future__ import annotations

from pathlib import Path

import datetime

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.parquet_utils import ensure_pixel_sorted, is_pixel_sorted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet(path: Path, groups: list[list[str]]) -> Path:
    """Write a parquet where each inner list is one row group of point_ids."""
    writer = None
    schema = pa.schema([("point_id", pa.string())])
    for group in groups:
        tbl = pa.table({"point_id": group}, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(path, schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    return path


def _write_full_parquet(path: Path, groups: list[list[str]], rng) -> Path:
    """Write a parquet with all band columns — required by sort_parquet_by_pixel.

    Point IDs must follow the '<row>_<col>' convention used by the real pipeline
    (e.g. '0001_0002') so that sort_parquet_by_pixel can extract row-coords.
    """
    from tam.core.dataset import BAND_COLS

    start = datetime.date(2023, 1, 1)
    writer = None
    for pids in groups:
        n = len(pids)
        dates = [start + datetime.timedelta(days=20 * i) for i in range(n)]
        df = pl.DataFrame({
            "point_id": pids,
            "date": [datetime.datetime(d.year, d.month, d.day) for d in dates],
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32).tolist() for b in BAND_COLS},
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        tbl = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(path, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    return path


# ---------------------------------------------------------------------------
# PS-1  Single row group → trivially sorted
# ---------------------------------------------------------------------------

def test_single_row_group_is_sorted(tmp_path):
    p = _write_parquet(tmp_path / "single.parquet", [["px1", "px2", "px3"]])
    assert is_pixel_sorted(p) is True


# ---------------------------------------------------------------------------
# PS-2  Two non-overlapping row groups → sorted
# ---------------------------------------------------------------------------

def test_non_overlapping_groups_are_sorted(tmp_path):
    p = _write_parquet(tmp_path / "sorted.parquet", [["px1", "px2"], ["px3", "px4"]])
    assert is_pixel_sorted(p) is True


# ---------------------------------------------------------------------------
# PS-3  Two row groups sharing a point_id → not sorted
# ---------------------------------------------------------------------------

def test_overlapping_groups_not_sorted(tmp_path):
    p = _write_parquet(tmp_path / "unsorted.parquet", [["px1", "px2"], ["px2", "px3"]])
    assert is_pixel_sorted(p) is False


# ---------------------------------------------------------------------------
# PS-4  Overlap in an unsampled middle pair is missed when n_pairs > n_check
#        (documents the known limitation)
# ---------------------------------------------------------------------------

def test_is_pixel_sorted_misses_overlap_beyond_n_check(tmp_path):
    # 20 row groups → 19 pairs. n_check=2 samples pairs {0, 10, 18} — pair 5
    # is never checked.  Place the overlap exactly at pair 5 (groups[5]/groups[6]).
    n_groups = 20
    groups = [[f"px{i}"] for i in range(n_groups)]
    groups[6] = ["px5"]  # groups[5]=["px5"], groups[6]=["px5"] → overlap at pair 5
    p = _write_parquet(tmp_path / "mid_overlap.parquet", groups)
    # n_check=2 misses pair 5 → false positive
    assert is_pixel_sorted(p, n_check=2) is True   # known gap
    # n_check=19 checks all pairs → overlap detected
    assert is_pixel_sorted(p, n_check=19) is False


# ---------------------------------------------------------------------------
# PS-5  ensure_pixel_sorted returns the original path when already sorted
# ---------------------------------------------------------------------------

def test_ensure_returns_original_when_sorted(tmp_path):
    p = _write_parquet(tmp_path / "54LWH.parquet", [["px1", "px2"], ["px3", "px4"]])
    result = ensure_pixel_sorted(p)
    assert result == p


# ---------------------------------------------------------------------------
# PS-6  ensure_pixel_sorted creates a -by-pixel sidecar for unsorted input
# ---------------------------------------------------------------------------

def test_ensure_creates_sidecar_when_unsorted(tmp_path):
    # 0001_0001 spans both row groups → genuinely unsorted.
    rng = np.random.default_rng(0)
    p = _write_full_parquet(
        tmp_path / "54LWH.parquet",
        [["px_0001_0001", "px_0001_0002"], ["px_0001_0001", "px_0001_0003"]],
        rng,
    )
    assert not is_pixel_sorted(p)

    result = ensure_pixel_sorted(p)

    sidecar = tmp_path / "54LWH-by-pixel.parquet"
    assert result == sidecar
    assert sidecar.exists()
    assert sidecar.stat().st_size > 0


# ---------------------------------------------------------------------------
# PS-7  The sidecar produced by ensure_pixel_sorted is actually pixel-sorted
# ---------------------------------------------------------------------------

def test_sidecar_is_pixel_sorted(tmp_path):
    rng = np.random.default_rng(1)
    # Each pixel spans multiple row groups → genuinely unsorted.
    p = _write_full_parquet(
        tmp_path / "tile.parquet",
        [
            ["px_0001_0001", "px_0001_0002"],
            ["px_0001_0002", "px_0001_0003"],
            ["px_0001_0003", "px_0001_0004"],
        ],
        rng,
    )
    assert not is_pixel_sorted(p)
    sidecar = ensure_pixel_sorted(p)
    assert is_pixel_sorted(sidecar) is True


# ---------------------------------------------------------------------------
# PS-8  Second call reuses the cached sidecar without re-sorting
# ---------------------------------------------------------------------------

def test_ensure_reuses_cached_sidecar(tmp_path):
    rng = np.random.default_rng(2)
    p = _write_full_parquet(
        tmp_path / "tile.parquet",
        [["px_0001_0001", "px_0001_0002"], ["px_0001_0001", "px_0001_0003"]],
        rng,
    )

    sidecar1 = ensure_pixel_sorted(p)
    mtime1 = sidecar1.stat().st_mtime

    sidecar2 = ensure_pixel_sorted(p)
    mtime2 = sidecar2.stat().st_mtime

    assert sidecar1 == sidecar2
    assert mtime1 == mtime2  # file was not rewritten


# ---------------------------------------------------------------------------
# PS-9  0-byte sidecar is deleted and re-sorted
# ---------------------------------------------------------------------------

def test_ensure_replaces_zero_byte_sidecar(tmp_path):
    rng = np.random.default_rng(3)
    p = _write_full_parquet(
        tmp_path / "tile.parquet",
        [["px_0001_0001", "px_0001_0002"], ["px_0001_0001", "px_0001_0003"]],
        rng,
    )

    # Plant a 0-byte sidecar to simulate a prior crashed sort.
    sidecar = tmp_path / "tile-by-pixel.parquet"
    sidecar.write_bytes(b"")

    result = ensure_pixel_sorted(p)

    assert result == sidecar
    assert sidecar.stat().st_size > 0
    assert is_pixel_sorted(sidecar) is True


# ---------------------------------------------------------------------------
# PS-10  Tile ID derivation: p.stem before ensure_pixel_sorted is correct
#         (guards the _cmd_score pattern: tid = p.stem; ensure_pixel_sorted(p))
# ---------------------------------------------------------------------------

def test_tile_id_stem_unaffected_by_ensure(tmp_path):
    """The tile ID derived from p.stem before calling ensure_pixel_sorted is correct.

    In _cmd_score: tid = p.stem; ... ensure_pixel_sorted(p)
    If p is already sorted, result == p and result.stem == p.stem (same tile ID).
    If p is unsorted, result.stem == "<stem>-by-pixel" — but tid was already
    captured from p.stem, so the tile_year_map key is correct regardless.

    This test documents that the pattern is safe and the sidecar stem would
    be wrong if someone mistakenly derived tid from the *return value* instead.
    """
    p = tmp_path / "54LWH.parquet"
    _write_parquet(p, [["px1", "px2"], ["px3", "px4"]])  # already sorted

    tid_before = p.stem
    result = ensure_pixel_sorted(p)
    tid_from_result = result.stem

    assert tid_before == "54LWH"
    assert tid_from_result == "54LWH"  # both agree when already sorted

    # Now make a genuinely unsorted file (0001_0001 spans both row groups).
    rng2 = np.random.default_rng(4)
    p2 = _write_full_parquet(
        tmp_path / "55KBA.parquet",
        [["px_0001_0001", "px_0001_0002"], ["px_0001_0001", "px_0001_0003"]],
        rng2,
    )

    tid_before2 = p2.stem
    result2 = ensure_pixel_sorted(p2)
    tid_from_result2 = result2.stem

    assert tid_before2 == "55KBA"
    # BUG CANARY: if someone derives tid from result2.stem they get "55KBA-by-pixel"
    assert tid_from_result2 == "55KBA-by-pixel"
    assert tid_before2 != tid_from_result2  # confirms the risk if pattern is changed


# ---------------------------------------------------------------------------
# PS-11  IDs without underscores sort without error (northing key falls back to None)
# ---------------------------------------------------------------------------

def test_sort_parquet_by_pixel_tolerates_no_underscore_ids(tmp_path):
    """sort_parquet_by_pixel no longer crashes on IDs without underscores.

    The northing key extraction (str.splitn("_", 4).list.get(2)) returns null
    for IDs like 'px1' — Polars sorts nulls first, then falls back to point_id
    order.  The output is valid and pixel-sorted.

    Real pipeline IDs follow 'px_<easting>_<northing>' so this is an edge case,
    but it should not crash.
    """
    from utils.parquet_utils import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(11)
    _start = datetime.date(2023, 1, 1)
    df = pl.DataFrame({
        "point_id": ["px2", "px1"],
        "date": [datetime.datetime(_start.year, _start.month, _start.day + 20 * i) for i in range(2)],
        "scl_purity": [1.0, 1.0],
        **{b: rng.uniform(0, 1, 2).astype(np.float32).tolist() for b in BAND_COLS},
    }).with_columns(pl.col("date").cast(pl.Datetime("us")))
    src = tmp_path / "any_ids.parquet"
    pq.write_table(df.to_arrow(), src, row_group_size=1)

    dst = tmp_path / "any_ids_sorted.parquet"
    sort_parquet_by_pixel(src, dst)
    assert dst.exists() and dst.stat().st_size > 0


# ---------------------------------------------------------------------------
# PS-12  sort_parquet_by_pixel sorts by northing row for real pipeline IDs
#
# Real IDs: "px_<easting_col>_<northing_row>"
# The sort key is the northing component so all pixels sharing a northing row
# are contiguous in the output stream, regardless of easting.  Lexicographic
# sort on the full point_id would group by easting instead.
# ---------------------------------------------------------------------------

def test_sort_groups_by_northing_not_easting(tmp_path):
    """All observations for pixels sharing a northing row must be contiguous
    in the sorted output — they must not be interleaved with observations from
    a different northing row.

    Grid layout (easting × northing):
      easting 0000: northing 0000, 0001
      easting 0001: northing 0000, 0001

    After sort: northing 0000 rows come before northing 0001 rows; within
    each northing, easting order is the secondary key.
    """
    from utils.parquet_utils import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(12)

    pids = [
        "px_0000_0000",  # easting 0, northing 0
        "px_0001_0000",  # easting 1, northing 0
        "px_0000_0001",  # easting 0, northing 1
        "px_0001_0001",  # easting 1, northing 1
    ]

    def _make_df(pid_list):
        n = len(pid_list)
        dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=20 * i) for i in range(n)]
        return pl.DataFrame({
            "point_id": pid_list,
            "date": dates,
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32).tolist() for b in BAND_COLS},
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))

    src = tmp_path / "tile.parquet"
    writer = None
    for _ in range(3):
        tbl = _make_df(pids).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(src, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()

    dst = tmp_path / "tile_sorted.parquet"
    sort_parquet_by_pixel(src, dst, row_group_size=1)  # 1 row/rg so each row is individually inspectable

    result = pl.read_parquet(dst)
    northings = [pid.split("_")[2] for pid in result["point_id"].to_list()]

    # Northing values must not alternate — once we leave northing 0000 we must
    # never return to it.
    seen = set()
    prev = None
    for n in northings:
        if n != prev:
            assert n not in seen, (
                f"Northing {n!r} reappears after other northings — sort is not grouping by northing"
            )
            seen.add(n)
            prev = n


# ---------------------------------------------------------------------------
# PS-13  All northing rows appear in the output and observations are contiguous
#        (inverse of PS-12: confirms no northing row is missing or split)
# ---------------------------------------------------------------------------

def test_sort_does_not_mix_northing_rows(tmp_path):
    """Pixels from different northing rows must not be interleaved in the output.
    With 3 distinct northing rows the sorted stream must contain all observations
    for northing 0000 before any for 0001, and all 0001 before any 0002.
    """
    from utils.parquet_utils import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(13)

    pids = ["px_0000_0000", "px_0000_0001", "px_0000_0002"]

    def _make_df(pid_list):
        n = len(pid_list)
        dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=20 * i) for i in range(n)]
        return pl.DataFrame({
            "point_id": pid_list,
            "date": dates,
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32).tolist() for b in BAND_COLS},
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))

    src = tmp_path / "tile.parquet"
    tbl = _make_df(pids * 4).to_arrow()
    pq.write_table(tbl, src)

    dst = tmp_path / "tile_sorted.parquet"
    sort_parquet_by_pixel(src, dst)

    result = pl.read_parquet(dst)
    northings = [pid.split("_")[2] for pid in result["point_id"].to_list()]

    assert set(northings) == {"0000", "0001", "0002"}

    seen: set[str] = set()
    prev: str | None = None
    for n in northings:
        if n != prev:
            assert n not in seen, f"Northing {n!r} reappears — rows are not contiguous by northing"
            seen.add(n)
            prev = n
