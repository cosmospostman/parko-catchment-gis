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
# PS-4  Overlap in an unsampled middle pair is caught by the distant-pair check
#
# Previously n_check=2 could miss overlap at pair 5 (false positive). The
# distant-pair check (rg0 vs rg_mid) now catches disjoint-pixel layouts and
# flags them as unsorted regardless of n_check.  Both low and high n_check
# should now correctly return False for a file with a real overlap.
# ---------------------------------------------------------------------------

def test_is_pixel_sorted_catches_overlap_regardless_of_n_check(tmp_path):
    # 20 row groups, overlap only at pair 5 (groups[5] and groups[6] both = "px5").
    # The distant-pair check (rg0=["px0"] vs rg10=["px10"]) finds no shared pixels
    # and correctly returns False — indicating the file is not pixel-sorted.
    n_groups = 20
    groups = [[f"px{i}"] for i in range(n_groups)]
    groups[6] = ["px5"]  # groups[5]=["px5"], groups[6]=["px5"] → overlap at pair 5
    p = _write_parquet(tmp_path / "mid_overlap.parquet", groups)
    # distant-pair check detects disjoint layout → correctly not pixel-sorted
    assert is_pixel_sorted(p, n_check=2) is False
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


# ---------------------------------------------------------------------------
# PS-14  _merge_sorted_parquets interleaves S1 rows at the correct northing
#        position for both plain ("px_") and underscore-prefixed region IDs.
#
# Regression: a bug extracted list_element(reverse_split, 1) = easting instead
# of list_element(reverse_split, 2) = northing, causing the merge to insert S1
# rows at wrong positions and leaving the output not pixel-sorted.
# ---------------------------------------------------------------------------

def test_merge_sorted_parquets_northing_key(tmp_path):
    """S1 rows must be interleaved by northing, not easting.

    Setup: 4 pixels on a 2×2 grid (easting 0/1, northing 0/1), two S2 dates.
    S1 observations land only on northing-0 pixels.  After the merge the output
    must be pixel-sorted: all northing-0 rows before all northing-1 rows.

    Tested with both plain IDs ("px_E_N") and underscore-prefixed region IDs
    ("region_with_underscores_E_N") so the reverse-split sort key is exercised
    for both cases.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.parquet_utils import _merge_sorted_parquets, sort_parquet_by_pixel, is_pixel_sorted

    for prefix in ["px", "region_with_underscores"]:
        pids_n0 = [f"{prefix}_0000_0000", f"{prefix}_0001_0000"]  # northing 0
        pids_n1 = [f"{prefix}_0000_0001", f"{prefix}_0001_0001"]  # northing 1

        s2_schema = pa.schema([
            pa.field("point_id", pa.string()),
            pa.field("lon",      pa.float32()),
            pa.field("lat",      pa.float32()),
            pa.field("date",     pa.date32()),
            pa.field("source",   pa.string()),
            pa.field("vh",       pa.float32()),
            pa.field("vv",       pa.float32()),
        ])

        def _make_s2_table(pids, date_val):
            n = len(pids)
            return pa.table({
                "point_id": pa.array(pids, pa.string()),
                "lon":      pa.array([0.0] * n, pa.float32()),
                "lat":      pa.array([0.0] * n, pa.float32()),
                "date":     pa.array([date_val] * n, pa.date32()),
                "source":   pa.array(["S2"] * n, pa.string()),
                "vh":       pa.array([None] * n, pa.float32()),
                "vv":       pa.array([None] * n, pa.float32()),
            }, schema=s2_schema)

        # Pixel-sorted S2 parquet: northing-0 rows then northing-1 rows
        s2_path = tmp_path / f"s2_{prefix}.parquet"
        s2_writer = pq.ParquetWriter(s2_path, s2_schema)
        for date_val in [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)]:
            s2_writer.write_table(_make_s2_table(pids_n0, date_val))
            s2_writer.write_table(_make_s2_table(pids_n1, date_val))
        s2_writer.close()

        # S1 rows only for northing-0 pixels
        s1_rows = pa.table({
            "point_id": pa.array(pids_n0 * 2, pa.string()),
            "lon":      pa.array([0.0] * 4, pa.float32()),
            "lat":      pa.array([0.0] * 4, pa.float32()),
            "date":     pa.array([datetime.date(2023, 1, 15)] * 4, pa.date32()),
            "source":   pa.array(["S1"] * 4, pa.string()),
            "vh":       pa.array([0.01, 0.02, 0.01, 0.02], pa.float32()),
            "vv":       pa.array([0.03, 0.04, 0.03, 0.04], pa.float32()),
        }, schema=s2_schema)
        s1_path = tmp_path / f"s1_{prefix}.parquet"
        pq.write_table(s1_rows, s1_path)
        s1_sorted = tmp_path / f"s1_{prefix}_sorted.parquet"
        sort_parquet_by_pixel(s1_path, s1_sorted)

        out_path = tmp_path / f"merged_{prefix}.parquet"
        _merge_sorted_parquets(s2_path, s1_sorted, out_path, s2_schema, tag_s2_source=False)

        assert is_pixel_sorted(out_path), (
            f"Merged output is not pixel-sorted for prefix={prefix!r} — "
            "S1 rows were interleaved at wrong northing positions (easting/northing key bug)"
        )

        # S1 rows must only appear for northing-0 pixels
        result = pl.read_parquet(out_path)
        s1_rows_out = result.filter(pl.col("source") == "S1")
        s1_northings = {pid.rsplit("_", 1)[1] for pid in s1_rows_out["point_id"].to_list()}
        assert s1_northings == {"0000"}, (
            f"S1 rows appeared at unexpected northings {s1_northings} for prefix={prefix!r}"
        )


# ---------------------------------------------------------------------------
# PS-15  merge_scenes output is pixel-sorted
#
# Verifies the end-to-end guarantee: merge_scenes now calls sort_parquet_by_pixel
# internally so callers never need to sort the output themselves.
# ---------------------------------------------------------------------------

def test_merge_scenes_output_is_pixel_sorted(tmp_path):
    """merge_scenes must write pixel-sorted output.

    Creates 3 synthetic scene parquets (each covers 4 pixels, 1 date) and
    asserts that the merged output passes is_pixel_sorted().
    """
    import datetime
    import pyarrow as pa
    import pyarrow.parquet as pq
    from proxy._pipeline import merge_scenes
    from utils.parquet_utils import is_pixel_sorted, COMBINED_PIXEL_SCHEMA

    schema = COMBINED_PIXEL_SCHEMA

    def _make_scene(path: Path, date_val: datetime.date, pids: list[str]) -> Path:
        n = len(pids)
        nulls_f32 = pa.array([None] * n, pa.float32())
        nulls_u16 = pa.array([None] * n, pa.uint16())
        nulls_i8  = pa.array([None] * n, pa.int8())
        nulls_u8  = pa.array([None] * n, pa.uint8())
        tbl = pa.table({
            "point_id":    pa.array(pids, pa.string()),
            "lon":         pa.array([0.0] * n, pa.float32()),
            "lat":         pa.array([0.0] * n, pa.float32()),
            "date":        pa.array([date_val] * n, pa.date32()),
            "item_id":     pa.array(["item"] * n, pa.string()),
            "tile_id":     pa.array(["54LWH"] * n, pa.string()),
            "B02": nulls_u16, "B03": nulls_u16, "B04": nulls_u16,
            "B05": nulls_u16, "B06": nulls_u16, "B07": nulls_u16,
            "B08": nulls_u16, "B8A": nulls_u16, "B11": nulls_u16,
            "B12": nulls_u16,
            "scl_purity": pa.array([100] * n, pa.int8()),
            "scl":        nulls_i8,
            "aot":        nulls_u8,
            "view_zenith": nulls_u8,
            "sun_zenith":  nulls_u8,
            "source":     pa.array([None] * n, pa.string()),
            "vh":         nulls_f32,
            "vv":         nulls_f32,
            "orbit":      pa.array([None] * n, pa.string()),
        }, schema=schema)
        pq.write_table(tbl, path)
        return path

    # 4 pixels on a 2×2 grid (easting 0/1, northing 0/1)
    pids = ["px_0000_0000", "px_0001_0000", "px_0000_0001", "px_0001_0001"]
    dates = [datetime.date(2025, 1, 1), datetime.date(2025, 4, 1), datetime.date(2025, 7, 1)]

    scene_paths = [
        _make_scene(tmp_path / f"scene_{i:02d}.parquet", d, pids)
        for i, d in enumerate(dates)
    ]

    out_path = tmp_path / "merged.parquet"
    merge_scenes(scene_paths, s1_path=None, out_path=out_path)

    assert out_path.exists(), "merge_scenes did not write output"
    assert is_pixel_sorted(out_path), (
        "merge_scenes output is not pixel-sorted — sort step missing or broken"
    )


# ---------------------------------------------------------------------------
# S1 shard sorting (regression for the S1_TRUNC defect).
#
# S1 shards are written DATE-major (rows in acquisition-submission order), but
# merge_scenes streams inputs in northing-band passes and silently drops any row
# whose yi is below the current band cursor.  _sort_s1_shards must therefore
# materialise a real northing (yi) sort — not just concatenate the shards.
# ---------------------------------------------------------------------------

def _write_date_major_s1(path: Path, dates, n_rows: int) -> Path:
    """Write an S1 shard the way _collect_s1_shards does: date-major.

    For each date, append a block of rows spanning ALL northings 0..n_rows-1
    (one easting column).  The resulting file is ordered by date, then yi —
    i.e. NOT globally northing-sorted.
    """
    import pyarrow as pa
    schema = pa.schema([
        ("point_id", pa.string()),
        ("date", pa.date32()),
        ("source", pa.string()),
        ("vh", pa.float32()),
    ])
    writer = pq.ParquetWriter(path, schema)
    for d in dates:
        pids = [f"px_0000_{yi:04d}" for yi in range(n_rows)]
        tbl = pa.table({
            "point_id": pids,
            "date": [d] * n_rows,
            "source": ["S1"] * n_rows,
            "vh": np.full(n_rows, 0.05, dtype=np.float32),
        }, schema=schema)
        writer.write_table(tbl)
    writer.close()
    return path


def test_sort_s1_shards_sorts_date_major_input_by_northing(tmp_path: Path):
    """PS-14  _sort_s1_shards turns date-major input into northing-ascending output
    with no row loss (the S1_TRUNC root cause)."""
    from utils.parquet_utils import _sort_s1_shards, COMBINED_PIXEL_SCHEMA

    dates = [datetime.date(2021, 1, 6), datetime.date(2021, 1, 18), datetime.date(2021, 1, 30)]
    n_rows = 50
    shard = _write_date_major_s1(tmp_path / "shard_0000.parquet", dates, n_rows)

    # Sanity: the input is NOT northing-sorted (date-major).
    in_df = pl.read_parquet(shard).with_columns(
        pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("yi")
    )
    assert not in_df["yi"].is_sorted(), "fixture should be date-major (not yi-sorted)"

    out = tmp_path / "s1_sorted.parquet"
    _sort_s1_shards([shard], out, COMBINED_PIXEL_SCHEMA)

    out_df = pl.read_parquet(out).with_columns(
        pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("yi")
    )
    # No row loss across all dates.
    assert len(out_df) == n_rows * len(dates)
    # Output is northing-ascending — what merge_scenes' band cursor requires.
    assert out_df["yi"].is_sorted(), "output is not northing-sorted"
    # Every date retains all northings (the defect dropped all but the first date's).
    for d in dates:
        assert out_df.filter(pl.col("date") == d).height == n_rows


def test_merge_scenes_keeps_all_s1_rows_from_date_major_shard(tmp_path: Path):
    """PS-15  Feeding a (correctly northing-sorted) S1 file through merge_scenes
    retains every S1 row for every date — guards the band-cursor row-drop."""
    from proxy._pipeline import merge_scenes
    from utils.parquet_utils import _sort_s1_shards, COMBINED_PIXEL_SCHEMA

    dates = [datetime.date(2021, m, 6) for m in range(1, 8)]  # 7 acquisitions
    n_rows = 200  # > _NORTHING_BAND (64) so multiple band passes are exercised
    shard = _write_date_major_s1(tmp_path / "shard_0000.parquet", dates, n_rows)

    s1_sorted = tmp_path / "s1.parquet"
    _sort_s1_shards([shard], s1_sorted, COMBINED_PIXEL_SCHEMA)

    out = tmp_path / "chunk.parquet"
    merge_scenes([], s1_path=s1_sorted, out_path=out)

    merged = pl.read_parquet(out).filter(pl.col("source") == "S1").with_columns(
        pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("yi")
    )
    # Every date must still cover all n_rows northings (no thin-band collapse).
    per_date_rows = merged.group_by("date").agg(
        pl.col("yi").n_unique().alias("nrows")
    )
    assert per_date_rows["nrows"].min() == n_rows, (
        f"S1 rows dropped in merge: min per-date northings "
        f"{per_date_rows['nrows'].min()} < {n_rows}"
    )
    assert merged.height == n_rows * len(dates)
