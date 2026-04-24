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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from signals._shared import ensure_pixel_sorted, is_pixel_sorted


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

    writer = None
    for pids in groups:
        n = len(pids)
        df = pd.DataFrame({
            "point_id": pids,
            "date": pd.date_range("2023-01-01", periods=n, freq="20D").astype("datetime64[us]"),
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32) for b in BAND_COLS},
        })
        tbl = pa.Table.from_pandas(df, preserve_index=False)
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
# PS-4  Overlap at pair (2,3) is missed by default n_check=2
#        (documents the known limitation)
# ---------------------------------------------------------------------------

def test_is_pixel_sorted_misses_overlap_beyond_n_check(tmp_path):
    # Groups 0+1 and 1+2 are clean; only 2+3 has overlap.
    p = _write_parquet(
        tmp_path / "late_overlap.parquet",
        [["px1"], ["px2"], ["px3"], ["px3"]],  # overlap at pair (2, 3)
    )
    # Default n_check=2 checks pairs (0,1) and (1,2) — misses (2,3).
    assert is_pixel_sorted(p, n_check=2) is True   # false positive — known gap
    # With n_check=3 the overlap is detected.
    assert is_pixel_sorted(p, n_check=3) is False


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
# PS-11  IDs without underscores crash sort_parquet_by_pixel (known fragility)
# ---------------------------------------------------------------------------

def test_sort_parquet_by_pixel_requires_underscore_ids(tmp_path):
    """sort_parquet_by_pixel silently finds zero row-coords for IDs without
    underscores (e.g. 'px1'), then divides by zero.

    This documents that the real pipeline pixel IDs must follow the
    'px_<easting>_<northing>' convention for the sort to work.
    """
    from signals._shared import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "point_id": ["px1", "px2"],
        "date": pd.date_range("2023-01-01", periods=2, freq="20D").astype("datetime64[us]"),
        "scl_purity": [1.0, 1.0],
        **{b: rng.uniform(0, 1, 2).astype(np.float32) for b in BAND_COLS},
    })
    src = tmp_path / "bad_ids.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), src, row_group_size=1)

    dst = tmp_path / "bad_ids_sorted.parquet"
    with pytest.raises(ZeroDivisionError):
        sort_parquet_by_pixel(src, dst)


# ---------------------------------------------------------------------------
# PS-12  sort_parquet_by_pixel groups by northing row for real pipeline IDs
#
# Real IDs: "px_<easting_col>_<northing_row>"
# The sort must bucket by the northing component (part index 2), not the
# easting component (part index 1).  After sorting, all pixels sharing a
# northing row coordinate must land in the same output row group —
# regardless of how many different easting columns they occupy.
# ---------------------------------------------------------------------------

def test_sort_groups_by_northing_not_easting(tmp_path):
    """Pixels in the same northing row but different easting columns must be
    written to the same output row group.

    Grid layout (easting × northing):
      easting 0000: northing 0000, 0001
      easting 0001: northing 0000, 0001

    Input: interleaved across 4 row groups so every pixel spans groups.
    Expected after sort: one row group per unique northing value (0000, 0001).
    If the sort mistakenly groups by easting, it produces one row group per
    easting (0000, 0001) instead — the assertion below would still pass for
    count but the northing-per-group check catches the axis confusion.
    """
    from signals._shared import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(12)

    # Two eastings, two northings → 4 pixels.
    # IDs follow the real pipeline convention: px_<easting>_<northing>.
    pids = [
        "px_0000_0000",  # easting 0, northing 0
        "px_0001_0000",  # easting 1, northing 0
        "px_0000_0001",  # easting 0, northing 1
        "px_0001_0001",  # easting 1, northing 1
    ]
    # Scatter observations: each row group contains one observation per pixel
    # so that no natural grouping by easting or northing pre-exists.
    n_obs = 4  # one obs per pixel per row group
    groups = []
    for _ in range(3):
        groups.append(pids)  # repeat all 4 pixels → 4 obs each, 3 row groups

    def _make_df(pid_list):
        n = len(pid_list)
        return pd.DataFrame({
            "point_id": pid_list,
            "date": pd.date_range("2023-01-01", periods=n, freq="20D").astype("datetime64[us]"),
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32) for b in BAND_COLS},
        })

    src = tmp_path / "tile.parquet"
    writer = None
    for pid_list in groups:
        tbl = pa.Table.from_pandas(_make_df(pid_list), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(src, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()

    dst = tmp_path / "tile_sorted.parquet"
    sort_parquet_by_pixel(src, dst)

    pf = pq.ParquetFile(dst)
    n_out = pf.metadata.num_row_groups

    # Collect the set of northing values present in each output row group.
    northings_per_group = []
    for i in range(n_out):
        tbl = pf.read_row_group(i, columns=["point_id"])
        ids = tbl.column("point_id").to_pylist()
        northings = {pid.split("_")[2] for pid in ids}
        northings_per_group.append(northings)

    # Every output row group must contain exactly one distinct northing value —
    # pixels from different geographic rows must never share a row group.
    for i, northings in enumerate(northings_per_group):
        assert len(northings) == 1, (
            f"Row group {i} contains mixed northing values {northings}; "
            "sort_parquet_by_pixel is grouping by easting instead of northing"
        )

    # Collect the set of easting values present in each output row group.
    eastings_per_group = []
    for i in range(n_out):
        tbl = pf.read_row_group(i, columns=["point_id"])
        ids = tbl.column("point_id").to_pylist()
        eastings = {pid.split("_")[1] for pid in ids}
        eastings_per_group.append(eastings)

    # Both easting columns (0000, 0001) must appear together in each group —
    # i.e. no row group is restricted to a single easting value.
    for i, eastings in enumerate(eastings_per_group):
        assert eastings == {"0000", "0001"}, (
            f"Row group {i} contains only eastings {eastings}; "
            "expected both easting columns to be co-located within a northing row"
        )


# ---------------------------------------------------------------------------
# PS-13  Each output row group spans exactly one northing row
#        (inverse of PS-12: catches the opposite axis confusion)
# ---------------------------------------------------------------------------

def test_sort_does_not_mix_northing_rows(tmp_path):
    """Pixels from different northing rows must never be placed in the same
    output row group.  With 3 distinct northing rows the output must have
    exactly 3 row groups (one per northing), not 1 (merged) or more than 3.
    """
    from signals._shared import sort_parquet_by_pixel
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(13)

    # 1 easting × 3 northings
    pids = ["px_0000_0000", "px_0000_0001", "px_0000_0002"]

    def _make_df(pid_list):
        n = len(pid_list)
        return pd.DataFrame({
            "point_id": pid_list,
            "date": pd.date_range("2023-01-01", periods=n, freq="20D").astype("datetime64[us]"),
            "scl_purity": [1.0] * n,
            **{b: rng.uniform(0, 1, n).astype(np.float32) for b in BAND_COLS},
        })

    src = tmp_path / "tile.parquet"
    # Write all 3 pixels into a single interleaved row group so the sort must
    # actually separate them.
    tbl = pa.Table.from_pandas(_make_df(pids * 4), preserve_index=False)
    pq.write_table(tbl, src, row_group_size=len(pids) * 4)

    dst = tmp_path / "tile_sorted.parquet"
    sort_parquet_by_pixel(src, dst)

    pf = pq.ParquetFile(dst)
    n_out = pf.metadata.num_row_groups

    assert n_out == 3, (
        f"Expected 3 output row groups (one per northing row), got {n_out}"
    )

    seen_northings = set()
    for i in range(n_out):
        tbl = pf.read_row_group(i, columns=["point_id"])
        northings = {pid.split("_")[2] for pid in tbl.column("point_id").to_pylist()}
        assert len(northings) == 1, f"Row group {i} mixes northing rows: {northings}"
        seen_northings.update(northings)

    assert seen_northings == {"0000", "0001", "0002"}, (
        f"Not all northing rows present in output: {seen_northings}"
    )
