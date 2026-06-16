"""Unit tests for utils/chunk_verify.py — chunkstore integrity verification.

All tests build tiny synthetic chunk parquets on disk (no network) so the S1
truncation detector is exercised deterministically.

Tests
-----
CV-1. A healthy chunk (every S1 date covers all rows) passes.
CV-2. A truncated chunk (most S1 dates cover a thin row band) is flagged S1_TRUNC.
CV-3. A chunk with no S1 rows is flagged S1_MISS.
CV-4. An empty parquet is flagged EMPTY.
CV-5. parse_chunk_stem extracts (row, col) and rejects non-chunk stems.
CV-6. verify_chunkstore walks {year}/{tile}/*.parquet and honours filters.
CV-7. verify_chunk infers year/tile from the path layout.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.chunk_verify import (
    S1_TRUNCATION_MED_FRAC,
    parse_chunk_stem,
    verify_chunk,
    verify_chunkstore,
)

_SCHEMA = pa.schema([
    ("point_id", pa.string()),
    ("lon", pa.float64()),
    ("lat", pa.float64()),
    ("date", pa.date32()),
    ("source", pa.string()),
    ("vh", pa.float32()),
])


def _write_chunk(
    path: Path,
    *,
    n_xi: int = 8,
    n_yi: int = 100,
    s1_dates: int = 10,
    s1_rows_per_date: int | None = None,
) -> Path:
    """Write a synthetic chunk parquet.

    Every pixel (xi, yi) gets one S2 row (source=None).  Each of ``s1_dates``
    dates writes S1 rows for the first ``s1_rows_per_date`` yi rows (defaults to
    all rows = full coverage).  point_id format matches the pipeline:
    ``px_{xi:04d}_{yi:04d}``.
    """
    if s1_rows_per_date is None:
        s1_rows_per_date = n_yi

    pid, lon, lat, date, source, vh = [], [], [], [], [], []
    import datetime

    for yi in range(n_yi):
        for xi in range(n_xi):
            p = f"px_{xi:04d}_{yi:04d}"
            # one S2 obs so the chunk has pixels/rows even with no S1
            pid.append(p); lon.append(145.0 + xi * 1e-4); lat.append(-16.0 - yi * 1e-4)
            date.append(datetime.date(2025, 1, 1)); source.append(None); vh.append(None)

    for d in range(s1_dates):
        dt = datetime.date(2025, 1, 1) + datetime.timedelta(days=12 * (d + 1))
        for yi in range(s1_rows_per_date):
            for xi in range(n_xi):
                pid.append(f"px_{xi:04d}_{yi:04d}")
                lon.append(145.0 + xi * 1e-4); lat.append(-16.0 - yi * 1e-4)
                date.append(dt); source.append("S1"); vh.append(0.05)

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"point_id": pid, "lon": lon, "lat": lat, "date": date,
                  "source": source, "vh": vh}, schema=_SCHEMA),
        path,
    )
    return path


def test_cv1_healthy_chunk_passes(tmp_path: Path):
    p = _write_chunk(tmp_path / "2025" / "54TST" / "54TST_r00_c00.parquet")
    rep = verify_chunk(p)
    assert rep.ok, rep.issues
    assert rep.s1_med_frac == pytest.approx(1.0)
    assert rep.s1_max_frac == pytest.approx(1.0)
    assert rep.pct_ge_min_obs == pytest.approx(100.0)


def test_cv2_truncated_chunk_flagged(tmp_path: Path):
    # Most dates cover only 10 of 100 rows -> 0% complete dates -> S1_INCOMPLETE.
    p = _write_chunk(
        tmp_path / "2025" / "54TST" / "54TST_r01_c00.parquet",
        n_yi=100, s1_dates=10, s1_rows_per_date=10,
    )
    rep = verify_chunk(p)
    assert not rep.ok
    assert any(i.startswith("S1_INCOMPLETE") for i in rep.issues)
    assert rep.s1_complete_date_frac == pytest.approx(0.0)


def test_cv2b_one_full_date_does_not_rescue_truncated(tmp_path: Path):
    # The real defect: 9 truncated dates + 1 full date.  Only 1/10 dates is
    # complete, far below the 95% completeness gate, so still flagged.
    import datetime
    base = tmp_path / "2025" / "54TST" / "54TST_r02_c00.parquet"
    p = _write_chunk(base, n_yi=100, s1_dates=9, s1_rows_per_date=10)
    # append one full-coverage date
    t = pq.read_table(p)
    pid, lon, lat, date, source, vh = ([] for _ in range(6))
    dt = datetime.date(2025, 12, 1)
    for yi in range(100):
        for xi in range(8):
            pid.append(f"px_{xi:04d}_{yi:04d}")
            lon.append(145.0); lat.append(-16.0); date.append(dt)
            source.append("S1"); vh.append(0.05)
    extra = pa.table({"point_id": pid, "lon": lon, "lat": lat, "date": date,
                      "source": source, "vh": vh}, schema=_SCHEMA)
    pq.write_table(pa.concat_tables([t, extra]), p)

    rep = verify_chunk(p)
    assert not rep.ok
    assert rep.s1_max_frac == pytest.approx(1.0)
    assert rep.s1_complete_date_frac == pytest.approx(0.1, abs=0.01)  # 1 of 10 dates


def test_cv2d_partially_damaged_chunk_caught_by_completeness_gate(tmp_path: Path):
    """A chunk the OLD median gate would PASS but is still missing rows on many
    dates — caught only by the strict complete-date gate.

    Half the dates are full (100 rows), half cover 60 rows.  median coverage =
    0.8 (> old 0.5 threshold → old gate passes), but only 50% of dates are
    'complete' (>=95%) → new gate FAILS.  This is the case the user worried about:
    'passes verify but may still be missing data'.
    """
    import datetime
    pid, lon, lat, date, source, vh = ([] for _ in range(6))
    n_yi, n_xi = 100, 8
    # S2 base layer (full grid)
    for yi in range(n_yi):
        for xi in range(n_xi):
            pid.append(f"px_{xi:04d}_{yi:04d}"); lon.append(145.0); lat.append(-16.0)
            date.append(datetime.date(2025, 1, 1)); source.append(None); vh.append(None)
    # 10 S1 dates: 5 full (100 rows), 5 partial (60 rows)
    for d in range(10):
        dt = datetime.date(2025, 1, 1) + datetime.timedelta(days=20 * (d + 1))
        nrows = 100 if d % 2 == 0 else 60
        for yi in range(nrows):
            for xi in range(n_xi):
                pid.append(f"px_{xi:04d}_{yi:04d}"); lon.append(145.0); lat.append(-16.0)
                date.append(dt); source.append("S1"); vh.append(0.05)
    p = tmp_path / "2025" / "54TST" / "54TST_r09_c00.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"point_id": pid, "lon": lon, "lat": lat, "date": date,
                             "source": source, "vh": vh}, schema=_SCHEMA), p)

    rep = verify_chunk(p)
    assert rep.s1_med_frac >= 0.5, "median should clear the OLD lenient gate"
    assert rep.s1_complete_date_frac == pytest.approx(0.5)  # only 5/10 dates complete
    assert not rep.ok, "strict completeness gate must flag the partially-damaged chunk"
    assert any(i.startswith("S1_INCOMPLETE") for i in rep.issues)


def test_cv2c_harmless_truncation_is_note_not_failure(tmp_path: Path):
    # Low per-date row coverage, but every pixel still reaches >=4 obs across the
    # year (each date covers a different thin band, cycling over all rows).  The
    # scorer keeps all pixels, so this must be an INFO note, not a FAIL.
    import datetime
    n_yi, n_xi, band = 20, 4, 5  # 20 rows; each date covers 5 rows
    pid, lon, lat, date, source, vh = ([] for _ in range(6))
    # S2 base layer
    for yi in range(n_yi):
        for xi in range(n_xi):
            pid.append(f"px_{xi:04d}_{yi:04d}"); lon.append(145.0); lat.append(-16.0)
            date.append(datetime.date(2025, 1, 1)); source.append(None); vh.append(None)
    # 16 dates, sliding 5-row window cycling over all rows -> each pixel covered
    # in 4 separate dates (>=4 obs) but no date covers > 25% of rows.
    for d in range(16):
        dt = datetime.date(2025, 1, 1) + datetime.timedelta(days=12 * (d + 1))
        start = (d * band) % n_yi
        for k in range(band):
            yi = (start + k) % n_yi
            for xi in range(n_xi):
                pid.append(f"px_{xi:04d}_{yi:04d}"); lon.append(145.0); lat.append(-16.0)
                date.append(dt); source.append("S1"); vh.append(0.05)
    p = tmp_path / "2025" / "54TST" / "54TST_r05_c00.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"point_id": pid, "lon": lon, "lat": lat, "date": date,
                             "source": source, "vh": vh}, schema=_SCHEMA), p)

    rep = verify_chunk(p)
    assert rep.s1_med_frac < S1_TRUNCATION_MED_FRAC      # truncated per-date
    assert rep.pct_ge_min_obs == pytest.approx(100.0)    # but all pixels scored
    assert rep.ok, rep.issues                            # not a failure
    assert any(n.startswith("S1_TRUNC_OK") for n in rep.notes)


def test_cv3_no_s1_flagged(tmp_path: Path):
    p = _write_chunk(tmp_path / "2025" / "54TST" / "54TST_r03_c00.parquet", s1_dates=0)
    rep = verify_chunk(p)
    assert not rep.ok
    assert any(i.startswith("S1_MISS") for i in rep.issues)


def test_cv4_empty_parquet_flagged(tmp_path: Path):
    p = tmp_path / "2025" / "54TST" / "54TST_r04_c00.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k.name: [] for k in _SCHEMA}, schema=_SCHEMA), p)
    rep = verify_chunk(p)
    assert not rep.ok
    assert any(i.startswith("EMPTY") for i in rep.issues)


def test_cv5_parse_chunk_stem():
    assert parse_chunk_stem("55KCB_r01_c00") == (1, 0)
    assert parse_chunk_stem("55KCB_r10_c03") == (10, 3)
    assert parse_chunk_stem("55KCB_summary") is None
    assert parse_chunk_stem("random") is None


def test_cv6_verify_chunkstore_walk_and_filters(tmp_path: Path):
    root = tmp_path / "store"
    _write_chunk(root / "2024" / "54AAA" / "54AAA_r00_c00.parquet")
    _write_chunk(root / "2025" / "54AAA" / "54AAA_r00_c00.parquet")
    _write_chunk(root / "2025" / "54BBB" / "54BBB_r00_c00.parquet")
    # a non-chunk parquet should be ignored
    _write_chunk(root / "2025" / "54BBB" / "54BBB_grid.parquet")

    assert len(verify_chunkstore(root)) == 3
    assert len(verify_chunkstore(root, year=2025)) == 2
    assert len(verify_chunkstore(root, tile="54AAA")) == 2
    assert len(verify_chunkstore(root, year=2025, tile="54BBB")) == 1


def test_cv8_progress_callbacks_fire(tmp_path: Path):
    root = tmp_path / "store"
    _write_chunk(root / "2025" / "54AAA" / "54AAA_r00_c00.parquet")
    _write_chunk(root / "2025" / "54AAA" / "54AAA_r01_c00.parquet")

    totals, progress = [], []
    verify_chunkstore(
        root,
        on_start=lambda total: totals.append(total),
        on_progress=lambda done, rep: progress.append((done, rep.tile)),
    )
    assert totals == [2]                       # on_start called once with the count
    assert [d for d, _ in progress] == [1, 2]  # on_progress called per chunk, in order


def test_cv7_year_tile_inferred_from_path(tmp_path: Path):
    p = _write_chunk(tmp_path / "2023" / "55KCB" / "55KCB_r07_c02.parquet")
    rep = verify_chunk(p)
    assert rep.year == 2023
    assert rep.tile == "55KCB"
    assert (rep.chunk_row, rep.chunk_col) == (7, 2)
