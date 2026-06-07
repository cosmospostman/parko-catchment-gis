"""Tests for tam.core.pixel_source — PixelSource abstraction.

Tests:
  1. test_parquet_pixel_source_delegates     — single-file wrapper
  2. test_chunk_pixel_source_concatenates    — multi-chunk row-group index mapping
  3. test_chunk_pixel_source_single_file     — single-path ChunkPixelSource == ParquetPixelSource
  4. test_score_pixels_chunked_accepts_chunk_source — end-to-end: ChunkPixelSource through scoring
  5. test_num_pixels_parquet                 — num_pixels() counts distinct point_ids
  6. test_num_pixels_chunk                   — num_pixels() unions across chunks, cached
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tam.core.pixel_source import ParquetPixelSource, PixelSource, ChunkPixelSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema([
    pa.field("point_id", pa.string()),
    pa.field("value", pa.float32()),
])


def _write_parquet(path: Path, tables: list[pa.Table]) -> None:
    """Write multiple row groups to a single parquet file."""
    writer = pq.ParquetWriter(path, schema=_SCHEMA)
    for tbl in tables:
        writer.write_table(tbl)
    writer.close()


def _make_rg(point_id: str, value: float) -> pa.Table:
    return pa.table(
        {"point_id": [point_id], "value": [np.float32(value)]},
        schema=_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Test 1 — ParquetPixelSource delegates correctly
# ---------------------------------------------------------------------------

class TestParquetPixelSource:
    def test_parquet_pixel_source_delegates(self, tmp_path):
        path = tmp_path / "single.parquet"
        rg0 = _make_rg("px_a", 1.0)
        rg1 = _make_rg("px_b", 2.0)
        _write_parquet(path, [rg0, rg1])

        src = ParquetPixelSource(path)

        assert src.num_row_groups == 2

        tbl0 = src.read_row_group(0, columns=["point_id", "value"])
        assert tbl0.to_pydict()["point_id"] == ["px_a"]
        assert tbl0.to_pydict()["value"] == pytest.approx([1.0])

        assert "point_id" in src.schema.names
        assert "value" in src.schema.names


# ---------------------------------------------------------------------------
# Test 2 — ChunkPixelSource concatenates row groups across strips
# ---------------------------------------------------------------------------

class TestChunkPixelSourceConcatenates:
    def test_strip_pixel_source_concatenates(self, tmp_path):
        # Three strips, each with 2 row groups → 6 total
        paths = []
        for i in range(3):
            p = tmp_path / f"strip_{i:02d}.parquet"
            rgs = [_make_rg(f"strip{i}_rg0", float(i * 10)), _make_rg(f"strip{i}_rg1", float(i * 10 + 1))]
            _write_parquet(p, rgs)
            paths.append(p)

        src = ChunkPixelSource(paths)

        assert src.num_row_groups == 6

        # Global rg index 4 → strip 2 (offset=4), local rg 0 → "strip2_rg0" with value=20.0
        tbl4 = src.read_row_group(4, columns=["point_id", "value"])
        assert tbl4.to_pydict()["point_id"] == ["strip2_rg0"]
        assert tbl4.to_pydict()["value"] == pytest.approx([20.0])

        # Global rg index 5 → strip 2, local rg 1 → "strip2_rg1" with value=21.0
        tbl5 = src.read_row_group(5, columns=["point_id", "value"])
        assert tbl5.to_pydict()["point_id"] == ["strip2_rg1"]
        assert tbl5.to_pydict()["value"] == pytest.approx([21.0])

        # Schema comes from first file
        assert "point_id" in src.schema.names


# ---------------------------------------------------------------------------
# Test 3 — Single-file ChunkPixelSource == ParquetPixelSource
# ---------------------------------------------------------------------------

class TestChunkPixelSourceSingleFile:
    def test_strip_pixel_source_single_file(self, tmp_path):
        path = tmp_path / "single_strip.parquet"
        _write_parquet(path, [_make_rg("px1", 1.5), _make_rg("px2", 2.5)])

        strip_src = ChunkPixelSource([path])
        parquet_src = ParquetPixelSource(path)

        assert strip_src.num_row_groups == parquet_src.num_row_groups

        for rg in range(strip_src.num_row_groups):
            t_strip = strip_src.read_row_group(rg, columns=["point_id", "value"])
            t_parq = parquet_src.read_row_group(rg, columns=["point_id", "value"])
            assert t_strip.to_pydict() == t_parq.to_pydict()

        assert set(strip_src.schema.names) == set(parquet_src.schema.names)


# ---------------------------------------------------------------------------
# Test 4 — score_pixels_chunked accepts ChunkPixelSource
# ---------------------------------------------------------------------------

# Number of observations per pixel-year — must clear min_obs_per_year threshold
_N_OBS = 15
_BAND_COLS_SUBSET = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "B01", "B09"]


def _stub_score_model():
    """Tiny S2-only TAMClassifier suitable for shape/smoke tests."""
    from tam.core.config import TAMConfig
    from tam.core.model import TAMClassifier
    from tam.core.dataset import ALL_FEATURE_COLS

    n_features = len(ALL_FEATURE_COLS)
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_features, use_s1=False, n_annual_features=0)
    torch.manual_seed(0)
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = False
    model._pixel_zscore = False
    model.eval()
    band_mean = np.zeros(n_features, dtype=np.float32)
    band_std = np.ones(n_features, dtype=np.float32)
    return model, band_mean, band_std


def _make_strip_parquet(tmp_path: Path, pixel_ids: list[str], year: int, filename: str) -> Path:
    """Write a minimal parquet strip with S2 band columns."""
    from tam.core.dataset import BAND_COLS

    rng = np.random.default_rng(42)
    rows = []
    start = datetime.date(year, 1, 15)
    for pid in pixel_ids:
        for i in range(_N_OBS):
            d = start + datetime.timedelta(days=23 * i)
            row = {
                "point_id": pid,
                "date": datetime.datetime(d.year, d.month, d.day),
                "scl_purity": 1.0,
            }
            for b in BAND_COLS:
                row[b] = float(rng.uniform(0.01, 0.5))
            rows.append(row)

    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    path = tmp_path / filename
    pq.write_table(df.to_arrow(), path, row_group_size=_N_OBS)
    return path


class TestScorePixelsChunkedAcceptsStripSource:
    def test_score_pixels_chunked_accepts_strip_source(self, tmp_path):
        from tam.core.score import score_pixels_chunked

        strip1 = _make_strip_parquet(tmp_path, ["px_strip1_a", "px_strip1_b"], 2022, "strip_00.parquet")
        strip2 = _make_strip_parquet(tmp_path, ["px_strip2_a", "px_strip2_b"], 2022, "strip_01.parquet")

        src = ChunkPixelSource([strip1, strip2])
        model, band_mean, band_std = _stub_score_model()

        result = score_pixels_chunked(
            source=src,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            device="cpu",
            mixed=False,
            decay=0.0,
        )

        assert "point_id" in result.columns
        assert "prob_tam" in result.columns

        scored_pids = set(result["point_id"].to_list())
        assert "px_strip1_a" in scored_pids
        assert "px_strip1_b" in scored_pids
        assert "px_strip2_a" in scored_pids
        assert "px_strip2_b" in scored_pids


# ---------------------------------------------------------------------------
# Test 5 — num_pixels() on ParquetPixelSource
# ---------------------------------------------------------------------------

class TestNumPixelsParquet:
    def test_num_pixels_parquet(self, tmp_path):
        path = tmp_path / "pixels.parquet"
        # 3 distinct point_ids, each repeated across 2 row groups
        rgs = [
            _make_rg("px_0001_0001", 1.0),
            _make_rg("px_0001_0002", 2.0),
            _make_rg("px_0001_0001", 3.0),  # duplicate — should not be double-counted
        ]
        _write_parquet(path, rgs)

        src = ParquetPixelSource(path)
        n = src.num_pixels()
        assert n == 2  # px_0001_0001 and px_0001_0002

        # Result is cached — second call returns same value without re-scanning
        assert src.num_pixels() is n or src.num_pixels() == n


# ---------------------------------------------------------------------------
# Test 6 — num_pixels() on ChunkPixelSource unions across strips
# ---------------------------------------------------------------------------

class TestNumPixelsStrip:
    def test_num_pixels_strip(self, tmp_path):
        # Strip 0: pixels A and B; Strip 1: pixels C and D (no overlap)
        p0 = tmp_path / "strip_00.parquet"
        p1 = tmp_path / "strip_01.parquet"
        _write_parquet(p0, [_make_rg("px_A", 1.0), _make_rg("px_B", 2.0)])
        _write_parquet(p1, [_make_rg("px_C", 3.0), _make_rg("px_D", 4.0)])

        src = ChunkPixelSource([p0, p1])
        assert src.num_pixels() == 4

    def test_num_pixels_strip_cached(self, tmp_path):
        path = tmp_path / "s.parquet"
        _write_parquet(path, [_make_rg("px_X", 1.0)])
        src = ChunkPixelSource([path])
        first = src.num_pixels()
        second = src.num_pixels()
        assert first == second == 1
        # Cache attribute should be set
        assert hasattr(src, "_num_pixels_cache")
