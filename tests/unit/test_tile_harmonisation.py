"""Unit tests for utils.tile_harmonisation.

All tests use synthetic in-memory DataFrames — no parquet files on disk.
The module under test does not exist yet; these tests define the contract.

Contract tests
--------------
  C1. calibrate(): exact 1.02× offset → scale ≈ 1/1.02 for non-reference tile
  C2. calibrate(): single tile present → empty correction table, no error
  C3. calibrate(): extreme ratio (2.0×) is clamped to 1.15 upper bound
  C4. load_corrections(): missing file → returns None (graceful fallback)
  C5. corrections applied: band values are multiplied by scale factor via join
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_overlap_parquet(tmp_path: Path, ratio: float = 1.02) -> Path:
    """Write a minimal pixel-sorted parquet with two tiles and known band ratio.

    Tile '54LWH' is the reference (more observations).
    Tile '54LWJ' has B07 values = ratio × reference B07.
    Both tiles observe the same 3 pixels on the same 5 dates in 2022.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    point_ids = ["p_0001_0001", "p_0001_0002", "p_0001_0003"]
    dates = [datetime(2022, 6, d) for d in [1, 5, 10, 15, 20]]
    bands = ["B04", "B05", "B07", "B08", "B11"]
    base_val = 0.10

    rows_h = []
    rows_j = []
    for pid in point_ids:
        for dt in dates:
            row_h = dict(
                point_id=pid,
                lon=142.0,
                lat=-15.5,
                date=dt,
                tile_id="54LWH",
                scl_purity=1.0,
                **{b: base_val for b in bands},
            )
            row_j = dict(
                point_id=pid,
                lon=142.0,
                lat=-15.5,
                date=dt,
                tile_id="54LWJ",
                scl_purity=1.0,
                **{b: base_val * (ratio if b == "B07" else 1.0) for b in bands},
            )
            rows_h.append(row_h)
            rows_j.append(row_j)

    # pixel-sorted: all observations for each pixel together, H before J
    all_rows = []
    for pid in point_ids:
        all_rows += [r for r in rows_h if r["point_id"] == pid]
        all_rows += [r for r in rows_j if r["point_id"] == pid]

    df = pl.DataFrame(all_rows).with_columns(pl.col("date").cast(pl.Datetime))
    out = tmp_path / "test.parquet"
    pq.write_table(df.to_arrow(), str(out))
    return out


# ---------------------------------------------------------------------------
# C1 — exact 1.02× offset → scale ≈ 1/1.02 for non-reference tile
# ---------------------------------------------------------------------------

def test_calibrate_exact_ratio(tmp_path):
    from utils.tile_harmonisation import calibrate

    parquet = _make_overlap_parquet(tmp_path, ratio=1.02)
    out = tmp_path / "corrections.parquet"

    result = calibrate(parquet, out, bands=["B07"])

    assert out.exists(), "calibrate() should write the output file"

    # 54LWH has more total observations (it is the reference) — no row for it.
    # 54LWJ row for B07 in 2022 should have scale ≈ 1/1.02
    j_row = result[(result["tile_id"] == "54LWJ") & (result["band"] == "B07") & (result["year"] == 2022)]
    assert len(j_row) == 1, "Expected exactly one correction row for (54LWJ, B07, 2022)"

    scale = j_row["scale_factor"].iloc[0]
    expected = 1.0 / 1.02
    assert math.isclose(scale, expected, rel_tol=0.005), (
        f"Expected scale ≈ {expected:.4f}, got {scale:.4f}"
    )


# ---------------------------------------------------------------------------
# C2 — single tile → empty correction table, no error
# ---------------------------------------------------------------------------

def test_calibrate_single_tile(tmp_path):
    from utils.tile_harmonisation import calibrate
    import pyarrow.parquet as pq

    rows = [
        dict(point_id="p_0001_0001", lon=142.0, lat=-15.5,
             date=datetime(2022, 6, 1), tile_id="54LWH", scl_purity=1.0,
             B04=0.1, B05=0.1, B07=0.1, B08=0.1, B11=0.1)
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime))
    parquet = tmp_path / "single.parquet"
    pq.write_table(df.to_arrow(), str(parquet))

    out = tmp_path / "corrections.parquet"
    result = calibrate(parquet, out, bands=["B07"])

    assert len(result) == 0, "Single tile should produce an empty correction table"


# ---------------------------------------------------------------------------
# C3 — extreme ratio is clamped to [0.85, 1.15]
# ---------------------------------------------------------------------------

def test_calibrate_clamping(tmp_path):
    from utils.tile_harmonisation import calibrate

    # ratio=2.0 → raw scale = 1/2.0 = 0.5, should be clamped to 0.85
    parquet = _make_overlap_parquet(tmp_path, ratio=2.0)
    out = tmp_path / "corrections_clamped.parquet"

    result = calibrate(parquet, out, bands=["B07"])

    j_row = result[(result["tile_id"] == "54LWJ") & (result["band"] == "B07")]
    assert len(j_row) == 1
    scale = j_row["scale_factor"].iloc[0]
    assert scale == pytest.approx(0.85, abs=1e-6), (
        f"Scale {scale:.4f} should be clamped to 0.85 (lower bound)"
    )


# ---------------------------------------------------------------------------
# C4 — load_corrections(): missing file → returns None
# ---------------------------------------------------------------------------

def test_load_corrections_missing_file(tmp_path):
    from utils.tile_harmonisation import load_corrections

    result = load_corrections(tmp_path / "nonexistent.parquet")
    assert result is None, "load_corrections() should return None for a missing file"


# ---------------------------------------------------------------------------
# C5 — corrections are applied: band values multiplied by scale factor
# ---------------------------------------------------------------------------

def test_corrections_applied_via_join():
    """Band values are scaled correctly by the join-based correction block."""
    from utils.tile_harmonisation import load_corrections
    import tempfile, os

    # Write a minimal correction table: 54LWJ B07 2022 → scale=0.98
    import pandas as pd
    corr_df = pd.DataFrame([
        {"tile_id": "54LWJ", "band": "B07", "year": 2022, "scale_factor": 0.98},
    ])
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    try:
        corr_df.to_parquet(tmp_path, index=False)
        corrections = load_corrections(Path(tmp_path))
    finally:
        os.unlink(tmp_path)

    assert corrections is not None
    assert ("54LWJ", "B07", 2022) in corrections
    assert math.isclose(corrections[("54LWJ", "B07", 2022)], 0.98)

    # Simulate the join-based correction block
    chunk = pl.DataFrame({
        "tile_id": ["54LWJ", "54LWH"],
        "year": [2022, 2022],
        "B07": [0.20, 0.20],
    })

    corr_rows = [
        {"tile_id": t, "band": b, "year": y, "scale": s}
        for (t, b, y), s in corrections.items()
    ]
    corr_pl = pl.DataFrame(corr_rows)
    band_corr = corr_pl.filter(pl.col("band") == "B07").select(["tile_id", "year", "scale"])

    chunk = (
        chunk
        .join(band_corr, on=["tile_id", "year"], how="left")
        .with_columns(
            pl.when(pl.col("scale").is_not_null())
              .then(pl.col("B07") * pl.col("scale"))
              .otherwise(pl.col("B07"))
              .alias("B07")
        )
        .drop("scale")
    )

    vals = dict(zip(chunk["tile_id"].to_list(), chunk["B07"].to_list()))
    assert math.isclose(vals["54LWJ"], 0.20 * 0.98, rel_tol=1e-6), (
        "54LWJ B07 should be scaled by 0.98"
    )
    assert math.isclose(vals["54LWH"], 0.20, rel_tol=1e-6), (
        "54LWH B07 has no correction entry — should be unchanged"
    )
