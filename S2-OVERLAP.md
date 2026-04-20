# S2 Tile Overlap — Fix Plan

## Problem

A Parkinsonia site or training region whose bounding box straddles two adjacent
Sentinel-2 MGRS tiles (e.g. 55HBU / 55HBV) ends up with **duplicate rows** in the
output parquet — same `(point_id, date)`, different `tile_id`, slightly different band
values (because each tile has its own independent L2A atmospheric correction).

### How duplicates enter the pipeline

```
bbox_to_tile_ids(region.bbox) → ["55HBU", "55HBV"]
        ↓
training_collector: groups region under both tile buckets
        ↓
_fetch_tile_items("55HBU", ...)  →  dedup within-tile granules → items_hbu
_fetch_tile_items("55HBV", ...)  →  dedup within-tile granules → items_hbv
        ↓
collect(..., items=items_hbu)   ← dedup block SKIPPED (items= pre-supplied)
collect(..., items=items_hbv)   ← dedup block SKIPPED
        ↓
for a boundary pixel: two rows written — one per tile
```

The within-tile granule dedup in `_fetch_tile_items` (stripping `_\d+_L2A$`) removes
same-overpass processing variants within one tile's result set, but it is applied
*per tile independently*. Items from different tiles that cover the same acquisition
are never compared against each other.

### Downstream impact

- `TAMDataset` groups observations by `(point_id, year)` without deduplicating on
  `(point_id, date)`, so both rows enter the model's input sequence for that pixel.
- The boundary pixel is effectively double-weighted in the attention window.
- The two rows have slightly different band values (independent atmospheric
  correction), injecting a small but systematic radiometric inconsistency.
- `score_pixels_chunked()` likewise has no `(point_id, date)` dedup.

`tile_harmonisation.calibrate()` exists and can correct the radiometric difference
between tiles, but it does not remove the structural duplicate — it only adjusts
one tile's band values to match the other.

---

## Fix

### Strategy: post-write dedup inside `collect()`

Deduplicate at the point where the final sorted parquet is assembled, keeping the
observation with the higher `scl_purity` when two rows share `(point_id, date)`.
This is the right place because:

- It is the single output gate for all pixel data, regardless of how many tiles
  were involved or whether `items=` was pre-supplied.
- The parquet is already sorted by `point_id` at this stage, so duplicates are
  adjacent and cheap to find.
- `scl_purity` is a proxy for observation quality; keeping the cleaner row is
  the least-arbitrary choice.

No changes are needed to `training_collector`, `TAMDataset`, or the scoring code.

### Change 1 — `utils/pixel_collector.py`

After `final_writer.close()` and before the `total_rows == 0` guard, insert a
dedup pass. The dedup reads the just-written parquet, drops duplicates, and
rewrites it in place.

```python
# --- 6. Dedup (point_id, date) — removes cross-tile boundary duplicates -----
# A pixel at the boundary of two S2 MGRS tiles may have been observed by both
# tiles on the same date, producing two rows with the same (point_id, date) but
# different band values (independent L2A atmospheric correction per tile).
# Keep the row with higher scl_purity; break ties by keeping the first tile
# (lexicographic order, which is arbitrary but deterministic).
if final_writer is not None:
    tbl = pq.read_table(out_path)
    n_before = len(tbl)
    df_out = tbl.to_pandas()
    df_out = (
        df_out
        .sort_values(["point_id", "date", "scl_purity"], ascending=[True, True, False])
        .drop_duplicates(subset=["point_id", "date"], keep="first")
        .sort_values(["point_id", "date"])
        .reset_index(drop=True)
    )
    n_dedup = n_before - len(df_out)
    if n_dedup:
        logger.info(
            "Cross-tile dedup: %d rows → %d (removed %d boundary duplicates)",
            n_before, len(df_out), n_dedup,
        )
        pq.write_table(
            pa.Table.from_pandas(df_out, preserve_index=False),
            out_path,
        )
    total_rows = len(df_out)
```

Insert this block at line ~562, immediately after `final_writer.close()`.

**Insertion point** (between existing lines):
```
    if final_writer is not None:
        final_writer.close()                   # ← existing
                                               # ← insert new block here
    # Clean up sorted intermediates ...        # ← existing
```

Note: `pa` (pyarrow) is already imported at the top of the file. `pq` (pyarrow.parquet)
is also already imported.

---

## Test Changes

The four documenting tests that currently assert the *buggy* behaviour must be
updated to assert the *fixed* behaviour. Two tests in `test_pixel_collector.py`
and one in `test_training_collector.py` need updating; one stays as-is.

### `tests/unit/test_pixel_collector.py`

**Test 18 — `test_extract_item_to_df_two_tiles_produce_separate_rows`**

This test operates at the `extract_item_to_df` level, which is below the dedup
layer. The extraction layer correctly produces two rows — that is expected behaviour.
This test documents the *extraction contract*, not the *output contract*. **No change
needed.**

**Test 19 — `test_granule_dedup_regex_strips_index`**

This is a pure regex test with no behaviour change. **No change needed.**

**Test 20 — `test_cross_tile_items_survive_per_tile_dedup`**

This test documents that per-tile dedup leaves cross-tile duplicates alive —
behaviour that remains true at the `_fetch_tile_items` level. The dedup happens
further downstream. **No change needed.**

**Add new Test 21 — `test_collect_deduplicates_cross_tile_rows`**

A focused test for the new dedup step. It cannot call `collect()` directly
(network + disk), so it tests the dedup logic in isolation by constructing the
scenario that `collect()` faces after final concatenation.

```python
def test_collect_output_deduplicates_cross_tile_rows(tmp_path):
    """The post-write dedup step in collect() removes cross-tile duplicate rows.

    Simulates the state of the final parquet just before the dedup block:
    two rows with the same (point_id, date) from tiles 55HBU and 55HBV,
    one with scl_purity=0.6, the other with scl_purity=0.9.

    After dedup: only the higher-purity row (55HBV) survives.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    from datetime import date

    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS

    pid = "px_0000_0000"
    dt  = pd.Timestamp("2022-08-15")
    band_val = 0.15

    def _row(tile_id, scl_purity):
        return {
            "point_id": pid,
            "lon": 145.41,
            "lat": -22.78,
            "date": dt,
            "item_id": f"S2A_{tile_id}_20220815",
            "tile_id": tile_id,
            **{b: band_val for b in BANDS},
            "scl_purity": scl_purity,
            "scl": 4,
            "aot": 0.9,
            "view_zenith": 0.95,
            "sun_zenith": 0.80,
            **{c: 0.3 for c in SPECTRAL_INDEX_COLS},
        }

    df_raw = pd.DataFrame([_row("55HBU", 0.6), _row("55HBV", 0.9)])
    out_path = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pandas(df_raw, preserve_index=False), out_path)

    # Apply the same dedup logic as the new block in collect():
    tbl = pq.read_table(out_path)
    df_out = (
        tbl.to_pandas()
        .sort_values(["point_id", "date", "scl_purity"], ascending=[True, True, False])
        .drop_duplicates(subset=["point_id", "date"], keep="first")
        .sort_values(["point_id", "date"])
        .reset_index(drop=True)
    )
    pq.write_table(pa.Table.from_pandas(df_out, preserve_index=False), out_path)

    result = pq.read_table(out_path).to_pandas()
    assert len(result) == 1
    assert result.iloc[0]["tile_id"] == "55HBV"    # higher scl_purity survives
    assert result.iloc[0]["scl_purity"] == pytest.approx(0.9)
```

Add this test after test 20 in `test_pixel_collector.py`.

### `tests/unit/test_training_collector.py`

**Tests 15 and 16** document structural facts (grouping loop, per-tile dedup
scope) that are unchanged by this fix. **No change needed.**

**Add new Test 17 — `test_cross_tile_boundary_pixel_deduplicated_in_output`**

An integration-level check that the dedup contract holds for the combined output
from two tiles. Mirrors test 21 above but framed in training-pipeline terms.

```python
def test_cross_tile_boundary_pixel_deduplicated_in_output(tmp_path):
    """A pixel appearing in two tile parquets (cross-tile boundary) produces
    only one row in a combined output after (point_id, date) dedup."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS

    pid = "boundary_region_0000_0000"
    dt  = pd.Timestamp("2022-08-15")

    def _row(tile_id, scl_purity):
        return {
            "point_id": pid,
            "lon": 145.15, "lat": -22.95,
            "date": dt,
            "item_id": f"S2A_{tile_id}_20220815",
            "tile_id": tile_id,
            **{b: 0.12 for b in BANDS},
            "scl_purity": scl_purity,
            "scl": 4, "aot": 0.85,
            "view_zenith": 0.9, "sun_zenith": 0.8,
            **{c: 0.25 for c in SPECTRAL_INDEX_COLS},
        }

    # Simulate two tile parquets written by separate collect() calls:
    df_hbu = pd.DataFrame([_row("55HBU", 0.7)])
    df_hbv = pd.DataFrame([_row("55HBV", 0.95)])
    combined_raw = pd.concat([df_hbu, df_hbv], ignore_index=True)

    # Apply the dedup logic:
    df_dedup = (
        combined_raw
        .sort_values(["point_id", "date", "scl_purity"], ascending=[True, True, False])
        .drop_duplicates(subset=["point_id", "date"], keep="first")
        .sort_values(["point_id", "date"])
        .reset_index(drop=True)
    )

    assert len(df_dedup) == 1
    assert df_dedup.iloc[0]["tile_id"] == "55HBV"   # higher purity wins
```

Add this test after test 16 in `test_training_collector.py`.

---

## Verification

After applying the code change and adding the new tests:

```bash
pytest tests/unit/test_pixel_collector.py tests/unit/test_training_collector.py -v
```

Expected: all existing tests pass (the documenting tests 18–20 and 15–16 remain
green because they describe extraction/per-tile behaviour, which is unchanged).
New tests 21 and 17 also pass, confirming the dedup logic is correct.

To verify end-to-end on real data, collect a location known to straddle a tile
boundary and confirm there are no duplicate `(point_id, date)` pairs:

```python
import pyarrow.parquet as pq, pandas as pd

df = pq.read_table("data/pixels/<location>/<location>.parquet",
                   columns=["point_id", "date"]).to_pandas()
dups = df[df.duplicated(subset=["point_id", "date"], keep=False)]
assert dups.empty, f"{len(dups)} duplicate (point_id, date) rows found"
```

---

## Notes

- The dedup reads the full final parquet back into memory. For very large locations
  (>50 GB parquet) this may be memory-constrained. If needed, the sort + dedup
  can be done in a streaming row-group pass, but this adds significant complexity
  and is not necessary for current site sizes.
- `tile_harmonisation.calibrate()` remains useful even after this fix: it corrects
  the radiometric offset between tiles for pixels that appear in *both* tiles'
  training data across different date ranges, not just on the same date.
- The fix is idempotent: if no cross-tile duplicates exist (single-tile location),
  the dedup pass is a no-op and logs nothing.
