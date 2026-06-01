# S2 Tile Harmonisation — Implementation Plan

## Context

Horizontal streaking in TAM inference (e.g. 55KBB vs 55KCB at Quaids, B11 median 0.325 vs 0.232) is caused by inter-tile radiometric offset. The fix — `tile_harmonisation.calibrate()` + scale-factor application — currently only runs inside `compute_features_chunked()`, which is not used by the TAM training or scoring paths. The goal is to bake corrected band values directly into the pixel parquet so every consumer (training, scoring, signals, describe, explore) gets harmonised values automatically with no code changes to those consumers.

The most efficient place to apply corrections is **during the existing concat pass** in `collect()` (`utils/pixel_collector.py` lines 625–751). This pass already streams every row group exactly once; folding in the correction eliminates what would otherwise be a separate read/write cycle over the full parquet.

The sequence becomes:
1. All shards written and sorted ✓ (unchanged)
2. **`calibrate()` scans the sorted shards** to compute scale factors (new — replaces a full-parquet scan)
3. Concat pass applies corrections inline, deduplicates, writes final parquet ✓ (extended)

## Why calibrate() can run on the shards

The sorted shards collectively contain all the data that will end up in the final parquet. Overlap pixels (same `point_id`, same `date`, different `tile_id`) exist within individual shards — each shard covers a row-coord band, and pixels near tile boundaries appear in whichever shard covers their row. `calibrate()` already handles the tail-buffer pattern for pixels split across row-group boundaries, so scanning shards rather than the final parquet is equivalent.

---

## Changes

### 1. `utils/tile_harmonisation.py` — accept multiple paths in `calibrate()`

`calibrate()` currently takes a single `parquet_path`. Change the signature to accept a **list of paths**, scanning them all as if they were one file:

```python
def calibrate(
    parquet_paths: Path | list[Path],   # was: parquet_path: Path
    out_path: Path,
    bands: list[str] = _DEFAULT_BANDS,
) -> pd.DataFrame:
```

The internal row-group loop iterates `[parquet_paths]` if a single `Path` is given, or the list directly. Existing callers (the CLI `_main()`) pass a single path and are unaffected.

---

### 2. `utils/pixel_collector.py` — extend the concat block (lines 625–751)

#### 2a. New parameter on `collect()`

```python
def collect(
    ...,
    calibration_out: Path | None = None,   # <-- new
) -> None:
```

If `None`, the harmonisation step is skipped entirely (backwards compatible for callers that don't go through `Location.fetch()`).

#### 2b. Before the concat loop — run `calibrate()` on the sorted shards

Insert after `sorted_shard_paths` is finalised and before `logger.info("Concatenating ...")`:

```python
from utils.tile_harmonisation import calibrate, load_corrections

_corrections: dict | None = None
if calibration_out is not None:
    calibrate(sorted_shard_paths, calibration_out)
    _corrections = load_corrections(calibration_out)
```

#### 2c. New private function `_apply_corrections(tbl, corrections)`

Add near `_dedup_rg` (around line 659):

```python
_CORRECT_BANDS = ["B04", "B05", "B07", "B08", "B11"]

def _apply_corrections(tbl: pa.Table, corrections: dict) -> pa.Table:
    """Apply per-(tile_id, band, year) scale factors, then recompute stored indices."""
    import pyarrow.compute as pc

    years = pc.year(tbl.column("date"))
    tile_ids = tbl.column("tile_id")

    for band in _CORRECT_BANDS:
        if band not in tbl.schema.names:
            continue
        col = tbl.column(band).cast(pa.float32())
        scale = pa.array(
            [corrections.get((t.as_py(), band, y.as_py()), 1.0)
             for t, y in zip(tile_ids, years)],
            type=pa.float32(),
        )
        tbl = tbl.set_column(tbl.schema.get_field_index(band), band,
                             pc.multiply(col, scale))

    # Recompute stored spectral indices from corrected band values
    b08 = tbl.column("B08").cast(pa.float32())
    b04 = tbl.column("B04").cast(pa.float32())
    b03 = tbl.column("B03").cast(pa.float32())
    b02 = tbl.column("B02").cast(pa.float32())
    ndvi = pc.divide(pc.subtract(b08, b04), pc.add(b08, b04))
    ndwi = pc.divide(pc.subtract(b03, b08), pc.add(b03, b08))
    evi  = pc.multiply(
        pa.scalar(2.5, pa.float32()),
        pc.divide(
            pc.subtract(b08, b04),
            pc.add(pc.add(b08, pc.multiply(pa.scalar(6.0, pa.float32()), b04)),
                   pc.add(pc.multiply(pa.scalar(-7.5, pa.float32()), b02),
                          pa.scalar(1.0, pa.float32()))),
        ),
    )
    for name, arr in [("NDVI", ndvi), ("NDWI", ndwi), ("EVI", evi)]:
        if name in tbl.schema.names:
            tbl = tbl.set_column(tbl.schema.get_field_index(name), name,
                                 arr.cast(pa.float32()))
    return tbl
```

The Python-level `zip` loop for scale factors is O(n) and operates on a 5M-row buffer — acceptable. If it shows up in profiling it can be replaced with a Polars join (same pattern as `compute_features_chunked`).

#### 2d. Apply inside `_flush_write_buf()`

```python
def _flush_write_buf() -> int:
    nonlocal final_writer, write_buf_rows
    if not write_buf:
        return 0
    out = pa.concat_tables(write_buf)
    write_buf.clear()
    write_buf_rows = 0
    if _corrections:
        out = _apply_corrections(out, _corrections)   # <-- new
    out = _optimise_schema(out)
    if final_writer is None:
        final_writer = pq.ParquetWriter(str(out_path), out.schema, **_WRITE_OPTS)
    final_writer.write_table(out)
    return len(out)
```

---

### 3. `utils/location.py` — `fetch()` passes `calibration_out`

```python
def fetch(self, ...) -> Path:
    from utils.s2_tiles import bbox_to_tile_ids

    _cal_out = None
    if len(bbox_to_tile_ids(tuple(self.bbox))) > 1:
        _cal_out = _PROJECT_ROOT / "data" / "calibration" / f"{self.id}.parquet"
        _cal_out.parent.mkdir(parents=True, exist_ok=True)

    collect(
        ...,
        calibration_out=_cal_out,
    )
    return _out
```

The tile check stays in `Location.fetch()` (which knows the location id); `collect()` just acts on the path it receives.

---

### 4. `signals/_shared.py` and `signals/__init__.py` — remove `calibration_path` wiring

Once the parquet is pre-corrected, the `calibration_path` parameter in `compute_features_chunked()` and `extract_parko_features()` is redundant. Remove it to avoid confusion. Existing callers pass `None` by default so removal is clean.

---

## Critical files

| File | Change |
|------|--------|
| `utils/pixel_collector.py:625–751` | Add `calibration_out` param, `_apply_corrections()`, extend `_flush_write_buf()`, run `calibrate()` on shards before concat |
| `utils/tile_harmonisation.py:41` | Accept `Path \| list[Path]` for first argument |
| `utils/location.py:190` | Pass `calibration_out` to `collect()` |
| `signals/_shared.py:81` | Remove `calibration_path` param |
| `signals/__init__.py:86` | Remove `calibration_path` param |

## What does NOT change

- `tam/pipeline.py` — reads corrected values from parquet automatically
- `utils/training_collector.py` — training parquets are single-tile, so `calibration_out=None` and no correction is applied (correct behaviour)
- `cli/describe.py`, `cli/explore.py` — automatically consume corrected parquet
- Existing `tests/unit/test_tile_harmonisation.py` — pass unchanged

---

## Verification

1. Run `loc.fetch()` for Quaids. Confirm:
   - `data/calibration/quaids.parquet` is written with non-trivial scale factors for 55KBB/55KCB
   - B11 values for 55KBB pixels in `data/pixels/quaids/quaids.parquet` are reduced (median should approach 55KCB's ~0.232)
2. Re-run `tam/pipeline.py score` for Quaids. Confirm the horizontal streaks at row `_0123` are absent or substantially reduced.
3. For a single-tile location (e.g. `frenchs`): confirm `calibration_out=None`, no calibration file written, parquet identical to current behaviour.
4. Run `tests/unit/test_tile_harmonisation.py` — should pass unchanged.
5. Spot-check: for a 55KBB-sourced pixel, verify stored B11 = original × scale factor from the calibration table.
