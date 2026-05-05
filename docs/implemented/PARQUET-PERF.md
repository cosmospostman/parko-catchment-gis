# Parquet Performance: Optimisation Plan

## Current state (kowanyama baseline)

| Property | Value |
|---|---|
| File size | 79 GB |
| Rows | 1,677,617,414 |
| Row groups | 2,084 |
| Compression | SNAPPY |
| Band columns | FLOAT (32-bit) |
| `lon`/`lat` | DOUBLE (64-bit) |
| `point_id`, `item_id`, `tile_id` | plain BYTE_ARRAY |

### Column-level compression breakdown (one row group, ~825k rows)

| Column | Compressed | Uncompressed | Ratio |
|---|---|---|---|
| B02–B12 (each) | ~3,385 KB | ~3,393 KB | 1.00× — incompressible |
| sun_zenith | 3,320 KB | 3,331 KB | 1.00× |
| view_zenith | 1,833 KB | 1,874 KB | 1.02× |
| date | 362 KB | 1,018 KB | 2.8× |
| item_id | 362 KB | 1,036 KB | 2.9× |
| tile_id | 6 KB | 25 KB | 4.2× |
| lon / lat | 13–15 KB | 15 KB | 1.1× |

The band columns dominate file size and are already near-incompressible as random
floats. Gains come from metadata columns and codec choice.

---

## Optimisations

### 1. ZSTD compression (level 3)

Replace SNAPPY with ZSTD level 3. SNAPPY is fast but achieves ~1× on float data.
ZSTD level 3 typically saves 15–25% on mixed numeric/string parquets at comparable
read speed. Level 3 is the sweet spot; higher levels yield diminishing returns on
float columns.

**Expected saving: ~12–20 GB on kowanyama.**

### 2. Dictionary encoding on `item_id`, `tile_id`, `point_id`

These are low-cardinality repeated strings. With dictionary encoding PyArrow writes
a small dictionary page and integer indices — `tile_id` (e.g. `54LWH`) repeats
millions of times. Currently these columns are plain BYTE_ARRAY.

Enable via `use_dictionary=["item_id", "tile_id", "point_id"]` in `ParquetWriter`.

**Expected saving: significant on metadata columns (~1–2 GB), negligible on bands.**

### 3. `lon`/`lat` → FLOAT32

Sentinel-2 10 m pixels are ~10 m apart. Float32 gives ~7 decimal digits of
precision (~1 mm at equator), which is more than sufficient. Halves these two
columns.

Cast at write time: `tbl.set_column(..., pa.array(..., type=pa.float32()))`.

**Expected saving: ~1–2 GB across a kowanyama-scale file.**

### 4. `date` → DATE32

Currently stored as `timestamp[us]` (INT64). Dates are daily granularity with no
sub-day component. `date32` (days since epoch, INT32) halves the column size and
improves dictionary/RLE encoding.

**Consumer impact:** `tam/core/score.py` was casting the date column directly to
`int64` assuming `timestamp[us]` encoding — fixed to go via
`pd.to_datetime(...).astype("datetime64[us]")` first, which handles both types.
All other consumers use `pd.to_datetime()` or Polars `.dt.*` ops which handle
`date32` natively without changes.

**Expected saving: ~0.5–1 GB.**

### 5. Row group consolidation

2,084 row groups in kowanyama is very fragmented (one per sort pass). Large row
groups improve compression (better statistics for predicate pushdown) and reduce
open/seek overhead. Target 5,000,000 rows per row group.

This is already parameterised in `sort_parquet_by_pixel` (`row_group_size=5_000_000`)
but appears not to be respected during the concat step in `pixel_collector.py`.

---

## Implementation

### Writer changes (`signals/_shared.py`, `utils/pixel_collector.py`)

```python
WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["point_id", "item_id", "tile_id"],
    write_statistics=True,
)

# Cast schema before writing
def _optimise_schema(tbl: pa.Table) -> pa.Table:
    for col in ("lon", "lat"):
        tbl = tbl.set_column(tbl.schema.get_field_index(col), col,
                             tbl.column(col).cast(pa.float32()))
    tbl = tbl.set_column(tbl.schema.get_field_index("date"), "date",
                         tbl.column("date").cast(pa.date32()))
    return tbl
```

Apply `_optimise_schema` before every `write_table` call in both `sort_parquet_by_pixel`
and the concat loop in `pixel_collector.py`.

### Migration order (given current disk constraints)

1. Migrate `kowtown` first (once fetch completes) — already on current disk.
2. Use the freed space to migrate `kowanyama` (79 GB → est. 55–65 GB).
3. Migrate remaining locations (longreach-8x8km, frenchs, etc.) in any order.

---

## Read throughput

These changes are primarily a **storage win, not a read-speed win**.

The band columns (B02–B12, sun_zenith) dominate file size and are already
near-incompressible float32. ZSTD decompresses slower than SNAPPY, so sequential
full-table scans — the typical pattern for feeding a location into a model — will
be neutral to marginally slower.

Where read performance does improve:

- **Row group consolidation** — the single biggest read win. 2,084 row groups means
  2,084 footer seeks even for a full scan. Consolidating to ~340 row groups (5M
  rows each) reduces open/metadata overhead and makes predicate pushdown effective.
- **Date range and pixel filtering** — larger row groups carry better min/max
  statistics, so `date`/`point_id` filters can skip more data without reading it.
- **Dictionary columns** (`item_id`, `tile_id`) — filter/group operations on these
  work on integers rather than strings; no benefit for pure scans.

If the access pattern is always "read all rows for a location sequentially", the
row group consolidation is the only meaningful throughput change. Everything else
is storage reduction only.

---

## Expected outcomes

| File | Current | Estimated post |
|---|---|---|
| kowanyama.parquet | 79 GB | 55–65 GB |
| kowtown.parquet | ~70 GB (est.) | 48–58 GB |
| Total /data/pixels | ~337 GB | ~230–270 GB |

These are conservative estimates; actual savings depend on spectral variability
and repetition in `item_id`/`point_id` for each location.
