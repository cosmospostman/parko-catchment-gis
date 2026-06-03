# Random Access Pixel Reads from Chunk Parquets

## Context

Chunk parquets from the proxy pipeline are stored on `/mnt/external/mitchell` at
`{root}/{year}/{tile_id}/{tile_id}_rNN_cNN.parquet`. These contain the full S2+S1
time-series for every pixel (~65M rows per chunk, 66 row groups × ~10 MB compressed).
The goal is a `utils/pixel_reader.py` module exposing two methods for interactive
UI feature development:

- **`query_point`**: single-pixel time-series (click-to-profile)
- **`query_bbox`**: all pixels within a bounding box

The two-level lookup (chunk grid → in-memory RG lat/lon scan → targeted disk reads)
keeps cold-query cost at 1–2 disk seeks × ~10 ms + ~10–20 MB read on the spinning
HDD = ~120–230 ms per chunk. The footer is paid once at index construction time.

---

## `utils/pixel_reader.py`

### `ChunkIndex` — constructed once, reused across queries

```python
ChunkIndex(root: Path, year: int, tile_id: str)
```

On construction:
- Glob `root/year/tile_id/*_rNN_cNN.parquet`
- For each file: open `pq.ParquetFile`, cache its metadata object keyed by
  `(chunk_row, chunk_col)` — pays the footer seek once per chunk file
- Extract and cache per-chunk envelope (union of all RG lat/lon stats) and
  per-RG lat/lon min/max — all in memory, no further disk I/O until a query hits

Chunk grid position comes from the filename (`_rNN_cNN`). No UTM math required.

### `query_point(lon, lat) -> pa.Table | None`

Returns all rows for the single pixel nearest to `(lon, lat)`.

1. **Chunk (O(n_chunks ≈ 8–10), in-memory)**: find the chunk whose cached envelope
   contains `(lon, lat)`.
2. **RG candidates (O(n_rg ≈ 66), in-memory)**: scan cached per-RG stats; keep RGs
   whose `[lat_min, lat_max] × [lon_min, lon_max]` overlaps the query point.
   Typically 1–2 RGs match (adjacent RGs overlap by ~1–2 pixel rows in lat).
3. **Disk read + filter**: read candidate RGs via `pq.ParquetFile.read_row_group()`,
   identify the `point_id` with the nearest `(lon, lat)`, then filter all rows for
   that `point_id`. Return the filtered `pa.Table`.

### `query_bbox(lon_min, lat_min, lon_max, lat_max) -> pa.Table`

Returns all rows for every pixel whose `(lon, lat)` falls within the bbox.

1. **Chunk candidates (O(n_chunks), in-memory)**: collect all chunks whose envelope
   overlaps the query bbox (may span multiple chunks).
2. **RG candidates per chunk (O(n_rg), in-memory)**: same lat/lon overlap scan.
3. **Disk reads + filter**: read candidate RGs from each matched chunk, filter rows
   to `lon_min ≤ lon ≤ lon_max AND lat_min ≤ lat ≤ lat_max`, concatenate results.
   Return a single `pa.Table`.

**Seek/read cost for bbox**: one disk seek per matched RG across all matched chunks.
A small viewport (e.g. 500 m × 500 m) typically hits 1 chunk and 2–4 RGs: ~4–8
seeks, ~40–80 MB read, ~500 ms–1 s cold.

### Return schema

All available columns from `COMBINED_PIXEL_SCHEMA`: `point_id`, `lon`, `lat`, `date`,
`item_id`, `tile_id`, `B02`–`B12`, `B8A`, `scl_purity`, `scl`, `aot`, `view_zenith`,
`sun_zenith`, `source`, `vh`, `vv`, `orbit`.

---

## Test: `tests/unit/test_pixel_reader.py`

Uses real data at `/mnt/external/mitchell` (marked with a fixture guard to skip if
the mount is absent).

- **`test_query_point`**: sample a known `(lon, lat)` from an existing chunk row;
  assert result is non-empty, `point_id` is unique, multiple dates returned.
- **`test_query_point_rg_boundary`**: query a lat in the overlap band between two
  adjacent RGs; assert the correct unique `point_id` is returned (no duplicates).
- **`test_query_bbox`**: query a small bbox; assert all returned pixels have
  `(lon, lat)` within the bbox, and result contains multiple distinct `point_id`s.
- **`test_query_bbox_multi_chunk`**: query a bbox that straddles a chunk boundary;
  assert pixels from both chunks are returned.

---

## Files

- **New**: `utils/pixel_reader.py`
- **New**: `tests/unit/test_pixel_reader.py`

---

## Verification

```bash
.venv/bin/pytest tests/unit/test_pixel_reader.py -v
```
