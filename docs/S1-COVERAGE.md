# S1 Coverage Gap — 55KCB / 55KBB (2025)

**Date investigated:** 2026-06-12
**Triggered by:** `score --tile-id 55KCB` producing only thin horizontal strips in PMTiles output.

---

## Symptom

Running score on the 2025 chunkstore for 55KCB (and 55KBB) produced scores for only ~7% of
pixels. In PMTiles the output appears as a repeating thin strip near the top of each spatial
chunk band — not random noise, but a consistent horizontal stripe ~64 rows wide per 1024-row
chunk.

## Root cause: Sentinel-1 IW burst boundary

55KCB sits at the boundary between two Sentinel-1 IW (Interferometric Wide) burst footprints
on the descending orbit track.

- **30 S1 descending passes** exist for the 55KCB bbox in 2025 (confirmed via STAC query).
- **No ascending-orbit data** is available — STAC returns zero ascending items for this bbox.
- The burst boundary is **purely yi-dependent** (horizontal in UTM space). Within a given
  chunk (e.g. `r01_c00`, yi 8956–9979):
  - `yi >= 9920`: 30 S1 obs/pixel — fully inside the burst footprint on every pass.
  - `yi < 9920`: 1–2 S1 obs/pixel — falls between burst extents, only caught incidentally.

The strip you see in the UI is exactly those ~64 rows per chunk that fall inside the burst
footprint on all 30 passes. The remaining ~960 rows per chunk have genuine no-data.

## Numbers

| Metric | Value |
|---|---|
| Total distinct pixels in 55KCB 2025 chunkstore | 34,580,480 |
| Pixels with ≥ 4 S1 obs (pass `MIN_S1_OBS_PER_YEAR`) | ~3,530,000 (~10%) |
| Pixels with exactly 1 S1 obs | 30,674,184 (88.7%) |
| Scored pixels in output | 2,459,352 (7.1%) |
| Comparison — 54LWH 2025 S1 obs/pixel | 45 (fully inside swath) |
| Comparison — 55KCB dense-swath pixels S1 obs | 30/pixel |
| Comparison — 55KCB sparse-swath pixels S1 obs | 1–2/pixel |

The scored fraction is lower than the ≥4 S1 obs fraction because the S2 `min_obs_per_year`
filter and the location geometry mask further reduce the pool.

## What was ruled out

- **Code bug in score pipeline** — pipeline is correct. `_preprocess` (score.py:613) filters
  `(n_s2_per_win >= min_obs_per_year) & (n_s1_per_win >= MIN_S1_OBS_PER_YEAR)`. With
  `MIN_S1_OBS_PER_YEAR = 4` (dataset.py:53), the 1-pass pixels are correctly dropped.
- **Tile_prefix / item_id filter** — S1 rows have `item_id=NULL` and are preserved by the
  `null_mask` in `score_pixels_chunked`. Not a factor.
- **Chunkstore fetch error** — fetch logs confirm `extract_s1` ran for every chunk and the row
  counts are consistent with the burst geometry. 29–62M S1 rows extracted per chunk, but only
  a minority of pixel positions fall inside the burst footprint.
- **Wrong orbit** — fetch is not filtered to descending only. STAC simply has no ascending
  items for this bbox in 2025.
- **Ascending orbit coverage** — STAC query (`_resolve_s1_items`, s1_collector.py:288)
  against the full year 2025-01-01/2025-12-31 returned exactly 30 items, all descending.
  There is no ascending SAR coverage for this geographic area.

## Key files / functions

- `tam/core/score.py:613` — `MIN_S1_OBS_PER_YEAR` filter in `_preprocess`
- `tam/core/dataset.py:53` — `MIN_S1_OBS_PER_YEAR = 4`
- `utils/s1_collector.py:288` — `_resolve_s1_items` (STAC query)
- `utils/s1_collector.py:344` — `_extract_s1_from_store` (NaN masking, only keeps pixels
  inside the burst footprint)

## Open questions

1. **Does 55KBB have the same burst boundary pattern?** 55KBB is adjacent to 55KCB and likely
   shares the same orbital geometry. Check S1 obs distribution in
   `/mnt/external/chunkstore/2025/55KBB/` before re-running score there.

2. **Which orbit relative number (orbit number in STAC) covers 55KCB?** Knowing the exact
   burst IDs might allow requesting additional S1 burst acquisitions if they exist under a
   different collection or product type.

3. **Is the burst boundary stable across years?** If 2024 data for 55KCB also only has
   burst-edge coverage, the tile may be fundamentally limited for mixed-mode scoring. Only
   two 2024 chunks exist in the chunkstore (`r05_c00`, `r05_c01`) with 13–14 S1 obs/pixel —
   those happen to be in a yi band that sits inside the burst. Need full 2024 tile coverage
   to assess.

4. **Model options for burst-edge tiles:** The v10 model requires both S2 and S1. If 55KCB
   is structurally SAR-poor, options are: (a) score only the covered fraction as-is, (b)
   train a S2-only fallback head, or (c) restrict the Mitchell catchment scoring domain to
   tiles with reliable SAR coverage.

## How to reproduce the key diagnostics

```python
import duckdb
from pathlib import Path

# S1 obs per pixel distribution for a chunk
result = duckdb.sql("""
WITH s1_counts AS (
  SELECT point_id, COUNT(*) FILTER (WHERE source = 'S1') as s1_obs
  FROM read_parquet('/mnt/external/chunkstore/2025/55KCB/55KCB_r01_c00.parquet')
  GROUP BY point_id
)
SELECT s1_obs, COUNT(*) as n_pixels FROM s1_counts GROUP BY s1_obs ORDER BY s1_obs
""").fetchall()

# S1 obs by yi band (shows burst boundary)
result = duckdb.sql("""
WITH s1_counts AS (
  SELECT CAST(split_part(point_id, '_', 3) AS INTEGER) as yi,
         COUNT(*) FILTER (WHERE source = 'S1') as s1_obs
  FROM read_parquet('/mnt/external/chunkstore/2025/55KCB/55KCB_r01_c00.parquet')
  GROUP BY point_id
)
SELECT (yi // 64)*64 as yi_band, AVG(s1_obs) as avg_s1, COUNT(*) as n_pixels
FROM s1_counts GROUP BY yi_band ORDER BY yi_band DESC
""").fetchall()
```

STAC query to confirm no ascending items:
```python
from utils.s1_collector import _resolve_s1_items
from pathlib import Path
items = _resolve_s1_items(
    [145.127, -16.366, 145.224, -16.279],
    '2025-01-01', '2025-12-31',
    Path('/mnt/external/chunkstore/2025/55KCB') / '.s1_stac_cache'
)
# Returns 30 items, all descending
```
