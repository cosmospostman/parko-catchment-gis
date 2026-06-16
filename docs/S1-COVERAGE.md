# S1 "Coverage Gap" — 55KCB / 55KBB (2025) — RESOLVED: stale truncated patches

**Date investigated:** 2026-06-12 (original, incorrect conclusion)
**Re-investigated:** 2026-06-16 — original "burst boundary" diagnosis was **wrong**.
**Triggered by:** `score --tile-id 55KCB` producing only thin horizontal strips in PMTiles output.

---

## Symptom

Running score on the 2025 chunkstore for 55KCB (and 55KBB) scored only ~7–11% of
pixels. In PMTiles the output appears as a repeating thin strip near the top of each
spatial chunk band. The `_preprocess` filter drops pixels with `< MIN_S1_OBS_PER_YEAR`
(=4) S1 observations, and ~90% of pixels have only 1–3 S1 obs, so they were dropped.

## TL;DR root cause

**The source data is fine. The chunkstore parquets are corrupt.** The S1 fetch stage
cached **truncated patches** — 2-D arrays only ~97 rows tall instead of the full ~1049
rows covering the chunk bbox. The extract stage faithfully read those short patches, so
for most S1 acquisitions only the top ~97 rows of each chunk received S1 values; every
other pixel mapped outside the (truncated) patch → NaN → dropped. There is **no SAR
coverage gap** — every acquisition fully covers the tile.

The parquets must be **rebuilt**. Re-running the current fetch+extract code against a
fresh patch produces full coverage (verified — see below).

## Why the original "burst boundary" diagnosis was wrong

The original doc claimed 55KCB straddles a Sentinel-1 IW burst boundary and that ~90%
of the tile genuinely lacks SAR. Three pieces of evidence disprove this:

1. **The dense band tracks each chunk's own edge, not a fixed geographic latitude.**
   In every chunk the "30-obs" pixels sit at the chunk's *top* (max-lat) edge, at a
   *different absolute latitude per chunk* (r00: −16.325→−16.280; r01: −16.418→−16.365;
   r02: −16.510→−16.458; …). A real burst footprint is fixed in geographic space; it
   would appear at one absolute latitude and be absent elsewhere. A defect pinned to
   each chunk's local frame is a per-chunk array/window artefact, not orbital geometry.

2. **Every S1 item is a full GRD scene that completely covers the chunk.** The STAC
   collection is `sentinel-1-rtc` and the items are `S1A_IW_GRDH_*` scenes with
   footprints ~2.6°×2° (e.g. `[143.77, −17.57, 146.44, −15.53]`), dwarfing the
   ~0.1°×0.09° chunk bbox. All 30 descending passes cover the entire tile.

3. **Reading the actual RTC COG over the chunk bbox returns 100% valid data.** For a
   "sparse" date (2025-03-04) `_read_bbox_patch` returns a 1049×1031 patch with
   `valid_frac = 1.000` over the whole bbox (top and bottom quarters both 1.000).
   Re-running `_extract_s1_from_store` against that fresh patch extracts
   **1,048,576 / 1,048,576** points — the full chunk — versus the **98,592** stored in
   the parquet for that date.

## The defect, precisely

Per-chunk breakdown of the 30 S1 dates in `55KCB_r01_c00.parquet`:

| Pattern | # dates | rows covered | pixels |
|---|---|---|---|
| Truncated patch | 28 | 97 (yi 9408–9504, top of chunk) | 98,592 |
| Full patch | 1 (2025-01-28) | 572 (full chunk) | 585,728 |
| Full patch | 1 (2025-02-09) | 549 | 561,440 |

The 28 truncated dates each cover *exactly* 97 contiguous rows at the chunk top, fully
populated across all 1024 columns — the signature of a 2-D patch array that was written
~97 rows tall instead of ~1049. Two dates happened to be read in full.

## Why the build didn't catch it

The fetch/extract split (`proxy/_pipeline.py:_stage_fetch_s1` / `_stage_extract_s1`)
caches one `.npz` patch per `(item_id, band)` under a **chunk-scoped** cache dir
(`scene_dir/s1_cache`) — that scoping is correct and was *not* the problem. The problem
is the cache-validity check:

- `fetch_patches` fast-path (`utils/fetch.py:298`) accepts a cached patch when
  `_patch_covers_bbox` returns True.
- `_patch_covers_bbox` (`utils/fetch.py:143`) compares the **stored `fetch_bbox`
  stamp** against the requested bbox. A truncated patch was fetched *for the correct
  bbox*, so its stamp matches and the check passes — **the actual array dimensions are
  never validated against the bbox**. A short read sails straight through and is reused
  on every subsequent run.

The likeliest origin of the short reads themselves: a partial/decimated COG read during
the fetch stage that returned a smaller array without raising (the fetch stage runs many
chunks concurrently and `fetch_patches` mutates global GDAL env vars —
`GDAL_HTTP_MAX_RETRY=0`, `GDAL_DISABLE_READDIR_ON_OPEN` — at line 263-264).

## Precise mechanism (and what is *not* proven)

The cached patches were **small, internally-consistent sub-windows** — e.g. a 97-row
array whose `transform` origin was shifted to match — not simple "bottom truncated, rest
zero" arrays. Evidence: the surviving band sits in the *middle* of the read window
(window-rows ~452–548 of ~1024) with fully valid VH and nothing above or below. A plain
short HTTP read drops the array *tail* and can't produce a valid mid-array band; only a
patch whose array+transform both describe a small sub-window does, because
`_pixel_coords` then projects all other chunk points out-of-bounds → NaN → dropped.

**Unproven:** *why* `_read_bbox_patch` computed a ~97-row window on the build run.
`utils/fetch.py` has not changed since before the build, and re-running it now reads the
*same* COGs/dates at full ~1041-row coverage (verified). The patches that caused it lived
in the transient per-chunk `s1_cache` and are gone, so it is **not currently
reproducible**. The defect is widespread (sampled chunks 2017–2025 all show
`med_frac ≈ 0.10`, `max_frac = 1.0`), pointing to a systemic runtime condition at fetch
time rather than a deterministic code bug. The earlier "short read" phrasing in this doc
overstated certainty and has been corrected.

## Fix / remediation (status)

1. **Write-time guard — DONE.** `_read_bbox_patch` (`utils/fetch.py`) now asserts the
   returned `arr.shape` equals the requested window `(round(height), round(width))` and
   raises (→ retry / None) on mismatch, so a short/sub-window read can never be cached.
   Tested: `tests/unit/test_fetch_cache.py::test_fc4_short_read_rejected`.
2. **Verification tool — DONE.** `python cli/chunk.py verify [--year Y] [--tile T]`
   (core in `utils/chunk_verify.py`) flags `S1_TRUNC` when a chunk's *median per-date S1
   row coverage* < 0.5 (healthy ≈ 1.0; defect ≈ 0.10). Non-zero exit on any failure, so
   it can gate a rebuild. Tested in `tests/unit/test_chunk_verify.py`.
3. **`_patch_covers_bbox` left as-is (intentional).** It still trusts the stored
   `fetch_bbox` stamp and does not re-check array dims, because a smaller-than-window
   array is *legitimate* for border-tile items. The write-time guard (1) closes the hole
   at the source, so weakening this check would only risk re-fetching valid edge patches.
4. **Rebuild still required** for 55KCB + 55KBB 2025 (and any tile `cli/chunk.py verify`
   flags). Delete each chunk's `s1_cache/*.npz` first so any cached truncated patches are
   not reused.

## Blast radius

**Wider than the June 11 run.** Sampling one chunk per year across the chunkstore
(`/mnt/gis-archive/chunkstore`) shows the defect in *every* year checked (2017–2025):
`med_frac` ≈ 0.10–0.34 with `max_frac` = 1.0 throughout. So this is not specific to the
2025 build — treat all existing S1 in the chunkstore as suspect until
`cli/chunk.py verify` clears it.

| Tile / year | `≥4 S1 obs` fraction |
|---|---|
| 55KCB 2025 r00_c00 | 8.8% |
| 55KBB 2025 r00_c00 | 11.2% |
| (2017–2024 sampled) | similar — see `cli/chunk.py verify` |

Run `python cli/chunk.py verify` (optionally `--year`/`--tile`) to enumerate exactly
which chunks are affected before trusting or rebuilding.

## How to reproduce the diagnostics

```python
import duckdb
p = '/mnt/gis-archive/chunkstore/2025/55KCB/55KCB_r01_c00.parquet'

# Per-date covered-row pattern — truncated dates show nrows≈97
duckdb.sql(f"""
WITH s AS (
  SELECT date, COUNT(DISTINCT CAST(split_part(point_id,'_',3) AS INT)) nrows,
         COUNT(*) npx
  FROM read_parquet('{p}') WHERE source='S1' GROUP BY date)
SELECT nrows, COUNT(*) n_dates, MIN(npx), MAX(npx) FROM s GROUP BY nrows
""").show()
```

Prove the source is good (full coverage from a fresh read):

```python
# See the investigation scripts: search_sentinel1 → sign → _read_bbox_patch over the
# chunk bbox returns a 1049×1031 patch with valid_frac=1.0; _extract_s1_from_store
# against it returns all 1,048,576 points (vs 98,592 stored in the parquet).
```

## Key files / functions

- `utils/fetch.py:143` — `_patch_covers_bbox` (trusts bbox stamp; **does not check array dims** — the gap)
- `utils/fetch.py:50` — `_read_bbox_patch` (where a short read should be rejected)
- `utils/fetch.py:298` — `fetch_patches` cached fast-path that accepts the patch
- `utils/chip_store.py:263` — `CachedNpzChipStore.get_all_points` (maps out-of-patch points → NaN)
- `proxy/_pipeline.py:1030` — `_stage_fetch_s1` / `_stage_extract_s1` (chunk-scoped cache; correct)
- `tam/core/score.py:613` — `MIN_S1_OBS_PER_YEAR` filter (correctly drops the 1–3-obs pixels)
