# Fetch Proxy — Cloud-side COG extraction

## Problem

The fetch pipeline is network-bound on the workstation side. For the Mitchell River catchment (72,000 km², 45 MGRS tiles, ~720M pixels) a single year of S2+S1 parquet is ~2.7 TB. Getting there requires fetching COG windows from AWS S3 — and those windows are 50–80× larger than the parquet they produce:

- A 10 km × 10 km catchment at 10 m resolution requires a 1000×1000 pixel window per band per scene
- 13 bands × ~100 scenes/tile/year × ~4 MB/window = **~5 TB of COG traffic** to produce ~60 GB of parquet per tile
- At 300 Mbps uplink, 5 TB takes ~37 hours per tile

A VM co-located with the Element84 S3 COGs reads those COG windows at Gbps rates. The proxy runs extraction there and streams only the compressed parquet back. The 300 Mbps WAN link then sees ~60 GB per tile instead of ~5 TB.

## Scale reference

Calibrated from Kowanyama 2021 (same tropical NQ climate, measured actuals):

| Metric | Value |
|---|---|
| Clear obs/pixel/year (S2+S1 combined) | ~115 (~85 S2, ~30 S1) |
| Final parquet per million pixels per year | ~3.7 GB |
| One MGRS tile (~121M pixels), one year | **~60 GB** |
| Mitchell River (45 tiles), one year | **~2.7 TB total** |

At 300 Mbps, 2.7 TB takes ~20 hours — the WAN transfer dominates. The VM's job is to keep the pipe saturated.

## VM location

The Element84 COG bucket is in us-west-2. A VM in the same region gets free, Gbps-rate S3 reads. A DigitalOcean VM in SFO is cross-cloud but the SFO→Oregon path is well-peered (~5–15 ms, 500–800 Mbps sustained) and DO charges outbound from DO rather than S3 egress. Either location works; the 300 Mbps workstation link is the bottleneck regardless of VM location or RTT.

## Why sort on the VM — compression mechanics

The goal is to minimise WAN bytes. Sorted+dict parquet achieves ~10× compression over raw unsorted because:

1. **Dictionary encoding on `point_id`**: sorted output groups all ~115 observations for each pixel consecutively. The dictionary stores each unique `point_id` string once; the index stream is long runs of the same integer — ZSTD compresses runs to near-zero. Per-scene parquets have every `point_id` appearing exactly once — no runs, no compression benefit, full string storage cost.

2. **Spectral locality in band columns**: all observations for a pixel are consecutive. A pixel's spectral signature is relatively stable across dates (same land cover), so band values have low variance within each run — ZSTD exploits that locality.

Per-scene transfer (deferring merge to workstation) would be ~57 GB/strip uncompressed vs ~5.5 GB sorted+dict — ~10× more WAN traffic, ~200 hours instead of ~20 for the full catchment. The sort must happen on the VM.

## Sensors handled

The proxy handles **both** Sentinel-2 and Sentinel-1. Both sensors are fetched, extracted, and sorted on the VM before streaming:

- **Sentinel-2** (~85 obs/pixel/year): STAC via Element84, COG bands (B02–B12 + SCL + B8A), polygon-masked, per-scene parquets sorted by `point_id`
- **Sentinel-1** (~30 obs/pixel/year): STAC via Microsoft Planetary Computer (MPC), VH+VV COGs, same polygon mask — fetched via `collect_s1_for_tile()` after all S2 scenes are extracted, producing one S1 parquet per strip

Each strip shard contains interleaved S2 and S1 rows, sorted by `(point_id, date)`, produced by `merge_scenes()` in `proxy/server.py`.

## Local fetch vs proxy fetch

The proxy does not reimplement the fetch pipeline — it runs the same `utils/` code on a different machine. The only new code is the `per_scene=True` mode in `collect()`, the `points=` kwarg on `collect_s1_for_tile()`, `merge_scenes()` in `proxy/server.py`, the frame protocol, and the HTTP client/server wrappers.

### Local fetch (current)

```
Location.fetch()
  └─ fetch_spec()
       ├─ Phase A (all years parallel, network-bound)
       │    └─ collect(phases={"fetch"})          — fetch_patches() → .npz cache on workstation disk
       │
       └─ Phase B (≤max_extract_years concurrent, memory-bound)
            ├─ collect(phases={"extract"})         — .npz → per-scene DataFrames → .s2.parquet
            ├─ collect_s1_for_tile()               — S1 STAC search, COG fetch, .s1.parquet
            ├─ merge_tile()                        — DuckDB S2+S1 sort-merge → <tile_id>.parquet
            └─ [merge_strips() if strip mode]      — N-way merge of strip shards
```

All compute and all I/O happen on the workstation. The WAN link sees raw COG traffic (~5 TB per tile/year).

### Proxy fetch (new)

```
Location.fetch(proxy_url=...)
  └─ proxy/client.py: fetch_tiles()
       ├─ For each tile × year: POST /run/tile
       │
       │   VM (proxy/server.py):
       │   ├─ stac.search_sentinel2() + S1 search  — same utils/stac.py
       │   ├─ collect(phases={"fetch"}, per_scene=True)
       │   │    └─ fetch_patches()                 — .npz at Gbps from S3, deleted after ACK
       │   ├─ collect(phases={"extract"}, per_scene=True)
       │   │    └─ extract_item_to_df()            — one parquet per scene, sorted by point_id
       │   ├─ collect_s1_for_tile(points=...)       — S1 COG fetch + extraction (after all S2 scenes done)
       │   └─ merge_scenes()                       — DuckDB N-way sort of S2 scene parquets + S1 parquet → strip_NNNN_sorted.parquet
       │
       │   Frame stream (VM → workstation):
       │   └─ 0x01 progress + 0x02 strip shards   — ~5.5 GB/strip at 300 Mbps
       │
       └─ workstation: merge_strips()             — N sorted shards → <tile_id>.parquet
```

The WAN link sees only compressed sorted parquet (~60 GB per tile/year instead of ~5 TB).

### What is shared / what is new

| Component | Local | Proxy VM | Proxy client (workstation) |
|---|---|---|---|
| `utils/stac.py` search | ✓ | ✓ same code | — |
| `fetch_patches()` | ✓ | ✓ same code | — |
| `extract_item_to_df()` | ✓ | ✓ same code | — |
| `collect()` | ✓ full | ✓ with `per_scene=True` (new flag) | — |
| `collect_s1_for_tile()` | ✓ | ✓ with `points=` kwarg + `s2_path` optional (new) | — |
| `COMBINED_PIXEL_SCHEMA` | — | new — canonical S2+S1 schema in `parquet_utils.py` | — |
| `merge_tile()` | ✓ | — | — |
| `merge_scenes()` | — | new — DuckDB N-way scene sort in `proxy/server.py` | — |
| `merge_strips()` | ✓ (strip mode) | — | ✓ same code |
| `parquet_utils` sort/dedup | ✓ | ✓ same code | ✓ same code |
| `.npz` fetch + delete | workstation | VM | — |
| Frame stream protocol | — | write frames | read frames |
| Atomic `.tmp`→`.parquet` | — | — | new (client only) |
| `Location.fetch()` proxy branch | — | — | new (thin dispatcher) |

**What is genuinely new:**
- `per_scene=True` flag in `collect()` — writes one parquet per scene instead of accumulating into a shard; sorts each scene by `point_id` on write; no stale-cache check (VM is freshly launched per tile). Return type changes to `Iterator[tuple[str, Path]]` only when `per_scene=True`; existing callers that omit the flag continue to receive `list[Path]` unchanged. `fetch_spec.py` is unaffected.
- `points: list[tuple[str, float, float]] | None = None` kwarg on `collect_s1_for_tile()`, and `s2_path` becomes `Optional[Path]` — when `points` is supplied, coord reading from `s2_path` is skipped entirely; `s2_path` is only accessed on the existing path when `points` is absent (backwards-compatible). Schema uses `COMBINED_PIXEL_SCHEMA` directly rather than calling `_extend_schema()` on a live file.
- `COMBINED_PIXEL_SCHEMA` in `parquet_utils.py` — hardcoded canonical combined schema (S2 columns + `source`, `vh`, `vv`, `orbit`); removes the file dependency from all VM-facing schema derivation. `_extend_schema()` is unchanged for existing callers.
- `merge_scenes()` in `proxy/server.py` — DuckDB N-way sort-merge of all S2 per-scene parquets + one S1 parquet → sorted strip shard; uses `COMBINED_PIXEL_SCHEMA` from `parquet_utils.py` for the output schema (no live-file schema derivation needed); local to proxy, does not modify `_extend_schema()` or other existing utilities
- Frame protocol encode/decode (`write_frame` / `read_frame`) — ~20 lines each
- `POST /run/tile` HTTP endpoint in `proxy/server.py` — orchestrates the existing functions
- `proxy/client.py` — tile loop, frame reader, atomic strip writes, calls existing `merge_strips()`
- `Location.fetch(proxy_url=...)` branch — calls `proxy/client.py` instead of `fetch_spec()`

## Division of labour

| Phase | Where | Why |
|---|---|---|
| STAC search (S2 + S1) | VM | One search per tile per year; item list passed to `collect()` |
| COG fetch → per-scene `.npz` (S2 + S1) | VM | Free/fast intra-region S3 reads |
| Per-scene pixel extraction + polygon mask | VM | CPU-bound, one scene at a time; mask applied here |
| DuckDB sort-merge → sorted strip shard | VM | Must happen before transfer for compression |
| **Stream sorted strip shards** | **VM → workstation** | Saturate 300 Mbps with compressed bytes |
| Strip-shard concat | Workstation | `merge_strips()` — same code as local strip mode |

## `collect()` — `per_scene=True` mode

`proxy/server.py` calls `collect()` from `utils/pixel_collector.py` with a new `per_scene=True` flag. Behaviour differences from the default mode:

| Behaviour | Default (`per_scene=False`) | Proxy (`per_scene=True`) |
|---|---|---|
| Output | One accumulated shard parquet per tile | One parquet per scene, sorted by `point_id` |
| Stale-cache check | Yes (workstation may have stale `.npz`) | No (VM is freshly launched, cache is always current) |
| STAC search | Internal, cached to `.pkl` | Skipped — `items` passed in by server |
| Polygon mask | Shapely geometry, bbox fallback | Shapely geometry decoded once from `polygon_wkb_b64` on request entry |
| Return type | `list[Path]` of tile parquets | `Iterator[tuple[str, Path]]` — yields `(scene_id, path)` as each scene completes. Existing callers omitting `per_scene` are unaffected; `fetch_spec.py` requires no changes. |

The server runs a single STAC search per tile per year, then passes the item list to `collect()` for each strip — no redundant searches across strips.

NBAR correction (`apply_nbar=True`) is applied on the VM inside `collect()`, using the same `extract_item_to_df()` path as local fetch. Output values match a direct `Location.fetch()` run on the workstation.

## COG block geometry

Both sensors have been verified against live tiles:

| Sensor | Source | Block size | Compression |
|---|---|---|---|
| Sentinel-2 | Element84 / AWS S3 | **1024×1024 px** | DEFLATE/LZW |
| Sentinel-1 RTC | MPC / Azure Blob | **512×512 px** | DEFLATE |

A range request for a window smaller than one block tall reads a full block. Strip height must therefore be a multiple of both block sizes to avoid wasting COG fetch bandwidth. Since 1024 = 2 × 512, **1024 px satisfies both sensors simultaneously** with zero over-fetch penalty for either.

| Strip height | S2 blocks fetched | S2 waste | S1 blocks fetched | S1 waste |
|---|---|---|---|---|
| 256 px | 1 (1024 px) | 75% | 1 (512 px) | 50% |
| 512 px | 1 (1024 px) | 50% | **1 — zero waste** | **0%** |
| **1024 px** | **1 — zero waste** | **0%** | **2 — zero waste** | **0%** |
| 2048 px | 2 — zero waste | 0% | 4 — zero waste | 0% |

**1024 px is the natural strip unit for both sensors.** Narrower strips waste COG fetch bandwidth and inflate producer time, making it harder to keep the WAN link saturated. Do not use sub-1024 px strips to work around a slow producer — the fix is always more parallelism or a larger VM.

## VM-side pipeline

The architecture has four concurrent workers. The goal is that the streamer **never goes idle** — there is always a sorted shard ready to send.

### Within a strip: three overlapping stages

For each 1024 px strip (~85 S2 scenes + ~30 S1 granules, 1024×11,000 px):

```
[S2 Fetcher]   scene 0   scene 1   scene 2  ...  scene 84
               ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓      ▓▓▓▓▓▓▓▓
[S2 Extractor]           ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ...  ▓▓▓▓▓▓▓▓
[S1 Collector]                                               ▓▓▓▓▓▓▓▓  (after all S2 scenes done)
[Merger]                                                               ▓▓▓▓ → strip_N.parquet
```

- **S2 Fetcher** (async, network-bound): fetches bands for one S2 scene at a time via `fetch_patches()` (13 bands → `scene_NNN.npz` ~570 MB), signals extractor, deletes `.npz` after extractor ACKs. Runs one scene ahead of extractor.
- **S2 Extractor** (CPU-bound, one scene at a time): `collect(phases={"extract"}, per_scene=True)` — loads `.npz`, applies polygon mask via `extract_item_to_df()`, writes `scene_NNN.parquet` sorted by `point_id`, ACKs fetcher. Peak RAM: ~1–2 GB (one `.npz` + one DataFrame).
- **S1 Collector** (after all S2 scenes extracted): `collect_s1_for_tile(points=pixel_grid, ...)` — STAC search, COG fetch and extraction for ~30 S1 granules, writes one `s1_strip.parquet`. Uses the pixel coordinate list derived from the strip bbox (same list used by the S2 fetcher), so no S2 parquet file is needed as input. Each S1 granule covers a ~250 km swath; the strip window is a small clip from a ~29,000×22,500 px raster, fetched as ~44 block reads per band (2 block rows × 22 block columns at 512 px blocks over an 11,000 px wide strip).
- **Merger** (after S1 collector completes): `merge_scenes()` — DuckDB N-way sort-merge of ~85 S2 scene parquets + 1 S1 parquet, sorted by `(point_id, date)`, written as ZSTD + dictionary-encoded `strip_NNNN_sorted.parquet` (~5.6 GB). Output schema is `COMBINED_PIXEL_SCHEMA` — no S2 parquet file is opened to derive the schema. Deletes all scene parquets and the S1 strip parquet. Peak RAM: bounded by DuckDB spill budget (~2 GB).

Each scene parquet is **already sorted by `point_id`** — the pixel grid is fixed within a strip, so extraction order is deterministic. The merge only interleaves by `date`. DuckDB exploits the existing per-file sort order, making this effectively O(n) in practice while running entirely in C++ — no Python row iteration. This is the same strategy used by `parquet_utils._merge_sorted_parquets()` and `_sort_s1_shards()`.

### Across strips: fetch N+1 overlaps stream of N

The fetcher for strip N+1 starts as soon as strip N's scene parquets are handed to the merger — it does not wait for the merge or stream to finish:

```
strip 0:  [fetch scenes ▓▓▓▓▓▓▓▓▓▓▓][extract ▓▓▓▓▓▓▓▓▓▓▓][merge ▓▓]
strip 1:                        [fetch scenes ▓▓▓▓▓▓▓▓▓▓▓][extract ▓▓▓▓▓▓▓▓▓▓▓][merge ▓▓]
stream:                                                            [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓][▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]
```

The merger is fast (~30 s, DuckDB C++ sort-merge) and the stream is slow (~147 s). As long as fetch+extract+merge for strip N+1 completes before the stream of strip N finishes, the streamer never goes idle. The stream window is the budget — everything else must fit inside it.

The fetcher is gated: it cannot start strip N+2 until strip N+1's scene parquets are deleted (pipeline discipline). This bounds simultaneous disk usage to one strip's scene parquets + one sorted shard.

### Peak disk

Peak disk at any moment is ~7 GB — far below the 20 GB VM disk:

| Component | Size | Present when |
|---|---|---|
| 1 S2 `.npz` in flight (deleted after extractor ACKs) | ~570 MB | During S2 fetch/extract phase |
| All S2 scene parquets accumulated (deleted after merge) | ~630 MB | From first S2 scene until `merge_scenes()` completes — includes the S1 collection window |
| S1 strip parquet (deleted after merge) | ~110 MB | From S1 collection until `merge_scenes()` completes |
| Strip N sorted shard (streaming) | ~5.6 GB | During stream to workstation |
| **Total** | **~7 GB** | — |

The per-scene `.npz` files are deleted one at a time as extraction proceeds — they never accumulate. The ~630 MB figure for scene parquets is: ~57k clear pixels × 85 S2 scenes × ~120 bytes/row. S2 scene parquets persist through S1 collection because the merger needs all of them together.

## VM spec

| Resource | Recommended | Notes |
|---|---|---|
| CPU | 4 vCPU | More cores → faster scene extraction (`n_workers`) |
| RAM | 8 GB | Merge heap + buffers; well within budget |
| Disk | 20 GB SSD | ~7 GB peak usage; headroom for OS, logs, pip cache |
| Network | 1+ Gbps | AWS/DO default; S3 reads are the fetch bottleneck |

Primary candidate: `t3a.xlarge` (4 vCPU, 16 GB, ~$0.05/hr spot). Minimum viable: `t3a.large` (2 vCPU, 8 GB, ~$0.025/hr spot) if extraction parallelism is not the bottleneck. DigitalOcean 8 GB droplet (~$48/mo) is an alternative with simpler pricing.

## Instrumentation and tuning

The architecture is designed so the streamer never goes idle — fetch+extract+merge for strip N+1 is always racing the stream of strip N. Every stage boundary is logged with wall-clock time so you can immediately see which stage is the bottleneck:

```
[strip 0000]  fetch:   t=0s → t=Xs    (COG patches via fetch_patches(), max_concurrent=32)
[strip 0000]  extract: t=0s → t=Ys    (~92 scenes, n_workers=4, overlaps fetch)
[strip 0000]  merge:   t=Ys → t=Zs    (DuckDB sort-merge, ~92 parquets → 5.5 GB)
[strip 0000]  stream:  t=Zs → t=Ws    (5.5 GB at measured Mbps)
[strip 0001]  fetch:   t=Ys → ...     (started as soon as strip 0 scenes handed to merger)
--- strip 0001 ready at t=?s, stream ended at t=Ws ---
    if ready < Ws: streamer was never idle  ✓
    if ready > Ws: idle gap = ready-Ws      ✗ → tune knobs below
```

**Tuning knobs — all runtime parameters (env vars or CLI flags, no code changes):**

| Bottleneck | Knob | Default |
|---|---|---|
| Fetch too slow | `PROXY_MAX_CONCURRENT` (async S3 requests per strip) | 32 |
| Extract too slow | `PROXY_N_WORKERS` (rasterio threads per scene, passed as `n_workers` to `collect()`) | unset — `collect()` auto-scales via `_auto_n_workers()`; set only to override after measuring |
| Both too slow | Upsize VM (more vCPU → higher `n_workers` ceiling) | — |
| Strip too large for budget | `PROXY_STRIP_HEIGHT` (px, must be multiple of 1024) | 1024 |

See **COG block geometry** above for why sub-1024 px strips waste bandwidth for both sensors.

## Workstation-side merge

After all strip shards for a tile are received, the workstation calls `merge_strips()` from `parquet_utils.py` — the same function used by the local strip-decomposed fetch path:

```
merge_strips(tmp/<tile>/<year>/strip_NNNN.parquet ...)
  → data/pixels/mitchell/<year>/<tile_id>.parquet
  → delete tmp/<tile>/<year>/
```

This is an O(n) row-group copy — no re-sort. Strip boundaries have no pixel overlap (`lat_max` of strip N = `lat_min` of strip N+1), so concatenation preserves the global pixel sort order with no dedup step needed.

## Recovery and resumability

### Strip-level checkpoints

Received shards are written atomically using the 4-byte `LENGTH` field in the `0x02` frame header for verification (FastAPI `StreamingResponse` does not set `Content-Length`):

1. Server streams strip N → client writes payload to `tmp/<tile>/<year>/strip_NNNN.tmp`
2. Client verifies bytes written matches `LENGTH` from the `0x02` frame header
3. On verified completion: `rename strip_NNNN.tmp → strip_NNNN.parquet`

Only `.parquet` files are treated as complete. A `.tmp` file is deleted and the strip re-requested on resume.

### Tile-level checkpoints

After workstation merge completes:

```
data/pixels/mitchell/<year>/<tile_id>.parquet   ← output
data/pixels/mitchell/<year>/<tile_id>.done      ← sentinel
```

### Resume flow

1. Scan for `<tile_id>.done` — skip those tiles
2. For each remaining tile, scan `tmp/<tile>/<year>/strip_NNNN.parquet` for highest complete strip N
3. Submit `POST /run/tile` with `resume_from_strip=N+1`
4. Server skips strips 0..N and re-enters the producer/consumer loop

The VM is stateless between strips — no server-side checkpoint needed.

### Failure table

| Failure | State left | Recovery |
|---|---|---|
| Connection drop mid-stream | `strip_NNNN.tmp` on workstation | `.tmp` deleted on restart; strip re-requested |
| VM crash mid-merge | Nothing (VM stateless) | Strip re-run from scratch |
| VM crash mid-fetch | Partial `.npz` on VM | Per-strip cache always re-fetched; stale overwritten |
| Workstation crash after merge | `.done` sentinel present | Tile skipped on next run |
| Workstation crash before merge | Complete strip shards in `tmp/` | Merge re-runs from existing shards |

## Protocol

### Request

`POST /run/tile` with JSON body:

```json
{
  "tile_id": "54LWH",
  "year": 2021,
  "polygon_wkb_b64": "<base64-encoded WKB of catchment polygon>",
  "cloud_max": 20,
  "apply_nbar": true,
  "strip_height_px": 1024,
  "max_concurrent": 32,
  "n_workers": 4,
  "resume_from_strip": 0
}
```

`resume_from_strip` (default 0) causes the server to skip strips 0..N−1 and start producing from strip N. The client sets this to the index of the first incomplete strip on resume. "Skipping" means the server runs the STAC search once as normal (it is per-tile, not per-strip, and is cheap) then simply does not enter the fetch/extract/merge/stream loop for those strip indices — no COG traffic, no disk I/O.

`polygon_wkb_b64` is the catchment boundary in WKB format, base64-encoded. The server decodes it to a Shapely geometry once on request entry and passes it as `geometry=` to `collect()`. Passing the polygon (not a location YAML) keeps the VM decoupled from the YAML format.

### Response — framing

The response is a chunked HTTP stream carrying an alternating sequence of **frames**:

```
[TYPE 1 byte][LENGTH 4 bytes big-endian][PAYLOAD LENGTH bytes]
```

| Type byte | Payload |
|---|---|
| `0x01` | UTF-8 JSON progress line: `{"strip": N, "stage": "fetch|extract|merge|stream", "t": seconds}` |
| `0x02` | Raw parquet bytes for completed strip shard N |

The client reads frames sequentially: on `0x01` it logs progress; on `0x02` it writes the payload to `strip_NNNN.tmp`, verifies byte count against the frame `LENGTH` field, and renames to `strip_NNNN.parquet`. End-of-stream signals job complete.

This framing is unambiguous — no content-type sniffing, no delimiter scanning, no SSE parser. A partial frame means the connection dropped; the client deletes the `.tmp` and resumes from the last complete strip. End-of-stream is the natural end of the HTTP chunked response — the FastAPI `StreamingResponse` generator is exhausted, the server closes the connection cleanly, and the client detects EOF. No explicit termination frame is needed.

### Auth and concurrency

- **Auth**: SSH tunnel only — proxy listens on `localhost:8765` and is accessed via `ssh -L 8765:localhost:8765 vm`. No Bearer token; the SSH key is sufficient.
- **Concurrency**: single-process uvicorn with one Uvicorn worker — one job at a time. A second `POST /run/tile` while one is running returns `HTTP 409 Conflict`. Tile-level parallelism via multiple VMs.

## Files to create / modify

| File | Action |
|---|---|
| `proxy/server.py` | FastAPI app: `/run/tile` endpoint, per-strip producer/consumer loop, calls `collect(per_scene=True)` + `collect_s1_for_tile(points=...)` + `merge_scenes()`, length-prefixed frame stream, purge |
| `proxy/client.py` | `fetch_tiles()`: POST per tile, read length-prefixed frames, atomic `.tmp`→`.parquet` strip writes, calls `merge_strips()` |
| `proxy/requirements-server.txt` | fastapi, uvicorn, duckdb, shapely, rasterio (+ all of `utils/` dependencies) |
| `proxy/requirements-client.txt` | httpx (+ all of `utils/` dependencies, for `merge_strips()`) |
| `proxy/README.md` | VM setup: OS, Python env, S3 credentials, SSH tunnel, running the server, tuning knobs |
| `utils/pixel_collector.py` | Add `per_scene=True` flag to `collect()` — writes one parquet per scene sorted by `point_id`, yields `(scene_id, Path)`, skips stale-cache check |
| `utils/s1_collector.py` | Add `points: list[tuple[str, float, float]] \| None = None` kwarg and make `s2_path: Path \| None = None` — when `points` is supplied, coord reading from `s2_path` is skipped; `s2_path` is only accessed when `points` is not supplied (backwards-compatible). Schema is taken from `COMBINED_PIXEL_SCHEMA` when `points` is given. |
| `utils/parquet_utils.py` | Add `COMBINED_PIXEL_SCHEMA: pa.Schema` — hardcoded canonical S2+S1 combined schema (all S2 columns from `pixel_collector.py` output schema + `source`, `vh`, `vv`, `orbit`). Used by VM-facing codepaths (`collect_s1_for_tile(points=...)` and `merge_scenes()`) instead of deriving the schema from a live parquet file. `_extend_schema()` unchanged for existing callers. |
| `utils/location.py` | `Location.fetch(proxy_url=None)` — if set, delegates to `proxy/client.fetch_tiles()` instead of `fetch_spec()` |
| `cli/location.py` | `--proxy URL` flag on `cmd_fetch()` |
| `tests/unit/test_proxy_pipeline.py` | Unit tests (see below) |
| `scripts/bench_proxy.py` | Benchmarks and simulations (see below) |

No changes to `fetch_spec.py` or training code.

## Unit tests — `tests/unit/test_proxy_pipeline.py`

Tests that run locally with no S3 access, using synthetic data:

| Test | What it checks |
|---|---|
| `test_collect_per_scene_parquets` | `collect(per_scene=True)` writes exactly one parquet per clear scene (S2+S1); combined row count matches `collect()` |
| `test_collect_per_scene_point_id_sorted` | Each per-scene parquet is sorted by `point_id` (pixel grid order is deterministic) |
| `test_collect_per_scene_polygon_mask` | Pixels outside the supplied Shapely geometry are absent from all per-scene parquets |
| `test_collect_per_scene_s1_rows` | S1 rows are present in output when S1 items are in the STAC result |
| `test_duckdb_merge_sorted_output` | DuckDB merge of N per-scene parquets (S2+S1) produces output sorted by `(point_id, date)` |
| `test_duckdb_merge_dictionary_encoded` | Merged strip parquet has dictionary encoding on `point_id` column |
| `test_duckdb_merge_row_count` | Merged strip row count = sum of per-scene row counts (no rows lost or duplicated) |
| `test_strip_purge_after_merge` | After merger completes, all `scene_NNN.parquet` files are deleted |
| `test_frame_roundtrip` | `write_frame()` / `read_frame()` roundtrip for both frame types (0x01 JSON, 0x02 parquet bytes) |
| `test_atomic_strip_write` | Client reads 0x02 frame, writes `.tmp`, verifies length against frame LENGTH field, renames to `.parquet` |
| `test_resume_skips_complete_strips` | Client with `resume_from_strip=N` skips strips 0..N−1 and requests from N onward |
| `test_workstation_merge_row_count` | `merge_strips()` on synthetic strip shards produces correct row count (sum of strip rows; no boundary overlap by construction) |
| `test_compression_ratio` | Sorted+dict strip is ≥5× smaller than unsorted equivalent (sanity-checks compression hypothesis) |

## Benchmarks and simulations — `scripts/bench_proxy.py`

Same ergonomics as `bench_score.py`, `bench_train.py`, etc.: standalone script, probe table of RSS + wall time per stage, `--assert-*` flags for CI thresholds.

```
python scripts/bench_proxy.py
python scripts/bench_proxy.py --n-scenes 100 --strip-px 1024 --n-pixels 11000
python scripts/bench_proxy.py --assert-merge-s 30 --assert-compression-ratio 5
python scripts/bench_proxy.py --sim-fetch-s 80 --sim-extract-s 60 --sim-merge-s 30 --sim-stream-s 150
```

| Benchmark / simulation | Hypothesis being tested | Pass condition |
|---|---|---|
| `bench_duckdb_merge` | DuckDB merges 100 per-scene parquets (57 k rows each) in ≤30 s | `--assert-merge-s 30` |
| `bench_extract_scene` | `collect(per_scene=True)` extraction on synthetic 1024×1024 npz completes in ≤2 s per scene | `--assert-extract-s 2` |
| `bench_compression_ratio` | Sorted+dict strip is ≥5× smaller than unsorted equivalent | `--assert-compression-ratio 5` |
| `bench_workstation_concat` | `merge_strips()` on 11 strip shards runs at ≥500 MB/s | `--assert-concat-mb-s 500` |
| `sim_pipeline` | Discrete-event simulation: given `--sim-{fetch,extract,merge,stream}-s`, compute idle gap per strip across N strips | idle = 0 when merge finishes before stream ends |

`sim_pipeline` is a pure-Python no-I/O simulation. Plug in measured or estimated stage times to confirm the pipeline stays saturated before deploying to a VM.

## Throughput targets

| Unit | Transfer time at 300 Mbps |
|---|---|
| One strip — 1024 px (~5.6 GB) | ~150 s |
| One tile/year (~60 GB, 11 strips — ⌈11,000 px ÷ 1024⌉) | ~27 min |
| Full Mitchell River, 1 year (~2.7 TB) | ~20 hours |

Producer budget per strip: ≤150 s for fetch+extract+merge. Whether this is achievable depends on S3→VM throughput, which must be measured on first use. If the producer can't keep pace, the knobs are `max_concurrent`, `n_workers`, and VM size — in that order.

## Verification

1. Run server locally against a 2×2 km sub-bbox of 54LWH for one year; confirm final parquet matches a direct `Location.fetch()` run (same rows, same order, same NBAR-corrected values).
2. After each strip, assert all scene parquets and `.npz` files are deleted.
3. Log bytes streamed per strip — confirm ~5.6 GB (sorted+dict), not ~57 GB (unsorted).
4. Monitor idle time between strips — target zero; any idle time identifies the bottleneck stage.
5. Time producer vs consumer per strip on first real tile; adjust knobs until idle ≈ 0.
