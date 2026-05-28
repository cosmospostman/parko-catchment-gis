# Fetch pipeline

## Unified model: `FetchSpec`

Both the location and training pipelines share the same fetch layer.
`FetchSpec` in `utils/fetch_spec.py` replaces the old `Location`/`TrainingRegion` split:

```python
@dataclass(frozen=True)
class FetchSpec:
    id: str                        # "longreach" or "lake_mueller_presence"
    bbox: list[float]              # [lon_min, lat_min, lon_max, lat_max]
    years: list[int]
    point_id_prefix: str           # "px" for locations, region.id for training
    geometry: object | None = None # polygon mask (locations); None for training regions
    label: str | None = None       # "presence"/"absence" (training); None for locations
    out_dir: Path | None = None    # caller controls output root
    cache_dir: Path | None = None  # explicit override; training uses per-region subdir
```

`fetch_spec(spec, ...)` in `utils/fetch_spec.py` runs the three-stage pipeline
and writes `<out_dir>/<year>/<tile_id>.parquet` for both pipelines:

```
fetch_spec(spec)
├── for year in spec.years (parallel):
│   ├── collect(spec, year)           → <out_dir>/<year>/<tile_id>.s2.parquet
│   ├── collect_s1_for_tile(...)      → <out_dir>/<year>/<tile_id>.s1.parquet
│   └── merge_tile(s2, s1)           → <out_dir>/<year>/<tile_id>.parquet
```

**`Location.fetch(years, proxy_url=None)`** — thin wrapper around `fetch_spec()`.
If `proxy_url` is set (e.g. `"http://localhost:8765"`), the work runs on the VM
via `proxy/client.py` instead of locally (see **Proxy fetch** below).

**`training_collector._collect_one_region(region)`** — constructs a `FetchSpec`
and calls `fetch_spec()`; STAC sharing across regions on the same tile stays in
`training_collector` and is passed via the `items=` parameter.

## Local fetch

```
Location.fetch()
  └─ fetch_spec()
       ├─ Phase A (all years parallel, network-bound)
       │    └─ collect(phases={"fetch"})          — fetch_patches() → .npz cache on workstation disk
       │
       └─ Phase B (≤max_extract_years concurrent, memory-bound)
            ├─ collect(phases={"extract"})         — .npz → per-scene DataFrames → .s2.parquet
            ├─ collect_s1_for_tile()               — S1 STAC search, COG fetch, .s1.parquet
            └─ merge_tile()                        — DuckDB S2+S1 sort-merge → <tile_id>.parquet
```

All compute and all I/O happen on the workstation. The WAN link sees raw COG
traffic (~5 TB per tile/year).

## Proxy fetch

### Problem

The fetch pipeline is network-bound on the workstation side. For the Mitchell
River catchment (72,000 km², 45 MGRS tiles, ~720M pixels) a single year of
S2+S1 parquet is ~2.7 TB. Getting there requires fetching COG windows from AWS
S3 — those windows are 50–80× larger than the parquet they produce:

- 13 bands × ~100 scenes/tile/year × ~4 MB/window = **~5 TB of COG traffic** to
  produce ~60 GB of parquet per tile
- At 300 Mbps uplink, 5 TB takes ~37 hours per tile

A VM co-located with the Element84 S3 COGs reads those COG windows at Gbps
rates. The proxy runs extraction there and streams only the compressed sorted
parquet back, so the 300 Mbps WAN link sees ~60 GB per tile instead of ~5 TB.

### Scale reference

Calibrated from Kowanyama 2021 (same tropical NQ climate, measured actuals):

| Metric | Value |
|---|---|
| Clear obs/pixel/year (S2+S1 combined) | ~115 (~85 S2, ~30 S1) |
| Final parquet per million pixels per year | ~3.7 GB |
| One MGRS tile (~121M pixels), one year | **~60 GB** |
| Mitchell River (45 tiles), one year | **~2.7 TB total** |

At 300 Mbps, 2.7 TB takes ~20 hours — the WAN transfer dominates. The VM's job
is to keep the pipe saturated.

### VM location

The Element84 COG bucket is in us-west-2. A VM in the same region gets free,
Gbps-rate S3 reads. A DigitalOcean VM in SFO is cross-cloud but the SFO→Oregon
path is well-peered (~5–15 ms, 500–800 Mbps sustained) and DO charges outbound
from DO rather than S3 egress. Either location works; the 300 Mbps workstation
link is the bottleneck regardless of VM location or RTT.

### Why sort on the VM — compression mechanics

The goal is to minimise WAN bytes. Sorted+dict parquet achieves ~10× compression
over raw unsorted because:

1. **Dictionary encoding on `point_id`**: sorted output groups all ~115
   observations for each pixel consecutively. The dictionary stores each unique
   `point_id` string once; the index stream is long runs of the same integer —
   ZSTD compresses runs to near-zero.

2. **Spectral locality in band columns**: all observations for a pixel are
   consecutive; band values have low variance within each run.

Per-scene transfer would be ~57 GB/strip uncompressed vs ~5.5 GB sorted+dict —
~10× more WAN traffic. The sort must happen on the VM.

### Proxy call path

```
Location.fetch(proxy_url=...)
  └─ proxy/client.py: fetch_tiles()
       ├─ For each tile × year: POST /run/tile
       │
       │   VM (proxy/server.py → proxy/_pipeline.py):
       │   ├─ stac.search_sentinel2()              — same utils/stac.py
       │   ├─ Pool A (fetch):  fetch_patches_to_tiff()  — one GeoTIFF per (item×band) on VM disk
       │   ├─ Pool B (extract): _extract_item_from_tiffs() → per-scene parquets sorted by point_id
       │   ├─ collect_s1_for_tile(points=...)      — S1 COG fetch + extraction (after all S2 scenes)
       │   └─ merge_scenes()                       — DuckDB N-way sort of scene parquets + S1 → strip_NNNN.parquet
       │
       │   Frame stream (VM → workstation):
       │   └─ 0x01 progress + 0x02 strip shards   — ~5.5 GB/strip at 300 Mbps
       │
       └─ workstation: merge_strips()             — N sorted shards → <tile_id>.parquet
```

### Shared vs new code

| Component | Local | Proxy VM | Proxy client (workstation) |
|---|---|---|---|
| `utils/stac.py` search | ✓ | ✓ same code | — |
| `fetch_patches_to_tiff()` | — | ✓ new | — |
| `_extract_item_from_tiffs()` | — | ✓ new | — |
| `collect()` | ✓ full | ✓ with `per_scene=True` | — |
| `collect_s1_for_tile()` | ✓ | ✓ with `points=` kwarg | — |
| `COMBINED_PIXEL_SCHEMA` | — | new — in `parquet_utils.py` | — |
| `merge_tile()` | ✓ | — | — |
| `merge_scenes()` | — | new — in `proxy/_pipeline.py` | — |
| `merge_strips()` | ✓ (strip mode) | — | ✓ same code |
| Frame protocol | — | write frames | read frames |
| Atomic `.tmp`→`.parquet` | — | — | new (client only) |
| `Location.fetch()` proxy branch | — | — | new (thin dispatcher) |

## COG block geometry

Both sensors have been verified against live tiles:

| Sensor | Source | Block size | Compression |
|---|---|---|---|
| Sentinel-2 | Element84 / AWS S3 | **1024×1024 px** | DEFLATE/LZW |
| Sentinel-1 RTC | MPC / Azure Blob | **512×512 px** | DEFLATE |

A range request for a window smaller than one block tall reads a full block.
Strip height must be a multiple of both block sizes. Since 1024 = 2 × 512,
**1024 px satisfies both sensors with zero over-fetch penalty for either**.

| Strip height | S2 blocks fetched | S2 waste | S1 blocks fetched | S1 waste |
|---|---|---|---|---|
| 256 px | 1 (1024 px) | 75% | 1 (512 px) | 50% |
| 512 px | 1 (1024 px) | 50% | **1 — zero waste** | **0%** |
| **1024 px** | **1 — zero waste** | **0%** | **2 — zero waste** | **0%** |
| 2048 px | 2 — zero waste | 0% | 4 — zero waste | 0% |

**1024 px is the natural strip unit for both sensors.** Do not use sub-1024 px
strips to work around a slow producer — the fix is always more parallelism or a
larger VM.

## VM-side pipeline (`run_tile_pipeline_v2` in `proxy/_pipeline.py`)

The v2 pipeline uses two concurrent thread pools to keep the network link
saturated throughout extraction:

- **Pool A (network→disk)**: `fetch_patches_to_tiff()` writes one GeoTIFF per
  `(item, band)` to disk immediately, dereferencing the array after each write.
  Peak RAM = O(one patch array) regardless of item count.
- **Pool B (disk→extract→parquet)**: `_extract_item_from_tiffs()` opens on-disk
  TIFFs, samples pixel values, writes per-scene parquets sorted by `point_id`.
  Peak RAM = O(n_points × n_bands × 4 bytes) per worker (~MB).

**Depth-2 prefetch across strips**: Pool A for strip N+2 runs concurrently with
Pool B for strip N, keeping the network link busy during extraction.

```
strip 0:  [Pool A fetch ▓▓▓▓▓▓▓▓][Pool B extract ▓▓▓▓▓▓▓▓][S1▓][merge▓▓]
strip 1:       [Pool A fetch ▓▓▓▓▓▓▓▓][Pool B extract ▓▓▓▓▓▓▓▓][S1▓][merge▓▓]
stream:                                                       [▓▓▓▓▓▓▓▓▓▓][▓▓▓▓▓▓▓▓▓▓]
```

On machines with ≤10 GB RAM the prefetch depth defaults to 1; set
`PREFETCH_DEPTH=2` on larger instances.

After all S2 scenes are extracted, `collect_s1_for_tile(points=pixel_grid, ...)`
fetches ~30 S1 granules for the strip. Then `merge_scenes()` runs a DuckDB
N-way sort-merge of all S2 scene parquets + the S1 parquet, sorted by
`(point_id, date)`, producing `strip_NNNN_sorted.parquet` (~5.6 GB, ZSTD +
dictionary-encoded).

Each scene parquet is **already sorted by `point_id`** — the pixel grid is fixed
within a strip, so the merge only interleaves by date. DuckDB exploits the
existing per-file sort order, making this effectively O(n) in practice.

### Peak disk

| Component | Size | Present when |
|---|---|---|
| 1 strip's GeoTIFFs (deleted after Pool B ACKs) | ~1–2 GB | During Pool A/B overlap |
| All S2 scene parquets accumulated | ~630 MB | From first scene until `merge_scenes()` completes |
| S1 strip parquet | ~110 MB | From S1 collection until `merge_scenes()` completes |
| Strip N sorted shard (streaming) | ~5.6 GB | During stream to workstation |
| **Total** | **~8 GB** | — |

### VM spec

| Resource | Recommended | Notes |
|---|---|---|
| CPU | 4 vCPU | More cores → higher `EXTRACT_WORKERS` ceiling |
| RAM | 16 GB | Pool A TIFFs + extraction + merge heap |
| Disk | 20 GB SSD | ~8 GB peak; headroom for OS, logs, pip cache |
| Network | 1+ Gbps | AWS/DO default; S3 reads are the fetch bottleneck |

Primary candidate: `t3a.xlarge` (4 vCPU, 16 GB, ~$0.05/hr spot).
Minimum viable: `t3a.large` (2 vCPU, 8 GB) — depth-1 prefetch only.

## Sensors handled

Both sensors are fetched, extracted, and sorted on the VM before streaming:

- **Sentinel-2** (~85 obs/pixel/year): STAC via Element84, COG bands
  (B02–B12 + SCL + B8A), polygon-masked, per-scene parquets sorted by
  `point_id`
- **Sentinel-1** (~30 obs/pixel/year): STAC via Microsoft Planetary Computer
  (MPC), VH+VV COGs, same polygon mask — fetched after all S2 scenes via
  `collect_s1_for_tile(points=...)`

## Workstation-side merge

After all strip shards for a tile are received, the workstation calls
`merge_strips()` from `parquet_utils.py`:

```
merge_strips(tmp/<tile>/<year>/strip_NNNN.parquet ...)
  → data/pixels/mitchell/<year>/<tile_id>.parquet
  → delete tmp/<tile>/<year>/
```

This is an O(n) row-group copy — no re-sort. Strip boundaries have no pixel
overlap (`lat_max` of strip N = `lat_min` of strip N+1), so concatenation
preserves global sort order with no dedup step.

## API

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
  "n_workers": null,
  "resume_from_strip": 0
}
```

`polygon_wkb_b64` — catchment boundary in WKB format, base64-encoded. Decoded
to a Shapely geometry once on request entry; the VM is decoupled from the YAML
format.

`resume_from_strip` (default 0) — server skips strips 0..N−1 and starts from N.
The STAC search still runs once as normal (it is cheap); only the
fetch/extract/merge/stream loop is skipped for earlier strips.

### Response — framing

The response is a chunked HTTP stream of length-prefixed frames:

```
[TYPE 1 byte][LENGTH 4 bytes big-endian][PAYLOAD LENGTH bytes]
```

| Type byte | Payload |
|---|---|
| `0x01` | UTF-8 JSON progress: `{"strip": N, "stage": "fetch\|extract\|merge\|stream", "t": seconds}` |
| `0x02` | Raw parquet bytes for completed strip shard N |

The client reads frames sequentially: on `0x01` it logs progress; on `0x02` it
writes the payload to `strip_NNNN.tmp`, verifies byte count against the frame
`LENGTH` field, and renames to `strip_NNNN.parquet`. End-of-stream (exhausted
`StreamingResponse` generator) signals job complete — no explicit termination
frame needed.

Frame encode/decode (`write_frame` / `read_frame`) lives in `proxy/_pipeline.py`
and is shared by both server and tests.

### Auth and concurrency

- **Auth**: SSH tunnel only — proxy listens on `localhost:8765`, accessed via
  `ssh -N -L 8765:localhost:8765 ubuntu@<vm-ip>`. No Bearer token.
- **Concurrency**: single uvicorn worker — one job at a time. A concurrent
  `POST /run/tile` returns `HTTP 409 Conflict`. Tile-level parallelism via
  multiple VMs.

## Recovery and resumability

### Strip-level checkpoints

1. Server streams strip N → client writes payload to `strip_NNNN.tmp`
2. Client verifies bytes written matches `LENGTH` from the `0x02` frame header
3. On verified completion: `rename strip_NNNN.tmp → strip_NNNN.parquet`

Only `.parquet` files are treated as complete. A `.tmp` file is deleted and the
strip re-requested on resume.

### Tile-level checkpoints

```
data/pixels/mitchell/<year>/<tile_id>.parquet   ← output
data/pixels/mitchell/<year>/<tile_id>.done      ← sentinel
```

### Resume flow

1. Scan for `<tile_id>.done` — skip those tiles
2. For each remaining tile, scan `tmp/<tile>/<year>/strip_NNNN.parquet` for
   highest complete strip N
3. Submit `POST /run/tile` with `resume_from_strip=N+1`
4. Server skips strips 0..N (STAC search still runs; no COG traffic for skipped
   strips)

The VM is stateless between strips — no server-side checkpoint needed.

### Failure table

| Failure | State left | Recovery |
|---|---|---|
| Connection drop mid-stream | `strip_NNNN.tmp` on workstation | `.tmp` deleted on restart; strip re-requested |
| VM crash mid-merge | Nothing (VM stateless) | Strip re-run from scratch |
| VM crash mid-fetch | Partial TIFFs on VM | Per-strip TIFFs always re-fetched; stale overwritten |
| Workstation crash after merge | `.done` sentinel present | Tile skipped on next run |
| Workstation crash before merge | Complete strip shards in `tmp/` | Merge re-runs from existing shards |

## Tuning

All tuning knobs are runtime parameters (env vars or per-request fields):

| Bottleneck | Knob | Default |
|---|---|---|
| S2 fetch too slow | `FETCH_WORKERS` (Pool A threads) | 16 |
| S2 extract too slow | `EXTRACT_WORKERS` (Pool B threads) | min(4, cpu_count) |
| Peak RAM too high | `PREFETCH_DEPTH` (strips fetched ahead) | 1 if ≤10 GB RAM, else 2 |
| Merge OOM | `PROXY_MERGE_MEM_GB` | auto (system RAM / 2) |
| S1 fetch too slow | `max_concurrent` in request (async S3 requests) | 32 |
| Strip too large | `strip_height_px` in request (must be multiple of 1024) | 1024 |

Log output includes wall-clock time for every stage boundary:

```
[v2 tile 54LWH 2021] [strip 0000] Pool A fetch → strip_0000_tiffs
[v2 tile 54LWH 2021] [strip 0000] Pool A done; starting Pool B (57200 pts, 85 items)
[v2 tile 54LWH 2021] [strip 0000] ready → strip_0000_sorted.parquet
```

The stream of strip N overlaps Pool A+B of strip N+1. If `ready` for strip N+1
arrives before the stream of strip N finishes, the streamer is never idle.

## Throughput targets

| Unit | Transfer time at 300 Mbps |
|---|---|
| One strip — 1024 px (~5.6 GB) | ~150 s |
| One tile/year (~60 GB, 11 strips) | ~27 min |
| Full Mitchell River, 1 year (~2.7 TB) | ~20 hours |

Producer budget per strip: ≤150 s for Pool A + Pool B + S1 + merge. If the
producer can't keep pace, the knobs are `FETCH_WORKERS`, `EXTRACT_WORKERS`,
`PREFETCH_DEPTH`, and VM size — in that order.

## Files

| File | Role |
|---|---|
| `proxy/_pipeline.py` | Shared pipeline logic: `run_tile_pipeline_v2`, `merge_scenes`, `write_frame`, `read_frame` |
| `proxy/server.py` | FastAPI app: `/run/tile` endpoint, frame streaming |
| `proxy/client.py` | `fetch_tiles()`: POST per tile, frame reader, atomic strip writes, calls `merge_strips()` |
| `proxy/requirements-server.txt` | fastapi, uvicorn, duckdb, shapely, rasterio, + utils deps |
| `proxy/requirements-client.txt` | httpx, + utils deps |
| `proxy/README.md` | VM setup: OS, Python env, credentials, SSH tunnel, running the server |
| `utils/fetch_spec.py` | `FetchSpec` dataclass + `fetch_spec()` orchestrator |
| `utils/pixel_collector.py` | `collect()` with `per_scene=True` flag; `_extract_item_from_tiffs()` |
| `utils/fetch.py` | `fetch_patches_to_tiff()` — patch-based S2 fetch writing GeoTIFFs to disk |
| `utils/s1_collector.py` | `collect_s1_for_tile()` with `points=` kwarg |
| `utils/parquet_utils.py` | `COMBINED_PIXEL_SCHEMA`, `merge_tile()`, `merge_strips()` |
| `utils/location.py` | `Location.fetch(proxy_url=None)` — delegates to `fetch_spec()` or proxy |
| `tests/unit/test_proxy_pipeline.py` | Unit tests for proxy pipeline (no S3 access) |
| `scripts/bench_fetch.py` | Benchmarks and discrete-event simulation |

## Verification

1. Run server locally against a 2×2 km sub-bbox of 54LWH for one year; confirm
   final parquet matches a direct `Location.fetch()` run (same rows, same order,
   same NBAR-corrected values).
2. After each strip, assert all scene parquets and TIFFs are deleted.
3. Log bytes streamed per strip — confirm ~5.6 GB (sorted+dict), not ~57 GB
   (unsorted).
4. Monitor idle time between strips — target zero; any idle time identifies the
   bottleneck stage.
5. Time producer vs consumer per strip on first real tile; adjust knobs until
   idle ≈ 0.
