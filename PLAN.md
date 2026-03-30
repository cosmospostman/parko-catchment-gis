# Two-pool fetch/compute refactor + pipeline abstraction

## Goal

Decouple network fetching from CPU processing so fetch connections stay continuously saturated and CPU workers never block the network. Extract shared pipeline boilerplate for steps 01 and 03.

## Architecture

`load_stackstac` is lazy — all HTTP I/O happens inside `.compute()`. The split point is:

- **Fetch worker**: `load_stackstac → apply_scl_mask → raw.compute()` — materialises raw band array, puts on queue
- **Compute worker**: pulls from queue, runs index math as pure numpy, writes tile `.tif`

```
main thread
  ├─ ThreadPoolExecutor(FETCH_WORKERS)  — fetch_tile() → q.put((idx, raw_array))
  └─ COMPUTE_WORKERS plain threads      — q.get() → index math → write .tif
```

Queue: `queue.Queue(maxsize=COMPUTE_WORKERS * 2)` — fetch threads block on `q.put()` when full, bounding memory and preventing connection bursts.

## New env vars

Replace `TILE_WORKERS` with two independent knobs:

| Var | Default | Role |
|---|---|---|
| `FETCH_WORKERS` | 16 | Network-bound, matches connection pool size |
| `COMPUTE_WORKERS` | `os.cpu_count()` | CPU-bound |
| `TILE_SIZE_PX` | 512 (up from 256) | Bytes per connection per request |

`--tile-workers` in `run.sh` kept as deprecated alias that sets both.

## New abstraction: `utils/pipeline.py`

Steps 01 and 03 share an identical skeleton — only the STAC search, bands loaded, and compute function differ. Extract to `run_tiled_pipeline(fetch_fn, compute_fn, ...)` in a new `utils/pipeline.py`. Each script becomes ~40 lines.

**Step 02 does not use `run_tiled_pipeline`** — it has two sequential tiled passes (baseline build + anomaly compute) with an intermediate cache, uses `odc-stac` instead of `stackstac`, and the anomaly compute worker needs two inputs (current tile + baseline slice). Forcing this into the abstraction would require special-case parameters for a single caller.

Instead, step 02 uses only the shared utility helpers extracted to `utils/pipeline.py`:
- `setup_gdal_env()` — GDAL HTTP settings, AWS env vars
- `setup_proj()` — PROJ_DATA bootstrap, pyproj `set_data_dir()`

These replace the verbatim copy-pasted blocks currently in all three scripts.

## Memory

At `TILE_SIZE_PX=512`, step 01 (~12 cloud-free scenes, 8 bands, float32):
512 × 512 × 8 × 12 × 4 bytes ≈ 100 MB/tile. Max in-flight: `FETCH_WORKERS` + `maxsize` + `COMPUTE_WORKERS` ≈ 40 tiles ≈ 4 GB on a 32 GB node. Acceptable.

Step 02 stays at `TILE_SIZE_PX=1024` (2 bands, 30 m — much lower footprint).

## Implementation order

1. **`utils/pipeline.py`** (new) — `setup_gdal_env()`, `setup_proj()`, `run_tiled_pipeline()`
2. **`analysis/01_ndvi_composite.py`** — use `run_tiled_pipeline`; split `process_tile` into `fetch_tile` + `compute_fn`; update defaults
3. **`analysis/03_flowering_index.py`** — same; flowering-specific `compute_fn`
4. **`analysis/02_ndvi_anomaly.py`** — use `setup_gdal_env()` + `setup_proj()` only; keep orchestration inline; apply two-pool pattern to `_build_baseline()` manually
5. **`run.sh`** — add `--fetch-workers` / `--compute-workers`; deprecate `--tile-workers`; update `TILE_SIZE_PX` default to 512; update banner
6. **Tests** — verify mock targets still valid after refactor; add backpressure test

## What this buys

| Before | After |
|---|---|
| 16 mixed workers, connections idle during CPU phase | 16 fetch threads continuously streaming |
| Connections burst-open at startup | Queue backpressure bounds concurrent fetches |
| CPU at 30%, network throttled at ~4.5 Gbps | CPU workers always have work; network stays saturated |
| Single `TILE_WORKERS` knob | Independent `FETCH_WORKERS` / `COMPUTE_WORKERS` tuning |
| GDAL/PROJ setup copy-pasted 3× | Single `setup_gdal_env()` / `setup_proj()` in `utils/pipeline.py` |
| ~150 lines of orchestration per script | ~40 lines per script; shared logic in `utils/pipeline.py` |
