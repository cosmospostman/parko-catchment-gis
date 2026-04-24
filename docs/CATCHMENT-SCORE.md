# Catchment-wide scoring — compute notes

Covers the full Mitchell River Catchment run:
**1,641,266,665 pixels · 71,622 km² · 30 S2 tiles · 10 m resolution**

---

## Pipeline overview

Scoring uses a two-phase tile-sharded approach (`score_tiles_chunked` in `tam/core/score.py`).

**Phase 1 — per-(tile, year) staging:**
`score_pixels_chunked` is called once per (tile, year) pair.  Within each call a
three-stage concurrent pipeline runs:

```
reader thread  →  raw_q  →  N prep threads (numba)  →  prep_q  →  main thread (torch inference)
```

Results (point\_id, year, prob\_tam\_raw) are written to a staging parquet:
`outputs/{loc_id}/{model_id}/{end_year}/staging/{tile_id}_{year}.parquet`

Staging files are written atomically (write to `.tmp.parquet`, then rename) so
a crash mid-write leaves no partial file. On restart, only fully-written staging
files are reused; tile-years with no staging file are scored again from scratch.

**Phase 2 — per-tile aggregation:**
All staging parquets for one tile are read, decay-weighted aggregation is applied
across years, scores are converted to uint8 [0–100], and the final parquet is written.
Staging files are deleted after the final write.

---

## Scale numbers

| | |
|---|---|
| Total pixels | 1,641,266,665 |
| S2 tiles covering catchment | 30 |
| Pixels per tile (approx.) | ~55M (edge tiles) – 100M (full tiles) |
| Years of imagery (2020–2025) | 6 |
| Total (tile, year) pairs | ~180 |

---

## Memory budget (32 GB machine)

Memory is dominated by Phase 1 inference and Phase 2 aggregation.

**Phase 1 — peak per worker per tile-year**

After `score_pixels_chunked` returns, the in-memory DataFrame holds
~80M rows (assuming ~80% pass the `min_obs_per_year` filter):

| Component | Size |
|---|---|
| `point_id` column (Python object strings, ~13 bytes/ptr) | ~1.0 GB |
| `prob_tam_raw` float32 | ~0.3 GB |
| Staging parquet write buffer | ~0.2 GB |
| Torch model weights + batch tensors in flight | ~0.1 GB |
| **Peak per worker** | **~1.5–2 GB** |

**Phase 2 — aggregation per tile**

All 6 staging parquets for one tile are concatenated before aggregation:

| Component | Size |
|---|---|
| Concatenated raw DataFrame (6 years × ~80M rows) | ~8–10 GB |
| Aggregated result (100M rows, point\_id + uint8) | ~1.5 GB |
| **Phase 2 peak** | **~10–12 GB** |

Phase 2 dominates memory. With `n_tile_workers=1` (sequential) the machine
comfortably handles this on 32 GB. With multiple workers, stagger tile sizes
so that large (full) tiles do not overlap in Phase 2 simultaneously.

**Safe worker counts on 32 GB:**

| `n_tile_workers` | Notes |
|---|---|
| 1 | Safe, no risk — good for validation |
| 2 | Safe if tiles are not all maximum-size simultaneously |
| 3–4 | Risky during Phase 2 aggregation; prefer if tiles are small |

---

## Inference throughput

The TAMClassifier is a small Transformer (d\_model=64, 2 layers, 4 heads).
Each forward pass processes a batch of 4096 pixel-year windows, each of shape
(128 time-steps × 13 features). One tile-year requires ~19,500 batches.

Throughput is **unmeasured on this hardware** — run a single tile first and
time it before committing to a full catchment run.  Rough expectations:

| Hardware | Estimated time per tile-year |
|---|---|
| Apple M-series (AMX, 10 cores) | 3–10 min |
| x86 CPU, AVX2, 8 cores | 10–30 min |
| c7g.4xlarge (Graviton3, 16 cores) | 5–15 min |
| GPU (any modern) | < 1 min |

At the low end (3 min/tile-year, 180 tile-years, 4 workers): **~2.5 hours**.  
At the high end (30 min/tile-year, sequential): **~90 hours**.

**Mitchell River Catchment — c7g.4xlarge (1 year, 17 tile-years)**

The Mitchell run is 17 tile-years (1 year × 17 S2 tiles, 1.64B pixels total).
At 5–15 min/tile-year on a c7g.4xlarge:

| Workers | Optimistic (5 min) | Realistic (10 min) | Pessimistic (15 min) |
|---|---|---|---|
| 1 | ~85 min | ~2.8 hrs | ~4.3 hrs |
| 4 | ~25 min | ~45 min | ~65 min |
| 6 | ~15 min | ~30 min | ~45 min |

Memory: Phase 2 peaks at ~1.5–2 GB/tile (single year), well within the 32 GB on a c7g.4xlarge.
Safe worker count: 6–8. Recommended starting point: `n_tile_workers=4` with 4 threads each,
then measure a single tile-year before committing to the full run.

The `torch.set_num_threads` call inside each worker divides available cores
evenly across workers. On a 10-core M-series with `n_tile_workers=4`, each
worker gets 2–3 threads — PyTorch's intra-op parallelism within a single
forward pass is modest at that thread count, so **2 workers** may outperform
4 in practice (each gets 5 threads, better BLAS utilisation).

---

## Disk budget

| Artefact | Size |
|---|---|
| Staging parquets (all tile-years, zstd) | ~68 GB peak |
| Final score parquets (30 tiles, uint8, zstd) | ~6 GB |
| Staging parquets after cleanup | deleted |

The staging directory peaks at ~68 GB if all tile-years complete before any
Phase 2 aggregation runs. In sequential mode (`n_tile_workers=1`) each tile's
staging files are cleaned up before moving to the next, so peak staging disk
is just ~2.3 GB (6 years × ~380 MB each).

---

## Invocation

```bash
# Fetch Mitchell catchment pixel parquets for 2025
python cli/location.py fetch mitchell --years 2025
```

```bash
# Sequential (safe, lower disk pressure)
python -m tam.pipeline score \
    --checkpoint outputs/tam-v1_spectral \
    --location mitchell \
    --years 2025 \
    --out-parquet \
    --n-tile-workers 1

# Parallel (measure single-tile time first)
python -m tam.pipeline score \
    --checkpoint outputs/tam-v1_spectral \
    --location mitchell \
    --years 2025 \
    --out-parquet \
    --n-tile-workers 2

# Output
# outputs/mitchell/tam-v1_spectral/2025/{tile_id}.scores.parquet
# Schema: point_id (string), prob_tam (uint8, 0–100)
```

Crash recovery: re-running the same command resumes from the last completed
tile-year. Staging files are written atomically (`.tmp.parquet` → rename), so
a mid-write crash leaves no partial file to confuse a restart.

---

## Next step — GeoTIFF assembly

The 30 tile parquets feed a rasterisation step (not yet implemented) that burns
`prob_tam` uint8 values into a single cloud-optimised GeoTIFF.  Expected output
size: ~400–700 MB (Deflate-compressed uint8, tiled 512×512, with overviews).
