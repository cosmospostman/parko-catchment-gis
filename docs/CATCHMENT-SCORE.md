# Catchment-wide scoring — compute notes

Covers the full Mitchell River Catchment run:
**1,641,266,665 pixels · 71,622 km² · 30 S2 tiles · 10 m resolution**

---

## Pipeline overview

Scoring uses a two-phase tile-sharded approach (`score_tiles_chunked` in `tam/core/score.py`).

**Phase 1 — per-(tile, year) staging:**
`score_tile_year` is called once per (tile, year) pair. Each call runs a four-stage
concurrent pipeline:

```
Pre-passes (parallel)
  S2 z-score ──┐
  S1 z-score ──┘  (both read parquet concurrently; scoring blocked until complete)
        ↓
reader thread  →  raw_q  →  N prep threads (numba)  →  prep_q  →  GPU inference  →  out_q  →  writer thread
                                                                                                      ↓
                                                                                             staging/{tile}_{year}.parquet
                                                                                             (flushed every 500K rows)
```

Pre-passes must complete before scoring starts because their output (per-pixel z-score
stats) is needed by every prep worker. They read the parquet concurrently (~20–60s at
full tile scale), then scoring streams results directly to a staging parquet via the
writer thread. Memory for the output path is bounded to ~80 MB regardless of tile size.

Staging files are written atomically (write to `.tmp.parquet`, then rename) so a crash
mid-write leaves no partial file. On restart, only fully-written staging files are
reused; tile-years with no staging file are scored again from scratch.

**Phase 2 — per-tile aggregation:**
All staging parquets for one tile are lazy-scanned, decay-weighted aggregation is
applied across years, scores are converted to uint8 [0–100], and the final parquet is
written. Staging files are deleted after the final write.

---

## Scale numbers

| | |
|---|---|
| Total pixels | 1,641,266,665 |
| S2 tiles covering catchment | 30 |
| Pixels per tile (approx.) | ~55M (edge tiles) – 100M (full tiles) |
| Years of imagery (2020–2025) | 6 |
| Total (tile, year) pairs | ~180 |

Quaids (the validation location) has ~1.5M pixels across 2 tiles — roughly 1/50th of a
single full catchment tile. Streaming to disk is essential: at 100M pixels per tile,
holding all scored rows in RAM during a single tile-year pass would require ~8 GB just
for the output arrays.

---

## Measured throughput (RTX 5060 Ti, v10 model, mixed S2+S1 mode)

| Stage | Time at 100M pixels |
|---|---|
| Pre-passes (S2 z-score + S1 z-score, parallel) | ~40–120 s |
| Scoring @ 2,011 px/s (mixed mode) | ~49,700 s = **~14 hours** |
| Phase 2 aggregation (6 years, lazy scan) | ~10–20 min |
| **Per tile total (1 year)** | **~14 hours** |
| **Full catchment (30 tiles × 6 years, sequential)** | **~100 days** |

The 14-hour-per-tile-year number makes clear that a single GPU on this hardware
cannot score the full catchment in operationally acceptable time. Options:

1. **Cloud GPU instance** — an A100 is roughly 5–10× faster than the RTX 5060 Ti on
   this workload (~1.5–3 hours/tile-year → ~27–54 days for 180 pairs, 1 GPU sequential).
2. **Parallel tile workers** with multiple GPUs or GPU instances.
3. **Reduce model complexity** — shorter MAX_SEQ_LEN (P3 from SCORE-PERF.md), which
   is the largest lever available. Halving T from 256→128 gives ~4× FLOP reduction.

For the Mitchell River pilot run (17 tile-years, 1 year):

| GPU | px/s estimate | Time (17 tile-years, ~97M px avg) |
|---|---|---|
| RTX 5060 Ti (z640) | 2,011 | ~230 hours |
| A10G (AWS g5) | ~10,000 | ~46 hours |
| A100 (AWS p4) | ~20,000 | ~23 hours |

These are extrapolated from the synthetic benchmark. Real tile times vary with
observation density (denser = closer to MAX_SEQ_LEN = more compute).

---

## Memory budget (streaming pipeline)

The streaming writer thread eliminates the in-memory output accumulator. Peak RSS
is now dominated by the **prep queue buffers and model weights**, not the scored output.

**Phase 1 — per tile-year (streaming)**

| Component | Size |
|---|---|
| Torch model weights (v10: d_model=256, 3 layers) | ~0.5 GB |
| Prep queue tensors in flight (raw_q + prep_q, ~16 chunks each) | ~0.5–1 GB |
| out_q buffer (8 × 500K rows) | ~80 MB |
| Pre-pass z-score stats (50–100M pixel dicts) | ~2–4 GB |
| **Peak Phase 1 RSS** | **~3–6 GB** |

Pre-pass stats dominate. At 100M pixels × 13 features × 2 arrays (mean + std), the
`_ZscoreArrays` holds ~10 GB of float32. This is unavoidable — every prep worker needs
per-pixel lookup. Reducing to `float16` halves it; using a sorted array + binary search
instead of a dict further reduces it.

**Phase 2 — per-tile aggregation**

All years are lazy-scanned with `pl.scan_parquet` to avoid materialising the full
multi-year DataFrame before the group-by.

| Component | Size (6 years × 100M pixels) |
|---|---|
| Lazy scan working memory | ~2–4 GB |
| Aggregated result (100M rows, point_id + uint8) | ~1.5 GB |
| **Phase 2 peak** | **~4–6 GB** |

**Safe worker counts on 32 GB:**

| `n_tile_workers` | Notes |
|---|---|
| 1 | Safe, no risk — good for validation |
| 2 | Safe — Phase 1 peak ~6 GB × 2 = 12 GB, Phase 2 adds ~6 GB = 18 GB |
| 4 | Marginal — depends on tile size overlap during Phase 2 |

---

## Disk budget

| Artefact | Size |
|---|---|
| Staging parquets (all tile-years, zstd) | ~68 GB peak |
| Final score parquets (30 tiles, uint8, zstd) | ~6 GB |
| Staging parquets after cleanup | deleted |

In sequential mode (`n_tile_workers=1`) each tile's staging files are cleaned up before
moving to the next, so peak staging disk is ~2.3 GB (6 years × ~380 MB each).

---

## Invocation

```bash
# Sequential (safe, lower disk pressure)
python -m tam.pipeline score \
    --checkpoint outputs/models/tam-v10 \
    --location mitchell \
    --years 2025 \
    --out-parquet \
    --n-tile-workers 1

# Parallel (2 tiles at once)
python -m tam.pipeline score \
    --checkpoint outputs/models/tam-v10 \
    --location mitchell \
    --years 2025 \
    --out-parquet \
    --n-tile-workers 2

# Output
# outputs/mitchell/tam-v10/2025/{tile_id}.scores.parquet
# Schema: point_id (string), prob_tam (uint8, 0–100)
```

Crash recovery: re-running the same command resumes from the last completed tile-year.
Staging files are written atomically (`.tmp.parquet` → rename) so a mid-write crash
leaves no partial file to confuse a restart.

---

## Next step — GeoTIFF assembly

The 30 tile parquets feed a rasterisation step (not yet implemented) that burns
`prob_tam` uint8 values into a single cloud-optimised GeoTIFF. Expected output
size: ~400–700 MB (Deflate-compressed uint8, tiled 512×512, with overviews).
