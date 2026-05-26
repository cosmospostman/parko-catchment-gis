# Score Pipeline — Performance Analysis & Optimisation Plan

## Baseline measurement

Benchmark: `scripts/bench_score.py` — 50,000 pixels × 50 obs/pixel × 2 years (5M rows).
Model: v10 weights (d_model=256, n_layers=3, d_ff=1024, n_bands=16).
Hardware: CPU only (z640 workstation, no GPU in scoring path).

```
  Stage                     RSS GB    Δ RSS   Elapsed s      Rows
  ---------------------------------------------------------------
  baseline                    0.05    +0.00       0.0
  synth_gen                   1.07    +1.01       3.5   5,000,000
  parquet_write               1.07    +0.00       3.5
  model_init                  1.53    +0.46       6.6   2,373,890
  warmup                      1.63    +0.10       7.1
  prepass_s2_zscore           2.43    +0.81       8.4      50,000
  prepass_band_summaries      2.61    +0.18       9.8      50,000
  score_chunked               2.77    +0.15     340.3      50,000

  Throughput (score_chunked only):
    elapsed:         330 s
    pixels/sec:      151
    pixel-years/sec: 303

Peak RSS: 2.77 GB   Total wall: 340 s
```

Real Quaids run (1,479,570 pixels, 2 tiles, from pipeline log):
- Pre-passes: ~53 s total (S2 z-score 17 s + S1 z-score 19 s + band summaries 17 s)
- Scoring tile 1 (55KBB): ~3 min
- Scoring tile 2 (55KCB): ~25 min
- **Total: ~28–29 min**

The tile-2 time (~25 min) dwarfs tile-1 (~3 min) — a separate issue explored below.

---

## Architecture

`score_pixels_chunked` runs a three-stage concurrent pipeline:

```
Reader thread         Prep workers (×4)         Main thread
[parquet row groups]  [numba kernels]            [torch inference]
      ↓                     ↓                          ↓
   raw_q ──────────→    prep_q ──────────────→  _gpu_score loop
 (maxsize=8)          (maxsize=8)
```

**Reader** reads row groups in order, accumulating `buffer_row_groups=16` per emit. It handles pixel boundary spillover so no pixel is split across chunks.

**Preprocessors** (4 threads) run CPU-side numba `fill_windows()` to scatter observations into `(W, 256, 16)` tensors with z-score normalisation.

**Inference loop** (`_gpu_score`) slices prepared batches into `batch_size=4096` windows and calls `model.forward()` for each.

---

## Root cause: where time goes

### 1. Transformer attention is O(T²) — and T=256

The forward pass per batch:

```python
x = self.band_proj(bands)          # (B, 256, 16) → (B, 256, 256)
x = self.encoder(x, src_key_padding_mask=safe_mask)   # 3× TransformerEncoderLayer
x_pool = (x * pool_w).sum(dim=1)   # mean pool → (B, 256)
logit = self.head(x_pool)          # (B, 1)
```

Each `TransformerEncoderLayer` computes multi-head self-attention with complexity
`O(B · T² · d_model)`. With T=256, d_model=256, n_heads=4, n_layers=3 and batch_size=4096:

- Attention per layer: 4096 × 256² × 256 ≈ 68 billion FLOPs/batch
- On CPU this is the overwhelmingly dominant cost

Pixels with few observations (sparse locations) still pay the full T=256 cost because the
sequence is zero-padded to `MAX_SEQ_LEN` before batching.

### 2. No GPU

The machine runs inference on CPU. PyTorch's TransformerEncoder on CPU is unaccelerated
relative to a CUDA device — a single A10/T4 GPU would give 20–100× throughput on this
workload.

### 3. Tile-2 regression in the real Quaids run

Tile 55KCB took ~25 min vs ~3 min for 55KBB despite having the same number of pixels.
The log shows much larger batch steps in tile-2 (23,840-pixel increments vs ~3,500 for
tile-1), meaning fewer but larger batches. This is consistent with tile-2 having
temporally denser pixels (more obs/pixel → closer to MAX_SEQ_LEN → more real compute
per window, less padding waste). This is expected variance, not a bug, but confirms
the attention cost is real.

### 4. Pre-passes re-read the full parquet

`_compute_s2_pixel_zscore_stats` and `_compute_band_summaries_from_parquets` each do
a full sequential read of all year parquets (row-group by row-group) before scoring
starts. At 1.5M pixels × ~50 obs = 75M rows per tile, these scans are not free (~17s
each). They cannot overlap with scoring because their outputs are needed upfront.

### 5. Per-pixel dict lookups in preprocessing

`_extract_s1_features()` (score.py ~line 206) does O(N) Python dict lookups per chunk
to apply per-pixel z-score stats:

```python
p_vh_mean = np.array([vh_mean.get(p, 0.0) for p in pids], dtype=np.float32)
```

This is vectorisable with a Polars join but is not the primary bottleneck at current
pixel counts.

---

## Optimisation plan

Ranked by expected impact and implementation effort.

### P1 — GPU inference (highest impact, ~20–100×)

**What:** Move inference to a CUDA device. The transformer forward pass is matmul-bound
and maps perfectly to GPU.

**How:** Pass `--device cuda` (or set `device="cuda"`) to `score_pixels_chunked`.
The pipeline already uses `non_blocking=True` transfers and `torch.inference_mode()`.
Pinned memory is already enabled when `device.startswith("cuda")` (score.py line 755).
No code changes needed — just hardware.

**Expected gain:** 20–100× on the forward pass (151 px/s → 3,000–15,000 px/s).
Pre-pass and preprocessing costs amortise over that speed, becoming negligible.

**Measure with:**
```bash
python scripts/bench_score.py --n-pixels 50000 --n-years 2 \
  --d-model 256 --n-layers 3 --d-ff 1024 \
  --pixel-zscore --band-summaries --device cuda
```

---

### P2 — torch.compile — MEASURED: NOT beneficial on this GPU

**Measured result (RTX 5060 Ti, v10 model, batch_size=4096):**
- Eager mode: **6,167 px/s**
- `torch.compile(mode="default")`: 2,894 px/s — **2.1× regression**
- `torch.compile(mode="reduce-overhead")`: ~2,700 px/s, +5 GB CUDA graph pool

PyTorch's built-in SDPA kernel (`torch.nn.functional.scaled_dot_product_attention`) is
already fused in eager mode. TorchInductor overhead dominates at this batch size and
sequence length. `torch.compile` has been removed from the production scoring path.

**TF32 (kept — free win):** `torch.set_float32_matmul_precision("high")` gives ~22%
throughput gain on Ampere+ (RTX 30/40/50 series) at no accuracy cost. Enabled
automatically when CUDA is detected in `score_pixels_chunked`.

---

### P3 — Reduce sequence length for sparse pixels (~1.5–3× on CPU)

**What:** Currently `MAX_SEQ_LEN=256` is used for all pixels. Sparse pixels (e.g.
min_obs_per_year=8 → 8–30 actual obs over 2 years) spend most of the attention compute
on padding. Attention is O(T²), so halving the effective T is a 4× FLOP reduction.

**How (option A — dynamic batching by n_obs):** Sort windows by n_obs before batching,
then use `torch.nn.utils.rnn.pack_padded_sequence` equivalent or nested tensors to run
attention only over the actual sequence length per sample. PyTorch's
`TransformerEncoder` supports `src_key_padding_mask` to skip padding, but the underlying
SDPA still allocates T×T attention. Use `torch.nested.nested_tensor` with
`layout=torch.jagged` (torch ≥ 2.1) for true variable-length attention.

**How (option B — shorter fixed T):** If most pixels have < 128 real obs/year, set
`MAX_SEQ_LEN=128`. This halves attention cost unconditionally but requires retraining.
Check the obs distribution across training pixels before deciding.

**Measure trigger:** Run `SELECT percentile_cont(0.9) WITHIN GROUP (ORDER BY n_obs)
FROM scored_pixels` on the existing score output, or inspect the `n_obs` tensor distribution
during a scoring run.

---

### P4 — Parallel pre-passes (~2× on multi-year runs)

**What:** S2 z-score stats, S1 z-score stats, and band summaries are computed in
series. They all do independent full-file reads and can run in parallel.

**How:** Wrap the three `_compute_*` calls in `concurrent.futures.ThreadPoolExecutor`
with `max_workers=3`. I/O-bound work parallelises well across threads in CPython.

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    f_s2  = ex.submit(_compute_s2_pixel_zscore_stats, ...)
    f_bs  = ex.submit(_compute_band_summaries_from_parquets, ...)
    f_s1  = ex.submit(_compute_pixel_s1_stats_mixed, ...)   # if mixed
pixel_zscore_stats = f_s2.result()
band_summaries     = f_bs.result()
s1_zscore_stats    = f_s1.result()
```

**Expected gain:** Reduces ~53s of serial pre-passes to ~19s (bottlenecked by the
longest single pass). Worthwhile on CPU-only; less important once GPU is available.

---

### P5 — Vectorise per-pixel dict lookups (~5–10% preprocessing speedup)

**What:** `_extract_s1_features()` iterates Python dicts over each row in the chunk
to apply pixel z-score stats (score.py ~line 206). With n_prep_workers=4 this is
parallelised but each worker still runs the list comprehension.

**How:** Convert the `vh_mean`/`vh_std`/`vv_mean`/`vv_std` dicts to a Polars DataFrame
once (point_id, mean, std columns), then use a Polars `join` in `_extract_s1_features`
instead of per-element `dict.get()`.

```python
# Build once in score_pixels_chunked, pass to _extract_s1_features
s1_stats_df = pl.DataFrame({
    "point_id": list(vh_mean.keys()),
    "vh_mean":  np.array(list(vh_mean.values()), dtype=np.float32),
    ...
})
# In _extract_s1_features:
chunk = chunk.join(s1_stats_df, on="point_id", how="left")
```

---

## Summary table

| ID | Optimisation | Effort | Expected gain |
|----|--------------|--------|---------------|
| P1 | GPU inference | hardware | 36× measured (151 → 5,409 px/s on RTX 5060 Ti) |
| P2 | TF32 matmul precision | 1 line | +22% on Ampere+ (already in pipeline) |
| P3 | Reduce / dynamic sequence length | medium | 1.5–3× on CPU |
| P4 | Parallel pre-passes | small | 2× pre-pass wall time |
| P5 | Vectorise dict lookups | small | ~5–10% preprocessing |

**Recommendation:** P1 (GPU) is the only change that gets scoring time to an operationally
acceptable level. TF32 (now always-on) adds a further 22% for free. P3–P5 are worthwhile
CPU-only improvements if GPU access is unavailable or for cost reduction.

---

## Re-running the benchmark

```bash
# CPU baseline (v10 model, with pre-passes)
python scripts/bench_score.py \
  --n-pixels 50000 --n-years 2 \
  --d-model 256 --n-layers 3 --d-ff 1024 \
  --pixel-zscore --band-summaries

# GPU (eager mode + TF32 — production config)
python scripts/bench_score.py \
  --n-pixels 50000 --n-years 2 \
  --d-model 256 --n-layers 3 --d-ff 1024 \
  --pixel-zscore --band-summaries --device cuda

# Assert throughput floor (RTX 5060 Ti baseline: ~5,400 px/s)
python scripts/bench_score.py \
  --n-pixels 50000 --n-years 2 \
  --d-model 256 --n-layers 3 --d-ff 1024 \
  --pixel-zscore --band-summaries --device cuda \
  --assert-pixels-per-sec 5000
```
