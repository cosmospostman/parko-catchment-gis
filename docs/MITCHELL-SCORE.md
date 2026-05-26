# Mitchell River Catchment — Scoring Feasibility

## Pixel count investigation

The first estimate for Mitchell River pixel count (~30M) was wrong by roughly 24×. That figure
likely came from confusing a row count in a training-region parquet with the full catchment
extent.

The correct figure was derived from the catchment polygon at
`data/catchments/mitchell_river.geojson` using shapely + pyproj (EPSG:3577 equal-area):

```
Area:  71,622 km²
Pixels at 10m:  716M  (7.16 × 10⁸)
```

For comparison, the two Quaids benchmark tiles (55KBB and 55KCB) each cover roughly 1.1M pixels
from the training chip geometry — about 1/650th of the full catchment.

---

## Throughput baselines

Benchmarked on an RTX 5060 Ti (sm_12.0, Blackwell) at T=64 padded:

```
bench_score.py --n-pixels 50 000 --n-years 2 --d-model 256 --n-layers 3 --d-ff 1024
→ 1,852 px/s
```

V10 is now trained and scored at **T=128** (Kowanyama reaches 106 obs/px/yr at single-year
cadence; T=64 truncated the tail). Attention cost is O(T²), so T=64→128 is a theoretical 4×
slowdown on the attention kernels. In practice the bound is memory-bandwidth vs. compute
dependent; 2–3× is a reasonable conservative estimate.

The RTX 5060 Ti is Blackwell (sm_12.0). flash-attn v4 beta does not yet support sm_12x, so
varlen attention is unavailable locally — the padded `forward()` path is always used. The
full 4× attention regression applies until flash-attn adds sm_12x support.

An A10G (sm_8.6, AWS `g5`) is fully supported by flash-attn v4. Varlen eliminates padding
overhead entirely — at p50 ≈ 48 obs/px for Quaids, that removes ~60% of the attention work.
Estimated A10G throughput at T=128 varlen: **~4,300 px/s**.

---

## Wall time estimates

| Location       | Pixels  | 5060 Ti padded T=128 | A10G varlen T=128 |
|----------------|---------|----------------------|-------------------|
| Quaids 55KBB   | ~1.1M   | ~40 min              | ~4 min            |
| Quaids 55KCB   | ~1.1M   | ~40 min              | ~4 min            |
| Mitchell full  | ~716M   | ~430 hrs (18 days)   | ~46 hrs           |

Notes:
- 5060 Ti estimate uses 463 px/s (= 1,852 × 0.25, i.e. full 4× T² penalty). If the
  real regression is 2–3×, the local estimate improves to 145–215 hrs.
- The 716M pixel figure is unique pixels (one pass per scored year). A 6-year Mitchell run
  scores the same pixels 6 times → 6 × 716M row-years of inference.
- The 46-hour A10G estimate is per year. A 6-year run would be ~275 hrs unless parallelised
  across multiple instances or years.

---

## Implications for production scoring

The full Mitchell catchment is not feasible on local hardware in a single run. Recommended
approach:

1. **Score year-by-year** using the streaming/resumable pipeline (plan:
   `how-will-the-score-functional-wilkinson.md`). Each tile-year writes an atomic staging
   parquet; crash recovery restarts only the failed tile-year.
2. **Use cloud GPU** (A10G / `g5.xlarge`) for production runs. See `docs/EBS-SETUP.md` for
   the EBS-based checkpoint and data transfer workflow.
3. **Benchmark T=128 padded locally first** against a single Quaids tile to validate the
   smart-sampling inference path before committing to cloud spend.

---

## Cascade gate: cheap pre-filter before full inference

At 716M pixels the A10G wall time is ~46 hrs per year. A two-stage cascade could reduce
the Stage 2 (full T=128) pixel count substantially if most of the catchment is
unambiguously non-Parkinsonia.

### Approach: same model, short sequence

Run V10 at T=8 or T=16 as a Stage 1 gate — same checkpoint, same weights, no retraining.
Pixels scoring below a threshold are discarded without running the full T=128 pass. Stage 1
attention cost is O(T²), so T=8 is 256× cheaper than T=128 per pixel.

### Investigation result (2026-05-26)

`scripts/bench_cascade.py` evaluated the T=64 v10 checkpoint on the held-out val set
(24,366 pixel-years; 6,980 presence / 17,386 absence):

| T gate | Threshold | Recall@gate | Absence discarded | Cascade speedup |
|--------|-----------|-------------|-------------------|-----------------|
| T=8    | 0.02      | 39.8%       | 78.5%             | 3.0×            |
| T=8    | 0.05      | 35.4%       | 81.5%             | 3.4×            |
| T=16   | 0.02      | 44.7%       | 69.7%             | 2.1×            |
| T=16   | 0.05      | 41.9%       | 73.0%             | 2.3×            |

Score correlation between T=8 gate and full T=64: **r=0.39**. Presence pixel p50 score at
T=8: **0.005** — the model pushes Parkinsonia pixels to near-zero at short sequence lengths.

**Conclusion: the current checkpoint is not usable as a cascade gate.** Recall of ~40% would
discard the majority of real Parkinsonia pixels. The model was trained exclusively on
full-length sequences (T=64) and has no representation for what a sparse 8-obs window looks
like — it defaults to near-zero confidence, which is exactly wrong for a high-recall gate.

### Fix: gate-augmented training

The cascade gate requires a model that is discriminative at both T=128 (full inference) and
T=8/16 (gate). The mechanism is a training augmentation — no architectural change:

**In `TAMDataset.__getitem__`**, after the existing obs-dropout block, with probability
`p_gate` (e.g. 0.3) subsample the sequence down to `T_gate` (e.g. 8 or 16) observations
using farthest-point DOY sampling. The loss is computed identically on the short view.

Key design decisions:

- **Augmentation, not split.** Every pixel is seen at full length every epoch. The short-window
  view is an additional augmentation that fires `p_gate` fraction of the time — not a 70/30
  held-out partition. Every pixel contributes to both full-length and gate-length learning.

- **Farthest-point DOY sampling, not truncation.** The first 8 obs of a year may all fall
  in the dry season before the wet-season green-up. Farthest-point sampling spreads the 8
  obs across the full annual arc, giving the model the best available seasonal signal in the
  budget. `subsample_obs_indices` (renamed from `subsample_s1_indices`) handles this.

- **`n_obs` scalar carries the sparsity signal.** The model already receives `n_obs =
  len(sequence) / max_seq_len` as a global scalar. At T=8 this is 0.063; at T=128 it is 1.0.
  The model can learn to calibrate confidence differently in each regime via this feature.

- **No threshold tuning until after retraining.** The optimal gate threshold depends on the
  retrained model's calibration at short T. The 0.02–0.05 range used above was exploratory.

### Expected speedup after gate-augmented retraining

If a retrained model achieves recall@gate ≥ 0.99 on presence pixels at T=8 while discarding
70–80% of absence pixels (plausible given the model will have seen short-window Parkinsonia
during training), the cascade speedup on Mitchell River would be:

```
Stage 1 cost:  716M × (8/128) = 44.7M T=128-equivalent pixel-passes
Stage 2 cost:  716M × 0.20 survivors × 1.0 = 143M T=128-equivalent pixel-passes
Total:         ~188M vs 716M  →  ~3.8× speedup
A10G estimate: 46 hrs / 3.8 ≈ 12 hrs per year
```

The 20% survivor fraction is a conservative assumption; if the catchment is 90%+ clearly
non-Parkinsonia habitat the real survivor rate could be lower and the speedup larger.

---

## Caveats

- The 1.1M pixel per-tile Quaids figure is derived from training chip geometry, not the full
  Sentinel-2 tile extent. Scored tile extents (full 100×100 km² tile minus water/cloud masks)
  will be larger.
- Mitchell bounding-box pixel count would be ~870M; the polygon-clipped figure (716M) is
  ~18% smaller due to coastal and off-catchment exclusion.
- Multi-year Mitchell scoring (e.g. 2019–2024) at 10m is a ~4.3 billion row-year job. An
  aggregated inference strategy (coarser resolution pre-filter → full 10m only in candidate
  areas) could reduce this by 1–2 orders of magnitude.
