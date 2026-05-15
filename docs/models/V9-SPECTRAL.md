# V9-SPECTRAL

S2-only TAM. Part of a three-way comparison (S2-only / S1-only / joint S1+S2).

**Current best val AUC: 0.7846** (`no_phase_shift`, sweep 2026-05-14)

---

## Feature set

11 bands: B02 B03 B04 B05 B07 B08 B8A B11 B12 + NDVI + NDWI.  
B06 and EVI excluded. Pixel z-score normalisation (identical to V8).

---

## Architecture

| Param | Value |
|---|---|
| `d_model` | 128 |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `dropout` | 0.5 |
| `weight_decay` | 0.1 |
| `lr` | 5e-5 |
| `n_epochs` | 60 |
| `patience` | 15 |
| `batch_size` | 1024 |
| `max_seq_len` | 64 |

---

## Training sites

Sites: Norman Road, Cloncurry (absence only), Etna Creek (train split), Landsend, Lake Mueller, Corfield, Roper River, Barcoorah, Hughenden, Frenchs.  
Val holdout: Etna Creek (presence + absence), Landsend, Barcoorah, Hughenden, Frenchs.

Frenchs added to address bare-soil confusion and restore monsoonal site diversity. Bare-soil absence regions (`frenchs_absence_bare_soil_2/3`) included after smoke-testing revealed bare/sparse ground was the primary false-positive class.

---

## Training history

### Initial — V9 created as S2-only experiment (`1c7bfe5`)

V9 launched as the S2-only leg of a planned three-way comparison (S2-only / S1-only / joint). Starting point: same `v8_roper` site set, same Transformer backbone, S1 columns disabled (`use_s1=False`). Feature set fixed at 11 bands (B06 and EVI dropped vs V8). `use_band_summaries` added as an ablation axis — appends [p5, p95, std] across the annual stack for each band as additional global inputs.

Scoring path refactored to support non-default feature column sets: V9's layout (no B06, no EVI) bypassed the Numba fast path used for V8, now routed through a Pandas fallback with per-pixel z-score applied in the score loop.

### Site expansion — Frenchs + Barcoorah added; val set restructured

Frenchs Creek (Cape York monsoonal savanna-riparian) added to training and val sets to address bare-soil confusion and restore monsoonal diversity. Bare-soil absence regions (`frenchs_absence_bare_soil_2/3`) included explicitly — same fix that improved V2. Barcoorah and Hughenden added to both train and val splits. Frenchs presence/absence regions split between train and val so monsoonal generalisation is assessed independently of the Etna holdout.

### OOM fix — band-summary computation refactored (`1f7d448`)

Training was OOM at startup on the expanded site set. Root cause: band-summary computation copied the full `pixel_df` into a thread pool (one copy per worker). Replaced with a single-pass groupby on a minimal S2-only column slice (`point_id` + feature columns only, view not copy). Also added per-step RSS logging to catch future memory regressions.

### `max_seq_len` reduced 128 → 64 (`afb4ee9`)

Sequence length cap reduced from 128 to 64 for better throughput on the M4000. Multi-year training stacks have ~330–650 obs/pixel; obs-dropout already subsamples these heavily, so reducing the hard cap from 128 to 64 costs little in sequence diversity but halves the memory footprint per batch.

### 2026-05-13 — per-year VH filter (+0.08 AUC)

**Val AUC: 0.755**

Previous filter computed a multi-year mean VH dry-season backscatter and dropped presence pixels below −18 dB. Multi-year averaging dragged expanding-front pixels below threshold (e.g. Corfield: 70/2,740 presence pixels survived). New filter applies the −18 dB threshold per-year independently, recovering those pixels in years they were woody. Total noise removed: ~51k pixel-years (10.5% of training data).

### 2026-05-13/14 — ablation sweep (null result on most axes)

Eight runs isolating one variable against the 0.755 baseline. Substantial run-to-run variance meant the baseline itself was not robustly reproducible at 0.755.

| run_id | change | val AUC |
|---|---|---|
| `vh_floor_m18` | baseline re-run | 0.701 |
| `no_phase_shift` | doy_phase_shift=False | 0.749 |
| `vh_floor_m17` | presence_min_vh_dry_db=−17.0 | 0.758 |
| `lr_1e4` | lr=1e-4 | 0.786 |
| `two_heads` | n_heads=2 | 0.737 |
| `dropout_03` | dropout=0.3 | 0.685 |
| `no_obs_dropout` | obs_dropout_min=0 | 0.654 |
| `vh_floor_m19` | presence_min_vh_dry_db=−19.0 | 0.645 |

### 2026-05-14 — phase-shift sweep (standout winner)

Sweep varying `doy_phase_shift`, `lr`, and `band_noise_std` against the expanded training set (Frenchs added, relaxed noise filter).

| run_id | doy_phase_shift | lr | band_noise_std | val AUC |
|---|---|---|---|---|
| baseline | True | 5e-05 | 0.05 | 0.6749 |
| **no_phase_shift** | **False** | 5e-05 | 0.05 | **0.7846** |
| lr_1e4 | True | 1e-04 | 0.05 | 0.6633 |
| lr_2e4 | True | 2e-04 | 0.05 | 0.6795 |

`doy_phase_shift=False` was the clear winner (+0.110 AUC over baseline). The DOY circular wraparound augmentation actively hurts S2 discrimination — likely because it destroys the absolute seasonal timing that is part of the Parkinsonia phenological signal. **`doy_phase_shift=False` is now the default in `v9_spectral.py`.**

---

## Known failure modes

Smoke-testing on Longreach showed the model assigning highest scores to bare/sparse ground. Two confirmed root causes:

1. **Obs-count mismatch** — training used multi-year stacks (~330–650 obs/pixel); inference parquets are single-year (~52 obs/pixel). Sparse wet-season curves look nothing like the training distribution.
2. **No bare-soil absence training examples** — fixed by adding `frenchs_absence_bare_soil_2/3`. V2 precedent: NDVI contribution jumped from −0.001 to +0.028 after this fix.
