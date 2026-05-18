# V10-SAR

S1-only TAM. Third leg of the three-way comparison (S2-only / S1-only / joint S1+S2).
Extends V8 (`v8_roper`) with V9's expanded site set and applicable training improvements.

---

## Relationship to V8 and V9

V8 (`v8_roper`) is the established S1-only baseline (val AUC 0.989 on Etna holdout).
V9 (`v9_spectral`) is the S2-only model trained on a significantly expanded site set with
several training improvements discovered during sweeps. V10 inherits V9's site set and
applies the V9 improvements that are relevant to S1, while excluding those that are S2-specific.

V8 training regions are a strict subset of V9 — no V8 regions are lost. The new regions
added by V9 (Frenchs, Hughenden, Burdekin, Maria Downs, Rupert Creek, expanded Landsend/Etna/
Corfield/Lake Mueller) are all included in V10.

---

## Changes from V8

### Applied from V9

**`doy_phase_shift=False`**
V8 uses `doy_phase_shift=True`. V9's sweep showed disabling it gave +0.110 AUC for S2, because
the augmentation destroys absolute seasonal timing that is part of the Parkinsonia phenological
signal. The argument is at least as strong for S1 — V8's attention analysis showed the model
attending to specific wet/dry season windows per site, which phase-shift augmentation would
actively corrupt. Treat as a primary sweep axis in the first V10 run.

**Per-year VH noise filter**
V8 computed a multi-year mean VH dry-season backscatter and dropped presence pixels below −18 dB.
Multi-year averaging dragged expanding-front pixels below threshold (Corfield: 70/2,740 presence
pixels survived). V9 switched to applying the threshold per-year independently, recovering pixels
in years they were woody. This fix is entirely S1-based and directly applicable. Apply the same
per-year logic (`presence_min_vh_dry_db` per year rather than on the multi-year mean).

**`max_seq_len` 128 → 64**
S1 sequences are similar in length to S2 (multi-year stacks, ~330–650 obs/pixel before
obs-dropout). The same memory/throughput argument applies. Obs-dropout already subsamples
heavily, so the hard cap costs little in sequence diversity. Apply directly.

**`band_noise_std=0.05`**
V8 has `band_noise_std=0.0`. S1 backscatter has higher per-observation noise than S2 reflectance,
so regularisation via band noise may be more beneficial here. Apply at the same level as V9 (0.05)
and treat as a sweep axis if needed.

**Expanded site set (V9 regions)**
All V9 train and val regions are included. See site list below.

**CVaR val metric**
V8's 0.989 AUC was on a single Etna holdout. The new multi-site val set enables the CVaR
(bottom-quartile site AUC) metric introduced in commit `19b6685`. This gives a more honest
picture of generalisation — a high Etna-only score may flatter the model.

### Excluded from V9 (S2-specific)

**`use_band_summaries=True`**
The band-summary computation explicitly filters `source == "S2"` rows. In an S1-only run
there are no S2 rows; the result would be an empty or all-NaN global feature table. Do not apply.

**`use_s1=False` / S2 feature columns**
V10 uses `use_s1="s1_only"` and `feature_cols=S1_FEATURE_COLS` (VH, VV, VH−VV, RVI).

**Frenchs bare-soil fix framing**
The bare-soil false-positive problem described in V9 was an S2 spectral issue. S1 has different
failure modes (low-VH wet areas, sparse ground). Frenchs bare-soil absence regions are still
included as training data — they are useful — but the motivation differs.

---

## Site coverage notes

### Multi-year sites (per-year VH filter fully active)
Frenchs, Maria Downs, Norman Road, Cloncurry, Landsend, Lake Mueller, Corfield, Roper River,
Etna Creek — all have 3–6 years of data. Per-year VH filter applies normally.

### Single-year sites (per-year VH filter is a single-year passthrough)
Hughenden (2024), Burdekin (2024), Rupert Creek (2020) — all configured with one year only.
The per-year filter degenerates to a single-year threshold: no multi-year averaging harm,
but also no cross-year recovery. Acceptable; just note that these sites contribute no
temporal diversity to the sequence model.

### S1 data availability for new sites
S1 pixel data has not yet been fetched for Frenchs, Hughenden, Burdekin, Maria Downs, or
Rupert Creek. This is the primary prerequisite before training can begin. Run the S1 collector
against all new region IDs and verify parquet outputs before launching V10.

---

## Proposed hyperparameters

| Param | V8 value | V10 value | Reason |
|---|---|---|---|
| `d_model` | 128 | 128 | No change |
| `n_layers` | 2 | 2 | No change |
| `n_heads` | 4 | 4 | No change |
| `dropout` | 0.5 | 0.5 | No change |
| `weight_decay` | 0.1 | 0.1 | No change |
| `lr` | 5e-5 | 5e-5 | Settled optimum from V8 sweeps |
| `n_epochs` | 60 | 60 | No change |
| `patience` | 15 | 15 | No change |
| `batch_size` | 1024 | 1024 | No change |
| `max_seq_len` | 128 | 64 | Throughput; obs-dropout makes 128 redundant |
| `doy_phase_shift` | True | False | Primary sweep axis — V9 showed +0.110 for S2 |
| `band_noise_std` | 0.0 | 0.05 | S1 backscatter noise warrants regularisation |
| `obs_dropout_min` | 4 | 4 | No change |
| `doy_density_norm` | True | True | No change |
| `pixel_zscore` | True | True | No change |
| `use_s1` | "s1_only" | "s1_only" | No change |
| `use_band_summaries` | False | False | S2-specific — excluded |
| `presence_min_vh_dry_db` | −21.0 (multi-year mean) | −18.0 per-year | V9 fix — recover expanding-front pixels |

---

## Training regions

### Train

```
# Norman Road
norman_road_presence_1 … norman_road_presence_9
norman_road_absence_1 … norman_road_absence_7 (excl. _6)

# Cloncurry
cloncurry_absence_1 … cloncurry_absence_7

# Etna Creek — train-only regions
etna_presence_2, etna_presence_5–9
etna_absence_6–12

# Landsend — train-only regions
landsend_presence_1–7, landsend_sparse_presence_1–5
landsend_absence_1–3, landsend_absence_grass_1–2, landsend_absence_riverbed_1–3

# Lake Mueller
lake_mueller_presence, lake_mueller_presence_2–4
lake_mueller_absence, lake_mueller_absence_2–6

# Corfield
corfield_presence_1–6, corfield_absence_1–3

# Roper River
roper_presence_1–4, roper_absence_1–3

# Hughenden — train-only regions
hughenden_presence_4, hughenden_absence_3, hughenden_absence_5

# Burdekin — train-only regions
burdekin_presence_1–2, burdekin_absence_1–3

# Maria Downs — train-only regions
maria_downs_presence, maria_downs_presence_2
maria_downs_absence, maria_downs_absence_2

# Rupert Creek — train-only regions
rupert_ck_presence_1–3, rupert_ck_presence_sparse_1
rupert_ck_absence_1–3

# Frenchs — Cape York Peninsula
frenchs_presence_1–4
frenchs_absence_bare_soil_2–3, frenchs_absence_mangrove, frenchs_absence_ocean
frenchs_absence_riparian_woodland, frenchs_absence_riparian
frenchs_absence_4–7, frenchs_absence_water_1–3
```

### Val

```
# Etna Creek
etna_presence_1, etna_presence_3–4
etna_absence_1–5

# Landsend
landsend_presence_8, landsend_absence_4–5

# Hughenden
hughenden_presence_1–3
hughenden_absence_1–2, hughenden_absence_6–7

# Burdekin
burdekin_val_presence_1, burdekin_val_absence_1–2

# Maria Downs
maria_downs_val_presence_1, maria_downs_val_absence_1

# Rupert Creek
rupert_ck_val_presence_1, rupert_ck_val_absence_1

# Frenchs
frenchs_presence_5–6
frenchs_absence_savanna, frenchs_absence_4 (val split), frenchs_absence_8
```

---

## Prerequisites

1. **Fetch S1 data for new sites** — Frenchs, Hughenden, Burdekin, Maria Downs, Rupert Creek
   have no S1 parquets yet. Run the pixel collector against all new region IDs and verify outputs.
2. **Verify per-year VH filter** — confirm the filter is applied per-year (not multi-year mean)
   before the first training run; check the `_apply_presence_filter` path in `train.py`.
3. Create `tam/experiments/v10_sar.py` mirroring `v9_spectral.py` structure but with
   `use_s1="s1_only"`, `feature_cols=S1_FEATURE_COLS`, and the hyperparams above.

---

## Planned sweeps

### Sweep 1 — phase shift and noise
Primary axes: `doy_phase_shift` (True/False) × `band_noise_std` (0.0/0.05) against the
full V10 site set. Baseline: V8 defaults with V10 site set and `max_seq_len=64`.

| run_id | doy_phase_shift | band_noise_std |
|---|---|---|
| `baseline` | True | 0.0 |
| `no_phase_shift` | False | 0.0 |
| `noise_only` | True | 0.05 |
| `no_shift_noise` | False | 0.05 |

---

## Training history

_(To be filled in as runs complete.)_
