# V9-SPECTRAL — Model Plan

## Context

V9-SPECTRAL is the S2-only leg of a three-way comparison planned in `docs/S2-AGAIN.md`:

1. **S2-only (this model)** — confirm phenological signal is discriminating before adding complexity
2. **S1-only (V8)** — reference point at val AUC 0.989 on Etna holdout
3. **Joint S1+S2** — interleaved sequence; assess whether combined modality beats either alone

All three use identical label sets on sites with both S1 and S2 coverage. V9 trains on the same `v8_roper` region set so the comparison is unconfounded by site distribution.

---

## Feature set

### Time-series bands — n_bands = 11

| Feature | Notes |
|---|---|
| B02, B03, B04, B05, B07, B08, B8A, B11, B12 | S2 L2A surface reflectance; B06 and B10 excluded |
| NDVI | (B08 − B04) / (B08 + B04); z-scored independently |
| NDWI | (B03 − B08) / (B03 + B08); z-scored independently |

EVI excluded — its additive denominator constant behaves differently under z-score normalisation and adds little over NDVI here.

### Normalisation

**Pixel z-score** (`pixel_zscore=True`): each observation normalised by the pixel's own temporal mean/std across the annual window. Identical to V8. Removes absolute reflectance offsets between tiles and sites; preserves phenological curve shape, which is the discriminating signal for Parkinsonia.

### Global features — sweep axis

The named S2 globals from prior models (`nir_cv`, `rec_p`, `peak_doy`, `peak_doy_cv`, `dry_ndvi`) are substantially redundant with band-level summary statistics: `dry_ndvi` ≈ NDVI p5, `rec_p` ≈ NDVI p95−p5, `peak_doy` is derivable from the z-scored curve shape. Rather than carry partially-redundant named scalars, the sweep uses **band summaries** as the ablation axis against a clean `n_global_features=0` baseline.

**`use_band_summaries=False`** — no global inputs; `n_global_features=0`.

**`use_band_summaries=True`** — appends [p5, p95, std] computed across the annual observation stack for each of the 11 bands (33 scalars total) as additional global inputs alongside the z-scored sequence. Gives the model explicit access to the phenological envelope per S2-AGAIN.md without a separate named-globals layer. Key question: do these add value over the sequence alone?

---

## Augmentation

Identical to V8:

| Augmentation | Value |
|---|---|
| `doy_phase_shift` | True — full-year circular wraparound; forces shape learning over calendar learning |
| `band_noise_std` | 0.03 — Gaussian in normalised space, independent per observation |
| `obs_dropout_min` | 4 — subsample to Uniform(4, n) observations per window |
| `doy_density_norm` | True — weight mean pool by inverse DOY frequency |

---

## Architecture

Identical Transformer backbone to V8 settled config:

| Param | Value |
|---|---|
| `d_model` | 128 |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `dropout` | 0.5 |
| `weight_decay` | 0.1 |
| `n_epochs` | 60 |
| `patience` | 15 |
| `batch_size` | 1024 |

`n_bands=11`. `n_global_features=0` (baseline) or `33` (band summaries ablation).

---

## Sweep

Etna held out as site holdout throughout — identical protocol to V8. Four runs over two axes:

| run_id | use_band_summaries | lr | expected |
|---|---|---|---|
| `s2_nosumm_1e-5` | False | 1e-5 | likely fails to converge (V8 precedent) |
| `s2_nosumm_5e-5` | False | 5e-5 | **primary baseline** |
| `s2_summ_1e-5` | True | 1e-5 | likely fails |
| `s2_summ_5e-5` | True | 5e-5 | band-summaries ablation |

Script: `sweeps/sweep_v9_spectral.py`

---

## Training data

Same as `v8_roper`:

| Site | Role | n_presence | n_absence |
|---|---|---|---|
| Norman Road | Train | 5,189 | 6,021 |
| Cloncurry | Train (absence only) | 0 | 2,228 |
| Landsend | Train | 3,954 | 5,147 |
| Lake Mueller | Train | 1,019 | 1,835 |
| Corfield | Train | 2,109 | 4,710 |
| Roper | Train | 937 | 3,635 |
| **Etna** | **Holdout** | **3,193** | **8,319** |

`use_s1="none"` — S1 columns ignored; no snapping step. All other filters (SCL, scl_purity ≥ 0.5, noise filter) unchanged.

---

## Key comparison

| Model | Modality | Val AUC (Etna) |
|---|---|---|
| V8 `v8_roper` | S1-only | 0.989 |
| V9-SPECTRAL `s2_nosumm_5e-5` | S2-only | TBD |
| V9-SPECTRAL `s2_summ_5e-5` | S2-only + band summaries | TBD |

S2-only is expected somewhat lower than V8 due to wet-season cloud gaps. If S2-only exceeds or matches S1-only, the case for the joint model weakens; if it underperforms, joint fusion is well-motivated.

---

## Implementation checklist

- [ ] `tam/core/config.py` — add `use_band_summaries: bool = False`
- [ ] `tam/core/dataset.py` — compute p5/p95/std per band and assemble global feature vector when `use_band_summaries=True`
- [ ] `tam/experiments/v9_spectral.py` — experiment definition
- [ ] `sweeps/sweep_v9_spectral.py` — sweep script
