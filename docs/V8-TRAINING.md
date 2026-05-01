# V8 Training — Status and Direction

## Where We Are

V8 is the first generation of TAM trained on Sentinel-1 only (no S2 bands). The goal is a model that discriminates Parkinsonia using SAR backscatter temporal signatures, which would complement S2-based models by operating through cloud cover and adding an independent signal.

### Best result so far: `sweep_zscore_etna_landsend` — 0.955 val AUC (etna holdout)

| run | lr | d_model | n_layers | val_auc |
|---|---|---|---|---|
| lr1e-05_dm64 | 1e-5 | 64 | 2 | 0.522 |
| lr5e-05_dm64 | 5e-5 | 64 | 2 | 0.899 |
| lr1e-05_dm128 | 1e-5 | 128 | 2 | 0.681 |
| **lr5e-05_dm128** | **5e-5** | **128** | **2** | **0.955** |

Val set: **etna held out entirely** (site holdout, not spatial split) — 3,193 presence + 8,319 absence pixels, geographically independent of training.

Key ingredients of the best config:
- **Sites (train):** Norman Road + Cloncurry + Landsend
- **Sites (val):** Etna Creek (held out)
- **Features:** VH, VV, VH−VV, RVI (4 S1 bands, no S2)
- **Per-pixel z-score normalisation** + **DOY phase-shift augmentation**
- **lr=5e-5, d_model=128, n_layers=2, dropout=0.5, weight_decay=0.1**
- **No global features, no S2**

`lr=1e-5` was definitively too small — dm64 peaked at epoch 1, dm128 plateaued at 0.681 after all 60 epochs. `d_model=128` added ~6 AUC points over 64 at the same lr. dm128 hadn't fully converged at epoch 50, suggesting headroom remains.

### Previous baseline: `v8_s1_zscore_nr` — 0.816 val AUC

- **Sites:** Norman Road presence (9 regions) + Cloncurry absence (7 regions) only
- Val AUC used a spatial (latitude) split — known to be unreliable; numbers not directly comparable to etna holdout results

### What we learned from the overnight sweep

18 experiments were run. Most runs with AUC < 0.5 are likely inversion artefacts from unlucky val splits rather than genuinely bad models. The multisite z-score run (`v8_s1_zscore`) collapsed at 0.495 — attributed to val split instability when mixing sites with very different backscatter regimes. The phase-shift experiments showed the most coherent attention patterns (Jun–Sep dry-season lean at NR), which is biologically plausible.

### Site similarity analysis

`utils/site_similarity.py` computes pairwise cosine distance between site presence-median vectors and within-site Bhattacharyya separability. Run against the sweep_zscore_etna_landsend cache:

- **Landsend** — most separable (1.253): presence and absence are spectrally distinct
- **Etna** — middling (0.437): mixed pixels, good challenging holdout
- **Norman Road** — least separable (0.200): presence/absence overlap heavily in S1 feature space
- Etna presence has distinctly lower S1 backscatter than landsend/norman_road (`s1_mean_vh_dry` z-score = −1.03) — explaining why the model struggled to generalise to etna from the other sites early in training

### S1 data pipeline fixes

- Fixed tile parquet rebuild that was missing Norman Road presence pixels
- Fixed S1 collector OOM for large location bboxes: point sharding (50k points/shard) + streaming parquet writes + 4-worker parallel item fetch
- Noise filter (dry_ndvi, rec_p, nir_cv) NaN-safe for S1-only runs: fillna(threshold) so missing S2 features don't silently pass or fail the filter

---

## Where We're Headed

### Immediate: `sweep_lm_corfield`

Adds **Lake Mueller** and **Corfield** to the training corpus. Grid explores the neighbourhood of the best config (lr=5e-5, d_model=128, n_layers=2):

| run | lr | d_model | n_layers |
|---|---|---|---|
| anchor | 5e-5 | 128 | 2 |
| lr_up | 1e-4 | 128 | 2 |
| wider | 5e-5 | 256 | 2 |
| deeper3 | 5e-5 | 128 | 3 |
| deeper4 | 5e-5 | 128 | 4 |

Etna remains the holdout site. Fetch data then run:
```
python -m cli.location training fetch --prefix corfield && \
python -m cli.location training fetch --prefix lake_mueller && \
python sweep_lm_corfield.py
```

### Medium term

- **Directions still needed:** north (Gulf lowlands), west (Barkly Tableland)
- **Ensemble with S2:** S1 model doesn't need to discriminate alone — even a weak orthogonal signal adds value in a fused model
- **Longreach comparison:** need S1 tile data for longreach to place it in the site similarity space; currently only S2 features (nir_cv=0.046, rec_p=0.279) are available

### Longer term

- **Input smoothing:** S1 output is noisy due to unsmoothed per-observation backscatter. A temporal median filter on the input (or learned smoothing) would clean up scoring maps
- **Revisit val strategy:** once more sites are added, consider rotating holdout across sites (LOSO) rather than fixing etna, to get a more robust estimate of generalisation
