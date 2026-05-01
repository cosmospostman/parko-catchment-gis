# V8 Training — Status and Direction

## Where We Are

V8 is the first generation of TAM trained on Sentinel-1 only (no S2 bands). The goal is a model that discriminates Parkinsonia using SAR backscatter temporal signatures, which would complement S2-based models by operating through cloud cover and adding an independent signal.

### Best result so far: `v8_s1_zscore_nr` — 0.816 val AUC

Key ingredients:
- **Sites:** Norman Road presence (9 regions) + Cloncurry absence (7 regions)
- **Features:** VH, VV, VH−VV, RVI (4 S1 bands, no S2)
- **Per-pixel z-score normalisation** — removes site-level backscatter offset (incidence angle, local geometry), leaving only the temporal shape
- **DOY phase-shift augmentation** — makes the model calendar-invariant, forcing it to learn signal shape rather than absolute timing
- **No global features, no S2**

Spatial scoring at Beaudesert and Flinders showed plausible spatial coherence. The model is noisy (unsmoothed S1 input) but spatially coherent at the aggregate level. Longreach showed limited discrimination, consistent with the model not having seen semi-arid vegetation during training.

### What we learned from the overnight sweep

18 experiments were run. Most runs with AUC < 0.5 are likely inversion artefacts from unlucky val splits rather than genuinely bad models. The multisite z-score run (`v8_s1_zscore`) collapsed at 0.495 — attributed to val split instability when mixing sites with very different backscatter regimes (Lake Mueller's strong Nov–Dec soil moisture signal dominated). The phase-shift experiments showed the most coherent attention patterns (Jun–Sep dry-season lean at NR), which is biologically plausible.

### S1 data pipeline fixes

- Fixed tile parquet rebuild that was missing Norman Road presence pixels
- Fixed S1 collector OOM for large location bboxes: point sharding (50k points/shard) + streaming parquet writes + 4-worker parallel item fetch
- Noise filter (dry_ndvi, rec_p, nir_cv) now runs for all experiments regardless of `n_global_features` — uses S2 data in the tile parquets to filter non-woody presence pixels even in S1-only training

---

## Where We're Headed

### Immediate: site expansion

The spatial expansion strategy is to add new training sites in different geographic directions from Norman Road, forcing the model to generalise its representation across biomes.

**Sites added so far (to be fetched and trained):**
- **Etna Creek** (semi-arid, ~142°E / 21.6°S) — 4 presence + 5 absence, southwest of NR
- **Landsend** (semi-arid/riparian, ~141.4°E / 20.45°S) — 4 presence + 3 absence, west of NR

**Next experiment:** `v8_s1_zscore_nr_etna_landsend` — identical hyperparams to the 0.816 baseline, adds Etna and Landsend. Directly comparable. Success criterion: val AUC stays near or above 0.816 while scoring plausibly at semi-arid sites (Winton, Longreach).

### Medium term

- **Directions still needed:** north (Gulf lowlands), west (Barkly Tableland), southeast (Cloncurry presence labels)
- **Multisite z-score:** revisit `v8_s1_zscore` once site coverage is broader and val split instability is resolved (fixed `--val-sites` holdout)
- **Ensemble with S2:** S1 model doesn't need to discriminate alone — even a weak orthogonal signal adds value in a fused model. Score both models on the same sites and check where S1 corrects S2 errors

### Longer term

- **Input smoothing:** S1 output is noisy due to unsmoothed per-observation backscatter. A temporal median filter on the input (or learned smoothing) would clean up scoring maps without touching the model architecture
- **More presence diversity:** all current presence training is from Norman Road. Etna and Landsend will add the first non-NR presence pixels — critical for breaking NR-specific overfitting
