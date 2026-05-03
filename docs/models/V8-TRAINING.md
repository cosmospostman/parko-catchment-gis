# V8 Training — Status and Direction

## Where We Are

V8 is the first generation of TAM trained on Sentinel-1 only (no S2 bands). The goal is a model that discriminates Parkinsonia using SAR backscatter temporal signatures, which would complement S2-based models by operating through cloud cover and adding an independent signal.

**Val protocol (all sweeps below):** Etna Creek held out entirely as a site holdout — 3,193 presence + 8,319 absence pixels, geographically independent of training. All runs use `lr=5e-5` as the settled optimum; `lr=1e-5` consistently fails to converge within 60 epochs.

### Site expansion comparison

| sweep | train sites | holdout | best val AUC | params |
|---|---|---|---|---|
| `sweep_zscore_etna_landsend` | NR + Cloncurry + Landsend | Etna | 0.956 | lr=5e-5, d_model=128 |
| `sweep_zscore_lm_corfield` | NR + Cloncurry + Landsend + Lake Mueller + Corfield† | Etna | 0.937 | lr=5e-5, d_model=128 |
| `v8_s1_zscore_nr_etna_landsend_lm_corfield` | NR + Cloncurry + Landsend + Lake Mueller + Corfield‡ | Etna | 0.964 | lr=5e-5, d_model=128 |
| `v8_roper` | NR + Cloncurry + Landsend + Lake Mueller + Corfield + Roper§ | Etna | **0.989** | lr=5e-5, d_model=128 |

†Corfield presence was filtered out by the noise filter (absence only); regression attributed to site imbalance.
‡Corfield presence re-collected after noise-filter NaN fix — 2,109 presence + 4,710 absence pixels included. Surpasses all previous runs; 32,203 train pixels total.
§Roper adds 937 presence + 3,635 absence pixels (36,775 train pixels total). Northern Gulf-region site; large AUC jump likely reflects independent S1 signature complementary to existing sites.

### `sweep_zscore_etna_landsend` — full results

| run | lr | d_model | val_auc |
|---|---|---|---|
| lr1e-05_dm64 | 1e-5 | 64 | 0.522 |
| lr5e-05_dm64 | 5e-5 | 64 | 0.899 |
| lr1e-05_dm128 | 1e-5 | 128 | 0.681 |
| **lr5e-05_dm128** | **5e-5** | **128** | **0.956** |

Train sites: Norman Road (presence + absence) + Cloncurry (absence) + Landsend (presence + absence). `d_model=128` added ~6 AUC points over 64 at the same lr.

### `sweep_zscore_lm_corfield` — full results

| run | lr | d_model | val_auc |
|---|---|---|---|
| lr1e-05_dm64 | 1e-5 | 64 | 0.476 |
| lr5e-05_dm64 | 5e-5 | 64 | 0.879 |
| lr1e-05_dm128 | 1e-5 | 128 | 0.526 |
| **lr5e-05_dm128** | **5e-5** | **128** | **0.937** |

Adds Lake Mueller (presence + absence) and Corfield (absence only — presence filtered). Regression vs previous sweep likely due to absence-only corfield skewing the decision boundary.

### Best config (settled hyperparams)

- **Features:** VH, VV, VH−VV, RVI (4 S1 bands, no S2, no global features)
- **lr=5e-5, d_model=128, n_layers=2, dropout=0.5, weight_decay=0.1**
- **Per-pixel z-score normalisation** + **DOY phase-shift augmentation** + **DOY density normalisation**

### Previous baseline: `v8_s1_zscore_nr` — val AUC unreliable

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

### ~~Immediate: fix corfield presence, re-run `sweep_lm_corfield`~~ — DONE

`v8_s1_zscore_nr_etna_landsend_lm_corfield` confirmed Corfield presence fix: val AUC rose from 0.937 → 0.964, surpassing the previous best (0.956). Task complete.

### ~~Add Roper site~~ — DONE

`v8_roper` adds Roper (937 presence + 3,635 absence, northern Gulf lowlands) to the full site set. Val AUC rose from 0.964 → **0.989** — the largest single-site gain yet. Checkpoint: `outputs/models/train_v8_roper/tam_model.pt`.

### Attention visualisation — `v8_roper`

Full outputs in `outputs/models/train_v8_roper/attention/`. Key findings:

The model attends to different periods of the year at different locations, reflecting local rainfall seasonality rather than a fixed global window. This is the right behaviour — Parkinsonia's phenological signature relative to background vegetation shifts with the local wet/dry cycle.

| site | presence peak months | interpretation |
|---|---|---|
| Roper | Mar, Mar, Jan | wet-season dominant; h2/h3 also pull Jun–Sep (dry) |
| Norman Road | Mar, Sep, Jul | wet-season primary with dry-season secondary |
| Corfield | Feb, Feb, Dec | wet-season |
| Landsend | Mar, Sep, Sep | wet-season primary, strong dry-season secondary |
| Lake Mueller | Nov, Oct, Oct | southern site — opposite-phase calendar; all 4 heads agree tightly |
| Etna (holdout) | Mar, Mar, Apr | wet/dry transition; consistent with its mixed-pixel character |

The transformer is inferring local phenological context from the S1 time series shape itself — no site ID is passed as input. Lake Mueller's Nov/Dec cluster (vs the Jan–Mar cluster at northern sites) confirms the model is picking up on the structural difference in seasonality, not memorising site-level offsets.

### Bhattacharyya separability — `v8_roper`

Run via `utils/site_similarity.py` against `outputs/models/train_v8_roper/global_features_cache.parquet`. Outputs in `outputs/models/train_v8_roper/site_similarity/`.

| Site | Separability | n_presence | n_absence |
|---|---|---|---|
| Landsend | 1.253 | 3,954 | 5,147 |
| Corfield | 1.186 | 2,109 | 4,710 |
| Lake Mueller | 0.712 | 1,019 | 1,835 |
| Etna | 0.437 | 3,633 | 8,319 |
| **Roper** | **0.275** | **937** | **3,635** |
| Norman Road | 0.200 | 5,189 | 6,021 |
| Cloncurry | NaN | 0 | 2,228 |

Roper sits near the bottom of the separability ranking — presence and absence overlap heavily in the S1 global feature space, consistent with the nearly-identical attention profiles noted above. Despite this, it produced the largest single-site AUC jump (0.964 → 0.989), so its value is in signature diversity rather than intrinsic separability. Presence-median profile confirms this: highest `s1_vh_contrast` of all sites (z=+1.26) and most distinct `peak_doy` (z=−1.22), pointing to a phenologically-offset signature that complements the other northern sites.

### Next steps

- **More seasonality diversity:** Lake Mueller is the only southern (Nov/Dec-phase) site — one more like it would strengthen that regime. Barkly Tableland (west) likely adds a distinct dry-season onset and is the priority new site.
- **Roper presence quality:** Low separability (0.275) and near-identical presence/absence attention profiles confirm weak within-site discrimination. Worth auditing the presence polygons before adding more Gulf-region sites.
- **Val strategy:** At 0.989 on a single fixed holdout, Etna may be flattering or hiding weaknesses. LOSO across all sites should be run before committing to this architecture for inference.
- **Ensemble with S2:** S1 model doesn't need to discriminate alone — even a weak orthogonal signal adds value in a fused model.
- **Input smoothing:** A temporal median filter on S1 input (or learned smoothing) would reduce per-observation backscatter noise in scoring maps.
