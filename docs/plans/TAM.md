# Temporal Attention Model (TAM) Classifier

## Context

The current Parkinsonia detection pipeline collapses per-pixel Sentinel-2 time-series into scalar features (nir_cv, rec_p, re_p10, swir_p10) and fits a logistic regression. This discards the phenological signal that is most diagnostic for Parkinsonia — its persistent green bark and post-wet-season flush when native vegetation senesces.

TAM replaces this with a Transformer encoder that operates on the raw annual time-series per pixel, outputting P(Parkinsonia present). The model must be interpretable (attention weights over time should map to ecologically meaningful phenophases) and sensitive to sub-pixel canopy fractions via temporal integration of weak but repeating signals.

---

## New Files

```
tam/
├── __init__.py        # empty
├── model.py           # TAMClassifier (nn.Module)
├── dataset.py         # TAMDataset, TAMSample, collate_fn
├── train.py           # train_tam(), spatial_split(), checkpoint save/load
└── pipeline.py        # run() — load → train → infer → plot
```

Outputs go to `outputs/tam-<location>/`.

---

## Data Pipeline (`tam/dataset.py`)

**Input:** pixel parquet (one row per pixel-date), loaded via `signals/_shared.py:load_and_filter()`.

**Annual windowing:** Group by `(point_id, year)`. Each window becomes one training sample. Labeled pixels only during training; all pixels during inference.

**Tensor per sample:**
- `bands`: shape `(MAX_SEQ_LEN=128, D=10)` — 10 S2 bands (B02 B03 B04 B05 B06 B07 B08 B8A B11 B12), zero-padded, Z-score normalised per band
- `doy`: shape `(MAX_SEQ_LEN,)` — integer DOY 1–365, 0 for padding positions
- `mask`: shape `(MAX_SEQ_LEN,)` — bool, `True` = padding (PyTorch convention)
- `label`: float32 scalar `{0.0, 1.0}`
- `weight`: float32 scalar, uniform `1.0` for PoC

**Band normalisation:** Compute per-band mean/std from all training observations after SCL filter. Save alongside checkpoint as `tam_band_stats.npz`.

**Label assignment:** Call `pipelines/common.py:label_pixels()` verbatim — no change needed.

---

## Model Architecture (`tam/model.py`)

```
TAMClassifier
├── band_proj  : Linear(10, 64)             # project bands to d_model
├── doy_embed  : Embedding(366, 64)         # learned DOY encoding, 0=padding
├── encoder    : TransformerEncoder(
│     TransformerEncoderLayer(
│         d_model=64, nhead=4, dim_feedforward=128,
│         dropout=0.1, batch_first=True
│     ), num_layers=2
│   )
├── pool       : mean over non-padded positions
└── head       : Linear(64, 1) → sigmoid
```

**PoC size:** ~100k parameters. Trains on CPU in seconds.

**Forward pass:**
```
bands (B,T,10) → band_proj → (B,T,64)
doy   (B,T)    → doy_embed → (B,T,64)
x = band_proj(bands) + doy_embed(doy)
x = encoder(x, src_key_padding_mask=mask)   # (B,T,64)
x_pool = mean over non-masked positions     # (B,64)
prob = sigmoid(head(x_pool))                # (B,)
```

**Attention extraction:** `get_attention_weights()` method — manually calls each layer's `self_attn` with `need_weights=True, average_attn_weights=False`, returning `list[tensor(n_heads, T, T)]` per layer.

---

## Training (`tam/train.py`)

**Loss:** `BCEWithLogitsLoss`. For class-imbalanced sites (Frenchs: 912 presence / 10155 absence), use `pos_weight = tensor([absence_count / presence_count])`.

**Optimizer:** AdamW lr=1e-3, weight_decay=1e-4. CosineAnnealingLR over 100 epochs.

**Spatial holdout split:** Hold out a geographic sub-region per class (not random) to avoid spatial autocorrelation leakage. At Longreach, split by latitude within each class (80/20). At Frenchs, hold out one full sub-bbox per class.

**Batch size:** 32. DataLoader shuffle=True. ~100 epochs for PoC.

**Checkpoint:** Save best val-AUC model to `outputs/tam-<location>/tam_model.pt` + `tam_config.json` (hyperparams) + `tam_band_stats.npz`.

---

## Inference (`tam/pipeline.py`)

**Small scenes (training sites):** Load full parquet in memory, run model over all pixels.

**Large scenes (future):** Row-group iterator following `signals/_shared.py:compute_features_chunked()` pattern — only current row group in RAM.

**Multi-year aggregation:** Per pixel, average `prob_tam` across all years meeting `min_obs_per_year >= 8`. Single `prob_tam` scalar output per pixel.

**Output compatibility:** Rename `prob_tam → prob_lr` before calling `utils/heatmap.py:plot_prob_heatmaps()`. Reuse `pipelines/common.py:save_pixel_ranking()` for CSV.

Output dir: `outputs/tam-<location>/`:
- `tam_model.pt`, `tam_band_stats.npz`, `tam_config.json`
- `tam_<location>_prob_vs_imagery.png`, `tam_<location>_prob_black.png`
- `tam_pixel_ranking.csv`
- `attention_viz/px_XXXX_XXXX_YYYY_attn.png`

---

## Attention Visualization

`plot_tam_attention(model, sample, out_path)` — for a single pixel:
- X-axis: DOY of each non-padded observation
- Y-axis: mean attention weight across heads (from final encoder layer)
- Overlay: NIR (B08) and NDVI trajectories

Validates ecological meaningfulness: high-attention dates should fall in dry-season senescence window or post-wet flush.

---

## Existing Utilities to Reuse

| Utility | File | Used for |
|---|---|---|
| `load_and_filter()` | `signals/_shared.py` | SCL filter + year/month columns |
| `ensure_pixel_sorted()` | `signals/_shared.py` | Chunked inference ordering |
| `label_pixels()` | `pipelines/common.py` | Presence/absence label assignment |
| `save_pixel_ranking()` | `pipelines/common.py` | CSV output |
| `plot_prob_heatmaps()` | `utils/heatmap.py` | Probability map PNGs |
| `Location`, `get()` | `utils/location.py` | Site config, parquet path |

TAM does **not** use `extract_parko_features()`, `ParkoClassifier`, or `StandardScaler` — these are bypassed entirely.

---

## Verification

1. `python -m tam.pipeline --location longreach --train` — trains on 748 labeled pixels, saves checkpoint
2. Inspect `outputs/tam-longreach/tam_longreach_prob_vs_imagery.png` — presence sub-bbox should score visibly higher than absence
3. Check `tam_pixel_ranking.csv` — verify presence pixels cluster at top of ranking
4. Run `plot_tam_attention()` on 3–5 high-scoring presence pixels — attention peaks should fall in dry season (DOY 150–300 for QLD)
5. Check val-AUC logged during training — target > 0.75 for PoC acceptance

---

## Effort Estimate

| Phase | Scope | Days |
|---|---|---|
| **PoC** | tam/dataset.py + tam/model.py + tam/train.py + tam/pipeline.py (small-scene path) + smoke tests + end-to-end run on Longreach | **~6 days** |
| **Build-out** | Chunked inference, multi-site training, attention viz, SHAP, hyperparameter sweep, soft label weighting, cross-site eval, docs | **~9 days** |
| **Total** | | **~15 days** |

PoC acceptance: trains without NaN loss, non-trivial probability maps, presence > absence, val-AUC > 0.75.
