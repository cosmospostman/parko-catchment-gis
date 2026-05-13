# Woody Classifier — Stage-1 Woody Mask

## Purpose

A stage-1 XGBoost classifier that separates woody vegetation (trees, shrubs, mangroves) from non-woody land cover (grass, bare soil, sand, water) at 10 m pixel resolution across Australia.

Used as a pre-filter before TAM inference: pixels classified as non-woody are suppressed, eliminating the false-positive problem observed in V9-SPECTRAL smoke-testing on Longreach bare ground.

**Design requirements:**
- High precision — borderline pixels default to non-woody (threshold 0.85)
- Retain mixed pixels with genuine woody component (sparse canopy, riparian fringe)
- Generalise across all Australian climate zones without per-region calibration
- No temporal sequence required — uses per-pixel annual summary statistics only (sidesteps the obs-count mismatch that breaks the TAM transformer at inference)

---

## Module Structure

```
woody-classifier/
├── __init__.py
├── features.py      # compute_woody_features() — band summaries + S1 stats
├── train.py         # load data → fit XGBoost → save model
├── score.py         # load model → score pixel parquets → write prob parquet
└── evaluate.py      # AUC, precision/recall curve, feature importances
```

Region IDs are read directly from `data/locations/woody-classifier.yaml` (no separate `regions.py`).

Outputs: `outputs/woody-classifier/`
Training bboxes: `data/locations/woody-classifier.yaml` (same schema as `training.yaml`; displayed in UI as a separate sub-accordion)

---

## Feature Set

~20 features per pixel, computed from the full annual observation stack.

### S2 optical (16)

| Feature | Derivation | Signal |
|---|---|---|
| `B11_p5` | 5th pct of SWIR-1 | Woody SWIR floor; bare soil stays high |
| `B11_p95` | 95th pct of SWIR-1 | SWIR ceiling |
| `B11_std` | Temporal std of SWIR-1 | Low = stable canopy |
| `B12_p5` | 5th pct of SWIR-2 | SWIR-2 axis |
| `B12_std` | Temporal std of SWIR-2 | Canopy stability |
| `B08_p95` | 95th pct of NIR | Peak canopy reflectance |
| `B08_std` | Temporal std of NIR | NIR stability |
| `B8A_p95` | 95th pct of NIR narrow | Canopy density cross-check |
| `B05_p95` | 95th pct of red-edge-1 | Chlorophyll saturation |
| `NDVI_p10` | 10th pct of NDVI | Dry-season vegetation floor — the "persistence gate" |
| `NDVI_p90` | 90th pct of NDVI | Peak greenness |
| `NDVI_std` | Temporal std of NDVI | Phenological variability |
| `NDWI_p5` | 5th pct of NDWI | Persistent water detection |
| `ndvi_amplitude` | `NDVI_p90 − NDVI_p10` | Large = grass; moderate = woody. Self-normalised. |
| `swir_nir_ratio_p5` | `B11_p5 / (B08_p95 + ε)` | Woody bark/cellulose ratio vs. bare soil |
| `nir_cv` | `B08_std / (mean B08 + ε)` | Relative NIR variability; low = evergreen woody |

### S1 SAR (4)

| Feature | Signal |
|---|---|
| `s1_mean_vh_dry` | Mean VH dB May–Oct. Woody structure persists through dry season; grass collapses. |
| `s1_vh_contrast` | Mean VH wet − mean VH dry. Large positive = grass; near-zero = woody. |
| `s1_vh_std` | Temporal std of VH. Low = stable canopy structure. |
| `s1_mean_rvi` | Mean Radar Vegetation Index. Integrated canopy volume. |

S1 features are adapted from `tam/core/global_features.py:_compute_s1_globals()` with no TAMConfig dependency.

---

## Training Data

All training bboxes are independent of the V9 region set — selected specifically for woody/non-woody discrimination, not Parkinsonia detection.

### Presence (woody) — target ~7,500 px across 5 types

| Type | Target px | Notes |
|---|---|---|
| Riparian woody (incl. Parkinsonia) | ~2,000 | Core use case |
| Open eucalypt/savanna woodland | ~2,000 | Cloncurry-style — label as presence here (V9 had as absence) |
| Dense brigalow / mulga | ~1,500 | New bboxes needed — central/SW QLD |
| Mangrove | ~500 | Woody canopy; `frenchs_absence_mangrove` flips to presence |
| Sparse / scattered woody (10–30% cover) | ~1,500 | Critical for mixed-pixel retention |

### Absence (non-woody) — target ~9,000 px across 6 types

| Type | Target px | Notes |
|---|---|---|
| Bare soil / red earth | ~3,000 | Largest class — the V9 failure mode |
| Dry-season grass (savanna) | ~2,000 | High wet-season NDVI but no woody structure |
| Irrigated cropland | ~1,000 | Seasonal NDVI but no SAR structure |
| Permanent water | ~500 | Low NDWI_p5 is a strong separator |
| Mitchell grass / dense pasture | ~1,500 | Hard case — moderate SAR signal |
| Saltpan / sand dune | ~500 | Bright SWIR; confirm model doesn't confuse with woody |

### Geographic coverage (both classes per zone)

| Zone | Climate | Required |
|---|---|---|
| Tropical monsoon | Cape York, Gulf, Top End | Riparian woody + savanna grass + bare floodplain |
| Semi-arid NW | NW QLD, NT | Sparse woody + dry grass + bare red soil |
| Arid central | Central QLD | Scattered shrubs + bare soil + gibber |
| Sub-tropical | SE/central west QLD | Brigalow/eucalypt + cropland + grass pasture |

Target ~20–28 presence bboxes and ~28–36 absence bboxes across all zones. Bbox size: 200–600 px each (~150–250 m side).

---

## Training Protocol

### Validation set

**Target: ≥800 presence + ≥800 absence val pixels**, balanced for directly comparable per-class metrics. Add more patches wherever cover-type diversity demands it — diversity of woody types within val is more important than hitting a pixel ceiling. Drawn from one complete holdout site per climate zone — never a random pixel split or spatial fraction of training regions (V5: spatial split gave AUC 0.88; site holdout gave 0.58 on same model).

| Zone | Holdout site | Why hard |
|---|---|---|
| Tropical monsoon | New patch — e.g. Pormpuraaw or Mitchell River | Tests tropical wet/dry phenology transfer |
| Semi-arid | Etna Creek | Consistent with V8/V9 protocol |
| Arid | Longreach bare-soil + sparse woody patch | The known V9 failure case |
| Sub-tropical | Barcoorah or new brigalow site | Tests new vegetation type |

Each holdout site contributes both presence and absence pixels. Use as many patches as needed to cover the full woody-type diversity at that site (dense riparian, sparse scattered trees, mangrove fringe, etc.) — don't stop at the first clear patch.

Geographic diversity across holdout sites matters as much as cover-type diversity within them. The four zones above span different soil colours, atmospheric paths, and rainfall seasonality — if the model has a regional bias it will only show up if val sites are geographically spread. Prefer holdout sites that are far from any training site in the same zone.

### XGBoost config

`n_estimators=500`, `max_depth=5`, `lr=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=<absence/presence ratio>`, `early_stopping_rounds=30`

### Prediction threshold

0.85 for the high-precision mask; 0.5 available for recall-optimised use.

---

## Verification Targets

1. **Val AUC ≥ 0.92** on site holdout
2. **Longreach inference**: >80% of confirmed Parkinsonia pixels pass; <10% of confirmed bare soil passes
3. **Feature importance**: `B11_p5` and `s1_mean_vh_dry` in top 5 (consistent with V1 permutation and V8 findings)
4. **Cross-climate spot-check**: score a temperate woodland tile and a desert tile — no systematic regional bias

---

## Integration

After the mask is validated, `tam/core/score.py` will read `prob_woody` scores and skip pixels below threshold before running TAM inference. This is deferred until the mask itself is validated end-to-end.
