# v1_spectral — Model Card

## Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 32 |
| n_heads | 4 |
| n_layers | 1 |
| d_ff | 64 |
| dropout | 0.3 |
| params | 9,025 |
| lr | 1e-4 |
| weight_decay | 1e-3 |
| batch_size | 1024 |
| patience | 15 |
| features | 13 (B02–B12 + NDVI + NDWI + EVI) |

## Training Data

23 regions across 6 S2 tiles — 9,762 presence / 22,198 absence pixels.

Spatial train/val split (80/20 by latitude): 7,810 presence / 17,759 absence train | 1,952 presence / 4,439 absence val.

**Presence sites:** Lake Mueller ×3 (arid riparian, Thomson River), Barcoorah (arid), Frenchs ×2 (Cape York monsoonal savanna), Maria Downs (Gulf savanna), Norman Road ×4 (Gulf savanna).

**Absence sites:** Lake Mueller, Barcoorah ×2, Frenchs ×5 (mangrove, ocean, riparian, savanna), Maria Downs, Mitchell River (monsoonal riparian), Norman Road ×2.

**Excluded:** nardoo_presence (uncertain label, no local absence); lake_mueller_presence_mixed (noisy — Parkinsonia mixed with native canopy, caused near-random AUC); stonehenge ×6 (patches <100px, insufficient signal); rockhampton ×2 (uncertain labels, swamp context).

## Best Run

**Checkpoint: epoch 3, val_auc=0.884** (workstation z640, 2026-04-20)

Across multiple runs on identical config and data, best val_auc ranged 0.844–0.884 — variance is due to random weight initialisation and spatial split randomness, not meaningful model differences.

## Training Log (best run)

```
2026-04-20 04:17:07  Labeled pixels — presence: 9762  absence: 22198
2026-04-20 04:17:08  Spatial split — train: 7810 presence / 17759 absence | val: 1952 presence / 4439 absence
2026-04-20 04:19:26  Train windows: 153414  |  Val windows: 36026
2026-04-20 04:19:26  Model: d_model=32 n_heads=4 n_layers=1 d_ff=64  params=9025
epoch  1/100  loss=0.8765  val_auc=0.625  *
epoch  2/100  loss=0.6773  val_auc=0.844  *
epoch  3/100  loss=0.5378  val_auc=0.884  *   ← best checkpoint saved
epoch  4/100  loss=0.4074  val_auc=0.873
epoch  5/100  loss=0.3083  val_auc=0.828
epoch  6/100  loss=0.2343  val_auc=0.798
epoch  7/100  loss=0.1800  val_auc=0.780
epoch  8/100  loss=0.1411  val_auc=0.773
```

Loss halves each epoch while val_auc peaks at epoch 3 — classic overfitting onset. The model ceiling is data-limited (noisy labels, ~9k presence pixels across 11 sites). More diverse GPS-verified training sites are the primary path to improvement.

# Feature Importance (Permutation)

Baseline AUC on full training set: 0.911

| Feature | AUC drop | Notes |
|---------|----------|-------|
| B11 (SWIR-1) | +0.047 | Most important — moisture/cellulose absorption |
| B05 (Red Edge) | +0.037 | Vegetation structure, canopy density |
| B12 (SWIR-2) | +0.030 | Woody biomass, bark/stem reflectance |
| B02 (Blue) | +0.026 | Background discrimination |
| B8A (NIR narrow) | +0.022 | Canopy structure |
| B04 (Red) | +0.014 | |
| B06 (Red Edge 2) | +0.013 | |
| NDWI | +0.011 | Modest moisture signal |
| B08 (NIR) | +0.009 | |
| B03 (Green) | +0.008 | |
| B07 (Red Edge 3) | +0.003 | |
| **NDVI** | **-0.001** | **Not used — shuffling improves AUC** |
| **EVI** | **-0.001** | **Not used — shuffling improves AUC** |

SWIR bands dominate. Model is detecting woody/structural signal, not greenness. NDVI and EVI contribute nothing — the model learned Parkinsonia's intrinsic SWIR reflectance rather than general vegetation indices.

# Attention Analysis (Temporal)

Mean attention weight by month — presence vs absence pixels:

| Month | Presence | Absence | Diff |
|-------|----------|---------|------|
| Jan | 0.020 | 0.020 | +0.001 |
| Feb | 0.020 | 0.029 | -0.009 |
| Mar | 0.023 | 0.027 | -0.004 |
| Apr | 0.017 | 0.016 | +0.001 |
| May | 0.022 | 0.017 | +0.004 |
| Jun | 0.015 | 0.013 | +0.002 |
| Jul | 0.018 | 0.013 | +0.004 |
| Aug | 0.017 | 0.013 | +0.004 |
| Sep | 0.017 | 0.017 | 0.000 |
| Oct | 0.018 | 0.021 | -0.002 |
| Nov | 0.019 | 0.021 | -0.002 |
| Dec | 0.020 | 0.022 | -0.003 |

Presence pixels attract more attention in the dry season (May–Aug); absence pixels attract more attention in the wet season (Oct–Feb). Consistent with Parkinsonia retaining its canopy/SWIR signal when native vegetation is senescent — the model has learned the correct phenological discriminant without supervision.