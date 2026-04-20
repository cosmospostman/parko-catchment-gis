# v2_spectral — Model Card

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

## Changes from V1

- Added **frenchs_presence_3/4/5/6** — four additional Frenchs presence sites
- Added **frenchs_absence_bare_soil_1/2/3** — bare/sparse soil absence sites to fix overfitting to bare soil signal
- Water absence sites (frenchs_absence_water_1/2/3) were already present in V1 — not new additions

## Training Data

27 regions across 6 S2 tiles — 12,339 presence / 36,761 absence pixels.

Spatial train/val split (80/20 by latitude): 9,872 presence / 29,409 absence train | 2,467 presence / 7,352 absence val.

**Presence sites:** Lake Mueller ×3 (arid riparian, Thomson River), Barcoorah (arid), Frenchs ×6 (Cape York monsoonal savanna), Maria Downs (Gulf savanna), Norman Road ×4 (Gulf savanna).

**Absence sites:** Lake Mueller, Barcoorah ×2, Frenchs ×8 (bare soil ×3, mangrove, ocean, riparian, riparian woodland, savanna, water ×3), Maria Downs, Mitchell River (monsoonal riparian), Norman Road ×2.

**Excluded:** nardoo_presence (uncertain label, no local absence); lake_mueller_presence_mixed (noisy — Parkinsonia mixed with native canopy); stonehenge ×6 (patches <100px, insufficient signal); rockhampton ×2 (uncertain labels, swamp context).

## Best Run

*(training log not retained)*

## Feature Importance (Permutation)

Baseline AUC on full training set: 0.989

| Feature | AUC drop | Notes |
|---------|----------|-------|
| B11 (SWIR-1) | +0.091 | Dominant — moisture/cellulose absorption |
| NDVI | +0.028 | Now contributes — likely separating bare soil from canopy |
| B05 (Red Edge) | +0.014 | Vegetation structure, canopy density |
| NDWI | +0.013 | Water discrimination — consistent with added water absence sites |
| B12 (SWIR-2) | +0.009 | Woody biomass, bark/stem reflectance |
| B04 (Red) | +0.005 | |
| B06 (Red Edge 2) | +0.004 | |
| **EVI** | **0.000** | **No contribution** |
| **B08 (NIR)** | **0.000** | **No contribution** |
| B8A (NIR narrow) | -0.001 | |
| B07 (Red Edge 3) | -0.002 | |
| B02 (Blue) | -0.003 | |
| B03 (Green) | -0.004 | |

B11 remains the single most important feature by a large margin. The baseline AUC increase from 0.911 (V1) to 0.989 reflects the additional training sites and reduced label noise from the bare soil and water absence sites — but with more diverse absence classes, the training AUC is less trustworthy as an indicator of generalisation.

**Notable shift from V1:** NDVI now contributes meaningfully (+0.028 vs −0.001 in V1). The added bare soil absence sites likely forced the model to use greenness as a discriminant against non-vegetated backgrounds, in addition to the SWIR-based structural signal it already had.

NDWI also now contributes (+0.013), consistent with the water absence sites pushing the model to learn that high moisture/water signal is absence, not presence.

## Attention Analysis (Temporal)

Mean attention weight by month — presence vs absence pixels (10% sample):

| Month | Presence | Absence | Diff |
|-------|----------|---------|------|
| Jan | 0.016 | 0.016 | -0.000 |
| Feb | 0.015 | 0.016 | -0.002 |
| Mar | 0.014 | 0.015 | -0.001 |
| Apr | 0.016 | 0.014 | +0.001 |
| May | 0.017 | 0.015 | +0.002 |
| Jun | 0.014 | 0.014 | +0.000 |
| Jul | 0.020 | 0.020 | -0.001 |
| Aug | 0.016 | 0.015 | +0.001 |
| Sep | 0.027 | 0.028 | -0.000 |
| Oct | 0.023 | 0.024 | -0.001 |
| Nov | 0.019 | 0.019 | -0.000 |
| Dec | 0.023 | 0.025 | -0.002 |

The dry-season preference seen in V1 (presence pixels attracting more attention in May–Aug) is present but attenuated. The dominant pattern in V2 is an overall attention peak in Sep–Dec (wet season onset) for both classes, likely driven by the larger Frenchs dataset which is monsoonal and has strong wet-season phenological signal. The presence/absence differential is much smaller than in V1, suggesting the model is now relying more on spectral identity (SWIR/NDVI) than on temporal phenological contrast to discriminate classes — a plausible consequence of the more diverse training set providing clearer spectral boundaries.
