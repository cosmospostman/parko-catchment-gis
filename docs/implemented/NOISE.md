# Presence Pixel Noise Filtering

Training bboxes are drawn generously around known Parkinsonia infestations. They inevitably
contain non-Parkinsonia pixels — water bodies, bare soil, grass patches, and SCL=6
misclassifications. These filters remove obvious noise before pixels reach the model.

## Filters applied

### 1. SCL=6 exclusion (all pixels)

Sentinel-2 SCL class 6 ("Dark Area Pixels") is frequently over-triggered in tropical
environments, misclassifying shaded vegetation understorey as dark surfaces. These
observations are stripped from `pixel_df` before any dataset construction.

**Effect:** ~13.9% of observations removed; ~7.3% of annual windows lost.

### 2. Dry-season median NDVI (presence only)

Pixels where the median NDVI during May–October is below `presence_min_dry_ndvi` (default
0.10) are excluded from presence training. This catches persistent water bodies and bare
soil that happen to fall inside a presence bbox.

**Motivation:** Analysis revealed 2,258 nassau presence pixels with dry-season NDVI ≈ -0.49,
caused by SCL=6 observations from tidal/flooded areas passing the purity filter. Even after
SCL=6 exclusion, persistent water produces strongly negative NDVI year-round.

### 3. Low NDVI amplitude (presence only)

Pixels where mean annual NDVI amplitude (rec_p) is below `presence_min_rec_p` (default
0.20) are excluded from presence training. Parkinsonia has a characteristic wet/dry seasonal
swing; pixels with near-zero amplitude are bare soil, standing water, or inactive ground.

### 4. High NIR variability (presence only)

Pixels where inter-annual dry-season NIR coefficient of variation exceeds
`presence_grass_nir_cv` (default 0.20) are excluded from presence training. High NIR CV
indicates year-to-year variability inconsistent with stable Parkinsonia canopy — typical of
grass or sparse shrub that senesces to different degrees each dry season.

## Configuration

All thresholds are in `TAMConfig` and overridable via CLI:

```
--presence-min-dry-ndvi   float  (default: 0.10)
--presence-min-rec-p      float  (default: 0.20)
--presence-grass-nir-cv   float  (default: 0.20)
```

## Intent

These filters lower the cost of drawing training bboxes — users can draw generously around
a known infestation without needing to exclude every water pixel or bare soil patch by hand.
Absence pixels are not filtered on NDVI or amplitude criteria since bare soil, grass, and
water are legitimate absence classes.

## Implementation

- `tam/core/global_features.py` — computes `dry_ndvi` (dry-season median NDVI) alongside
  the other global features; cached to `global_features_cache.parquet` in the output dir
- `tam/core/config.py` — filter threshold fields
- `tam/core/train.py` — SCL=6 strip applied to `pixel_df`; presence noise filter applied
  after global features are loaded, before `TAMDataset` construction
- `tam/pipeline.py` — CLI args for threshold overrides
