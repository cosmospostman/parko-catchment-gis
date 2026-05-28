# Pseudo Cross Multivariate Variogram (PCMV) — Research Notes

## What it is

The PCMV is a geostatistical image texture feature that measures the spatial variance
of spectral-temporal change within a local neighbourhood. It was introduced by
Myers (1991) and Papritz (1993) in cokriging theory and adapted to remote sensing
image texture by Chica-Olmo & Abarca-Hernández (2000) and Chica-Olmo et al. (2009).

For two image bands (or the same band at two dates) Z_i and Z_j, the PCMV texture
at pixel x using spatial lag h and window W is:

```
γ_ij(h) = 1/(2N(h)) Σ_{x_α ∈ W} [Z_i(x_α) - Z_j(x_α + h)]²
```

where N(h) is the number of pixel pairs separated by lag h within the window.

The indices i and j can refer to:
- **Two spectral bands at the same date** — captures spectral-spatial texture
- **The same band at two different dates** — captures phenological-spatial texture
  (the multitemporal PCMV)

The multitemporal form is the most relevant here:

```
γ(h) = 1/(2N(h)) Σ [Z(x_α, t_wet) - Z(x_α + h, t_dry)]²
```

This measures how spatially consistent the phenological change between wet and dry
seasons is across the neighbourhood. Low values = spatially homogeneous phenological
response (monospecific stand). High values = heterogeneous response (mixed vegetation).

## Why PCMV is distinct from Pearson coherence

Pearson correlation between pixel time series is scale-invariant — it discards the
absolute variance σ² and measures only co-variation shape. The variogram retains σ²:

```
γ(h) = σ²(1 - ρ(h))
```

For a dense Parkinsonia stand: σ² is low (all pixels similar) and ρ(h) is high
(spatially coherent) → PCMV is very low.

For mixed savanna: σ² is higher (spectral variation between species) and ρ(h) is
lower → PCMV is higher.

Pearson coherence (`tc_p95`) misses the σ² component. Two neighbourhoods with the
same Pearson correlation structure but different absolute variances are
indistinguishable to Pearson but distinguishable by PCMV. This is the key advantage
for monoculture detection.

## Key papers

**Rodriguez-Galiano et al. (2012)** — "Random Forest Classification of Mediterranean
Land Cover Using Multi-Seasonal Imagery and Multi-Seasonal Texture." *Remote Sensing
of Environment*, 121, pp. 93–107.

The most directly applicable paper. Used Random Forest with all five geostatistical
texture estimators (variogram, madogram, rodogram, cross-variogram, PCMV) computed
at multiple window sizes (5×5, 15×15, 31×31) and lags, applied to Landsat TM spring
and summer imagery over 14 Mediterranean land cover classes. Key results:

- Multi-seasonal PCMV was ranked #1 feature by Gini importance — the single most
  discriminative texture feature class overall
- kappa = 0.92 (geostatistical texture) vs 0.89 (GLCM) vs 0.83 (maximum likelihood)
- Geostatistical features outperformed GLCM by 3 kappa points
- PCMV between visible and NIR was most important for general classification
- Multi-seasonal PCMV (same band, two dates) dominated for vegetation classes

**Chica-Olmo et al. (2009)** — "Multivariate Image Texture by Multivariate Variogram
for Multispectral Image Classification." *Photogrammetric Engineering & Remote
Sensing*, 75(2).

Introduced the multivariate extension of the variogram texture to multispectral data.
Compared Euclidean, Mahalanobis, and Spectral Angle Distance versions. The PCMV
(bivariate pairwise form) is one flavour of this. Spectral Angle Distance version
gave the best results on Landsat data.

**Jin & Li (2012)** — "Land Cover Classification Using Multitemporal CHRIS/PROBA
Images and Multitemporal Texture." *International Journal of Remote Sensing*, 33(1).

Explicitly derived PCMV as a *multitemporal* texture measure — two dates treated as
Z_i and Z_j. Found +3.3–4.3% OA improvement and +4.9–6.6% kappa improvement over
spectral-only classification, with the largest gains for vegetation classes. Validated
at 18m resolution, directly analogous to Sentinel-2 10m use case.

## Our implementation: raw time series, not seasonal composites

The literature implements PCMV from two seasonal composites because that is all those
authors had. With a full dense time series (Sentinel-2 every 5 days, cloud-filtered)
we can do better.

For each pixel pair (target pixel i, neighbour j), compute the mean squared difference
across all dates where both pixels have a valid cloud-free observation:

```
γ_ij = 1/N Σ_{t ∈ shared_dates} (NDVI_i(t) - NDVI_j(t))²
```

where `shared_dates` is the intersection of cloud-free observation dates for the two
pixels. Since both pixels are within a 50m window in the same S2 scene, their cloud
masks are nearly identical — most observations will be shared.

This has two advantages over seasonal composites:

1. **Preserves asynchrony signal.** If two adjacent eucalypts flush two weeks apart,
   there is a period where one is at peak NDVI and the other hasn't started. That
   divergence exists in raw observations and is captured by the squared difference on
   those dates. Fortnightly or monthly compositing smears this signal away entirely.

2. **No aggregation decisions.** No choice of wet/dry season window, no compositing
   method, no cloud threshold for composite inclusion — just the raw observations with
   their existing SCL cloud filter.

Aggregate across neighbours using the **minimum** (most similar neighbour) rather than
the mean — analogous to the 95th percentile choice for Pearson coherence, and for the
same reason: avoids encoding neighbourhood density as a proxy for presence.

The result is one scalar per pixel: `pcmv_ndvi` — the minimum mean squared NDVI
difference between this pixel and any neighbour within r=2. Low values indicate at
least one neighbour with a near-identical time series. High values indicate no similar
neighbours exist in the window.

## Relationship to Pearson coherence

Both features measure neighbourhood phenological homogeneity but via different
mathematics:

| Feature | What it measures | Captures σ²? | Temporal resolution |
|---|---|---|---|
| `tc_p95_r2` (Pearson) | Shape of co-variation with neighbours | No — normalised out | Full time series |
| `pcmv_ndvi` (this) | Magnitude of difference from neighbours | Yes | Full time series |

PCMV replaces Pearson coherence rather than complementing it. The σ² component is
the key advantage: two pixels with identical phenological shape but different absolute
NDVI levels (e.g. denser vs sparser canopy of the same species) are indistinguishable
by Pearson but distinguishable by PCMV. For native eucalypt woodland where individual
trees vary in canopy density, this matters — Pearson may still find high correlation
while PCMV correctly identifies the absolute divergence between neighbours.

## Implementation path

Computable from existing training parquets with no data collection changes:

1. For each labelled pixel, find neighbours within r=2 using spatial coordinates
2. For each neighbour, compute NDVI on shared observation dates and take mean squared
   difference
3. Take the minimum across all neighbours → `pcmv_ndvi`
4. Drop into `compute_global_features()` in `tam/core/global_features.py` alongside
   existing features

The diagnostic script (`tam/experiments/temporal_coherence.py`) can be updated to
compute `pcmv_ndvi` instead of `tc_p95` with minimal changes — same spatial window
logic, different aggregation function.

## Band selection

Start with NDVI. Candidates for incremental value beyond NDVI:

- **B11/B12 (SWIR)** — canopy water content and leaf structure; captures differences
  that NDVI saturates over in dense canopy
- **B05/B06/B07 (red-edge)** — chlorophyll concentration; may separate species with
  similar NDVI but different leaf physiology
- **VH, VV, RVI (Sentinel-1)** — canopy structure independent of phenology; SAR
  backscatter responds to physical geometry (stem density, branch angles, foliage
  orientation) rather than chlorophyll, so PCMV on SAR features captures structural
  homogeneity rather than phenological homogeneity — genuinely complementary

The remaining optical bands (B02, B03, B04, B08, B8A) are likely largely redundant
with NDVI for this discrimination and risk giving the model features to overfit on.

## Window size

5×5 pixels (50m) as default. Parkinsonia stands in riparian corridors are typically
>100m across, so interior pixels of a dense stand will have a full same-species window.
Test 7×7 (70m) once NDVI PCMV is validated.

## Limitations

- Silent for isolated individual plants — same limitation as Pearson coherence,
  handled by the propagation step (see `docs/PROPAGATION.md`)
- At stand boundaries PCMV will be elevated regardless of species — it is a spatial
  homogeneity measure, not a species classifier; discrimination relies on the full
  feature set
- Native monocultures (dense buffel grass, uniform coolibah woodland) will also score
  low — same conclusion as Pearson coherence, expected and acceptable

## Open questions

- Does Parkinsonia's PCMV separate from native savanna vegetation in practice?
  Test by running the updated diagnostic over Norman Road and Frenchs bboxes.
- Does the minimum-neighbour aggregation outperform mean across all neighbours?
- Does PCMV on NDVI add discriminative value beyond the existing `nir_cv`, `rec_p`,
  `dry_ndvi` features which already capture some of the same phenological signal?
