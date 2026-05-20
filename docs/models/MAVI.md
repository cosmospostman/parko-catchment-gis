# MAVI — Moisture-Adjusted Vegetation Index

## Motivation

Sparse and early-stage Parkinsonia infestations are the hardest detection case: at
low canopy fractions the per-pixel spectral signal is dominated by background soil
and grass, and the sub-pixel canopy contribution is below the noise floor.

The hypothesis investigated here is that Parkinsonia's eco-hydrological footprint
is spatially larger than its canopy footprint. Deep-rooted woody invaders draw down
soil moisture in a radial zone beyond the canopy drip line (root lateral extent
typically exceeds 2× crown diameter, wider still in arid/semi-arid climates;
Tumber-Dávila et al. 2022). This "drawdown halo" — a zone of accelerated soil
moisture depletion around even isolated or sparse plants — may be detectable in
Sentinel-2 SWIR bands even when the canopy itself is sub-pixel.

The research question:

> Can the soil moisture drawdown halo around sparse Parkinsonia infestations be
> detected in Sentinel-2 SWIR temporal profiles, and does it improve discrimination
> of sparse and early-stage infestations?

## Index formulation

```
MAVI = (B08 - B04) / (B08 + B04 + B11)
```

- **B08** (NIR, 842 nm): sensitive to green vegetation structure
- **B04** (Red, 665 nm): absorbed by chlorophyll
- **B11** (SWIR-1, 1610 nm): peak soil and canopy water content sensitivity;
  high SNR relative to B12; less contaminated by NPV/litter than B12 (2190 nm)

MAVI is an NDVI-like index with SWIR in the denominator. When soil or canopy
moisture decreases, SWIR reflectance rises, which depresses MAVI. When
Parkinsonia roots draw down moisture from surrounding soil, the inter-canopy
pixels should show an earlier and steeper MAVI decline during the wet-to-dry
transition than equivalent absence pixels.

The temporal derivative ΔMAVI/Δt (per day, normalised for variable observation
cadence) was also investigated as an explicit rate-of-change feature.

## Investigation

### Data and method

`tam/viz_mavi.py` — analysis script that:
- Loads raw S2 band data (B04, B08, B11) and S1 VH for presence pixels
- Applies the training woody filter (drop presence pixel-years where mean dry-season
  VH < −21 dB, unless rescued by dry-season NDVI ≥ 0.50)
- Computes MAVI per observation
- Plots temporal profiles (mean ± std by DOY) and per-bbox MAVI histograms
  across five seasonal windows

Regions tested: Landsend (semi-arid, riparian), Corfield (semi-arid), Hughenden
(semi-arid). Landsend has explicit sparse presence labels (`landsend_sparse_presence_*`)
and grass absence labels (`landsend_absence_grass_*`), making it the most controlled
comparison.

### Temporal profile results

At Landsend (tag-grouped, woody filter applied):

| Class | MAVI dry-season mean |
|---|---|
| presence (dense) | 0.193 |
| absence (riparian matrix) | 0.186 |
| presence/sparse | 0.145 |
| absence/grass | 0.099 |

The ordering is consistent and physically interpretable. Sparse presence sits
midway between dense presence and grass absence — as expected for mixed pixels
with partial canopy cover.

The ΔMAVI/Δt signal was flat across all classes at the bbox-aggregate level. No
dry-down rate separation was detectable in mean trajectories.

### Histogram results

Per-bbox MAVI histograms across seasonal windows revealed a signal invisible in
the aggregated means:

**Sparse presence bboxes (Landsend):** clear bimodal distributions during the
wet-to-dry transition (Apr–May). A low-MAVI cluster (~0.10) representing
grass/soil-dominated pixels and a high-MAVI cluster (~0.35–0.40) representing
canopy-dominated pixels. The bimodality collapses by mid-dry season as both modes
converge downward.

**Absence/grass bboxes:** consistently unimodal and low throughout, with a tight
distribution tracking steadily downward (median 0.107 → 0.065 through dry season).
No right-tail spread.

The Apr–May window is the highest-discrimination period, not mid-dry — consistent
with the halo hypothesis: Parkinsonia contrast against the drying matrix is
greatest at the wet-to-dry transition.

**Corfield presence (dense, riparian with reliable groundwater):** MAVI increases
into the dry season (median 0.32 → 0.55), the opposite of Landsend sparse. Dense
canopy with deep groundwater access stays hydrated as the surrounding matrix
senesces, pushing NIR up and SWIR down.

### Key finding: MAVI is canopy-fraction-weighted

MAVI does not read purely soil moisture or purely canopy moisture — it reads a
canopy-fraction-weighted combination:

- **Low canopy cover (sparse presence):** MAVI tracks inter-canopy soil moisture,
  decreases into dry season
- **High canopy cover (dense presence):** MAVI tracks canopy water content,
  increases into dry season as surrounding vegetation senesces

This means the *direction* of seasonal MAVI change may be as discriminative as
the absolute value. It also means MAVI is not a universal discriminator — its
interpretation is landscape- and canopy-fraction-dependent.

## Rationale for adding to v10

The model already has B11 and NDVI separately. MAVI adds value by:

1. Making the canopy-fraction-weighted moisture relationship explicit — the model
   doesn't need to learn the ratio from raw band values
2. Providing an additional axis that correlates differently with S1 VH (canopy
   structure) and NDVI (greenness) depending on canopy fraction — multi-signal
   reasoning that attention is suited for
3. The bimodal within-bbox distributions mean individual sparse-presence pixels
   sit in discriminable positions in MAVI space even when the bbox mean is
   uninformative

The temporal derivative ΔMAVI/Δt is not recommended for v10 — no separation was
observed at the pixel-year aggregate level. The TAM attention mechanism can in
principle learn to compare adjacent timesteps from raw MAVI values. If explicit
derivative features are added later, they should be normalised by Δt (days between
observations) to account for irregular S2 cadence.

## Proposed feature

Add `MAVI` to `V9_FEATURE_COLS` in `tam/core/dataset.py` and compute it in
`analysis/constants.py:add_spectral_indices`. Run ablation on v10 vs v9 baseline,
evaluated specifically on sparse-presence validation pixels.

## References

- Zhu, G., Ju, W., Chen, J. M., & Liu, Y. (2014). A Novel Moisture Adjusted
  Vegetation Index (MAVI) to Reduce Background Reflectance and Topographical
  Effects on LAI Retrieval. *PLoS ONE*, 9(7), e102560.
  https://doi.org/10.1371/journal.pone.0102560
  — Original MAVI formulation and validation.

- Robinson, T. P., Trotter, L., & Wardell-Johnson, G. W. (2024). Uncertainty
  Modelling of Groundwater-Dependent Vegetation. *Land*, 13(12), 2208.
  https://doi.org/10.3390/land13122208
  — Applies MAVI-family temporal moisture indices to groundwater-dependent
  vegetation mapping; supports use of SWIR-based indices for detecting
  subsurface moisture access in semi-arid landscapes.

- Tumber-Dávila et al. (2022). Plant sizes and shapes above and belowground and
  their interactions with climate. *New Phytologist*, 235(3), 1032–1056.
  — Root lateral extent meta-analysis; basis for the drawdown halo spatial scale
  argument.

- Persson et al. (2018). Tree Species Classification with Multi-Temporal Sentinel-2
  Data. *Remote Sensing*, 10(11), 1794.
  — Multi-temporal S2 SWIR for woody vegetation discrimination.
