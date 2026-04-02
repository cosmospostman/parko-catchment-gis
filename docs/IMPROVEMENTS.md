# Improvements

## Stage 5: False positives from woody riparian species — and a concrete fix

### The problem

The three signals used in the plausibility map (NDVI anomaly, flowering index, HAND)
were designed to discriminate Parkinsonia from *dry-season senescent grass*, not from
other woody riparian species. The result is that Stage 5 is effectively identifying
"low-lying pixels with persistent woody greenness" — which includes:

- *Melaleuca* spp. — obligate-riparian, evergreen, flowers Aug–Nov
- *Callistemon* spp. (bottlebrush) — obligate-riparian, evergreen, flowers Sep–Nov
- *Casuarina cunninghamiana* (river sheoak) — riparian, persistent green canopy
- *Pandanus* spp. — evergreen, dense, common in tropical riparian corridors
- *Eucalyptus camaldulensis* (river red gum) — dominant floodplain tree, evergreen
- *Leucaena leucocephala* — invasive legume, similar ecological niche to Parkinsonia, yellow-ish flowers
- *Prosopis* spp. (mesquite) — invasive, riparian-favouring, sometimes co-occurs with Parkinsonia

All of these score high on NDVI anomaly (evergreen) + HAND (riparian) + flowering index
(reflectance above senescent grass background). Without a signal that discriminates
Parkinsonia *within the riparian woody stratum*, Stage 5 is a prioritised riparian woody
vegetation map, not a Parkinsonia detector.

### What makes Parkinsonia spectrally distinct from obligate evergreens

The key biological difference is **phenological behaviour**. Native and naturalised
riparian evergreens (Melaleuca, Casuarina, Pandanus, river red gum) are *persistently*
green year-round — their NDVI is flat across the dry season. Parkinsonia, by contrast,
exhibits a marked **seasonal pulse**: it greens up rapidly at the start of the dry
season, reaches peak greenness and flowering in Aug–Oct, then partially defoliates in
late dry/early wet. This pulse is detectable as a *change* in NDVI between early dry
season (May–Jul) and peak flowering window (Aug–Oct).

### Concrete fix: phenological pulse signal

Add a fourth signal — **NDVI pulse magnitude** — computed as:

```
ndvi_pulse = ndvi_aug_oct_median − ndvi_may_jul_median
```

A high positive value means the pixel greened up significantly between early and late dry
season, consistent with Parkinsonia's flush behaviour. Persistently evergreen species
(Melaleuca, river red gum) will have a pulse near zero. Senescing grasses will have a
negative pulse (they brown off through the dry season).

**Implementation notes:**

- Stage 1 already fetches Sentinel-2 scenes for the full May–Oct window. Splitting the
  existing fetch into two sub-window medians (May–Jul and Aug–Oct) requires minimal
  additional compute — the STAC search is already being done, only the temporal
  aggregation window changes.
- `ndvi_may_jul_median` and `ndvi_aug_oct_median` would be written as new Stage 1
  outputs alongside the existing `ndvi_median_YYYY.tif`.
- Stage 5 would load both and compute `ndvi_pulse` as a fourth normalised input to the
  plausibility score, with appropriate weighting (or as a multiplicative gate: suppress
  pixels where pulse < threshold regardless of other signals).

**Expected effect:**
- Parkinsonia: pulse ≈ +0.10 to +0.25 (strong flush)
- Melaleuca / river red gum: pulse ≈ −0.05 to +0.05 (flat)
- Leucaena / Prosopis: intermediate — partial defoliation but less pronounced than Parkinsonia
- Senescent grass: pulse ≈ −0.15 to −0.30 (already suppressed by NDVI anomaly)

### Secondary signal: Red/Green ratio during flush

Parkinsonia's saturated yellow flowers lift Sentinel-2 B4 (red) relative to B3 (green)
in a way that white/cream-flowered species (Melaleuca, Callistemon) do not. A
`red_green_ratio = B4 / (B3 + 1e-9)` computed over the Aug–Oct flowering composite
would add discriminating power against non-yellow-flowered riparian species. Stage 3
already fetches B03; adding B04 to that fetch is trivial.

This signal is less powerful than the phenological pulse for discriminating against
Leucaena and Prosopis (also yellow-ish flowers), but it strengthens the case against
Melaleuca and native riparian species.

### Pipeline impact

Only three stages are affected — Stages 2, 4, and 6 are untouched:

- **Stage 1:** compute two NDVI medians instead of one — May–Jul and Aug–Oct sub-windows
  from the existing fetch. Write both alongside the existing full-window median. No
  additional data download.
- **Stage 3:** add B04 (red) to the existing Aug–Oct fetch (B03 already retrieved).
  Write a Red/Green ratio raster alongside the existing flowering index. No additional
  data download.
- **Stage 5:** gains two new inputs (`ndvi_pulse`, `red_green_ratio`) and combines four
  signals instead of three.

### What to do now vs defer to Stage 6

The phenological pulse signal is worth adding to Stage 5 because:
1. The data is already being fetched — it requires only a sub-window split in Stage 1
2. It directly addresses the structural weakness of the current approach
3. It reduces the false positive burden on drone survey planning

The trained classifier (Stage 6) will still provide empirical weighting of all signals
once ground truth is available. But the drone survey ground truth should explicitly
include confirmed non-Parkinsonia riparian woody patches (Melaleuca stands, river red
gum groves) as labelled absences — not just Parkinsonia presences and bare-ground
absences — so Stage 6 can learn within-riparian discrimination.

---

## Additional signals for improved Parkinsonia discrimination

The following signals could be added to the pipeline to improve discrimination of
Parkinsonia from co-occurring woody riparian species. They are ranked by value vs
implementation cost.

### Priority 1 — Sentinel-2 SWIR bands (B11, B12)

**What it adds:** SWIR reflectance is sensitive to canopy water content and leaf
structure. Parkinsonia's green photosynthetic bark and bipinnate leaflets have a
different SWIR response to sclerophyll eucalypts and dense-canopy Melaleuca.
SWIR also helps distinguish woody vegetation from bare soil and shallow water within
low HAND positions.

**Implementation cost:** Minimal. B11 and B12 are available in the same Sentinel-2
STAC items already being queried in Stages 1 and 3 — adding them requires only an
extra band selection in the existing fetch. They are 20 m native resolution (vs 10 m
for B03/B04/B08) so a reproject-match step is needed, consistent with the red edge
bands already handled in Stage 3.

**Pipeline impact:** Stage 1 or Stage 3 fetch; new output rasters `swir1_median_YYYY.tif`
and `swir2_median_YYYY.tif`; Stage 5 gains SWIR ratio as an additional input.

### Priority 2 — Red edge slope (B05, B06, B07)

**What it adds:** The position and steepness of the red edge inflection point is
sensitive to chlorophyll concentration and canopy structural density. Parkinsonia's
feathery, open bipinnate canopy produces a shallower, earlier red edge than
broad-leaved dense-canopy species (river red gum, Melaleuca). B05 and B06 are already
fetched in Stage 3; adding B07 (Red Edge 3, 20 m) completes the slope calculation.

**Implementation cost:** Very low — one additional band in the Stage 3 fetch. The red
edge slope index is `(B07 − B05) / (B07 + B05)` or a three-point derivative. No new
data source.

**Pipeline impact:** Stage 3 fetch; new output `red_edge_slope_YYYY.tif`; Stage 5 gains
one additional input.

### Priority 3 — DEA Water Observations (WOfS)

**What it adds:** Replaces the terrain-modelled HAND signal with *observed* inundation
frequency from the full Landsat archive (1986–present). Parkinsonia tolerates periodic
inundation but not permanent or very frequent flooding. WOfS directly captures where
water actually pools, which is more ecologically accurate than HAND in channels with
complex micro-topography.

**Implementation cost:** Moderate. WOfS is a DEA product on the same infrastructure as
the Landsat baseline used in Stage 2, so access is straightforward. It requires a new
fetch function and output raster but fits naturally alongside the Stage 4 HAND
computation. The two signals are complementary: HAND captures terrain position, WOfS
captures observed hydrological behaviour.

**Pipeline impact:** New sub-stage alongside Stage 4; new output
`wofs_frequency_YYYY.tif`; Stage 5 gains WOfS frequency as an input (likely replacing
or complementing HAND).

### Priority 4 — GEDI canopy height

**What it adds:** Spaceborne lidar canopy height estimates at ~25 m footprint resolution
(GEDI L2A, 2019–present). Parkinsonia is typically 3–8 m tall — canopy height
discriminates it from low shrubs and grass (below ~2 m) and from tall closed-canopy
riparian forest (river red gum, large Melaleuca > 10 m). This is the only signal that
directly addresses canopy structure without requiring drone surveys.

**Implementation cost:** Higher. GEDI data is sparse (orbital track footprints, not
continuous coverage) and requires a separate fetch from NASA EarthData or via the
`gedi_l2a` DEA product. Coverage over the Mitchell catchment will be patchy — useful
as a soft constraint or prior rather than a per-pixel signal. Spatial interpolation or
kriging between footprints introduces uncertainty.

**Pipeline impact:** New optional Stage (e.g. Stage 4b); output is a sparse point layer
or interpolated raster. Most useful as an input to Stage 6 classifier rather than the
rule-based Stage 5.

### Signals assessed and rejected

- **SAR (Sentinel-1 C-band backscatter):** Can distinguish woody from non-woody
  vegetation but within the riparian woody stratum the structural differences between
  Parkinsonia and Melaleuca are below C-band discrimination threshold. Not worth the
  additional complexity.
- **Thermal / land surface temperature (Landsat Band 10, ECOSTRESS):** Parkinsonia's
  open canopy and transpiration rate produces a distinct thermal signature in theory, but
  at 100 m resolution a mixed pixel in the riparian corridor will be dominated by bare
  soil and water temperature rather than canopy temperature. Resolution is too coarse
  to be useful here.
- **Hyperspectral (DESIS, PRISMA):** Coverage over northern Queensland is too sparse
  and opportunistic to be reliable as a pipeline input.
- **Commercial high-resolution imagery (Planet, Maxar):** Crown-level discrimination
  becomes possible at <3 m resolution but requires a data purchase and is better
  handled at the drone survey stage (Stage 6).

---

## Stage 5: Sparse ALA training data in the Mitchell catchment

The ALA database contains only ~16 georeferenced *Parkinsonia aculeata* records within
the catchment bounding box. This is insufficient for reliable Random Forest training —
the model will technically run but cross-validation accuracy is unreliable and the
classifier is likely to be overfit or noisy.

Expanding the bbox to pull more records from broader Australia is not a valid fix,
because the other signals (NDVI anomaly, flowering index, distance to watercourse) are
only computed for the catchment — there would be no feature data to pair with distant
occurrence records.

### Partial mitigation implemented

`06_classifier.py` now supplements the ALA records with **ecological bootstrap
samples** — high-confidence synthetic presences and absences derived from known
Parkinsonia biology:

- **Synthetic presences**: within 500m of watercourse + NDVI anomaly in top quartile
  + flowering index above scene median + moderate NDVI median (0.15–0.65)
- **Synthetic absences**: beyond 2km from water, or bare ground, or low anomaly +
  low flowering + beyond 500m from water
- Synthetic samples are down-weighted at 0.3 vs 1.0 for real ALA records, so ground
  truth anchors the model

This brings the training set from ~16 to ~600+ samples but introduces circularity —
the features used to generate training labels are the same features the model learns
from. The model may become overconfident in the ecological prior rather than
discovering new signal.

### Remaining fixes needed

**Option 1 — Extend the GIS analysis to ecologically comparable areas**

Run stages 1–4 over additional catchments or regions with similar ecology (tropical
savanna, riparian corridors) where ALA has denser Parkinsonia sightings. Train the
Random Forest on the combined multi-region dataset, then apply it to the Mitchell
catchment. This is the most principled fix as it preserves the integrity of the
feature-label pairing.

**Option 2 — Supplement with manually collected presence records**

Incorporate field survey GPS points or manually digitised presence polygons from
land managers, NRM groups, or remote sensing interpretation. These can be merged with
the ALA records before training. A GeoPackage drop-in at
`{CACHE_DIR}/manual_occurrences.gpkg` would be a natural integration point in
`fetch_ala_occurrences.py`.
