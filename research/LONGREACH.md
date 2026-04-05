# Longreach Parkinsonia Infestation Site

## Confirmed infestation patch

Dense *Parkinsonia aculeata* infestation on floodplain/gilgai clay, visible in both
wet-season (green canopy) and dry-season (individual crowns distinguishable) imagery.

- **Centre:** -22.763402, 145.425009
- **bbox:** lon [145.4213, 145.4287], lat [-22.7671, -22.7597]
- **Size:** ~820 × 820 m
- **Quicklook:** `input-img/spot_longreach/qld_20cm_infestation.png` (4096×4096px, 0.2m/px)

### Spectral notes (Sentinel-2, Aug–Oct 2025)

- NIR reflectance consistently +0.02–0.04 higher than surrounding landscape across all
  19 dry-season scenes — stable signal
- Flowering index (FI) higher in April (post-wet, ~0.15 mean) than August–September
  (~-0.03 mean) — flowering window may be Jan–Apr for this latitude, not Aug–Oct
- Water mask (NIR DN < 1500) removes 0.2–4.3% of pixels; waterholes present in scene

---

## High-density scene

Cropped from the confirmed infestation patch — dense Parkinsonia crowns individually
visible at 20cm/px.

- **bbox:** lon [145.4240, 145.4250], lat [-22.7640, -22.7610]
- **Quicklook:** `outputs/Longreach-infestation-qglobe.png`

### Characterisation (20cm Queensland Globe imagery)

- Individual crowns clearly distinguishable — rounded green canopies against pale gilgai clay
- High density but not a solid canopy; significant bare soil between crowns
- Crown diameter approximately 1.5–3 m (~15–30 px at 20cm/px), consistent with mature Parkinsonia
- Uniform crown size and spacing throughout — a consistent, mature infestation
- Inter-crown soil is pale/grey gilgai clay — high visible reflectance, will mix into every S2 pixel
- A cleared track runs through the middle of the scene (thin dark feature)
- At S2 10m resolution, most pixels will be a mix of Parkinsonia crown and inter-crown gap;
  pixels crossing the track will have lower canopy fraction
- The majority of S2 pixels within the bbox contain significant Parkinsonia signal,
  though a small number of edge pixels or bare patches may not

### S2 dataset size (2020–2025 archive)

- **Pixels:** 374 (11 × 34 on the 10m S2 grid, covering the 110m × 340m bbox)
- **Observations per pixel:** 383–390 (median 387) over the full 6-year archive
- **Seasonal cadence:** ~3 usable acquisitions/month in wet season (Dec–Feb peak cloud),
  ~6/month in dry season (Jun–Oct) — August consistently hits the 5-day S2 revisit ceiling
- **Total rows:** ~144,800 (point × date)
- **Source file:** `data/longreach_pixels.parquet` — one row per (pixel, date), all 10 S2 bands plus quality scores

### S2 signal investigation notes

Every S2 pixel in this scene is a spectral mixture of Parkinsonia canopy and bare gilgai
clay — crown cover is roughly 30–40% of ground area. The signal we are looking for is
not pure Parkinsonia reflectance but the effect of that canopy fraction on pixel-level
spectral and temporal behaviour.

**Sub-pixel canopy fraction as a natural gradient**

Pixels vary in canopy fraction depending on where crowns happen to fall relative to the
10m grid. This creates a natural gradient from high-fraction (strongly Parkinsonia-like)
to low-fraction (mostly bare clay) within the bbox — without needing to sample external
negatives. Signal exploration should treat this as a continuum rather than binary
presence/absence.

**Priority signals to investigate (2020–2025 time series)**

1. **Dry-season NIR stability** — does elevated NIR persist across all dry seasons, or
   is it variable year to year? Stability across 5 years is a strong discriminator from
   ephemeral vegetation responses.

2. **Wet/dry seasonal amplitude** — how much does NIR or NDVI swing between wet and dry
   season compared to surrounding bare/grass pixels? Parkinsonia's deep roots sustain
   canopy through dry season; grasses collapse. Lower amplitude = more persistent canopy.

3. **Seasonality shape** — the full annual waveform (greenup timing, peak, recession)
   may be more discriminating than any single-date band value. Parkinsonia greenup
   likely leads or lags surrounding grass in a consistent, detectable way.

4. **Red-edge ratio (B07/B05)** — measures active chlorophyll independently of canopy
   structure. Senescent dry-season grasses collapse toward ratio ≈ 1; Parkinsonia
   retaining active chlorophyll stays elevated.

5. **SWIR water index (B08−B11)/(B08+B11)** — Parkinsonia's deep roots sustain higher
   canopy water year-round relative to surrounding dry grass and bare soil.

**What crown-level mixing implies for modelling**

A pixel with higher canopy fraction will show stronger signal on all of the above.
The classifier output will naturally approximate canopy fraction / infestation density
rather than hard presence/absence — which is the intended "Parkinsonia presence
probability" output.

---

## Galilee Basin East 2021 imagery

Queensland Government SISP aerial ortho, publicly available via WMS (CC-BY-SA).

- **Dataset name:** `Galilee_Basin_East_2021_20cm_SISP`
- **Resolution:** 20 cm/px
- **Capture dates:** July–October 2021 (dry season)
- **Extent:** lon [143.375, 147.469], lat [-24.001, -20.218]
- **Coverage:** ~454 km E-W × 420 km N-S (~190,000 km², irregular polygon)
- **WMS endpoint:** `https://spatial-img.information.qld.gov.au/arcgis/services/Basemaps/LatestStateProgram_AllUsers/ImageServer/WMSServer`
- **Layer name:** `LatestStateProgram_AllUsers`
- **Estimated size:** ~1.7 TB as COG JPEG; ~5 TB lossless

The Longreach infestation site falls within this extent. Captured in dry season —
Parkinsonia crowns individually resolvable (~10–20px diameter at 20cm/px), suitable
for hand-labelling polygons in QGIS.

### ALA sightings within this extent (2005–2020)

586 records total. Key clusters for ground-truth labelling:

| Cluster | Centroid (lat, lon) | n | Years |
|---------|-------------------|---|-------|
| 3 | -22.443, 144.652 | 41 | 2015–16 |
| 6 | -22.538, 144.560 | 36 | 2015 |
| 1 | -22.462, 144.775 | 16 | 2015 |
| 2 | -22.492, 144.724 | 12 | 2016 |
| 7 | -22.164, 144.645 | 13 | 2015 |
| 2005-0 | -22.115, 145.194 | 16 | 2005 |
| 4 | -20.920, 143.856 | 11 | 2014 |
