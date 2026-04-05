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
- **Quicklook:** `outputs/Longreach - test.PNG`

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
