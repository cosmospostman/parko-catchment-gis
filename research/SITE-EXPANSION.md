# Site Expansion Plan — Galilee Basin East

## Purpose

Apply the Longreach-derived analysis pipeline to additional sites within the
Queensland Globe Galilee Basin East 2021 WMS coverage. The goals are:

1. **Test cross-site transfer** of the Longreach-trained classifier — do the
   feature distributions for Parkinsonia and background shift with soil type,
   climate regime, or vegetation structure?
2. **Build a multi-site feature dataset** that will support a more robust
   classifier prototype than a single-site logistic regression.
3. **Identify failure modes** — sites where the Longreach thresholds produce
   systematic false positives or misses, and diagnose why.

---

## ALA occurrence data

Source: `outputs/australia_occurrences/ala_australia_occurrences.gpkg`
(39,863 Australia-wide records, columns: stateProvince, decimalLatitude,
decimalLongitude, coordinateUncertaintyInMeters)

Within the Galilee Basin East region (lon 144–147, lat -23.5–-20), 862 records
exist across 15 spatial clusters (DBSCAN eps=0.05°, min_samples=3). The clusters
below are within or near the 2021 WMS coverage and have sufficient record density
to anchor a site fetch.

---

## Candidate sites

Sites are ranked by priority. All coordinates are WGS84.

---

### Site 1 — Barcaldine corridor (Priority 1)

| Parameter | Value |
|-----------|-------|
| ALA centroid | −21.438, 145.004 |
| ALA records | 472 |
| Coordinate uncertainty (median) | 100 m |
| Cluster spread | ~23 km N–S × ~19 km E–W |
| Proposed fetch bbox | 144.88, −21.55, 145.07, −21.30 |
| Approximate area | ~19 km × ~28 km |

**Why first:** Largest ALA cluster in the region by a wide margin. The spread
(~23 km N–S) suggests a diffuse infestation across multiple land-cover types
rather than a single dense patch — likely a mix of riparian colonisation and open
floodplain infestations. If the classifier handles this heterogeneity, it
generalises. The 100m uncertainty is adequate for confirming presence at S2
resolution.

**Context:** Barcaldine is ~180 km north of Longreach on the Barcoo River system.
Soil types and vegetation structure will differ from the Longreach gilgai clay
site. This is the most important cross-environment test available within the WMS
coverage.

**Fetch strategy:** The bbox is large (~530 km²). Consider fetching at the full
bbox but checking pixel observation counts before committing — the 2020–2025 S2
archive should cover it but cloud fraction may be higher than Longreach.
Alternatively, fetch a ~5 km × 5 km core sub-bbox centred on the ALA centroid
first as a pilot, then expand if results are coherent.

---

### Site 2 — Aramac Road cluster (Priority 2)

| Parameter | Value |
|-----------|-------|
| ALA centroid | −22.104, 145.205 |
| ALA records | 109 |
| Coordinate uncertainty (median) | 100 m |
| Cluster spread | ~6 km N–S × ~4 km E–W |
| Proposed fetch bbox | 145.16, −22.17, 145.24, −22.07 |
| Approximate area | ~9 km × ~11 km |

**Why second:** Compact cluster with 109 records and tight spatial spread (~6 km
N–S), suggesting a more localised infestation similar in character to Longreach.
Located ~120 km SSW of Barcaldine. The 100m uncertainty is adequate. A tighter
cluster is easier to interpret as a first cross-site validation than the diffuse
Barcaldine population.

---

### Site 3 — Jericho area cluster (Priority 3)

| Parameter | Value |
|-----------|-------|
| ALA centroid | −22.452, 144.693 |
| ALA records | 78 |
| Coordinate uncertainty (median) | 1 m |
| Cluster spread | ~13 km N–S × ~18 km E–W |
| Proposed fetch bbox | 144.57, −22.53, 144.80, −22.36 |
| Approximate area | ~21 km × ~19 km |

**Why third:** The 1m coordinate uncertainty indicates GPS-logged field survey
records — the highest spatial precision in the Galilee East dataset. This makes
it the most reliable ground truth for confirmed-presence pixels. The spread is
larger than Site 2, suggesting a riparian corridor infestation pattern rather
than a single patch.

**Note:** This cluster overlaps with or is adjacent to the cluster previously
identified in LONGREACH-STAGE2.md as the Priority 3 candidate (−22.443, 144.652,
n=41, 2015–16). That record count is a subset of the 78 records here; the
boundary is a clustering artefact. The full 78-record cluster should be used.

---

### Site 4 — Muttaburra (Priority 4)

| Parameter | Value |
|-----------|-------|
| ALA centroid | −22.538, 144.558 |
| ALA records | 37 |
| Coordinate uncertainty (median) | 1 m |
| Cluster spread | ~4 km N–S × ~7 km E–W |
| Proposed fetch bbox | 144.548274, −22.546983, 144.567726, −22.529017 |
| Approximate area | 2 km × 2 km (~40,000 pixels, comparable to Longreach) |

**Why fourth:** High-precision GPS records (1m uncertainty). Small, compact
cluster. Located ~30 km SW of Site 3 — close enough that the two sites could
be jointly analysed if both fetch successfully.

**Fetch note:** Bbox is a 2 km × 2 km square centred exactly on the ALA
centroid (−22.538000, 144.558000), giving ~40,000 S2 pixels and ~1.9 GB raw
parquet — the same scale as the Longreach expansion fetch. No stride needed;
fetch at native 10 m resolution.

---

### Site 5 — Longreach South cluster (Priority 5)

| Parameter | Value |
|-----------|-------|
| ALA centroid | −22.773, 145.428 |
| ALA records | 6 |
| Coordinate uncertainty (median) | 13 m |
| Cluster spread | ~2 km N–S × ~1 km E–W |
| Proposed fetch bbox | 145.41, −22.79, 145.45, −22.75 |
| Approximate area | ~4 km × ~4.5 km |

**Why fifth:** Only 6 records — marginal for a standalone site — but it is the
closest ALA cluster to the Longreach training site (~20 km SSE along the Thomson
River). If this site shows similar feature distributions to Longreach, it
corroborates the classifier without introducing new soil/climate variation. If it
shows differences, the distance and river corridor provide a mechanistic
explanation. Treat as a supplementary validation rather than a primary new site.

---

### Site 6 — Kowanyama (Priority 6)

| Parameter | Value |
|-----------|-------|
| Location | Mitchell River delta, Gulf of Carpentaria coast |
| Proposed fetch bbox | 141.47528, −15.51051, 141.66922, −15.41736 |
| Approximate area | ~21 km × ~10 km |
| Source | Queensland Globe WMS export (14.2 m/px, EPSG:3857) |

**Why included:** Reported to be a large-scale infestation. The image shows dense
riparian Parkinsonia along the lower Mitchell River and its distributaries, with
the characteristic dark-green crown texture visible against bare/sparse floodplain
interfluve. The coastal Gulf setting (monsoonal climate, seasonally inundated
floodplain) represents a very different environment from the Galilee Basin
clay-plain sites — this is a high-value out-of-distribution test for the
classifier.

**Note:** This site is outside the Queensland Globe Galilee Basin East 2021 WMS
footprint. Confirm which WMS layer covers this region before fetching S2 data.
Sentinel-2 coverage should be fine; cloud fraction will be higher in the wet
season — restrict fetches to dry season (May–October).

---

## Excluded sites

| Cluster | Reason for exclusion |
|---------|---------------------|
| −20.665, 146.995 (n=28) | East of Galilee Basin, different bioregion (Burdekin) |
| −20.752, 146.977 (n=26) | Same — Burdekin bioregion |
| −23.418, 144.249 (n=8) | South of likely WMS coverage; low record count |

---

## Execution plan

For each site, in priority order:

1. **Visual check** — Open the proposed fetch bbox in Queensland Globe WMS before
   fetching. Confirm Parkinsonia crown structure is visible in the 20cm imagery at
   the ALA centroid location. If the site appears to have been treated (no crowns
   visible), note it and move to the next site.

2. **Fetch S2 time series** — Use `scripts/collect_pixel_observations.py` with the
   proposed bbox, date range 2020-01-01–2025-12-31, `--cloud-max 30`. Output to
   `data/<site_id>_pixels.parquet`.

3. **Run expansion pipeline** — Apply `longreach/expansion-map.py` (or a
   generalised version of it) to the new parquet. This produces per-pixel
   `(nir_cv, rec_p, re_p10)` features and scores each pixel with the
   Longreach-trained logistic regression.

4. **Evaluate transfer** — Check whether:
   - ALA-confirmed pixels score high (prob ≥ 0.5)
   - Open grassland pixels in the same bbox score low (prob ≤ 0.1)
   - The spatial pattern of high-scoring pixels is coherent with visible crown
     structure in the WMS imagery (accounting for the ~4-year imagery lag)

5. **Anchor calibration if needed** — If the Longreach boundary misclassifies
   systematically (e.g. all probabilities compressed toward 0.5), select a small
   number of confirmed-presence and confirmed-absence pixels from the new site
   and retrain or fine-tune the boundary. Record the feature-space shift.

---

## What to look for across sites

| Question | Diagnostic |
|----------|-----------|
| Do Parkinsonia `nir_cv` values shift between sites? | Compare infestation-pixel nir_cv distributions across sites |
| Does `rec_p` compress in wetter/drier years? | Compare rec_p distributions vs mean annual rainfall at each site |
| Is native riparian woodland correctly scored low at all sites? | Visual check of high-scoring pixels against WMS imagery in riparian corridors |
| Do the two GPS-survey sites (Site 3, Site 4) show tighter infestation clusters than the 100m-uncertainty sites? | Compare spatial coherence of high-scoring pixels |

---

## Relationship to LONGREACH-STAGE2.md priorities

- **Priority 3** (second site validation) → Site 3 (Jericho) or Site 2 (Aramac
  Road) depending on WMS visual check result
- **Priority 5** (native riparian ground truth) → All sites provide additional
  native riparian pixels if riparian woodland is visible in the WMS imagery; Sites
  1 and 3 are most likely given their spread along river corridors
