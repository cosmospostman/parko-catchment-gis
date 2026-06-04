# Mitchell River Training Region Design

## Context

V10 was trained with zero Mitchell River presence data and a single small absence region
(`mitchell_river_absence`, ~1 km² riparian patch at 142.15°E, 15.80°S). When run over the
broader Mitchell River mouth and surrounds (Gulf of Carpentaria coast), the model produces
false positives on mangroves, bare floodplain ground, and native riparian woodland — the
dominant non-Parkinsonia land covers in this landscape.

The root cause: monsoonal vegetation pushes `rec_p` (wet–dry NDVI amplitude) far beyond
the training range, so the model cannot discriminate Parkinsonia from any other green
vegetation. The fix is adding Mitchell-region training data covering confirmed presence
and hard-negative absence for the specific false-positive cover types.

---

## Using Claude to inspect candidate bboxes

Claude can query the pixel timeseries for any bbox you draw in the UI. To use this:

1. Draw a candidate bbox in the UI
2. In a new chat, say: **"Inspect bbox: [lon_min, lat_min, lon_max, lat_max]"**

Claude will run:

```bash
source .venv/bin/activate && python utils/pixel_timeseries.py \
  --root /mnt/external/chunkstore --year 2025 \
  --tile 54LWH --bbox <lon_min,lat_min,lon_max,lat_max>
```

If the result is empty, Claude retries with `--tile 54KWG`. Tile coverage:
- `54LWH`: roughly lat -15.5 to -16.3, lon 141.5–142.1 (Mitchell mouth and north)
- `54KWG`: roughly lat -16.3 to -16.7, lon 141.3–141.9 (further south/inland)

Only 2025 data is available in the chunkstore at present.

### Signal profiles for each cover type

Claude interprets the output against these expected profiles:

| Cover type | Wet NDVI | Dry NDVI | Wet–dry Δ | MAVI | Dry VH/VV |
|---|---|---|---|---|---|
| **Parkinsonia presence** | 0.75–0.90 | 0.20–0.35 | 0.50–0.65 | moderate (0.45–0.60) | -3 to -6 dB (stable) |
| **Mangrove absence** | 0.90–0.97 | 0.65–0.75 | < 0.30 | high stable | -3 to -5 dB |
| **Bare ground / floodplain** | 0.50–0.75 | 0.10–0.15 | > 0.55 | low (< 0.25 dry) | -7 to -9 dB dry |
| **Native riparian / savanna** | 0.70–0.85 | 0.25–0.45 | 0.40–0.55 | moderate | -4 to -6 dB |

Key discriminators:
- **Mangrove vs Parkinsonia**: mangrove wet NDVI > 0.90, dry floor > 0.60, stable VH/VV —
  Parkinsonia wet NDVI < 0.90, dry floor < 0.35
- **Bare ground vs Parkinsonia**: bare ground dry VH/VV crashes to -7 to -9 dB (no woody
  structure), Parkinsonia VH/VV stays at -3 to -6 dB year-round
- **Native woodland vs Parkinsonia**: these are spectrally similar and the hardest to
  separate — rely on WMS imagery confirmation; do not commit presence without visual
  verification

### Bbox quality checks

Before committing any bbox, Claude will report:

- **p25–p75 width**: narrow band (< 0.05) = spatially clean patch; wide band (> 0.10)
  suggests spatial mixing — shrink the bbox or move it
- **Anomalous single-date dips**: occasional NDVI near zero mid-season = cloud/shadow
  contamination on one scene, not a real signal — safe to ignore
- **Wet-season NDVI anomalously high** (> 0.90 for a supposed presence region): likely
  clipping a mangrove or dense riparian edge — inspect imagery and tighten the bbox

---

## What we're looking for

### Presence regions

Target: dense, visually unambiguous Parkinsonia crown clusters confirmed on WMS imagery.
Each bbox should be spatially compact (a few hundred metres per side, ~0.001° per edge).

Expected spectral profile:
- Wet NDVI 0.75–0.90, peaking late wet season (Apr–May)
- Dry NDVI 0.20–0.35 (substantial crash — deciduous leaf drop)
- Wet–dry Δ ≥ 0.50
- MAVI/NDVI ratio ~0.60–0.70 in wet season (woody canopy water content)
- VH/VV -3 to -6 dB, **seasonally stable** — persistent woody structure even when
  optically bare. This is the key radar discriminator vs bare ground.

**Variation needed**: aim for presence regions at different positions along the river
(different reaches, not clustered in one spot) and if possible covering both dense
interior crown patches and slightly sparser edge patches. Dense-only training risks
missing real infestations at lower canopy cover.

**Quantity**: 3–5 presence regions for training, 1–2 held out for validation.
Each region typically yields 20–100 pixels × multiple years. Target ~300–500 presence
training pixels total across all regions and years.

### Absence — mangrove (highest priority)

Mangroves are the dominant false-positive class. Spatially clustered along tidal channels
and the Gulf shoreline — identifiable in WMS as dark-green dense fringe.

Expected spectral profile:
- Wet NDVI 0.90–0.97 (higher than any Parkinsonia bbox)
- Dry NDVI > 0.65 (stays green — no wet–dry crash)
- Wet–dry Δ < 0.30
- MAVI high and stable
- VH/VV -3 to -5 dB (closed woody canopy), no seasonal swing

**Variation needed**: 2–3 mangrove absence regions at different positions along the
coast/tidal channels to capture spatial variability in mangrove density. One tight-fringe
patch and one broader stand would cover the range.

**Quantity**: 2–3 training regions, 1 validation. Target ~200–400 absence pixels from
mangrove across regions and years.

### Absence — bare ground / floodplain (high priority)

Bare floodplain and exposed seasonal ground score high in v10 — a hard false-positive
class the model has never seen.

Expected spectral profile:
- Wet NDVI 0.50–0.75 (moderate greenup from annual grass flush)
- Dry NDVI 0.10–0.15 (near-bare, effectively no canopy)
- Wet–dry Δ > 0.55
- VH/VV crashes in dry season to -7 to -9 dB (no woody structure)

The VH/VV dry-season crash is the clearest discriminator from Parkinsonia, which
maintains -3 to -6 dB year-round due to persistent woody stems.

**Variation needed**: 1–2 regions; one on exposed interfluve and one on seasonally
inundated floodplain fringe if available.

**Quantity**: 1–2 training regions. Target ~150–300 absence pixels.

### Absence — native riparian / savanna woodland

Spectrally the most similar to Parkinsonia — deciduous woodland with a real wet–dry NDVI
swing. Important for teaching the model local contrast near presence patches.

Expected spectral profile:
- Wet NDVI 0.70–0.85
- Dry NDVI 0.25–0.45 (higher floor than Parkinsonia — retains more dry-season greenness)
- VH/VV -4 to -6 dB, slight seasonal variation but not the bare-ground crash
- Wider p25–p75 band than Parkinsonia (more canopy structural heterogeneity)

**Variation needed**: 1–2 regions adjacent to or near presence patches so the model
learns local contrast. The existing `mitchell_river_absence` (142.15°E) covers native
riparian; add more only if presence patches are on different river reaches.

**Quantity**: 1–2 training regions (the existing region counts). Target ~150–250 pixels.

---

## Summary target for v11

Based on the v10 training set (~390K presence / ~720K absence train pixel-years), Mitchell
additions should be ~0.5–1% of the existing corpus per class — enough to shift the
monsoonal decision boundary without destabilising existing performance. At 6 years and
10 m pixels this translates to roughly 30–70 spatial pixels per training bbox and 15–30
per val bbox.

| Category | Train bboxes | Train pixel-years | Val bboxes | Val pixel-years |
|---|---|---|---|---|
| Presence | 3–4 | 2,000–4,000 | 1–2 | 500–1,000 |
| Mangrove absence | 2–3 | 1,500–3,000 | 1 | 400–800 |
| Bare ground absence | 1–2 | 1,000–2,000 | 1 | 300–600 |
| Riparian absence | 1–2 | 1,000–2,000 | 1 | 300–600 |

---

## Candidate regions

Working table for evaluating candidate bboxes before committing to training.yaml. "Pixels" is spatial pixel count (single year, ~10 m); multiply by years in training range for pixel-years.

| # | Category | Bbox | Pixels | Signal quality | Notes |
|---|---|---|---|---|---|
| 2 | Mangrove absence | [141.432703, -15.823780, 141.434759, -15.819690] | — | Excellent: wet NDVI 0.93–0.96, dry floor 0.78–0.85, Δ ~0.12, VH/VV stable −5.3 to −6.1 dB, IQR < 0.05 | Train. Textbook mangrove. Two cloudy scene dips (Feb-17, Oct-10) — scene artefacts, safe to ignore. |
| 3 | Mangrove absence | [141.522892, -15.867121, 141.526076, -15.864479] | — | Good: wet NDVI 0.86–0.87, dry floor 0.75–0.79, Δ ~0.11, VH/VV stable. IQR slightly wider in Feb–Mar. | Train. Fringe stand — wet NDVI just below 0.90 but visually confirmed dense mangrove. Different reach from Bbox 2. |
| 4 | Mangrove absence | [141.425301, -16.057730, 141.427833, -16.054378] | — | Good: wet NDVI 0.88–0.94, dry floor 0.63–0.72, Δ ~0.22, VH/VV stable −5.6 to −6.4 dB. IQR moderate (~0.05–0.12). | Train. Southern reach — adds spatial diversity. Slightly more heterogeneous but unambiguously mangrove. |
| 5 | Mangrove absence | [141.416778, -15.894227, 141.419044, -15.891997] | — | Excellent: wet NDVI 0.84–0.93, dry floor 0.77–0.87, Δ ~0.12, VH/VV stable −4.5 to −5.6 dB, IQR ~0.02–0.04. | Val. Separate tidal inlet between Bboxes 3 and 4 — geographically independent from all train patches. Several cloudy scene dips (Feb-17, May-13, Oct-10) — scene artefacts only. |
| 6 | Bare ground absence | [141.490979, -16.057881, 141.494836, -16.055694] | — | Excellent: wet NDVI 0.49–0.55, dry floor 0.15–0.19, VH/VV crashes to −7.2 to −8.4 dB dry (Jul–Nov), IQR tight ~0.01–0.03. | Train. Textbook bare ground radar signature. Southern reach near Bbox 4 mangroves. Wet–dry Δ modest (~0.33) because wet peak is already suppressed — consistent with grass flush over bare substrate, not a canopy. Single cloud dip Sep-05, ignore. |
| 7 | Bare ground absence | [141.546159, -15.953764, 141.548024, -15.951436] | — | Good: wet NDVI 0.52–0.55, dry floor 0.17–0.19, VH/VV reaches −5.7 to −6.2 dB dry (marginal — short of −7 to −9 dB target), IQR tight ~0.01–0.02. | Val. NDVI profile consistent with bare ground; radar slightly warmer than ideal, suggesting sparse grass/forb retention. Geographically independent from Bbox 6. |
| 8 | Bare ground absence | [141.637069, -15.392288, 141.640681, -15.389818] | — | Excellent: wet NDVI 0.68, dry floor 0.09–0.14, VH/VV crashes to −7 to −9.8 dB dry (Sep–Nov), IQR tight ~0.02–0.05. | Train. Northern floodplain — different reach and inundation regime from Bbox 6. Deep dry-season radar crash and very low NDVI floor suggest seasonally inundated floodplain that dries completely. Complements Bbox 6 spatially and spectrally. |
| 9 | Riparian absence | [141.757433, -15.904606, 141.761327, -15.902438] | — | Good: wet NDVI 0.73–0.82, dry floor 0.47–0.49, Δ ~0.33, VH/VV stable −4.4 to −6.1 dB year-round, IQR moderate ~0.14–0.18. | Train. Textbook riparian/savanna — dry-season NDVI floor well above Parkinsonia range (0.20–0.35), no VH/VV crash. Moderate IQR consistent with mixed woodland canopy. Oct-30 scene dip (IQR spike) is cloud artefact, ignore. |
| 10 | Riparian absence | [141.989778, -15.915257, 141.993131, -15.912756] | — | Good: wet NDVI 0.67–0.70, dry floor 0.36–0.37, Δ ~0.33, VH/VV stable −4.0 to −5.5 dB year-round, IQR tight ~0.03–0.06. | Train. Drier open woodland end-member — dry-season floor lower than Bbox 9 but radar confirms persistent woody structure (no crash). Tight IQR indicates spatially clean patch. Spectrally proximal to Parkinsonia at dry-season minimum; included deliberately to teach the lower bound of riparian absence. Geographically independent from Bbox 9 (141.99°E vs 141.76°E). |
| 11 | Water absence | [141.563041, -15.858784, 141.563571, -15.857719] | — | Excellent: NDVI −0.46 to −0.94 year-round, VH/VV noisy −0.7 to −3.4 dB (specular SAR artefact on narrow creek). | Train. ~60 px. Permanent creek — strongly negative NDVI unambiguous. SAR noise is expected specular reflection from narrow water body, not a signal problem. |
| 12 | Water absence | [141.561771, -15.858072, 141.562854, -15.857077] | — | Excellent: NDVI −0.34 to −0.85 year-round, VH/VV noisy (specular). | Train. ~110 px. Same creek reach as Bbox 11, different meander segment. |
| 13 | Water absence | [141.566770, -15.861490, 141.567707, -15.860561] | — | Excellent: NDVI −0.43 to −0.92 year-round, VH/VV noisy (specular). | Train. ~90 px. Adjacent creek reach — adds spatial coverage. Combined train water pixels ~260. |
| 14 | Water absence | [141.563797, -15.853290, 141.565461, -15.852743] | — | Good: NDVI −0.12 to −0.70 year-round (water-dominated), one Dec anomaly NDVI +0.30 suggesting minor vegetated bank contamination. | Val. ~90 px. Geographically independent reach upstream of train patches. Dec anomaly is edge contamination — majority of pixels cleanly water. |
| 15 | Riparian absence | [141.626992, -15.864820, 141.628852, -15.863669] | — | Good: wet NDVI 0.79–0.89, dry floor 0.48–0.53, VH/VV stable −3.7 to −6.0 dB year-round, IQR tight ~0.04–0.07. | Val. ~230 px. Clean riparian woodland — dry floor well above Parkinsonia range, no radar crash. Geographically independent from train patches (141.63°E). May-13 scene dip is cloud artefact, ignore. |
| 16 | Presence | [141.363789, -16.336770, 141.365969, -16.335197] | — | Excellent: wet peak 0.875 (Feb-22, IQR 0.019), dry floor 0.29–0.31, Δ ~0.58, VH/VV stable −4.1 to −5.5 dB year-round (no crash), IQR tight throughout. | Train. Textbook Parkinsonia profile. Wet peak clears 0.75+ bar cleanly; dry floor in target band; radar persistent. Southern reach (54KWG tile, lat −16.34°S). |
| 17 | Presence | [141.364065, -16.339440, 141.364817, -16.336892] | — | Excellent: wet peak 0.891 (Feb-22, IQR 0.013), dry floor 0.30–0.33, Δ ~0.56, VH/VV stable −3.8 to −5.5 dB year-round (no crash). IQR widens May–Jun (~0.13–0.15) during senescence. | Val. Wet peak and dry floor both textbook; IQR widening at senescence suggests within-patch density variation — suitable for validation. Adjacent to Bbox 16, geographically paired. |
| 18 | Presence | [141.400465, -16.324751, 141.402291, -16.323115] | — | Excellent: wet peak 0.899 (Apr-28, IQR 0.013), dry floor 0.24–0.26, Δ ~0.65, VH/VV stable −4.6 to −5.4 dB year-round (no crash). Tightest IQR and deepest dry floor in the southern cluster. | Train. Southern reach (54KWG). Best-quality signal of the batch — clean patch, large seasonal swing, woody radar signature year-round. |
| 19 | Presence | [141.379285, -16.345988, 141.380974, -16.341960] | — | Excellent: wet peak 0.881 (Apr-28, IQR 0.016), dry floor 0.27–0.31, Δ ~0.60, VH/VV stable −4.1 to −4.9 dB year-round (no crash). | Train. Southern reach (54KWG), different sub-patch ~2.1 km from Bbox 18. Paired to add intra-reach spectral diversity. |
| 20 | Presence | [141.394728, -16.320307, 141.398059, -16.319024] | — | Excellent: wet peak 0.868 (Apr-23/28, IQR 0.013), dry floor 0.25–0.30, Δ ~0.59, VH/VV stable −4.3 to −5.5 dB year-round (no crash). Flagged as sparse patches by observer. | Val. Southern reach (54KWG), geographically close to Bbox 18 but spectrally independent — sparse canopy cover makes this a useful sparse-presence validation case. |
| 21 | Presence | [141.389167, -16.199947, 141.390500, -16.197966] | — | Good: wet peak 0.831 (Feb-17, IQR 0.014), dry floor 0.25–0.28, Δ ~0.56, VH/VV stable −3.9 to −4.9 dB year-round (no crash). Very tight IQR throughout. | Train. Middle reach (54LWH, lat −16.20°S) — distinct river section ~13 km north of southern cluster. Adds reach diversity. |
| 22 | Presence | [141.417057, -16.209358, 141.420611, -16.206871] | — | Good: wet peak 0.820 (Feb-17/Apr-28, IQR 0.014), dry floor 0.27–0.30, Δ ~0.55, VH/VV stable −4.2 to −5.1 dB year-round (no crash). Tight IQR, slightly lower wet peak. | Val. Middle reach (54LWH), ~2.8 km east of Bbox 21 — geographically paired for middle-reach validation. |
| 23 | Presence | [141.463662, -15.990899, 141.465735, -15.987729] | — | Excellent: wet peak 0.868 (Feb-27/Mar-9, IQR 0.025), dry floor 0.26–0.28, Δ ~0.55, VH/VV stable −4.0 to −5.0 dB year-round (no crash). Trimmed from original K bbox to tighten patch. | Train. Northern reach (54LWH, lat −15.99°S) — ~22 km north of middle cluster, different river section. Adds the largest geographic separation in the presence set. |
| 24 | Presence | [141.474997, -15.959035, 141.477229, -15.956661] | — | Good: wet peak 0.793 (Feb-17, IQR 0.013), dry floor 0.26–0.30, Δ ~0.51, VH/VV stable −4.3 to −5.4 dB year-round (no crash). Trimmed from original N bbox to remove edge pixels. | Val. Northern reach (54LWH, lat −15.96°S) — furthest north of all presence candidates, ~3 km from Bbox 23. Geographically independent val for the northern reach. |

---

## Adding to training.yaml

Add new regions to [data/locations/training.yaml](data/locations/training.yaml) after
line 343 (the existing `mitchell_river_absence` entry), following this schema:

```yaml
- id: mitchell_presence_1
  name: "Mitchell River — presence 1"
  label: presence
  bbox: [lon_min, lat_min, lon_max, lat_max]
  years: [2020, 2021, 2022, 2023, 2024, 2025]
  tags: [monsoonal, riparian]

- id: mitchell_absence_mangrove_1
  name: "Mitchell River — absence (mangrove 1)"
  label: absence
  bbox: [lon_min, lat_min, lon_max, lat_max]
  years: [2020, 2021, 2022, 2023, 2024, 2025]
  tags: [monsoonal, mangrove]

- id: mitchell_absence_bare_1
  name: "Mitchell River — absence (bare ground 1)"
  label: absence
  bbox: [lon_min, lat_min, lon_max, lat_max]
  years: [2020, 2021, 2022, 2023, 2024, 2025]
  tags: [monsoonal, bare_soil]
```

Then add the new region IDs to the experiment's `train_region_ids` list in the v11
experiment file. Run the pixel collector to fetch training data, retrain, and evaluate
the Kowanyama/Mitchell probability map — mangrove fringes and bare interfluve should
collapse toward 0, confirmed presence patches should read > 0.7.

---

## Verification

Score the Mitchell River bbox after retraining and compare the probability map against
WMS imagery:
- Mangrove fringes and bare interfluve: < 0.2
- Confirmed presence patches: > 0.7
- Check that existing site val metrics (Etna, Landsend, Frenchs val) do not regress
