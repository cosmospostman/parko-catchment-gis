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
