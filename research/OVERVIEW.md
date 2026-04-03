# Research Overview: Spectral Time Series Approach to Parkinsonia Detection

## Motivation

The current pipeline detects *Parkinsonia aculeata* using single-season NDVI anomaly composites and a rule-based plausibility map. Two weaknesses prompted this investigation:

1. **The 2025 wet-year problem:** An anomalously wet season suppressed the NDVI anomaly signal because native vegetation also stayed green, removing the contrast that makes Parkinsonia detectable.
2. **Species discrimination:** Co-occurring riparian species (Melaleuca, Eucalyptus, Acacia) can produce similar spectral signatures, leading to false positives.

The proposed solution is to exploit the ~20,000 ALA occurrence records nationally as a labelled spectral library, and to move from single-season composites to **individual-acquisition time series analysis**.

---

## Core Investigative Approach

Use ALA-confirmed Parkinsonia locations to examine what the full Sentinel-2 time series looks like at those points. Rather than compositing over a window (which smears temporal signals), extract per-acquisition values and analyse the resulting waveform.

---

## Key Biological Signals and Satellite Detectability

### 1. Flowering flush (Aug–Oct, tropical QLD)
- Parkinsonia produces dense yellow flowers on nearly leafless branches — brief, synchronised, driven by photoperiod
- Elevates green and red bands, suppresses NIR → distinctive green/NIR ratio spike
- Signal of interest: a **sharp, brief, annually-recurring spike** in the flowering index
- Bands to investigate: B3 (green), B4 (red), B5/B6 red-edge

### 2. Dry-season greenness persistence ("stays green" anomaly)
- Parkinsonia is phreatophytic — deep taproot maintains canopy greenness after natives senesce
- Already exploited in the NDVI anomaly layer
- **Multi-year persistence** is more reliable than single-season anomaly — a pixel anomalously green across multiple dry seasons is far stronger evidence than a single year

### 3. Canopy structure and texture
- Fine-textured, semi-transparent pendant canopy creates characteristic sub-pixel shadow/reflectance mix
- GLCM texture features may be more stable than spectral indices for separating from dense-canopy natives

### 4. Post-wet-season green-up timing
- Parkinsonia flushes rapidly after inundation, producing an early-dry-season greenness peak (May–June) before grasses senesce
- Timing of NDVI peak may be detectably earlier than surrounding vegetation

### 5. Post-disturbance resilience
- Resprouts aggressively after fire; native grasses show dramatic NDVI suppression post-fire, Parkinsonia does not
- Multi-year NDVI stability in fire-affected areas could be discriminative

---

## Co-occurring Species: Discriminating Characteristics

### Melaleuca / Paperbark
- Evergreen, phreatophytic — persistently high NDVI, similar to Parkinsonia
- **Discriminators:** Steeper red-edge slope (dense, waxy leaves); lower SWIR (drier leaves); more homogeneous texture; no synchronised Aug–Oct flowering spike

### Riparian Eucalyptus
- NDVI typically declines through dry season — contrast with Parkinsonia generally works
- In wet years both stay green, which is why 2025 was problematic
- **Discriminator:** Rate of NDVI decline through dry season (July vs. September values)

### Native grasses and sedges
- High inter-annual NDVI variance — strongly rainfall-coupled
- Very different SWIR behaviour (dry grass has extremely high SWIR reflectance)
- **Discriminator:** Multi-year NDVI consistency (Parkinsonia stable, grasses variable)

### Acacia / Vachellia (hardest case)
- Some acacias also yellow-flowering in similar seasons
- **Discriminators:** Parkinsonia's flowering spike is stronger, more spectrally consistent, and more calendar-locked year-to-year; Acacia canopy is grey-green with different red-edge behaviour; flowering duration and timing differ

---

## The Mixed-Pixel Problem

At 10m resolution, a "Parkinsonia pixel" almost certainly contains multiple species. This is not a flaw — it's the nature of the operational detection target.

**Key implications:**
- Don't try to learn pure-Parkinsonia spectral signatures; learn what a Parkinsonia-*containing* pixel looks like vs. one without it
- Features should respond linearly to fractional cover, not threshold-based
- Multi-year NDVI persistence is well-suited to mixed pixels — even 40% Parkinsonia in a pixel produces a consistent anomaly
- Texture features implicitly encode density/cover fraction
- Prefer spatially clustered ALA training points (likely denser stands) over isolated single records

---

## Regional Phenological Variation

Parkinsonia flowering is triggered by wet-to-dry season transition, which operates on different calendars across Australia:

| Region | Typical flowering window | Notes |
|---|---|---|
| Tropical QLD (Mitchell catchment) | Aug–Sep | Sharp monsoon boundary, consistent |
| Northern Territory | Jul–Aug | Earlier dry season establishment possible |
| Subtropical QLD/NSW | May–Jul | No monsoon; several months earlier |
| Arid zone (SA/WA) | Opportunistic post-rain | Fixed window meaningless |

**Solution:** Search for flowering peak within a wide window (June–November) rather than compositing over a fixed window. Extract peak value, day-of-year of peak, and spike duration as features. Inter-annual consistency of peak timing (low SD of peak DoY across years) is a strong discriminator.

---

## From Composites to Time Series

**The critical methodological shift:** instead of averaging over a seasonal window, extract per-acquisition values and analyse the waveform shape.

### What the time series gives you that a composite cannot:

| Feature | What it captures |
|---|---|
| Peak flowering index value | Spike amplitude — sensitive to fractional cover |
| Day-of-year of peak | Timing — comparable across years and regions |
| Spike duration (days above threshold) | Species discrimination (Parkinsonia is brief and sharp) |
| Spike rise/fall rate | Biological consistency check |
| Inter-annual SD of peak DoY | Calendar-locking — strongest discriminator |
| Fraction of years with detectable spike | Robustness metric across wet/dry years |

### Wet-year robustness
Multi-year consistency solves the 2025 problem: you're characterising a waveform shape and its calendar consistency, not comparing a single season against a baseline. Wet years contribute one data point in the time series, not the whole signal.

---

## Proposed Feature Set (expanded from current pipeline)

| Feature | Source | New? |
|---|---|---|
| NDVI anomaly (current year) | Existing | No |
| NDVI persistence score (N years above threshold) | Multi-year Sentinel-2 | Yes |
| Rainfall-normalised anomaly | NDVI anomaly weighted by seasonal rainfall | Yes |
| Peak flowering index | Individual acquisitions, not composite | Yes |
| Day-of-year of flowering peak | Time series extraction | Yes |
| Spike duration | Time series extraction | Yes |
| Inter-annual SD of peak DoY | Multi-year time series | Yes |
| Red-edge bands (B5/B6/B7) at peak | Individual acquisition | Yes |
| SWIR (B11/B12) dry-season mean | Sentinel-2 | Yes |
| HAND | Existing | No |
| GLCM texture | Existing | No |
| Distance to watercourse | Existing | No |

---

## Training Data Strategy

### Development phase (process verification)
- **Goal:** Verify the time series extraction, waveform feature derivation, and RF training pipeline work correctly. Overfitting is acceptable — even desirable as a signal that features are discriminative.
- **Geometry:** Single densest cluster of ALA records (Mitchell catchment or nearest equivalent in tropical QLD)
- **Scale:** ~20–50 presence points within a compact spatial area (~1–2 Sentinel-2 tiles)
- **Absence points:** Drawn from the same tile — same sensor geometry and phenological context
- **Hard negatives:** Small set (~50–100) from known Melaleuca/Acacia stands within the same tile

### Workstation constraints (16 CPU, 32GB RAM)
- Extract point values only via COG windowed reads — never load full scenes
- Accumulate time series in DataFrames, not rasters
- Cache scenes locally once fetched; reuse across all point extractions
- Point-level parallelism maps cleanly to 16 workers
- Memory estimate for development: ~700MB (500 points × 70 acquisitions × 5 bands × float32)

### Production phase (after process verification)
- Expand to 2,000–5,000 presence points, multi-catchment or national
- Stratify by climate zone before training (tropical, subtropical, semi-arid)
- Use 2017–2024 Sentinel-2 archive (8 years; avoid early S2 calibration issues)
- Full 10-band stack

### ALA record filtering heuristics
- Prefer spatially clustered records (within 50m of at least one other record)
- Exclude records from before 2015 (stands may have been cleared)
- Exclude points near water bodies (specular reflection)
- Prefer human-verified observations over modelled/bulk-uploaded records

---

## Literature Findings and Refinements

The following are drawn from a targeted literature review (April 2026).

### Confirmed by literature

- **Individual acquisitions over composites** — directly validated. Brief flowering spikes are erased by compositing; uncomposited time series are necessary to capture them.
- **Red-edge and SWIR bands** — B5/B6/B7/B8a for chlorophyll and vigour (avoiding NDVI saturation in dense biomass); B11/B12 for leaf water content. NDCI (Normalized Difference Chlorophyll Index) specifically validated for high-biomass discrimination.
- **Peak day-of-year as primary discriminator** — explicitly confirmed as more stable than absolute index values. Absolute reflectance fluctuates with soil nutrition and moisture; phenological timing is regulated by biological clocks and environmental triggers.
- **Climate-stratified modelling** — confirmed necessary. Continent-wide single models explicitly cautioned against. Parkinsonia dormancy release is driven by "wet heat" (temperature + soil moisture threshold), not calendar position — arid-zone populations behave fundamentally differently from tropical ones.
- **Random Forest for mixed pixels** — validated as operationally more reliable than spectral unmixing for broad-area detection. RF implicitly handles mixed-pixel problem without endmember extraction.
- **Inter-annual consistency** — confirmed to improve classification accuracy by several percentage points over single-season approaches.
- **ALA filtering requirements** — strongly cautioned: roadside/tourist bias, missing dates, coordinate transcription errors. Exhaustive filtering required before use as training data.

### New findings not previously considered

**1. Cloud gap-filling is a first-order problem**
In monsoonal tropical environments, cloud cover renders the majority of wet-season and build-up acquisitions unusable. Critically: *standard smoothing filters (Savitzky-Golay, Whittaker) will underfit brief flowering spikes* — they are designed for broad seasonal patterns and smooth away exactly the signal we need. Options:
- Kalman Filter + LSTM: high accuracy but computationally heavy
- **Sentinel-1 SAR fusion**: weather-independent, maintains continuity through cloud gaps — viable given existing S1 infrastructure in the pipeline

**2. Biocontrol operations distort expected waveforms**
Over 45% of surveyed Parkinsonia plants in northern Australia show defoliation from biocontrol moths (*Eueupithecia* spp.); bioherbicide (*Di-Bak Parkinsonia*) causes localised canopy dieback. Implications:
- Defoliated/dying stands show neither greenness persistence nor flowering spike — appear as bare soil or dead wood
- Some ALA records may correspond to stands that have since been treated and are now spectrally invisible
- Multi-year consistency metrics may flag treated stands as absences even if weed is still present
- This is a hard detection limit: the approach detects *living, untreated* Parkinsonia

**3. Yellow-flowering Acacia false positives are a documented problem**
Explicitly confirmed in literature: mapping yellow-flowering legumes in areas dominated by native Acacia spp. yields *"an intolerable number of false positives"* unless highly species-specific indices are calibrated. The High-Resolution Flowering Index (HrFI) exists for this purpose but has limited success at medium satellite resolution due to sub-pixel mixing. The flowering index alone cannot carry the discrimination — it must be combined with temporal consistency features (calendar-locking, multi-year persistence) to separate Parkinsonia from native acacias.

**4. Temporal autocorrelation inflates accuracy metrics**
Standard t-tests and GLMs show empirical Type I error rates of ~25% in autocorrelated time series (vs. nominal 5%). Only Generalised Least Squares maintains controlled false-positive rates. For RF trained on multi-year data, accuracy metrics will be inflated if years are treated as independent samples. **Training/test splits must be temporal** (hold out entire years) not random.

**5. Detection threshold: ~10% fractional cover**
Below ~10% pixel fractional cover, standard vegetation indices fail to overcome background noise. This is confirmed from analogous riparian shrubland work (Lignum, Murray-Darling Basin). Isolated individual plants or very sparse infestations are below the detection limit of this approach regardless of technique.

**6. Upland infestations are real and underrepresented**
Parkinsonia establishes beyond riparian zones via two main pathways confirmed in the literature:
- **Livestock dispersal:** Seeds consumed and passed by cattle and horses (gut passage aids germination); deposited at upland yards, mustering routes, and water points across entire station properties
- **Man-made water infrastructure:** Dense thickets documented along bore drains, station dams, and stock water points — upland by definition, no drainage connection

Implications:
- **HAND is not a valid hard exclusion mask.** The current pipeline uses HAND ≤5m as a gating feature; bore drain infestations score high HAND and are invisible to the model by design. HAND should remain as a probabilistic feature but not as an exclusion filter.
- **Absence sampling is at risk:** Drawing absences from "upland, high HAND" pixels may accidentally label genuine bore drain presences as absences.
- **ALA underrepresents upland infestations:** Records are biased toward roads and waterways; remote bore drain infestations on station properties are systematically under-recorded.
- **Two distinct infestation types** may require separate treatment in the classifier: riparian (low HAND, floodplain groundwater) vs. upland water infrastructure (high HAND, artificial water source). Performance should be evaluated separately across both.

---

## Per-Observation Quality Weighting

Each Sentinel-2 acquisition at a point receives its own quality score, so the time series is a **weighted scatter of (value, weight) pairs** rather than a uniform sequence. This means the waveform carries uncertainty at every point — a genuine flowering peak from a clean acquisition outranks a slightly higher value from a hazy or off-nadir scene.

### Quality score components (per observation)

| Component | Source | What it captures |
|---|---|---|
| SCL neighbourhood purity | S2 SCL band, 3×3 or 5×5 window | Cloud edges, adjacency effects, undetected thin cloud |
| Aerosol Optical Thickness | S2 L2A AOT band (per-pixel) | Smoke and haze — suppresses NIR, elevates red, distorts flowering index |
| View zenith angle | Scene metadata | BRDF effects from off-nadir acquisitions |
| Sun zenith angle | Computed from time + location | Shadow fraction within canopy at low sun elevations |
| Scene-relative greenness z-score | Computed vs. same calendar-month archive | Detects cloud contamination, anomalous wet conditions, fire recovery — no external rainfall data needed |

Combined score: product of [0,1] scalars — any single severe degradation collapses the overall weight.

```
observation = {
    date:          acquisition date,
    flowering_idx: computed band value,
    ndvi:          computed NDVI,
    quality:       w_scl * w_aot * w_vza * w_sza * w_greenness
}
```

### How weighting changes waveform analysis

- **Peak detection:** Highest quality-weighted value, not simply the maximum raw value
- **Duration above threshold:** Only quality-passing observations count; low-quality gaps are treated as missing data, not as "below threshold"
- **Inter-annual consistency:** Each year's peak DoY is weighted by the quality of the acquisition that identified it — a peak from a single marginal observation is less reliable than one confirmed by multiple clean acquisitions
- **Visual interpretation:** Time series plots encode quality as point size or opacity — immediately interpretable and consistent with the weighted computation

This also avoids the Savitzky-Golay problem identified in the literature: rather than fitting a smooth curve through all observations equally (which erases the spike), the weighted scatter preserves high-quality observations at full weight without allowing poor observations to drag the peak estimate down.

---

## Research Question for Literature Review

> *How can the full Sentinel-2 time series archive (individual acquisitions, not seasonal composites) be exploited to extract phenologically-discriminative spectral waveforms at known Parkinsonia aculeata occurrence locations — and what does the existing literature say about the reliability, limitations, and species-discrimination power of time-series approaches to invasive riparian shrub detection in mixed-pixel tropical and subtropical savanna environments?*

### Sub-questions
- What methods exist for detecting brief flowering events (spikes) in dense-revisit time series with irregular cloud gaps?
- How have red-edge and SWIR bands been used to discriminate species with similar NDVI profiles in savanna/riparian contexts?
- At what fractional cover does a target species become detectable in mixed riparian pixels?
- Has inter-annual phenological consistency been used as a classification feature for invasive species detection?
- How do wet/dry year anomalies affect NDVI-based invasive detection, and what normalisation strategies have been validated?
- What spectral and temporal features best separate Parkinsonia from Melaleuca, riparian Eucalyptus, and Acacia in northern Australia?
- Is flowering phenology timing (peak DoY) a more stable inter-species discriminator than peak spectral index value?

### Cautions to look for in literature
- Time-series approaches degrading in cloud-prone tropical environments due to insufficient cloud-free acquisitions at critical phenological moments
- Spatial uncertainty or taxonomic error in ALA/citizen science records propagating into training datasets
- Yellow-flowering natives (Acacia spp.) producing false positives in flowering-index detection
- Temporal autocorrelation in multi-year training data inflating accuracy metrics
