# Mitchell River Catchment — Parkinsonia GIS Analysis
## Design Document

**Version:** 2.0
**Status:** Draft
**Prepared for:** Kowanyama Aboriginal Land and Natural Resources Management Office (KALNRMO)
**Last updated:** April 2026

---

## 1. Purpose

This document describes the design and implementation plan for an automated satellite-based analysis pipeline to map Parkinsonia aculeata distribution across the Mitchell River catchment (~72,229 km²) and produce annual treatment priority outputs for the KALNRMO Parkinsonia management program.

The analysis addresses a critical strategic unknown: whether the upstream Mitchell catchment infestation is concentrated in a tractable number of accessible locations (supporting a coordinated eradication program) or pervasively distributed across all sub-catchments (supporting a long-term suppression strategy). No catchment-scale systematic survey has been published since Deveze (2004), which recorded only four known Parkinsonia outbreaks on the entire Cape York west coast. Twenty years of potential spread make this the highest-priority evidence gap in the program.

---

## 2. Strategic context

The Mitchell River is Australia's highest-discharge river system, distributing Parkinsonia seed pods from the full 72,229 km² upstream catchment across the Kowanyama DOGIT floodplain every wet season. On-DOGIT treatment alone cannot achieve zero reproductive plants in watercourse-adjacent zones while that upstream seed flux continues. The satellite analysis is therefore not primarily a surveillance tool — it is the evidence base for deciding what kind of program to run and where to run it.

**The pivotal question the analysis must answer:** Are the flood-connected upstream Parkinsonia source populations concentrated or diffuse?

---

## 3. Why satellite detection is well-suited to Parkinsonia

Parkinsonia aculeata is unusually detectable from satellite compared to most invasive weed species, for three independent reasons that can be exploited simultaneously:

### 3.1 Dry-season NDVI anomaly (primary signal)
Parkinsonia photosynthesises through its green bark and flattened rachises year-round, including when leaflets are shed in the dry season. In the Cape York tropical savanna, native grasses senesce to straw-colour by June–July while Parkinsonia maintains anomalously high NDVI throughout the May–October dry season. This creates a persistent, strong spectral contrast against surrounding vegetation detectable in Sentinel-2 10 m imagery.

### 3.2 Flowering spectral spike (secondary signal)
During flowering (approximately August–October at ~17°S), yellow flowers lift reflectance in the green band (560 nm) significantly above co-occurring native vegetation. This produces a measurable spectral anomaly in August–October composites, confirmed in published research combining Sentinel-2 with higher-resolution imagery.

### 3.3 Canopy texture (supporting discriminator)
Parkinsonia's open, pendulous canopy creates measurable Grey-Level Co-occurrence Matrix (GLCM) texture statistics that differ from native woodland, reducing false positives in areas where other dry-season green species (e.g. Melaleuca) occur. Note: at Sentinel-2's 10 m resolution, GLCM statistics are only meaningful for patches large enough to span multiple pixels with a sufficient kernel. This feature should be treated as supplementary in year 1 and evaluated against classifier performance before being retained.

### 3.4 Red edge discrimination (Melaleuca separation)
The red edge spectral region is the most discriminating zone for separating Parkinsonia from Melaleuca, the primary false-positive risk in the Mitchell catchment. Sentinel-2 provides red edge bands only at 20 m (requiring upsampling that degrades spatial precision); Planet SuperDove provides red edge at native 3 m. This is addressed in the Tier 2 pipeline (Section 6.2).

Requiring coincident evidence from two or more signals simultaneously reduces false positives dramatically. The three signals are biologically independent, so a non-Parkinsonia pixel matching all three by coincidence is a low-probability event in the Mitchell catchment landscape. Multi-temporal Random Forest classifiers using this approach achieve 85–95% overall accuracy for woody invasives in savanna contexts in published literature.

---

## 4. Two-tier pipeline design

The pipeline is structured in two tiers that build on each other sequentially.

| Tier | Data | Purpose | Status |
|---|---|---|---|
| 1 | Sentinel-2 + DEA Landsat | Catchment-scale prioritisation, flood connectivity analysis, strategic eradication/suppression decision | Build now — no data access barriers |
| 2 | Planet SuperDove (3 m, 8-band) | Pre-reproductive cluster detection, early dry season compositing, Melaleuca false-positive reduction | Pending SISP access; fallback via research partnership |

**Sequencing rationale:** Tier 1 is built first because it has no data access barriers and produces the primary strategic deliverable. The completed Tier 1 catchment-scale probability map is also the most compelling justification for the Tier 2 data access request — a demonstrated output is a far stronger case than a prospective proposal.

### Why Planet SuperDove (Tier 2) is worth pursuing

Planet SuperDove offers three specific advantages over Sentinel-2 that are directly relevant to this program:

1. **Resolution:** A pre-reproductive Parkinsonia cluster of 3–5 plants covering ~20 m² occupies approximately 0.2 Sentinel-2 pixels — effectively invisible in background noise — but registers across ~2 Planet pixels, detectable as a spectral anomaly particularly during the August–October flowering window. This extends effective detection by an estimated 6–12 months, pushing detection into year 1 of establishment rather than year 2.

2. **Red edge at native 3 m:** The SuperDove 8-band sensor includes a red edge band at native 3 m resolution. This directly enables Melaleuca discrimination that Sentinel-2 cannot achieve at its 20 m red edge resolution.

3. **Near-daily revisit:** Enables cloud-free composite construction in May–June, the early dry season transition when new seedling cohorts from wet season flooding are establishing and most treatable, but when residual cloud frequently gaps Sentinel-2 coverage.

### Planet data access pathway

The Queensland Government already holds this data under an existing Planet Labs contract providing a near-daily feed covering the entire state. Access via the Spatial Imagery Services Program (SISP) through the Department of Resources is therefore not a request for new expenditure or new tasking — it is a request to use existing archived imagery for a documented biosecurity purpose.

Parkinsonia is a Category 3 Restricted Invasive Plant under the Biosecurity Act 2014 (Qld), the Cape York Peninsula Regional Biosecurity Plan identifies it as a regional priority weed, and KALNRMO carries a general biosecurity obligation over the DOGIT. Access for this purpose falls squarely within the program's mandate.

**Fallback:** A Planet Education and Research account (~3,000 km²/month) via a JCU or CSIRO research partnership provides sufficient quota for targeted validation of the highest-confidence Sentinel-2 anomalies — the primary Tier 2 use case. A named contact and formal approach to one of these institutions should be identified before the SISP request is submitted.

**Note:** Tier 2 requires a separate classifier trained at 3 m resolution on Planet-specific band indices. It is not a drop-in upgrade to the Tier 1 classifier. Allocate additional development effort for this if/when Planet access is confirmed.

---

## 5. Data sources

### 5.1 Primary sources

| Source | Role | Resolution | Revisit | Cost | Access |
|---|---|---|---|---|---|
| Sentinel-2 L2A | NDVI anomaly, flowering index, red edge discrimination | 10 m (optical), 20 m (red edge) | 5 days (2A+2B) | Free | Element84 STAC / CDSE |
| Sentinel-1 GRD | Wet season flood extent mapping | 10 m | 12 days (1A only) | Free | Element84 STAC / CDSE |
| DEA — Landsat Collection 3 ARD | Long-term NDVI baseline (back to 1986); atmospherically corrected, BRDF-normalised, cross-calibrated across Landsat 5/7/8/9 for Australian conditions | 30 m | 8–16 days (varies by mission) | Free | DEA AWS S3 / `odc-stac` (preferred); Element84 STAC (fallback) |

**Why DEA over raw Landsat:** Building a multi-decade baseline from raw Landsat scenes requires inter-sensor calibration across five instruments with different spectral response functions, gain settings, and atmospheric correction approaches. DEA (run by Geoscience Australia) has already solved this problem for the Australian continent to a standard that would take significant analyst effort to replicate independently. Using DEA Collection 3 ARD makes Step 2 both less work and higher quality simultaneously.

### 5.2 Supporting sources

| Source | Role | Resolution | Access |
|---|---|---|---|
| DEA — Fractional Cover | Pre-computed annual green/brown/bare ground fractions since 1987; directly encodes dry-season greenness contrast; supplements the NDVI baseline | 25 m | Free, DEA AWS S3 / Open Web Services |
| DEA — Wetlands Insight Tool (Qld) | Tracks inundation extent for 270,000 Queensland wetlands since 1987; cross-validation for Sentinel-1 flood extent output | Variable | Free, DEA Open Web Services |
| ALOS-2 PALSAR annual mosaic | One-off structural layer for classifier training; L-band penetrates canopy, sensitive to woody biomass | 25 m | Free download, JAXA |
| Atlas of Living Australia (ALA) | Georeferenced Parkinsonia occurrence records for classifier training | Point data | Free, ALA API |
| Biosecurity Queensland records | Additional ground-truth points from historical control programs | Point data | Request from Biosecurity Qld |
| Queensland Globe / QImagery | Visual exploration and ground-truth checking; historical aerial photography downloadable via QImagery | Variable (~30 cm–1 m aerial) | Free browser access |

### 5.3 Alternatives considered

**Planet SuperDove:** See Section 4 (Tier 2). Not rejected — access pathway identified, sequenced after Tier 1 completion.

**Google Earth Engine (GEE):** GEE's Python API can execute equivalent analysis on Google's cloud infrastructure. It is fully scriptable and Git-committable. However, it requires Google account authentication and OAuth token management (friction in CI/CD environments), exports results to Google Drive rather than local storage, and provides no data access advantage over open STAC endpoints. Not selected.

**OpenEO (Copernicus backend):** A viable alternative execution platform — backend-independent code, free EU compute, fully scriptable. The open STAC stack selected here is more mature and better documented for this use case. Remains a viable fallback if CDSE access becomes preferred.

**MODIS:** 250 m–1 km resolution. Too coarse to detect individual Parkinsonia patches or distinguish infestation boundaries at sub-catchment scale. Not suitable.

**Queensland Globe / QImagery:** Useful for visual ground-truth checking and historical aerial photo review. Current annual statewide Planet mosaics exist under the state government's Planet contract but are available only to select agencies — not for programmatic bulk access. Not a suitable programmatic data source for this pipeline.

### 5.4 Data endpoints

```
Element84 Earth Search:  https://earth-search.aws.element84.com/v1
CDSE STAC:               https://catalogue.dataspace.copernicus.eu/stac
DEA AWS S3:              s3://dea-public-data/  (via odc-stac)
DEA Open Web Services:   https://ows.dea.ga.gov.au/
```

Element84 and CDSE are free, require no authentication for read access, and carry Sentinel-1, Sentinel-2, and Landsat. The pipeline defaults to Element84 for Sentinel data; CDSE is the fallback. DEA AWS S3 is the preferred source for the Landsat historical baseline; `odc-stac` is the recommended loader as it natively understands the DEA datacube format and metadata conventions.

---

## 6. Technical stack

All components run natively on Linux. No cloud accounts, proprietary platforms, or browser authentication required.

```
Python 3.11+
pystac-client       # STAC catalogue search
stackstac           # STAC items → lazy xarray DataArrays via COG range requests
odc-stac            # Alternative loader, preferred for DEA data and Sentinel-1
rioxarray           # Raster I/O with xarray
dask[distributed]   # Parallel computation, memory-flat chunked processing
scikit-learn        # Random Forest classifier
geopandas           # Vector data (catchment boundary, output polygons)
rasterio            # GeoTIFF read/write
numpy
fsspec              # Filesystem abstraction + local COG tile cache
```

### 6.1 Execution environment

Steps 1–4 run on an EC2 `c7gn.4xlarge` (16 vCPU ARM64/Graviton3, 32 GB) in `us-west-2`, co-located with the Sentinel-2 COG bucket. Steps 5–7 work entirely from the GeoTIFFs and GeoPackages written by steps 1–4 and can be run locally.

Step 4 (Sentinel-1 flood extent) was originally designed to run locally, but the S1 GRD scenes for the Mitchell catchment flood season total ~128 GB from `s3://sentinel-s1-l1c` (us-east-1). Streaming this volume locally is impractical; caching to EBS on the EC2 instance keeps all S3 transfer within AWS and makes annual re-runs fast (only new scenes need syncing). The pipeline's step 0 handles S1 EBS sync automatically when `LOCAL_S1_ROOT` is set, alongside the existing S2 sync.

The pipeline processes the catchment in 512 px spatial tiles via `ProcessPoolExecutor` with 16 worker processes. Each worker runs fetch + compute for one tile independently, giving true CPU parallelism (each worker has its own GIL). Peak memory: ~86 MB per tile × 16 workers ≈ 1.4 GB arrays in flight, well within the 32 GB budget.

### 6.2 Storage layout

```
/data/mrc-parko/               # BASE_DIR — all operational data
├── mitchell_catchment.geojson
├── cache/                      # (unused — no fsspec tile cache)
├── working/                    # Intermediate tile scratch files (cleaned up after each step)
├── outputs/YYYY/               # Final GeoTIFFs and quicklooks
└── logs/                       # Per-run logs and tile stats CSVs

/mnt/ebs/s2cache/                  # EBS volume (2 TB gp3) — Sentinel-2 COG local cache
└── sentinel-cogs/
    └── sentinel-s2-l2a-cogs/  # Mirrors s3://sentinel-cogs key structure

/mnt/ebs/s1cache/                  # EBS volume (500 GiB gp3) — Sentinel-1 GRD local cache
└── sentinel-s1-l1c/
    └── GRD/                   # Mirrors s3://sentinel-s1-l1c key structure
```

Both EBS volumes are created once, snapshotted after the pipeline run, and restored the following year. Only new scenes need syncing each year. See [EBS-SETUP.md](EBS-SETUP.md) for full lifecycle instructions.

---

## 7. Analysis pipeline

### Step 1 — Sentinel-2 dry-season composite

```python
# Search all S2 L2A scenes intersecting catchment boundary, May–Oct
# Filter: cloud cover < 20%
# Load: B03 (Green), B04 (Red), B05 (Red edge 1), B06 (Red edge 2),
#        B07 (Red edge 3), B08 (NIR), SCL (cloud mask)
# Cloud mask: SCL classes 4, 5, 6 (clear vegetation, bare soil, water)
# Additionally apply s2cloudless probability mask to catch cirrus shadows
# and aerosol contamination from May–June biomass burning events
# Compute NDVI = (B08 - B04) / (B08 + B04)
# Median composite across all clear-sky time steps
# Output: ndvi_median_YYYY.tif
```

Note: B05, B06, B07 are at 20 m native resolution. Resample to 10 m when combining with B04/B08. `stackstac` handles this with a single `resolution=10` parameter.

**Cloud masking note:** SCL alone misses cirrus shadows and aerosol haze from early dry season biomass burning. Apply `s2cloudless` in addition to SCL, and consider excluding May–June scenes if the July–October window provides sufficient coverage.

### Step 2 — Multi-year NDVI anomaly

```python
# Load DEA Landsat Collection 3 ARD via odc-stac
# (fallback: Element84 STAC Landsat scenes, requires manual inter-sensor calibration)
# DEA provides atmospherically corrected, BRDF-normalised surface reflectance
# validated for Australian conditions — Landsat 5/7/8/9, 1986–present
# Build per-pixel long-term median NDVI for dry-season window only (COMPOSITE_START–COMPOSITE_END)
# Also load DEA Fractional Cover product (green fraction) as supplementary baseline
# Anomaly = current year NDVI - long-term median
# Persistent positive anomaly = primary Parkinsonia detection signal
# Output: ndvi_anomaly_YYYY.tif
```

The long-term baseline is computed once and cached; only the current-year composite changes annually. Using DEA Collection 3 ARD eliminates inter-sensor calibration work across five Landsat missions — this is technically the most complex step in the pipeline and DEA has already solved it for Australian conditions.

**Baseline temporal sampling:** The STAC search is performed per-year (one query per calendar year, 1986–YEAR-1) restricted to the dry-season window (`COMPOSITE_START`–`COMPOSITE_END`, typically May–October). This ensures two things. First, the baseline phenology matches the current-year composite: comparing a dry-season current NDVI against a full-year baseline would embed a seasonal bias rather than a vegetation-change signal. Second, per-year queries with a per-year item cap (12 items/year, evenly sampled across the window) guarantee even temporal representation across all four Landsat missions — a single wide query with a total item cap would return items in chronological order and truncate LS8/LS9 data (2013–present) entirely once the cap is reached, leaving the most recent and highest-quality mission absent from the baseline.

**Back-analysis performance note:** The cache invalidation logic keys the baseline on `1986–YEAR-1`, so each year of a back-analysis (e.g. 2017–2025) triggers a full baseline rebuild — approximately 8× redundant Landsat fetches and compute across a 9-year run. Before running a multi-year back-analysis, the baseline end year should be decoupled from `YEAR` and built once to the full extent (e.g. 1986–2024), then reused across all years. This is not yet implemented.

**Known confound:** Dense Wet Tropics forest in the upper Mitchell headwaters (Atherton Tablelands, Walsh and Lynd headwaters) creates persistently high NDVI that may produce false anomalies in those zones. Apply a habitat mask (watercourse buffer + vegetation type exclusion) to reduce but not eliminate this.

### Step 3 — Flowering spectral index

```python
# Primary index: B03 (Green) / B08 (NIR) ratio for Aug–Oct window
# Parkinsonia yellow flowers lift green reflectance relative to NIR
#
# Supplementary: NDRE = (B06 - B05) / (B06 + B05)
# Red edge normalised difference — better discriminates flowering canopy
# from background green vegetation; consider as replacement or addition
#
# Mask with NDVI anomaly layer to focus on already-flagged pixels
# Output: flowering_index_YYYY.tif
```

**Validation check:** The flowering index should show a different spatial pattern from the NDVI anomaly — patchier, more temporally specific. If the two rasters are highly correlated (>0.7), they are measuring the same thing and only one should enter the classifier feature stack.

### Step 4 — Flood extent (HAND-based)

See [DESIGN-FLOOD-CONNECTIVITY.md](DESIGN-FLOOD-CONNECTIVITY.md) for the full design.

The original Sentinel-1 / Otsu thresholding approach was abandoned — C-band GRD cannot penetrate the dense grassland/sedgeland canopy in the Mitchell catchment. Step 4 now derives flood connectivity from terrain geometry using the Height Above Nearest Drainage (HAND) index computed from the Copernicus GLO-30 DEM.

### Step 5 — Random Forest classifier

```python
# Feature stack per pixel:
#   - NDVI anomaly (magnitude)
#   - Flowering index
#   - Red edge ratio B06/B08 (or NDRE)
#   - GLCM texture variance from B08 (supplementary — evaluate before retaining)
#   - Distance to nearest watercourse (from drainage network layer)
#   - Flood connectivity (binary, from Step 4)
#
# Training data:
#   - Positive class: ALA occurrence records + KALNRMO confirmed sites
#   - Negative class: random sample of non-Parkinsonia pixels
#     stratified by vegetation type, constrained by habitat mask,
#     with minimum spatial buffer from known positives
#
# Validation: spatial block cross-validation (GroupKFold on ~50 km grid)
# Standard random train/test splits are optimistic due to spatial autocorrelation
#
# Output: parkinsonia_probability_YYYY.tif  (Float32, 0–1)
```

**ALA data caveat:** ALA records are presence-only and spatially biased toward roadsides and surveyed areas. A standard RF trained on these will over-predict near infrastructure and under-predict in remote areas. Options: use MaxEnt or a presence-background approach (e.g. `elapid`) for initial habitat suitability modelling; or constrain pseudo-absence generation to plausible habitat with a minimum distance buffer from known positives. Year 1 outputs should be treated as first-pass screening, not confirmed detections.

**Feature importance check:** NDVI anomaly should be the dominant feature. If distance-to-watercourse dominates, the model is fitting geography rather than spectral signal.

Target accuracy: >85% overall, >80% recall on positive class (missing a plant is worse than a false alarm). Validate annually against new KALNRMO ground-truth surveys. Retrain when ground-truth dataset grows materially.

### Step 6 — Vectorisation and prioritisation

```python
# Threshold probability raster (default 0.6 — tunable, see note)
# Morphological smoothing (binary erosion/dilation) before vectorising
# to reduce jagged patch boundaries
# Vectorise contiguous patches > 0.25 ha minimum mapping unit
# Attribute each polygon:
#   area_ha, mean_prob, max_prob,
#   flood_connected (intersect with Step 4 output),
#   dist_to_kowanyama_km,
#   stream_order (Strahler, from drainage network),
#   estimated_seed_flux_rank (area_ha * mean_prob * stream_order_weight / dist_km)
#
# Classify into treatment tiers A–E per strategy document
# Output: candidate_infestations_YYYY.gpkg
#         priority_zones_YYYY.gpkg
```

**Threshold note:** A fixed 0.6 threshold creates a hard decision boundary that will produce different recall/precision tradeoffs depending on annual imagery quality. Consider delivering the full probability raster to field teams with threshold sliders in QGIS, and using probability quintiles for Tier A–E classification rather than a hard cutoff.

**Seed flux ranking note:** The formula weights flood-connected populations on higher-order streams more heavily than equidistant tributary populations of the same area, reflecting the difference in downstream seed delivery to the DOGIT. Stream order should be derived from the drainage network layer, not from distance alone.

### Step 7 — Change detection (year 2 onwards)

```python
# Diff of current vs previous year probability rasters
# Verify CRS and resolution match exactly before differencing
# New high-confidence detections = likely new infestations or previously missed
# Declining probability in treated zones = treatment success signal
# Output: change_detection_YYYY_vs_YYYY-1.tif
#         summary_statistics_YYYY.csv
```

**Interpretation note:** Large apparent changes should be checked against cloud cover and imagery quality differences between the two years before being attributed to real vegetation change.

---

## 8. Output verification

Each pipeline step produces a verification report alongside its primary output. The run script fails immediately if any verification step returns a non-zero exit code, preventing downstream steps from consuming bad inputs.

### Repository structure

```
outputs/
└── products/
    ├── ndvi_median_2025.tif
    ├── verification_report_2025.json
    └── quicklooks/
        ├── ndvi_median_2025.png
        ├── ndvi_anomaly_2025.png
        └── ...
```

### Step 1 — NDVI composite verification

```python
import numpy as np, rioxarray as rxr
ndvi = rxr.open_rasterio("ndvi_median_YYYY.tif").squeeze()

assert ndvi.min().item() >= -1 and ndvi.max().item() <= 1
nan_frac = ndvi.isnull().mean().item()
assert nan_frac < 0.20, f"NaN fraction {nan_frac:.1%} exceeds 20% — check cloud mask"

catchment_median = float(ndvi.median())
assert 0.15 < catchment_median < 0.50, \
    f"Catchment median NDVI {catchment_median:.2f} outside expected 0.15–0.50 range for dry-season savanna"
```

Visually: load in QGIS with a diverging colour ramp. River corridors should show persistently higher NDVI than surrounding savanna. Rectangular NaN blocks or striping indicate SCL over-triggering.

### Step 2 — NDVI anomaly verification

```python
anom = rxr.open_rasterio("ndvi_anomaly_YYYY.tif").squeeze()

# Anomaly should be centred near zero — it is relative to a long-run baseline
assert abs(anom.mean().item()) < 0.05, "Anomaly mean far from zero — baseline may be miscalculated"
assert 0.03 < anom.std().item() < 0.20, "Anomaly std outside expected range for stable savanna"
```

Cross-check: overlay ALA occurrence points on the anomaly raster. Known Parkinsonia locations should cluster in positive anomaly zones. If they don't, investigate the baseline or current composite.

### Step 3 — Flowering index verification

```python
fi = rxr.open_rasterio("flowering_index_YYYY.tif").squeeze()
anom = rxr.open_rasterio("ndvi_anomaly_YYYY.tif").squeeze()

valid = ~(np.isnan(fi.values) | np.isnan(anom.values))
correlation = np.corrcoef(fi.values[valid].ravel(), anom.values[valid].ravel())[0, 1]
assert correlation < 0.70, \
    f"Flowering index / NDVI anomaly correlation {correlation:.2f} too high — likely measuring same signal"
```

### Step 4 — Flood extent verification

```python
import geopandas as gpd
flood = gpd.read_file("flood_extent_YYYY.gpkg")

assert flood.geometry.is_valid.all(), "Invalid flood geometries"
flood_km2 = flood.geometry.area.sum() / 1e6
# Cross-check against DEA Wetlands Insight Tool for the same wet season
print(f"Total flood area: {flood_km2:.0f} km²")
```

Visually: overlay on a Sentinel-2 RGB composite from the same period. Systematic spatial offset suggests terrain correction error.

### Step 5 — Classifier verification

```python
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
import pandas as pd

# Spatial block cross-validation — assign training points to ~50 km grid blocks
# GroupKFold ensures train/test blocks are spatially separated
gkf = GroupKFold(n_splits=5)
scores = []
for train_idx, test_idx in gkf.split(X, y, groups=spatial_block_id):
    ...  # fit, predict, score

# Feature importance — NDVI anomaly should dominate
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top_feature = importances.index[0]
assert top_feature in ("ndvi_anomaly", "flowering_index"), \
    f"Unexpected dominant feature '{top_feature}' — model may be fitting geography not spectral signal"

# Probability calibration
fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
# Plot and inspect — significant deviation from diagonal means probabilities are not well-calibrated
```

Visually: high-probability zones should follow watercourses. Clustering near roads or town boundaries indicates ALA survey bias being fitted.

### Step 6 — Vector output verification

```python
patches = gpd.read_file("candidate_infestations_YYYY.gpkg")

assert patches.geometry.is_valid.all(), "Invalid patch geometries"
assert (patches["area_ha"] >= 0.25).all(), "Patches below 0.25 ha MMU present"
assert patches.crs.to_epsg() == 7844, f"Wrong CRS: {patches.crs}"  # GDA2020

# Tier distribution sanity check
tier_counts = patches["tier"].value_counts()
assert len(tier_counts) > 1, "All patches assigned to same tier — check threshold/classification logic"
print(patches[["area_ha", "mean_prob", "max_prob"]].describe())
```

### Step 7 — Change detection verification

```python
change = rxr.open_rasterio("change_detection_YYYY_vs_YYYY-1.tif").squeeze()

# Distribution should be approximately symmetric around zero
assert abs(change.mean().item()) < 0.05, "Change raster mean far from zero — possible CRS/resolution mismatch between years"
```

Cross-check: KALNRMO-treated sites from the previous dry season should show declining probability. Absence of decline in known treated areas is either a treatment outcome signal or a classifier sensitivity issue — investigate before reporting.

---

## 9. Outputs

| File | Format | Audience | Description |
|---|---|---|---|
| `parkinsonia_probability_YYYY.tif` | GeoTIFF | GIS analyst | Full-catchment probability raster, 0–1 |
| `candidate_infestations_YYYY.gpkg` | GeoPackage | GIS analyst, coordinators | Vectorised patch polygons with attributes |
| `priority_zones_YYYY.gpkg` | GeoPackage | Rangers (QGIS / Avenza) | Tier A–E treatment priority polygons |
| `priority_zones_YYYY_fieldmap.pdf` | PDF | Rangers (printed field use) | A3 field map for dry-season operations |
| `upstream_sources_ranked_YYYY.gpkg` | GeoPackage | Program managers, MRWMG | Upstream seed sources ranked by flux contribution |
| `change_detection_YYYY.tif` | GeoTIFF | GIS analyst, program review | Year-on-year probability change |
| `summary_statistics_YYYY.csv` | CSV | Funders, program reporting | Total area by tier, flood-connected vs non |
| `verification_report_YYYY.json` | JSON | Pipeline operator | Pass/fail and key statistics for each step |
| `quicklooks/` | PNG | Pipeline operator | Thumbnails for rapid visual QA without opening QGIS |
| `archive/YYYY/` | — | Long-term record | Copy of all products for the year |

The **upstream sources ranked** output is the primary strategic deliverable — it directly answers the eradication vs suppression question by revealing whether flood-connected seed sources are concentrated or diffuse across the catchment.

The **priority zones** GeoPackage and field map PDF are the primary operational deliverables — they are what KALNRMO rangers use to plan drone survey routes each dry season.

---

## 10. Running the pipeline

The full pipeline is a standard Python project in a Git repository, runnable from the command line on any Linux machine.

```
mitchell-parkinsonia/
├── README.md
├── requirements.txt
├── config.sh                   # Base directory and derived paths (sourced by run.sh and config.py)
├── config.py                   # Thresholds, date ranges; imports paths from config.sh via env
├── run.sh                      # Pipeline entry point
├── data/
│   └── mitchell_catchment.geojson
├── analysis/
│   ├── 01_ndvi_composite.py
│   ├── 02_ndvi_anomaly.py
│   ├── 03_flowering_index.py
│   ├── 04_flood_extent.py
│   ├── 05_classifier.py
│   ├── 06_vectorise_prioritise.py
│   └── 07_change_detection.py
├── verify/
│   ├── 01_verify_ndvi_composite.py
│   ├── 02_verify_ndvi_anomaly.py
│   ├── 03_verify_flowering_index.py
│   ├── 04_verify_flood_extent.py
│   ├── 05_verify_classifier.py
│   ├── 06_verify_vectors.py
│   └── 07_verify_change_detection.py
└── outputs/
    └── .gitkeep
```

### config.sh

```bash
# Base directory for all data outside the repository.
# Override by setting BASE_DIR in the environment before sourcing.
BASE_DIR=${BASE_DIR:-/data/mitchell-parko}

CACHE_DIR=$BASE_DIR/cache          # fsspec COG tile cache (sentinel2/, sentinel1/, dea-landsat/)
WORKING_DIR=$BASE_DIR/working      # Intermediate rasters — overwritten each run
OUTPUTS_DIR=$BASE_DIR/outputs      # Final products (versioned by year)

export BASE_DIR CACHE_DIR WORKING_DIR OUTPUTS_DIR
```

Python scripts import these via `os.environ` in `config.py`:

```python
from pathlib import Path
import os

BASE_DIR    = Path(os.environ["BASE_DIR"])
CACHE_DIR   = Path(os.environ["CACHE_DIR"])
WORKING_DIR = Path(os.environ["WORKING_DIR"])
OUTPUTS_DIR = Path(os.environ["OUTPUTS_DIR"])
```

All four variables are always set by `config.sh` before any script runs, so no defaults are needed in `config.py`. Override `BASE_DIR` (or any individual path) in the environment before calling `run.sh` to relocate data without editing any file.

### run.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=config.sh
source "$(dirname "$0")/config.sh"

YEAR=${1:?Usage: ./run.sh YYYY [COMPOSITE_START] [COMPOSITE_END]}
COMPOSITE_START=${2:-05-01}
COMPOSITE_END=${3:-10-31}

export YEAR COMPOSITE_START COMPOSITE_END

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Starting analysis for $YEAR (dry season $COMPOSITE_START – $COMPOSITE_END)"

run_step() {
    local step=$1
    log "Running $step"
    python "analysis/$step.py" || { echo "FAILED: $step"; exit 1; }
    log "Verifying $step"
    python "verify/${step/analysis\//}_verify.py" || { echo "VERIFICATION FAILED: $step"; exit 1; }
}

run_step 01_ndvi_composite
run_step 02_ndvi_anomaly
run_step 03_flowering_index
run_step 04_flood_extent
run_step 05_classifier
run_step 06_vectorise_prioritise
run_step 07_change_detection

log "Analysis complete. Outputs in $OUTPUTS_DIR/$YEAR/"
log "Tag this run: git tag analysis/$YEAR $(git rev-parse --short HEAD)"
```

Run as:

```bash
./run.sh 2025
./run.sh 2025 06-01 10-31   # adjusted composite window
```

### Per-step resource estimates

Data download figures are for first run; subsequent runs fetch only newly available scenes (cache hit rate typically >90% on re-runs). Runtime assumes a 16-core workstation with 16 GB RAM and a 100 Mbps connection, processing with Dask over the full 72,229 km² catchment extent.

| Step | Description | Data download (first run) | Estimated runtime | Output size |
|---|---|---|---|---|
| 1 | S2 dry-season composite | ~25 GB (6 bands × ~20 scenes via COG range requests) | 40–60 min | ~4 GB (NDVI median + band stack, Float32 10 m) |
| 2 | NDVI anomaly | ~15 GB (DEA Landsat, ~700 scenes × 3 bands; computed once, reused annually) | 50–70 min | ~2 GB (anomaly raster resampled to 10 m) |
| 3 | Flowering index | ~5 GB (S2 Aug–Oct scenes; largely already cached from Step 1) | 10–20 min | ~600 MB (single-band ratio, Float32 10 m) |
| 4 | S1 flood extent | ~8 GB (~15 GRD scenes × ~500 MB each) | 50–70 min | ~50 MB (vector .gpkg) |
| 5 | Random Forest classifier | Negligible (ALA training points via API) | 20–40 min | ~3 GB (probability raster, Float32 10 m) |
| 6 | Vectorise and prioritise | None | 10–20 min | ~20 MB (two .gpkg files) |
| 7 | Change detection | None | 5–10 min | ~3 GB (difference raster) + negligible CSV |
| **Total** | | **~53 GB first run; ~5–10 GB annually thereafter** | **3–5 hours** | **~13 GB per annual run** |

Step 2's Landsat baseline is the largest one-time cost — once computed and cached it does not need to be redownloaded unless the cache is cleared. Step 4 is the most time-intensive per GB because S1 GRD preprocessing (terrain correction) is CPU-bound rather than I/O-bound.

**Design notes:**
- `set -euo pipefail` means any failed command or failed verification exits immediately. Downstream steps never run on bad inputs.
- Year and composite window are arguments so the pipeline can be re-run for any year or adjusted window without touching code.
- Each analysis step and its verify script are paired — if the verify script exits non-zero, the run halts at that step with a clear message.
- The suggested `git tag` at the end ties the outputs to the exact commit that produced them, providing the same audit trail as a CI artifact without requiring any external service.

---

## 11. Limitations and caveats

**Sentinel-2 cloud cover:** The 5-day revisit is a theoretical maximum. Early dry season (May–June) cloud and biomass burning aerosol can reduce usable scenes. Expect 15–30 cloud-free scenes per dry season. The median composite is robust to this, but very early-season gaps may affect composite quality. SCL masking alone is insufficient — apply `s2cloudless` for aerosol/cirrus contamination.

**Sentinel-1B decommissioned:** S1B went offline August 2022. Flood extent mapping now relies on S1A alone (12-day revisit). Adequate for seasonal flood mapping.

**S1 preprocessing dependency:** Step 4 requires radiometric calibration and terrain correction of raw S1 GRD data. This adds setup complexity not required for the optical steps.

**Classifier accuracy limited by ground-truth data:** ALA occurrence records are presence-only and spatially biased toward surveyed areas. Year 1 outputs should be treated as first-pass screening. Accuracy will improve as KALNRMO rangers contribute confirmed ground-truth from field surveys. Use spatial block cross-validation — standard random splits are optimistic.

**Minimum mapping unit:** The pipeline detects patches above 0.25 ha at 10 m resolution. Individual plants and pre-canopy seedlings are below detection threshold. This is a catchment-scale prioritisation tool, not a substitute for on-ground survey within high-priority zones. Tier 2 (Planet, 3 m) extends detection to pre-reproductive clusters but does not eliminate this constraint entirely.

**Upland forested headwaters:** Dense Wet Tropics forest in the upper Mitchell headwaters creates persistently high NDVI that may confound dry-season anomaly detection. The habitat mask reduces but does not eliminate this.

**GLCM texture at 10 m:** Texture statistics are only meaningful for patches large enough to span multiple pixels with a sufficient kernel at Sentinel-2 scale. Evaluate feature importance before retaining GLCM in the classifier.

**Seed flux ranking:** The ranking formula is an index for prioritisation, not a hydrological model. It should inform field investigation, not substitute for it.

---

## 12. Key references

- Deveze, M. (2004). *Parkinsonia: National Case Studies Manual.* Queensland DNRM, Cloncurry. ISBN 1 920920 67 6.
- van Klinken, R.D. et al. (2009). The biology of Australian weeds 54: *Parkinsonia aculeata* L. *Plant Protection Quarterly* 24(1).
- Geoscience Australia. (2024). Digital Earth Australia — Surface Reflectance Collection 3 (Landsat 5/7/8/9, Sentinel-2). https://knowledge.dea.ga.gov.au/
- Dhu, T. et al. (2017). Digital Earth Australia — unlocking new value from Earth observation data. *Big Earth Data* 1(1–2), 64–74. https://doi.org/10.1080/20964471.2017.1402490
- Royimani, L. et al. (2021). Can Sentinel-2 be used to detect invasive alien trees and shrubs in savanna and grassland biomes? *International Journal of Applied Earth Observation and Geoinformation* 103, 102472.
- Everitt, J.H. & Villarreal, R. (1987). Detecting huisache and Mexican palo-verde by aerial photography. *Weed Science* 35(3), 427–432.
- Panetta, F.D. (2015). Weed eradication feasibility: lessons of the 21st century. *Weed Research* 55(3), 226–238.

---

*This design document is a living record. Update it when: ground-truth data materially changes the classifier; Planet access is confirmed or denied; new satellite sources become available; or the eradication vs suppression strategic decision is made following the first complete analysis run.*
