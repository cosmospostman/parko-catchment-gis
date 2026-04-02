# NT Training Region — Approach, Investigation, and Justification

**Status:** Plan
**Related scripts:** `analysis/fetch_ala_australia.py`, `analysis/plot_training_candidates.py`, `analysis/05_classifier.py`

---

## Problem

The Mitchell River catchment contains only ~13 recorded *Parkinsonia aculeata* sightings in the Atlas of Living Australia. This is statistically insufficient to train a robust random forest classifier: a model fit on 13 presence points cannot learn a generalised decision boundary and will almost certainly overfit to those specific locations. The true population within the catchment is vastly larger — the low count reflects survey effort and access constraints rather than genuine scarcity.

The proposed solution is to train the classifier on a geographically separate NT region with abundant, high-quality sightings, then transfer the fitted model to predict over the Mitchell catchment feature rasters.

---

## ALA Data Investigation

A standalone script (`analysis/fetch_ala_australia.py`) was written to fetch all *Parkinsonia aculeata* occurrence records within the Australian bounding box from the ALA biocache API. The ALA biocache API enforces a hard per-query cap of 5,000 records (requests beyond `startIndex=5000` return empty pages silently). This was worked around by tiling the bounding box into a 8×6 grid and recursively subdividing any tile that exceeds the cap into four quadrants.

The fetch returned **39,863 records** across Australia, of which 22,510 had valid coordinates after deduplication. The spatial distribution was visualised as a hexbin density map and a point-level map zoomed to NT and western QLD.

**A critical caveat:** ALA sightings are not representative of true distribution. They are strongly biased toward areas accessible by road or boat — roads, stock routes, and river corridors attract disproportionate survey effort. Sighting counts should be treated as a proxy for survey effort as much as true infestation density. This has direct implications for training region selection (see below).

---

## Candidate Training Regions

Four candidate regions were identified and assessed:

| Region | Sightings | Notes |
|---|---|---|
| Katherine / Daly River (130–133.5°E, 13.5–16.5°S) | ~4,700 | Coastal/escarpment NT; perennial baseflow from Tindall/Oolloo aquifers; dense tropical savanna; strong monsoonal signal |
| McArthur River (135.5–138.5°E, 16.5–19°S) | ~5,700 | Compact, strongly linear riparian cluster; Barkly Tablelands escarpment; channelised corridors |
| Georgina / Barkly Tablelands (133.5–139°E, 17.5–22°S) | ~15,600 | Large low-gradient inland drainage; most geomorphologically similar to Mitchell megafan; but see below |
| W QLD — Cloncurry / Flinders | ~114 | Too few records; discarded |

The Georgina/Barkly count of 15,600 was the highest but is likely the most survey-effort-biased: the spatial pattern on the distribution map is diffuse and does not show clear riparian linearity, consistent with road-corridor sampling along the Barkly Highway and stock routes rather than genuine floodplain infestation mapping. Training on this scatter would teach the RF to associate Parkinsonia with accessible paddocks rather than drainage lines.

---

## Justification for Region Selection

The optimal training region depends on which features dominate the random forest. This is an empirical question — the answer should be obtained by inspecting feature importances from the serialised Mitchell model (saved to `CACHE_DIR/rf_model_{year}.pkl`) before committing to a training region.

**If NDVI anomaly and flowering index carry the highest importance:** Katherine/Daly River is preferred. The tropical savanna background is spectrally similar to Mitchell, minimising covariate shift in optical features. The main risk is that the Daly River has perennial baseflow driven by the Tindall/Oolloo aquifers — the RF may learn a stronger dependence on persistent dry-season soil moisture than is appropriate for the highly seasonal Mitchell system.

**If HAND-derived flood connectivity carries the highest importance:** The McArthur River is preferred over Katherine. Its Barkly Tablelands setting has more comparable low-gradient floodplain behaviour to the Mitchell megafan than the escarpment-constrained Daly system. The Georgina/Barkly region would be the geomorphological best-match but is disqualified by data quality concerns (see above).

**Riparian sightings are more valuable than upland sightings.** *Parkinsonia aculeata* disperses primarily by hydrochory — seeds are buoyant and transported by floodwater, establishing most aggressively in floodplain depressions and backwater areas adjacent to active drainage. Field survey in the Cape River catchment (north QLD) found 80% of the population within 1 km of a waterway, with the highest densities in the 50 m–1 km riparian zone rather than the immediate bank (which is scoured by high-energy flow). Training on corridor sightings forces the RF to correctly weight hydrological proximity, preventing commission errors in dry upland savanna. Given the ALA access bias, corridor sightings along major rivers are also likely to be more accurately georeferenced and representative of true infestation structure than scattered inland points.

**Practical recommendation:** Use the McArthur River region as the primary training domain. It has the highest sighting density of the credible candidates, the most clearly riparian spatial pattern, and a reasonable latitudinal match to Mitchell (~17°S vs ~15–17°S). Katherine/Daly is a useful secondary option if feature importance analysis indicates spectral features dominate.

---

## Phenological Window Considerations

The current pipeline uses fixed temporal windows calibrated to the Mitchell catchment:

- **Dry season composite:** May–October
- **Flowering detection window:** August–October
- **Flood season:** January–May

These windows cannot be applied unchanged to NT training regions:

**Flowering:** *Parkinsonia aculeata* flowering is triggered by rising temperatures after winter and responds opportunistically to rainfall pulses. The August–October window is likely valid for Katherine (~13.5°S, similar to Mitchell) and probably adequate for McArthur (~17°S), but should be verified against phenological observations. At higher latitudes (Barkly, ~19°S), peak flowering may be delayed and more erratic.

**Flood season / wet season onset:** The Australian summer monsoon arrives progressively later with distance from the tropical coast. The coastal NT and Cape York Peninsula experience wet season onset in late October to early November. For inland regions like the southern Barkly Tablelands, median onset is mid-January or later. Applying the January–May flood window to Katherine is reasonable; applying it to McArthur may clip the early wet season, and it would be substantially misaligned for the Barkly/Georgina system. The `FLOOD_SEASON_START` should be shifted to approximately February for McArthur to ensure the imagery captures full inundation rather than the transitional pre-wet state.

The `COMPOSITE_START` / `COMPOSITE_END` (dry season) are overridable via environment variables in `config.sh`. `FLOWERING_WINDOW_START` / `FLOWERING_WINDOW_END` and `FLOOD_SEASON_START` / `FLOOD_SEASON_END` are currently hardcoded constants in `config.py` and would need to be made env-overridable for an NT training run.

---

## Pipeline Transfer Approach

The pipeline currently trains and predicts in a single Stage 5 pass, serialising the fitted model to `CACHE_DIR/rf_model_{year}.pkl`. The transfer workflow requires splitting this into two modes:

1. **Train mode (NT region):** Run stages 1–5 with NT bbox and adjusted phenological windows → fitted model written to `.pkl`
2. **Predict mode (Mitchell):** Run stages 1–4 as normal on Mitchell → Stage 5 loads the NT `.pkl` instead of training → writes Mitchell probability raster

Implementation requires a `--model` flag in `analysis/05_classifier.py` that, when supplied, skips label assembly and model fitting and proceeds directly to prediction using the loaded classifier. The `.pkl` stores `feature_names` alongside the model; these should be asserted to match `FEATURE_NAMES` at load time.

The feature set (NDVI anomaly, flowering index, VV/VH backscatter, NDVI median, GLCM texture, distance to watercourse) is derived entirely from Sentinel-2, Sentinel-1, and the drainage network — nothing Mitchell-specific — so the feature stack is directly comparable between regions.

Stage 6 (priority patches, distance-to-Kowanyama scoring) is Mitchell-specific and does not need to run on the NT training region. The NT run only needs to reach Stage 5.

---

## Known Risks

- **Spectral covariate shift:** Even within tropical savanna, background NDVI, soil brightness, and seasonal amplitude vary between regions. If the shift is severe, unsupervised domain adaptation (histogram matching or canonical correlation analysis) may be needed to align the NT feature distributions to Mitchell before prediction.
- **False absences:** Pseudo-absence sampling assumes ALA records have reasonable spatial coverage. In remote, inaccessible areas, pixels sampled as absences may contain uninventoried Parkinsonia. This risk is lower for well-surveyed NT regions (Katherine, McArthur) than for the Mitchell catchment itself.
- **CRS zone mismatch:** The pipeline uses `EPSG:7855` (GDA2020 / MGA Zone 55) as `TARGET_CRS`. Katherine/Daly and McArthur fall in Zone 52–53. The fallback CRS in `utils/dem.py` and `utils/sar.py` is hardcoded to Zone 55 and must be corrected before running in the NT. `TARGET_CRS` should be set to `EPSG:7852` (Zone 52) or `EPSG:7853` (Zone 53) in `config.sh` for the NT run.
