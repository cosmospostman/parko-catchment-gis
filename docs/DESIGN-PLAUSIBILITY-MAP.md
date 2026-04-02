# Plausibility Map — Implementation Plan

**Status:** Plan
**Depends on:** Stages 1–4 outputs (ndvi_anomaly, flowering_index, hand raster)
**Output:** `plausibility_map_YYYY.tif`, `plausibility_map_YYYY.gpkg`

---

## Purpose

Produce a ranked plausibility surface for Parkinsonia presence across the Mitchell catchment using only the three most ecologically grounded signals available from existing Stage 1–4 outputs — no training data required. This serves two immediate purposes:

1. **Direct drone survey zone selection** before confirmed Mitchell ground truth exists
2. **Validate signal coherence** by cross-referencing flagged pixels against the 13 known ALA sightings — if the map misses known locations, a threshold or feature is miscalibrated before surveys begin

The plausibility map is an interim product. Once drone survey ground truth is available it is superseded by the trained classifier (Stage 6), which produces calibrated probabilities suitable for seed dispersal modelling. The plausibility map is fit for purpose as a survey-direction tool but not for the eradication vs management decision.

---

## Signals used

| Feature | Source | Ecological rationale |
|---|---|---|
| `ndvi_anomaly` | Stage 2 output | Parkinsonia maintains anomalously high NDVI through dry season while native grasses senesce; persistent positive anomaly is the primary detection signal |
| `flowering_index` | Stage 3 output | Yellow flower flush (Aug–Oct) lifts green reflectance above co-occurring native vegetation; biologically independent of NDVI anomaly |
| `hand` | Stage 4 output | Parkinsonia establishes primarily in floodplain depressions and low-drainage positions; high HAND pixels are ecologically implausible regardless of spectral signal |

These three signals suppress the two main commission error types:
- Spectrally green but non-riparian pixels → killed by HAND threshold
- Low-lying riparian pixels that are not distinctively green → killed by NDVI anomaly + flowering index

GLCM texture and dist_to_watercourse are deliberately excluded: texture is unreliable at Sentinel-2 10 m for small patches, and dist_to_watercourse would require the drainage network layer and introduces geography-fitting risk without a trained model to weight it correctly.

---

## Scoring approach

Rather than hard binary thresholds (which are brittle to cutoff choice), compute a continuous plausibility score per pixel:

```python
# Normalise each feature to [0, 1] over valid pixels
ndvi_norm     = percentile_scale(ndvi_anomaly, lo=2, hi=98)
flower_norm   = percentile_scale(flowering_index, lo=2, hi=98)
hand_inv_norm = 1.0 - percentile_scale(hand, lo=2, hi=98)  # invert: low HAND = high score

# Equal-weight combination
plausibility = (ndvi_norm + flower_norm + hand_inv_norm) / 3.0
# Result: Float32 raster, 0–1, higher = more plausible
```

Equal weighting is a deliberate starting assumption, not a tuned value. The drone survey validation and feature importance analysis will reveal whether any signal should be weighted more heavily before the trained classifier is built.

A secondary binary mask at a fixed threshold (e.g. plausibility > 0.6) is also produced for the drone survey planning GeoPackage — field teams need a polygon layer, not a continuous raster.

---

## Implementation

Implement as `analysis/05a_plausibility_map.py`, inserted between the existing Stage 4 and Stage 5 scripts. It reads existing Stage 2–4 outputs and writes two new outputs:

```
outputs/YYYY/plausibility_map_YYYY.tif     # continuous score, Float32 0–1
outputs/YYYY/plausibility_zones_YYYY.gpkg  # polygons where score > threshold
```

### Script structure

```python
"""Stage 5a — Rule-based plausibility map from NDVI anomaly, flowering index, and HAND."""
import numpy as np
import rioxarray as rxr
import geopandas as gpd
from rasterio.features import shapes
import config

PLAUSIBILITY_THRESHOLD = 0.60   # tunable; controls polygon output only
MIN_PATCH_HA = 0.25             # consistent with Stage 6 MMU


def percentile_scale(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Scale arr to [0,1] using percentile clipping. NaN-safe."""
    valid = arr[np.isfinite(arr)]
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    scaled = (arr - p_lo) / (p_hi - p_lo + 1e-9)
    return np.clip(scaled, 0.0, 1.0)


def main():
    year = config.YEAR
    ndvi_da    = rxr.open_rasterio(config.ndvi_anomaly_path(year)).squeeze()
    flower_da  = rxr.open_rasterio(config.flowering_index_path(year)).squeeze()
    hand_da    = rxr.open_rasterio(config.hand_raster_path(year)).squeeze()

    # Reproject-match to NDVI anomaly grid (reference)
    flower_da = flower_da.rio.reproject_match(ndvi_da)
    hand_da   = hand_da.rio.reproject_match(ndvi_da)

    ndvi_arr   = ndvi_da.values.astype(np.float32)
    flower_arr = flower_da.values.astype(np.float32)
    hand_arr   = hand_da.values.astype(np.float32)

    # NaN mask: pixel must be valid in all three inputs
    valid = np.isfinite(ndvi_arr) & np.isfinite(flower_arr) & np.isfinite(hand_arr)

    plausibility = np.full_like(ndvi_arr, np.nan)
    plausibility[valid] = (
        percentile_scale(ndvi_arr)[valid]
        + percentile_scale(flower_arr)[valid]
        + (1.0 - percentile_scale(hand_arr))[valid]
    ) / 3.0

    # Write continuous raster
    out_da = ndvi_da.copy(data=plausibility)
    out_path = config.OUTPUTS_DIR / str(year) / f"plausibility_map_{year}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_da.rio.to_raster(str(out_path), dtype="float32", compress="deflate")

    # Vectorise pixels above threshold
    binary = (plausibility >= PLAUSIBILITY_THRESHOLD).astype(np.uint8)
    transform = ndvi_da.rio.transform()
    crs = ndvi_da.rio.crs
    geoms = [
        geom for geom, val
        in shapes(binary, mask=binary, transform=transform)
        if val == 1
    ]
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gdf = gdf[gdf["area_ha"] >= MIN_PATCH_HA].reset_index(drop=True)

    zones_path = config.OUTPUTS_DIR / str(year) / f"plausibility_zones_{year}.gpkg"
    gdf.to_file(str(zones_path), driver="GPKG")
```

---

## Verification

```python
import numpy as np
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import config

year = config.YEAR

# Raster checks
plaus = rxr.open_rasterio(config.OUTPUTS_DIR / str(year) / f"plausibility_map_{year}.tif").squeeze()
vals = plaus.values
valid = vals[np.isfinite(vals)]

assert valid.min() >= 0.0 and valid.max() <= 1.0, "Plausibility values outside [0, 1]"
assert len(valid) > 0, "All pixels are NaN"

nan_frac = np.isnan(vals).mean()
assert nan_frac < 0.30, f"NaN fraction {nan_frac:.1%} too high — check input rasters"

# Spatial coherence: high-plausibility pixels should cluster near drainage
# (visual check — load in QGIS and overlay drainage network)

# ALA cross-check: at least 8 of 13 known sightings should fall in top quartile
ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
if ala_cache.exists():
    ala = gpd.read_file(str(ala_cache)).to_crs(plaus.rio.crs)
    import rasterio
    from rasterio.transform import rowcol
    transform = plaus.rio.transform()
    hits = 0
    threshold_75 = float(np.percentile(valid, 75))
    for pt in ala.geometry:
        row, col = rowcol(transform, pt.x, pt.y)
        H, W = vals.shape
        if 0 <= row < H and 0 <= col < W and np.isfinite(vals[row, col]):
            if vals[row, col] >= threshold_75:
                hits += 1
    hit_rate = hits / len(ala)
    print(f"ALA sightings in top quartile: {hits}/{len(ala)} ({hit_rate:.0%})")
    if hit_rate < 0.5:
        print("WARNING: fewer than 50% of known sightings fall in the top quartile — review thresholds")
```

The ALA cross-check is diagnostic, not a hard assertion — the 13 sightings are too few and too spatially biased to be treated as ground truth. A low hit rate should prompt investigation of which signal is failing (overlay each feature individually against the sighting locations) rather than automatic threshold adjustment.

---

## Integration with run.sh

Stage 5 is a standard pipeline step — it runs between Stage 4 and Stage 6 and is already wired into `run.sh`:

```bash
run_step_or_abort 5 "05_plausibility_map"  "05_verify_plausibility_map"
```

This step is fast (pure numpy operations on existing rasters, < 5 min) and produces no large intermediate files.

---

## verify-input script

`verify-input/05_verify_plausibility_inputs.py` checks that all three upstream inputs exist and are internally consistent before Stage 5 runs. It follows the same two-section pattern as Stage 04:

**Section 1 — Input files:**
- NDVI anomaly raster exists and is non-empty
- Flowering index raster exists and is non-empty
- HAND raster exists and is non-empty
- All three rasters share the same CRS and grid dimensions (pixel count, transform) — mismatched grids produce silent `reproject_match` artifacts

**Section 2 — Scientific sanity checks:**
- NDVI anomaly: mean close to zero (|mean| < 0.05), std in expected range (0.03–0.20) — large deviation indicates baseline miscalculation or wrong year
- Flowering index: reasonably decorrelated from NDVI anomaly (correlation < 0.70) — high correlation means both features are measuring the same signal and one should be dropped
- HAND: at least 5% of valid pixels below 5 m (confirms the raster contains floodplain terrain, not just uplands); void fraction < 5%
- CRS check: all three rasters in the same projected CRS (expected EPSG:7855)

**Usage:**
```bash
source config.sh
python verify-input/05_verify_plausibility_inputs.py
```

No arguments needed — all paths are resolved via `config.py`.

---

## Test cases

`tests/analysis/test_05_plausibility_map.py` covers two categories matching Stage 04's pattern.

### Mechanics tests

These test the pure functions in `analysis/05_plausibility_map.py` with synthetic numpy inputs — no real rasters required.

**`TestPercentileScale`**
- `test_output_range_is_zero_to_one` — all outputs clipped to [0, 1]
- `test_uniform_array_returns_constant` — flat input returns all-0.5 (or constant after clip)
- `test_nan_values_propagate` — NaN inputs produce NaN outputs, not zeros
- `test_monotone_increasing_input` — strictly increasing input maps to strictly increasing output
- `test_low_percentile_clip` — values below the 2nd percentile all map to 0.0

**`TestPlausibilityScore`**
- `test_score_range` — composite score is always in [0, 1] for all-finite inputs
- `test_high_ndvi_high_flower_low_hand_scores_high` — pixel with ndvi_norm=1, flower_norm=1, hand_inv_norm=1 → score=1.0; confirms the ecological ranking is correct
- `test_low_ndvi_scores_low` — pixel with ndvi_norm=0, others moderate → score < 0.4
- `test_high_hand_scores_low` — pixel at HAND=200 m (upland) → hand_inv_norm≈0 → low composite regardless of spectral signal
- `test_nan_any_input_produces_nan` — if any of the three inputs is NaN, output is NaN (pixel excluded from map)
- `test_equal_weighting` — score equals simple mean of the three normalised inputs

**`TestBinaryThreshold`**
- `test_threshold_produces_binary_output` — output contains only True/False, no intermediate values
- `test_pixels_above_threshold_are_true` — score=0.8 with threshold=0.6 → True
- `test_pixels_below_threshold_are_false` — score=0.4 with threshold=0.6 → False
- `test_nan_pixels_are_false` — NaN score → False (never flagged as plausible)
- `test_min_patch_removal` — isolated single-pixel patches below MMU are removed from the polygon output

**`TestVectorisation`**
- `test_output_is_valid_gpkg` — vectorised output is readable, has valid geometries and correct CRS
- `test_area_ha_attribute_correct` — `area_ha` attribute matches geometry area / 10000 within 1%
- `test_empty_scene_produces_empty_geodataframe` — all-zero score raster produces empty GeoDataFrame without raising
- `test_polygon_area_within_10pct_of_raster_area` — vector area consistent with pixel count × pixel area

### Scientific theory tests

These test that the scoring approach encodes correct ecological reasoning about *Parkinsonia aculeata* in the Mitchell catchment. They use synthetic rasters designed to represent specific ecological scenarios.

**`TestEcologicalSignalEncoding`**

- `test_riparian_green_pixel_ranks_highest` — a pixel with high NDVI anomaly + high flowering + low HAND (archetypal riparian Parkinsonia) scores higher than any partial match
- `test_upland_green_pixel_suppressed_by_hand` — a spectrally positive pixel (high NDVI anomaly, high flowering) at HAND=80 m scores below a weaker spectral pixel at HAND=2 m; confirms HAND is doing its job as an ecological gate
- `test_dry_season_greenness_required` — a low-HAND, high-flowering pixel with ndvi_anomaly=0 (not anomalously green) scores below threshold; confirms spectral signal is necessary, not just topographic position
- `test_flowering_signal_is_additive` — adding a positive flowering signal to an already-plausible pixel increases its score; confirms the independence of the flowering signal from NDVI anomaly
- `test_all_signals_must_be_weak_for_low_score` — a pixel must be weak on all three signals simultaneously to score near zero; being weak on one signal alone doesn't suppress it below threshold (prevents over-suppression)
- `test_known_parkinsonia_habitat_type_scores_above_threshold` — a synthetic pixel parameterised to match published Parkinsonia habitat characteristics (NDVI anomaly > 0.15, flowering_index in 80th percentile, HAND < 5 m) scores above the default 0.60 threshold; this is the key ecological contract test

**`TestALACoherence`**

- `test_ala_sightings_in_top_half` — using real cached ALA occurrences (if present, otherwise skip), at least 50% of the 13 Mitchell sightings should fall in the top half of the plausibility distribution on a synthetic score raster parameterised to match expected Mitchell conditions; this test is marked `pytest.mark.skipif` when the ALA cache is absent so CI passes without data

---

## Limitations

- Equal feature weighting is an assumption. The trained classifier (Stage 6) will provide empirical weights once ground truth exists — if it reveals that HAND dominates heavily, the plausibility map threshold for HAND should be tightened retrospectively.
- The plausibility score is not a probability. It should not be used for seed dispersal modelling or eradication/management decisions. It is a ranked map for survey direction only.
- **The current three signals do not discriminate Parkinsonia from other woody riparian species.** NDVI anomaly, flowering index, and HAND were chosen to separate Parkinsonia from dry-season senescent grass — they do not distinguish it from obligate-riparian evergreens (Melaleuca, Casuarina, Pandanus, river red gum) or co-occurring invasive legumes (Leucaena, Prosopis) that occupy the same floodplain positions and maintain similar dry-season greenness. In its current form the plausibility map identifies "low-lying pixels with persistent woody greenness", not Parkinsonia specifically. See `IMPROVEMENTS.md` for a concrete fix (phenological pulse signal) that would address this within Stage 5.
