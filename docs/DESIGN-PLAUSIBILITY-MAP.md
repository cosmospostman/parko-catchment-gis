# Plausibility Map — Implementation Plan

**Status:** Plan
**Depends on:** Stages 1–4 outputs (ndvi_anomaly, flowering_index, hand raster)
**Output:** `plausibility_map_YYYY.tif`, `plausibility_map_YYYY.gpkg`

---

## Purpose

Produce a ranked plausibility surface for Parkinsonia presence across the Mitchell catchment using only the three most ecologically grounded signals available from existing Stage 1–4 outputs — no training data required. This serves two immediate purposes:

1. **Direct drone survey zone selection** before confirmed Mitchell ground truth exists
2. **Validate signal coherence** by cross-referencing flagged pixels against the 13 known ALA sightings — if the map misses known locations, a threshold or feature is miscalibrated before surveys begin

The plausibility map is an interim product. Once drone survey ground truth is available it is superseded by the trained classifier (Stage 5), which produces calibrated probabilities suitable for seed dispersal modelling. The plausibility map is fit for purpose as a survey-direction tool but not for the eradication vs management decision.

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

Add as an optional step between Stage 4 and Stage 5:

```bash
# In run.sh, after run_step 04_flood_extent:
run_step 05a_plausibility_map   # interim product — no training data required
```

This step is fast (pure numpy operations on existing rasters, < 5 min) and produces no large intermediate files.

---

## Limitations

- Equal feature weighting is an assumption. The trained classifier (Stage 5) will provide empirical weights once ground truth exists — if it reveals that HAND dominates heavily, the plausibility map threshold for HAND should be tightened retrospectively.
- The plausibility score is not a probability. It should not be used for seed dispersal modelling or eradication/management decisions. It is a ranked map for survey direction only.
- Melaleuca and other riparian species with similar dry-season greenness and low HAND positions will produce false positives. The drone surveys will characterise the false positive rate in the Mitchell context, which informs how aggressively to filter plausibility zones before fieldwork.
