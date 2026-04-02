# Step 4 Redesign — HAND-Based Flood Connectivity

**Status:** Plan
**Supersedes:** Step 4 (Sentinel-1 flood extent) in DESIGN.md
**Reason:** Sentinel-1 C-band GRD is physically incapable of detecting under-canopy inundation in the dense grassland and sedgeland communities that dominate the Mitchell River floodplain. Validation against actual data confirms no bimodal water/land signature in VV backscatter — the entire catchment returns a single broad peak centred around −16 to −17 dB regardless of season. This is consistent with the literature: C-band at 5.5 cm wavelength saturates within the upper canopy and cannot reach standing water beneath it. Per-scene Otsu thresholding is therefore splitting within the land distribution, not separating water from land.

The replacement approach derives flood connectivity from terrain geometry using the Height Above Nearest Drainage (HAND) index. HAND is a physics-based, sensor-independent measure of how elevated each pixel is above its nearest downslope drainage channel. Low HAND values correspond to pixels that are topographically connected to the drainage network and inundated first during overbank flow. This directly encodes the flood connectivity question the classifier needs to answer.

**Scientific basis:** Rennó et al. (2008), *Remote Sensing of Environment* — original HAND formulation. Nobre et al. (2011) — validation across Amazon megafan floodplains (directly analogous large tropical river system). HAND is used operationally by the NOAA National Water Model for continental-scale flood inundation mapping.

**Known research gap:** HAND has been validated in humid tropical river systems including the Amazon but has not, to our knowledge, been independently validated on Gulf of Carpentaria floodplains specifically. The Mitchell megafan's highly dynamic macro-channels and shifting thalweg present a complex topographic surface that may reduce HAND precision near the active channel belt. Results should be documented as a geomorphic flood probability layer, not a confirmed inundation extent, and treated as such in classifier feature importance analysis.

---

## 1. Data source

### DEM

**Source:** Copernicus DEM GLO-30 (Global 30 m DEM)
**Provider:** ESA / Copernicus Land Service
**Access:** Free, no authentication. Available via AWS S3 at `s3://copernicus-dem-30m/` (requester-pays) or direct HTTPS tile download from the Copernicus Land Service portal.
**Resolution:** ~30 m at the equator (1 arc-second)
**Coverage:** Global, including full Mitchell catchment
**Format:** GeoTIFF tiles (1° × 1°)

The Copernicus GLO-30 is preferred over SRTM for this application. SRTM voids in the Cape York tropics are filled with coarser data; GLO-30 uses a superior void-filling algorithm. More importantly, GLO-30 was derived from TanDEM-X X-band radar (2.4 cm wavelength), which does not penetrate vegetation canopy — it measures the top-of-canopy surface. For HAND, a consistent surface model is more important than bare-earth accuracy, and GLO-30 provides better consistency than SRTM across the catchment's mixed vegetation cover.

**Is it already on the EBS cache?** No. The current EBS volumes hold only Sentinel-1 GRD scenes (`/mnt/ebs/s1cache/`) and Sentinel-2 COGs (`/mnt/ebs/s2cache/`). The Copernicus DEM tiles covering the Mitchell catchment (approximately 20 × 1° tiles across ~141–146°E, ~15–19°S) total roughly 200 MB and need to be downloaded once and added to the EBS layout. They are static — no annual refresh required.

**Proposed EBS path:**
```
/mnt/ebs/dem/
└── copernicus-dem-30m/
    └── <tile>.tif   # e.g. Copernicus_DSM_COG_10_S16_00_E141_00_DEM.tif
```

Add a `scripts/dem_cache.py` download script (equivalent to `scripts/s1_sync_manifest.py`) to fetch the required tiles on first setup. The tile list is fixed and can be hardcoded from the catchment bbox.

### Drainage network

HAND requires a drainage network raster (flow accumulation threshold) to define "nearest drainage." Two options:

**Option A — derive from the DEM itself** using a flow accumulation algorithm (D8 or MFD). Standard practice: threshold flow accumulation at a minimum upstream area (e.g. 1 km²) to define stream pixels. This is fully self-contained but sensitive to DEM artefacts in flat floodplain areas.

**Option B — use the Geoscience Australia GEODATA TOPO 250K stream network** as a raster burn to condition the DEM before computing flow accumulation. This produces a hydrologically consistent drainage network grounded in cartographic survey rather than pure DEM-derived flow paths. The TOPO 250K watercourse layer is freely available from data.gov.au.

Option B is preferred: burning the known drainage network into the DEM before HAND computation improves accuracy in low-relief megafan settings where flat topography produces unstable DEM-derived flow paths. The cartographic drainage captures the macro-channel network that drives lateral connectivity on the Mitchell floodplain.

---

## 2. Validation script changes

**File:** `validate_hand_inputs.py` (new script, replacing `validate_s1_inputs.py` for this step)

The new validation script tests the assumptions of HAND-based flood connectivity rather than SAR backscatter. It operates on the downloaded DEM tiles and the derived HAND raster.

### Assumptions to test

**Assumption 1: DEM void coverage is acceptable**
Expected: <2% of catchment pixels are NoData voids. Voids require interpolation that introduces uncertainty into HAND values near the void boundary. Report void fraction and locations.

**Assumption 2: DEM vertical accuracy is consistent across the catchment**
The Copernicus GLO-30 metadata reports a target vertical accuracy of ±4 m LE90 globally. For HAND, the relevant question is relative accuracy within the floodplain — absolute elevation error matters less than whether the DEM correctly ranks adjacent pixels by elevation.
Expected: DEM elevation histogram over the floodplain (pixels within 5 km of the drainage network) should show a smooth distribution with no obvious artefact spikes. Report histogram and check for bimodal artefacts (common at tile boundaries).

**Assumption 3: Tile seam alignment**
Expected: pixel values at tile boundaries differ by <1 m from their neighbours. Seam mismatches propagate directly into HAND values and can produce linear artefacts in the flood connectivity layer. Report max absolute difference across all tile seams within the catchment.

**Assumption 4: Flow accumulation drainage network is spatially coherent with the known Mitchell River channel**
Expected: DEM-derived stream network (flow accumulation > threshold) overlaps the GA TOPO 250K Mitchell River centreline within 500 m for >90% of the main channel length. Report overlap percentage and plot divergence hotspots. This is the primary check that the HAND computation is anchored to the correct drainage network.

**Assumption 5: HAND value distribution is consistent with a megafan floodplain**
Expected: HAND values within the mapped floodplain extent (using the existing Mitchell catchment boundary clipped to the lower 30% by elevation as a proxy) should have a median below 5 m and a p90 below 20 m. Very high HAND values uniformly across the floodplain would indicate a DEM or flow-routing error.
Report: HAND percentiles (p10, p25, p50, p75, p90, p99) over the full catchment and over the floodplain zone separately.

**Assumption 6: HAND threshold produces a geomorphically plausible flood extent**
At a candidate HAND threshold (e.g. 5 m), the flood connectivity mask should:
- Be spatially continuous along the river corridor
- Not exceed ~30% of the total catchment area (the Mitchell floodplain is extensive but not the whole catchment)
- Show higher coverage in the lower megafan than the upper headwaters

Report: flood connectivity fraction at HAND thresholds of 2 m, 5 m, 10 m, and 15 m. Flag if the 5 m threshold produces <5% or >40% coverage as likely indicating a routing error.

---

## 3. Test case changes

**File:** `tests/analysis/test_04_flood_extent.py`

The existing tests cover SAR-specific logic: `_otsu_threshold`, `_focal_mean_inplace`, `flood_mask_from_scene`, `build_dry_season_reference_mask`, and the accumulation/vectorisation pipeline. All of these are removed or substantially rewritten.

### Tests to remove
- `TestOtsuThreshold` — SAR Otsu logic no longer used in Step 4
- `TestFocalMeanInplace` — speckle filter no longer used in Step 4
- `TestFloodMaskFromScene` — entire SAR classification pipeline removed
- `TestBuildDrySeasonReferenceMask` — dry-season SAR reference mask removed
- Accumulation tests for `flood_count` / `obs_count` / `MIN_OBS` — replaced by static HAND raster

### Tests to add

**`TestDemLoading`**
- `test_dem_tiles_merge_without_seams` — given two synthetic adjacent DEM tiles, verify the merge produces no value discontinuity at the seam
- `test_dem_void_fill_preserves_valid_pixels` — void fill must not alter valid pixels, only fill NoData
- `test_dem_reproject_to_target_crs` — reprojected DEM must cover the catchment bbox in EPSG:7855 at 30 m

**`TestFlowRouting`**
- `test_flow_accumulation_drains_to_outlet` — on a synthetic sloped DEM, all flow must accumulate to the lowest outlet pixel
- `test_stream_burn_lowers_channel_pixels` — after burning the drainage network, channel pixels must have lower elevation than their neighbours
- `test_stream_network_connected` — DEM-derived stream network must form a connected graph with no isolated stream pixels above the flow accumulation threshold

**`TestHandComputation`**
- `test_hand_is_zero_at_stream_pixels` — pixels identified as stream must have HAND = 0
- `test_hand_increases_with_distance_from_stream` — on a synthetic V-shaped valley DEM, HAND must increase monotonically away from the channel
- `test_hand_flat_floodplain` — on a perfectly flat floodplain DEM adjacent to a channel, HAND must equal the elevation difference between the floodplain and channel pixels
- `test_hand_nodata_propagation` — DEM voids must produce NoData HAND values, not zeros

**`TestFloodConnectivityMask`**
- `test_threshold_produces_binary_mask` — HAND threshold must produce a boolean raster with no intermediate values
- `test_low_hand_pixels_are_flood_connected` — pixels at HAND=0 (stream) and HAND < threshold must be flagged True
- `test_high_hand_pixels_excluded` — pixels at HAND >> threshold must be flagged False
- `test_mask_clips_to_catchment` — output must contain no True pixels outside the catchment boundary
- `test_output_is_valid_gpkg` — vectorised output must be a valid GeoPackage readable by geopandas with correct CRS

---

## 4. Implementation

### 4.1 New utility: `utils/dem.py`

Replaces `utils/sar.py` for Step 4. Responsibilities:

```python
def download_dem_tiles(bbox_wgs84: list, out_dir: Path) -> list[Path]:
    """Download Copernicus GLO-30 tiles covering bbox. Returns tile paths."""

def merge_and_reproject_dem(
    tile_paths: list[Path],
    catchment_geom,
    target_crs: str,
    resolution: int,
) -> xr.DataArray:
    """Merge tiles, fill voids, reproject to target_crs at resolution metres.
    Returns a DataArray with x/y coords in target_crs."""

def burn_drainage_network(
    dem: xr.DataArray,
    drainage_gpkg: Path,
    burn_depth_m: float = 10.0,
) -> xr.DataArray:
    """Lower DEM values along the cartographic drainage network by burn_depth_m.
    Ensures DEM-derived flow paths follow the known channel network."""

def compute_flow_accumulation(dem: xr.DataArray) -> xr.DataArray:
    """D8 flow routing. Returns upstream contributing area in pixels."""

def compute_hand(
    dem: xr.DataArray,
    flow_accumulation: xr.DataArray,
    min_upstream_px: int,
) -> xr.DataArray:
    """Compute Height Above Nearest Drainage.

    For each pixel, HAND = elevation(pixel) - elevation(nearest upstream
    stream pixel in the flow network), where stream pixels are those with
    flow_accumulation >= min_upstream_px.

    Returns a DataArray of HAND values in metres, NoData where DEM is void."""

def flood_connectivity_mask(
    hand: xr.DataArray,
    threshold_m: float,
) -> xr.DataArray:
    """Return boolean mask: True where HAND <= threshold_m."""
```

### 4.2 Rewritten `analysis/04_flood_extent.py`

```python
# Step 4 — HAND-based flood connectivity
#
# 1. Download / load Copernicus GLO-30 DEM tiles for catchment bbox
# 2. Merge tiles, fill voids, reproject to EPSG:7855 at 30 m
# 3. Burn GA TOPO 250K drainage network into DEM (optional, recommended)
# 4. Compute D8 flow accumulation
# 5. Compute HAND raster
# 6. Apply HAND threshold (HAND_FLOOD_THRESHOLD_M) → binary flood connectivity mask
# 7. Morphological closing (150 m radius) to merge adjacent floodplain blobs
# 8. Remove patches < MIN_PATCH_PX pixels
# 9. Vectorise → union → simplify (100 m tolerance) → clip to catchment
# Output: flood_extent_YYYY.gpkg  (same filename/format as before — no downstream changes)
# Output: hand_YYYY.tif           (HAND raster for diagnostics)
```

Key constants to add to `config.py`:
```python
HAND_FLOOD_THRESHOLD_M: float = 5.0   # metres above nearest drainage
HAND_MIN_UPSTREAM_KM2: float  = 1.0   # minimum upstream area to define a stream pixel
DEM_BURN_DEPTH_M: float       = 10.0  # channel burn depth for flow conditioning
```

The `HAND_FLOOD_THRESHOLD_M = 5.0` initial value is consistent with HAND-based flood mapping literature for low-relief tropical floodplains and produces flood extents in the range expected for the Mitchell megafan. It should be treated as a calibration parameter: if gauge data or independent flood extent mapping becomes available, recalibrate against it.

### 4.3 Changes to `config.py`

- Add `HAND_FLOOD_THRESHOLD_M`, `HAND_MIN_UPSTREAM_KM2`, `DEM_BURN_DEPTH_M` constants
- Add `hand_raster_path(year)` output path function
- Remove `dry_season_mask_path(year)` — no longer needed
- `flood_extent_path(year)` is unchanged — same output filename

### 4.4 Changes to `scripts/`

- Add `scripts/dem_cache.py` — downloads Copernicus GLO-30 tiles for the catchment bbox to `/mnt/ebs/dem/copernicus-dem-30m/`. Tiles are static; script is idempotent (skips already-downloaded tiles). Add to EBS-SETUP.md as a one-time setup step.
- `scripts/s1_sync_manifest.py` — no longer needed for Step 4, but retain as it may still be used for future SAR work or archival purposes.

### 4.5 What does not change

- Output filename: `flood_extent_{year}.gpkg` — no changes to Step 5 or downstream steps
- Output CRS: EPSG:7855
- Vectorisation and simplification logic in `04_flood_extent.py` is reused unchanged
- `config.py` `flood_extent_path()` and `flood_obs_count_path()` — the obs count path is removed (no longer meaningful); all other output paths unchanged

### 4.6 DESIGN.md update

The Step 4 section of DESIGN.md should be updated to:
- Replace the SAR-based pseudocode with HAND-based pseudocode
- Document the C-band physical limitation finding and cite the validation work
- Note the HAND research gap for Gulf of Carpentaria floodplains
- Update the data sources table to replace Sentinel-1 GRD with Copernicus GLO-30 DEM
- Remove the S1B decommission note; remove the EBS S1 cache from the storage layout (or note it is no longer required for Step 4)
- Update the cross-validation note: compare HAND flood connectivity extent against DEA WIT for the lower floodplain as a sanity check, noting WIT's limitations as described in the research
