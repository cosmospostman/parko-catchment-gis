# Plan: NBAR correction for cross-tile radiometric consistency

## Context

The Kowanyama scene parquet mixes observations from two overlapping S2 MGRS tiles (54LWH and 54LWJ). The tiles see the landscape at different viewing angles; even after L2A atmospheric correction, B05/B07 (and derived NDVI, SWIR) disagree between tiles. This produces visible N-S banding in classifier output wherever only one tile contributes.

The principled fix is BRDF normalisation (NBAR): apply a per-observation c-factor correction that removes the viewing-angle dependence, normalising all observations to a common nadir/fixed-solar-zenith geometry (Roy et al. 2016, same coefficients used in Landsat ARD and recommended for S2).

**Key constraints discovered:**
- `sen2nbar` is xarray-only and won't integrate with the patch-based COG pipeline
- earth-search does **not** serve angle raster assets (VZA/SZA) as COGs
- Each STAC item has a `granule_metadata.xml` on the element84 S3 bucket (e.g. `s3://sentinel-cogs/.../granule_metadata.xml`) containing 23×23 solar/view angle grids at 5 km spacing — these are the source of truth
- `requests` is already a dependency; `xml.etree.ElementTree` is stdlib
- The correction belongs in `pixel_collector.py`'s `extract_item_to_df()` — apply c-factor to each pixel's band values before writing to parquet

## Architecture

Two new modules, one change to `extract_item_to_df()`:

```
utils/granule_angles.py   — fetch + parse granule_metadata.xml → interpolated angle arrays
utils/nbar.py             — Roy 2016 BRDF kernels + c-factor computation
utils/pixel_collector.py  — wire NBAR into extract_item_to_df()
```

The NBAR correction is **opt-in** via a boolean flag to `collect()`. Existing parquet files are unaffected. The parquet schema gains no new columns — the corrected reflectances replace the raw values in the same band columns.

---

## Critical files

| File | Role |
|---|---|
| `utils/pixel_collector.py` | `extract_item_to_df()` at line 141 — apply correction here |
| `utils/granule_angles.py` | **new** — angle grid fetch and interpolation |
| `utils/nbar.py` | **new** — c-factor formula and band coefficients |
| `requirements.txt` | no changes needed (`requests` already present) |

---

## Step 1 — New module: `utils/granule_angles.py`

Fetches the `granule_metadata.xml` asset for a pystac Item, parses the 23×23 Sun/View angle grids, and returns arrays interpolated to the patch pixel coordinates.

### 1a. Fetch XML
Use `item.assets["granule_metadata"].href` to get the URL (already confirmed available in STAC fixture). Fetch synchronously with `requests.get()` (called once per item, lightweight). Cache in memory via an `lru_cache` keyed on item ID to avoid re-fetching if `extract_item_to_df` is called multiple times for the same item.

### 1b. Parse angle grids
The XML contains:
- `<Sun_Angles_Grid>` → `<Zenith><Values_List>` and `<Azimuth><Values_List>`: 23×23 grids, one per granule
- `<Viewing_Incidence_Angles_Grids>` → `<Zenith>` / `<Azimuth>` per (bandId, detectorId) — use mean across detectors for each band

Parse with `xml.etree.ElementTree`. For each grid, `<Values_List>` contains space-separated floats per row.

### 1c. Interpolate to pixel coordinates
Given a list of `(lon, lat)` point coordinates and the item's CRS/transform (from the patch metadata), interpolate the 23×23 grids (defined on a 5 km UTM grid) to the point coordinates using `scipy.interpolate.RegularGridInterpolator`. `scipy` is already in requirements.

**Public API:**
```python
def get_item_angles(
    item,                          # pystac.Item
    lons: np.ndarray,              # shape (N,), WGS84 longitudes
    lats: np.ndarray,              # shape (N,), WGS84 latitudes
    utm_crs: str,                  # e.g. "EPSG:32754"
    bands: list[str],              # e.g. ["B04", "B05", "B07"]
) -> dict[str, dict[str, np.ndarray]]:
    """Return {band: {'sza': ndarray, 'vza': ndarray, 'saa': ndarray, 'vaa': ndarray}} shape (N,)."""
```

Returns `None` (and logs a warning) on any fetch/parse failure — callers fall back to no correction.

---

## Step 2 — New module: `utils/nbar.py`

Self-contained Roy et al. 2016 c-factor implementation. ~100 lines, no external dependencies beyond numpy.

### 2a. BRDF coefficients
```python
# Roy et al. 2016, Table 1 — Sentinel-2 RossThick-LiSparse coefficients
BRDF_COEFFICIENTS: dict[str, dict[str, float]] = {
    "B02": {"fiso": 0.0774, "fgeo": 0.0079, "fvol": 0.0372},
    "B03": {"fiso": 0.1306, "fgeo": 0.0178, "fvol": 0.0580},
    "B04": {"fiso": 0.1690, "fgeo": 0.0227, "fvol": 0.0574},
    "B05": {"fiso": 0.2085, "fgeo": 0.0256, "fvol": 0.0845},
    "B06": {"fiso": 0.2316, "fgeo": 0.0273, "fvol": 0.1003},
    "B07": {"fiso": 0.2599, "fgeo": 0.0294, "fvol": 0.1197},
    "B08": {"fiso": 0.3093, "fgeo": 0.0330, "fvol": 0.1535},
    "B8A": {"fiso": 0.3430, "fgeo": 0.0453, "fvol": 0.1154},  # proxy from B11
    "B11": {"fiso": 0.3430, "fgeo": 0.0453, "fvol": 0.1154},
    "B12": {"fiso": 0.2658, "fgeo": 0.0387, "fvol": 0.0639},
}

# Nadir target geometry (Roy 2016 standard)
TARGET_SZA_DEG = 45.0
TARGET_VZA_DEG = 0.0
```

### 2b. Kernel functions (vectorised, operate on radians)
```python
def _kvol(sza, vza, raa):   # RossThick kernel
def _kgeo(sza, vza, raa):   # LiSparse-R kernel
def brdf(sza, vza, raa, fiso, fvol, fgeo):
    return fiso + fvol * _kvol(sza, vza, raa) + fgeo * _kgeo(sza, vza, raa)
```

All inputs in radians, all operate on numpy arrays (vectorised across N pixels).

### 2c. C-factor
```python
def c_factor(
    sza_deg: np.ndarray,   # solar zenith, shape (N,)
    vza_deg: np.ndarray,   # view zenith, shape (N,)
    raa_deg: np.ndarray,   # relative azimuth (saa - vaa), shape (N,)
    band: str,
) -> np.ndarray:
    """Return per-pixel c-factor = BRDF(target) / BRDF(observed)."""
    coef = BRDF_COEFFICIENTS[band]
    sza = np.deg2rad(sza_deg)
    vza = np.deg2rad(vza_deg)
    raa = np.deg2rad(raa_deg)
    target_sza = np.full_like(sza, np.deg2rad(TARGET_SZA_DEG))
    target_vza = np.zeros_like(sza)
    target_raa = np.zeros_like(sza)
    brdf_target = brdf(target_sza, target_vza, target_raa, **coef)
    brdf_obs    = brdf(sza, vza, raa, **coef)
    # Guard against degenerate BRDF (near-zero denominator)
    safe_denom = np.where(brdf_obs < 1e-6, 1.0, brdf_obs)
    cf = brdf_target / safe_denom
    # Clamp to physically plausible range [0.5, 2.0] to suppress outliers
    return np.clip(cf, 0.5, 2.0)
```

---

## Step 3 — Wire NBAR into `utils/pixel_collector.py`

### 3a. Add `apply_nbar: bool = False` parameter to `collect()` (line ~211)
Pass it through to each `extract_item_to_df()` call.

### 3b. Extend `extract_item_to_df()` signature
```python
def extract_item_to_df(
    item,
    store: MemoryChipStore,
    point_ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    apply_nbar: bool = False,       # <-- new
) -> pd.DataFrame | None:
```

### 3c. Insert NBAR block after spectral bands are read, before clear-pixel filtering (after line 182)

```python
    # --- NBAR c-factor correction (optional) --------------------------------
    if apply_nbar:
        from utils.granule_angles import get_item_angles
        from utils.nbar import c_factor as compute_cf
        angles = get_item_angles(item, lons, lats, utm_crs=UTM_CRS, bands=list(BANDS))
        if angles is not None:
            for band in BANDS:
                if band not in angles or band_arrays[band] is None:
                    continue
                a = angles[band]
                raa = a["saa"] - a["vaa"]   # relative azimuth
                cf = compute_cf(a["sza"], a["vza"], raa, band)
                band_arrays[band] = np.clip(band_arrays[band] * cf, 0.0, 1.0)
        # If angles is None (fetch failed), band_arrays are left uncorrected
```

> **Why before clear-pixel filtering**: the correction is applied to the full pixel grid then the clear-pixel mask selects the rows to write. This keeps the logic identical to the existing AOT path.

### 3d. Update the `view_zenith` / `sun_zenith` columns

When `apply_nbar=True` and angles were fetched successfully, populate `view_zenith` and `sun_zenith` with the actual angle-based quality scores (consistent with the existing `1.0 - deg/90` formula used in `analysis/timeseries/extraction.py`) instead of `np.ones()`:

```python
        if angles is not None:
            # Use mean SZA/VZA across bands for the quality columns
            sza_mean = np.mean([angles[b]["sza"] for b in BANDS if b in angles], axis=0)
            vza_mean = np.mean([angles[b]["vza"] for b in BANDS if b in angles], axis=0)
            sun_zenith_col  = np.clip(1.0 - sza_mean / 90.0, 0.0, 1.0).astype(np.float32)
            view_zenith_col = np.clip(1.0 - vza_mean / 90.0, 0.0, 1.0).astype(np.float32)
        else:
            sun_zenith_col  = np.ones(n, dtype=np.float32)
            view_zenith_col = np.ones(n, dtype=np.float32)
```

Then use these in the DataFrame construction instead of the bare `np.ones()`.

### 3e. CLI flag
Add `--nbar` flag to the `__main__` argparse block so the correction can be toggled on the command line when re-collecting Kowanyama data.

---

## Re-collection required

The existing Kowanyama parquet was collected without NBAR correction. To fix the banding, the data must be re-collected with `--nbar`:

```bash
python -m utils.pixel_collector \
    --location kowanyama \
    --start 2019-01-01 --end 2025-12-31 \
    --nbar
```

The corrected parquet replaces the existing one. All downstream pipeline scripts (`pipelines/kowanyama_pormpuraaw.py`) are unchanged — they read the same parquet path.

---

## Verification

1. **Unit test `utils/nbar.py`**: at nadir (VZA=0, SZA=45°), c-factor should be 1.0 for all bands (target equals observed geometry).
2. **Diagnostic before/after**: extract `re_ratio = B07/B05` statistics for pixels north vs. south of lat -15.46. After NBAR correction, the values should be consistent across the tile boundary.
3. **Run the pipeline**: `python -m pipelines.kowanyama_pormpuraaw` — the N-S stripe at lat -15.46 should be absent in the output heatmap.
4. **Angle fetch fallback**: deliberately break the XML URL for one item and confirm the pipeline continues without crashing, with a warning logged.
