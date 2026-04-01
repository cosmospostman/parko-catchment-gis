# Terrain Correction — Investigation & Outcome

## Goal

Step 04 (`04_flood_extent.py`) maps wet-season flood extent from Sentinel-1 GRD scenes.
The original design used **sarsen** for terrain-corrected SAR processing (gamma-naught
radiometric correction using the Copernicus DEM), which is important for accurate
backscatter thresholding in hilly terrain.

## What was tried

### 1. sarsen + xarray_sentinel (original approach)

**Versions:** sarsen 0.9.5, xarray_sentinel 0.9.5

sarsen reads the SAFE product root, parses the annotation XML via `xarray_sentinel`
to extract GCPs and burst metadata, then performs range-Doppler terrain correction
against the COP-DEM-30m.

**Failure:** `ValueError: zero-size array to reduction operation fmin which has no identity`

Traceback led to `xarray_sentinel.sentinel1.open_gcp_dataset` →
`get_footprint_linestring` → `.min()` on an empty `azimuth_time` array.

**Root cause:** The S3 bucket (`sentinel-s1-l1c`) stores annotation XML with child
elements (`<azimuthTime>2025-...</azimuthTime>`), but `xarray_sentinel`'s
`esa_safe.parse_tag_as_list` expects XML attributes. It returns empty dicts for every
GCP, so `azimuth_time` is an empty list.

Confirmed by:
```python
from xarray_sentinel import esa_safe
gcps = esa_safe.parse_tag_as_list(annotation_path, ".//geolocationGridPoint")
# → [{}, {}, ...]  — 210 GCPs present in XML but all parsed as empty
```

The raw XML has the correct structure:
```xml
<geolocationGridPoint>
  <azimuthTime>2025-05-27T19:52:14.619073</azimuthTime>
  <line>0</line>
  <pixel>0</pixel>
  ...
</geolocationGridPoint>
```

This is a bug (or unsupported variant) in `xarray_sentinel` 0.9.5 — it cannot parse
the child-element style annotation format used by this S3 bucket. No fix was available
in the current release.

### 2. Intermediate dead ends

- **stackstac fallback:** `sentinel-1-grd` items on Element84 earth-search v1 lack
  `proj:shape`/`proj:transform` STAC metadata, so stackstac returns a 1×1 pixel array.
- **odc-stac fallback:** Same issue — `resolution=10` in `EPSG:4326` is interpreted as
  10 degrees, producing a 1×1 result. Switching to `EPSG:7855` produced correct bounds
  but the underlying GeoTIFFs have no embedded CRS/geotransform (raw ESA SAFE
  measurement files), so odc-stac cannot warp them.
- **Missing annotation files:** The S1 sync manifest originally guessed annotation
  filenames (`iw-vv.xml`→`s1a-iw-grd-vv-...-001.xml`) which don't exist on S3. Fixed
  to sync `iw-vv.xml`/`iw-vh.xml` directly, plus runtime symlinks from the long ESA
  names (referenced in `manifest.safe`) to the short S3 names.

## Current approach (GCP warp, no terrain correction)

`utils/sar.py` now uses `rasterio.warp.reproject` with GCPs parsed directly from the
annotation XML child elements (bypassing `xarray_sentinel` entirely):

1. Parse `<geolocationGridPoint>` elements from `annotation/iw-vv.xml` using stdlib
   `xml.etree.ElementTree`.
2. Build `rasterio.control.GroundControlPoint` objects from pixel/line/lat/lon/height.
3. Reproject the raw uint16 measurement TIFF to EPSG:7855 at 10 m using bilinear
   resampling.
4. Convert DN² / 1e8 as a linear sigma-naught proxy (no calibration LUT applied).

**Limitation:** No radiometric terrain correction. Backscatter values will be affected
by local incidence angle variation in hilly terrain, potentially causing false positives
(layover/shadow) or suppressed flood signal on slopes. For the Mitchell catchment, which
is predominantly flat to gently undulating, this is acceptable for flood extent mapping.

## Re-enabling terrain correction

If terrain correction becomes necessary:

1. **Fix xarray_sentinel:** Patch `esa_safe.parse_tag_as_list` to handle child-element
   style XML (or upgrade if a fix is released upstream).
2. **Apply calibration LUT:** Use `annotation/calibration/calibration-iw-vv.xml` to
   convert DN to calibrated sigma-naught (σ⁰ = DN² / β²(pixel)) before thresholding.
3. **Alternative:** Use `pyroSAR` or `s1reader` which parse the same XML format
   correctly and can produce terrain-flattened gamma-naught.
