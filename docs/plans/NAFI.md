# Plan: NAFI Fire Log Pipeline

## Context
Fire history is a major confounding factor for Parkinsonia discrimination in northern Australia. Post-fire pixels have suppressed NIR, elevated SWIR, and altered phenology for months to years. Without knowing when a pixel last burned — and how often — it's impossible to cleanly interpret spectral signals.

NAFI (North Australian Fire Information) publishes monthly fire scar polygons at ~30m resolution back to ~2000, covering all of Cape York and the Gulf Country. The goal is to maintain a per-pixel fire log parquet alongside each location's S2 parquet, enabling fire-based signals (frequency, time-since-fire, fire seasonality) to be computed like any other signal.

## Output schema

```
data/pixels/<location_id>/<location_id>_fire.parquet
```

Columns:
```
point_id  | lon | lat | burn_year | burn_month | source
int64     | f32 | f32 | int16     | int8       | str
```

- `source = "nafi"` — leaves room for future sources (MODIS burned area, manual mapping)
- Month resolution only — NAFI doesn't provide sub-monthly dates
- One row per pixel per burn event (sparse; most pixels have few entries)

## NAFI data source

NAFI fire scars are available as monthly GeoTIFF or shapefile downloads from:
- `https://firenorth.org.au/nafi3/` (web map, not directly scriptable)
- **Preferred**: The underlying WFS/WMS or direct file server — needs investigation

**Action required before implementation**: Determine the programmatic access URL. Options:
1. NAFI WFS endpoint (GeoServer-based — likely queryable by bbox + date)
2. Direct shapefile/GeoPackage downloads if available
3. Geoscience Australia / DEA fire scar products as fallback (different source, similar coverage)

This is the main unknown. The rest of the pipeline is straightforward once we have a reliable fetch URL.

## Pipeline design

### `utils/nafi.py` — new utility module

```python
def fetch_nafi_scars(
    bbox: tuple[float, float, float, float],  # lon_min, lat_min, lon_max, lat_max
    year_from: int,
    year_to: int,
    cache_dir: Path,
) -> list[shapely.Geometry]:
    """Fetch NAFI fire scar polygons for bbox + date range. Caches locally."""
    ...

def build_fire_log(
    pixel_parquet: Path,
    bbox: tuple[float, float, float, float],
    year_from: int = 2000,
    year_to: int | None = None,  # defaults to current year
    cache_dir: Path = Path("data/nafi_cache"),
) -> pd.DataFrame:
    """Spatial join NAFI scars against pixel centroids. Returns fire log DataFrame."""
    ...
```

### `pipelines/fire_log.py` — CLI entry point

```bash
python pipelines/fire_log.py --location frenchs
python pipelines/fire_log.py --location frenchs --year-from 2010 --year-to 2024
python pipelines/fire_log.py --location frenchs --force  # re-fetch even if cache exists
```

Flow:
1. Load location → get bbox + parquet path
2. Extract unique pixel centroids from parquet (point_id, lon, lat) — read only those 3 columns
3. Call `fetch_nafi_scars(bbox, year_from, year_to, cache_dir)` — fetches monthly, caches per month
4. For each month's scar polygons: point-in-polygon test against all pixel centroids
5. Collect hits → write `<location_id>_fire.parquet`

### Spatial join approach

Use `shapely` + numpy vectorisation:
```python
from shapely.vectorized import contains  # fast batch point-in-polygon
```
Or geopandas `sjoin` if available. Avoid iterating per-pixel — batch by month's polygon set.

## Fire-based signals (future — not in scope now)

Once the fire log exists, these become simple aggregations:

| Signal | Description |
|---|---|
| `fire_frequency` | Burns per year over observation period |
| `time_since_fire` | Months since most recent burn (relative to query date) |
| `fire_seasonality` | Fraction of burns in early dry (May–Jul) vs late dry (Aug–Oct) |
| `post_fire_recovery` | Join fire dates to NDVI curve; measure NDVI recovery slope |

These would be implemented as `FireFrequencySignal`, `TimeSinceFireSignal` etc. in `signals/fire.py`, following the standard Signal API.

`describe.py` and `explore.py` can then incorporate fire signals once implemented — no architectural changes needed.

## Integration with SCL composition

The fire log should be integrated with `SclCompositionSignal` (see SCL.md). Post-fire pixels appear
as bare soil (5) or unclassified (7) in SCL, making it impossible to distinguish seasonal bare ground
from fire disturbance without fire context. The `plot_scl_composition` chart should overlay burn events
as markers (e.g. vertical lines) at the months they occurred, joined from the fire log via `point_id`.
The chart works without fire data but is significantly more interpretable with it.

## Caching strategy

- Cache NAFI downloads per month: `data/nafi_cache/nafi_<YYYY>_<MM>.gpkg`
- `build_fire_log` is idempotent — safe to re-run; appends new months, doesn't duplicate
- Fire log parquet is versioned by re-running the pipeline (overwrite, not append)

## Dependencies

- `shapely` — point-in-polygon (likely already present via geopandas)
- `geopandas` — for reading shapefiles/GeoPackage if NAFI provides those formats
- `requests` or `httpx` — HTTP download (already used in `utils/fetch.py`)

## Files to create
- `utils/nafi.py` — fetch + spatial join logic
- `pipelines/fire_log.py` — CLI pipeline

## Open questions
1. **NAFI programmatic URL** — needs investigation before implementation begins
2. **Historical coverage** — confirm NAFI goes back far enough to cover the S2 archive start (~2017); if not, consider MODIS burned area as a supplement for pre-2017
3. **Polygon vs raster** — if NAFI provides rasters rather than polygons, the spatial join approach changes (sample raster at pixel centroids instead)
