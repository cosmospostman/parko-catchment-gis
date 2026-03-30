# STACKING — Fix for Full-Extent Eager Stacking

## Problem

Steps 01, 02 (baseline build), and 03 call `stackstac.stack()` or `odc.stac.load()`
over the full spatial extent of the Mitchell catchment (~375 km × 452 km) before
calling `.compute()`. Memory requirement scales as `n_scenes × spatial_extent`,
which OOMs even on 32 GB RAM:

- Step 01: 1129 scenes × 8 bands × 37508 × 45234 px × 4 B ≈ 61 TB uncompressed
- Step 03: similar scene count, fewer bands, same spatial extent
- Step 02 baseline: ~40 years of Landsat × 2 bands, same spatial extent

The Dask distributed scheduler cannot help because the *task graph itself* (held
in the single-threaded scheduler process) grows with `n_scenes × n_chunks` and
exhausts memory before any pixel data is processed.

## Fix: Spatial Tiling

Process the catchment in spatial tiles. Each tile:
1. Clips the STAC items to the tile bbox at load time (stackstac/odc-stac support this natively)
2. Computes the reduction (median NDVI, median ratio, etc.) in memory
3. Writes a tile GeoTIFF to a scratch directory
4. Discards all arrays before moving to the next tile

Tiles are independent — N tiles run concurrently via `ThreadPoolExecutor`,
saturating the network pipe without any single tile exceeding the memory budget.
The scratch tiles are merged into the final COG at the end.

## Memory Budget (32 GB system)

Working backwards from 32 GB with 4 concurrent tiles:

```
usable RAM per tile = 32 GB × 0.6 / 4 concurrent = ~4.8 GB
n_scenes = 1129 (worst case, step 01)
n_bands  = 8
bytes    = 4 (float32)

max_pixels_per_tile = 4.8 GB / (1129 × 8 × 4) = ~132,000 px²
tile_side = sqrt(132,000) ≈ 360 px → round down to 256 px (conservative)
```

`TILE_SIZE_PX = 256` at 10 m resolution = 2.56 km × 2.56 km tiles.
Over the 37508 × 45234 px extent that is ~(147 × 177) = ~26,000 tiles.
With 4 concurrent tiles and fast S3, this is still the right approach —
network concurrency comes from the 4 simultaneous tile loads, each issuing
many S3 range requests in parallel via the threaded Dask scheduler.

The constant is exposed as `TILE_SIZE_PX` in each analysis script and as the
`TILE_SIZE_PX` env var so it can be tuned for different hardware.

## New Module: `utils/tiling.py`

```python
def make_tile_bboxes(
    full_bbox: list[float],   # [minx, miny, maxx, maxy] in degrees (EPSG:4326)
    resolution_m: int,        # e.g. 10
    tile_size_px: int,        # e.g. 256
) -> list[list[float]]:
    """Return a list of [minx, miny, maxx, maxy] tile bboxes covering full_bbox."""
```

Tiles are computed in projected space (EPSG:7855) then converted back to
EPSG:4326 for passing to stackstac, which takes `bounds_latlon`.

Also expose:

```python
def merge_tile_rasters(
    tile_paths: list[Path],
    out_path: Path,
    nodata: float,
    crs: str,
) -> None:
    """Mosaic tile GeoTIFFs into a single COG using rioxarray.merge_arrays."""
```

## Changes to Analysis Scripts

### `analysis/01_ndvi_composite.py`

Replace the single `load_stackstac → .compute()` block with:

```python
TILE_SIZE_PX = int(os.environ.get("TILE_SIZE_PX", "256"))
TILE_WORKERS = int(os.environ.get("TILE_WORKERS", "4"))

tile_bboxes = make_tile_bboxes(bbox, config.TARGET_RESOLUTION, TILE_SIZE_PX)
scratch_dir = Path(config.WORKING_DIR) / f"tiles_ndvi_{config.YEAR}"
scratch_dir.mkdir(exist_ok=True)

def process_tile(tile_bbox):
    stack = load_stackstac(items, bands=load_bands, bbox=tile_bbox, ...)
    # mask → NDVI → median → write tile → return path

with ThreadPoolExecutor(max_workers=TILE_WORKERS) as pool:
    tile_paths = list(pool.map(process_tile, tile_bboxes))

merge_tile_rasters(tile_paths, out_path, nodata=np.nan, crs=config.TARGET_CRS)
scratch_dir.rmdir()  # clean up
```

The `LocalCluster` added in the previous iteration should be **removed** —
the threaded Dask scheduler (`scheduler="threads"`) inside each tile worker
is sufficient and avoids the distributed scheduler overhead.

The GDAL env vars (`GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` etc.) remain and
should be set as OS env vars before the `ThreadPoolExecutor` starts so all
worker threads inherit them (rasterio reads env at open time).

### `analysis/03_flowering_index.py`

Same pattern as step 01. `TILE_SIZE_PX` default is the same (same worst-case
scene count). Step 03 has fewer bands (5 vs 8) so memory per tile is lower —
the default tile size is still conservative.

### `analysis/02_ndvi_anomaly.py` — `_build_baseline()`

`odc.stac.load()` has the same problem. Replace with the same tiled pattern,
but note:
- Baseline uses 2 bands (vs 8) and 30 m resolution (vs 10 m), so tiles can
  be larger: `TILE_SIZE_PX = 1024` default for the baseline build.
- The `REBUILD_BASELINE` cache check stays unchanged — tiling only affects the
  compute path inside `_build_baseline()`.
- The `LocalCluster` added in the previous iteration should also be removed here.

## Changes to `run.sh`

Add `TILE_SIZE_PX` and `TILE_WORKERS` to the `export` block and the
configuration table printed at startup:

```bash
export TILE_SIZE_PX="${TILE_SIZE_PX:-256}"
export TILE_WORKERS="${TILE_WORKERS:-4}"
```

Add optional CLI flags:

```
--tile-size N     Spatial tile size in pixels (default 256, tune for RAM)
--tile-workers N  Concurrent tiles (default 4, tune for CPU/network)
```

## Tests

### New: `tests/unit/test_utils_tiling.py`

- `test_make_tile_bboxes_covers_full_extent`: tiles union == full bbox
- `test_make_tile_bboxes_no_overlap`: adjacent tiles share only an edge
- `test_make_tile_bboxes_single_tile`: bbox smaller than one tile → returns one tile
- `test_merge_tile_rasters_values`: merge of two synthetic tiles produces correct
  pixel values at the boundary (no gap, no double-counting)
- `test_merge_tile_rasters_crs_preserved`: output CRS matches input

### Update: `tests/analysis/` — add step 01 and 03 unit tests

Currently there are no unit tests for steps 01 or 03 (only step 07 has one).
Add `tests/analysis/test_01_ndvi_composite.py` and
`tests/analysis/test_03_flowering_index.py` that:
- Mock `load_stackstac` to return a tiny synthetic stack (3 scenes, 2×2 px)
- Mock `make_tile_bboxes` to return a single tile (the full bbox)
- Assert the output raster has the correct shape, CRS, and value range
- Assert that `process_tile` is called once per tile (use `unittest.mock.patch`)

### Update: `tests/integration/test_harness.py`

The existing sentinel/resume tests do not need changes — the tiled approach
writes the same output path and uses the same sentinel file. No changes needed.

## Documentation Updates

### `README.md`

1. **Configuration table** — add `TILE_SIZE_PX` and `TILE_WORKERS` rows with
   defaults and a note that lower values reduce peak RAM at the cost of more
   tiles and merge overhead.

2. **Run pipeline steps 1–3 on DigitalOcean** section — update the droplet
   spec note to explain *why* the fat pipe is now fully utilised: concurrent
   tile loads issue many simultaneous S3 range requests.

3. **Optional flags table** in the Running the pipeline section — add
   `--tile-size` and `--tile-workers`.

### `config.py`

`TILE_SIZE_PX` and `TILE_WORKERS` are intentionally **not** added to
`config.py` — they are operational tuning knobs (hardware-dependent), not
scientific parameters. They live as env vars with defaults in each script and
in `run.sh`.

## Implementation Order

1. `utils/tiling.py` + `tests/unit/test_utils_tiling.py` — standalone, no deps
2. `analysis/01_ndvi_composite.py` refactor + `tests/analysis/test_01_ndvi_composite.py`
3. `analysis/03_flowering_index.py` refactor + `tests/analysis/test_03_flowering_index.py`
4. `analysis/02_ndvi_anomaly.py` (`_build_baseline`) refactor
5. `run.sh` flag additions
6. `README.md` documentation updates
