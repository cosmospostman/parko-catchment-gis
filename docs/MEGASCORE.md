# Plan: Strip-aware scoring + PMTiles output + simplified UI renderer

## Context

The 54LWH tile (Mitchell River catchment) arrives on disk as 10 strip parquets (~89 GB). Rather
than a one-time megamerge, the scoring pipeline should consume strip files natively. The existing
`.bin` + server-side PNG renderer also breaks down at Mitchell-catchment scale (must load the
whole grid into RAM, opaque to external tools, no zoom overviews). We replace it with raster
PMTiles: client applies the colormap, server just byte-ranges a file.

---

## Component 1 — PixelSource abstraction

### Motivation

`score_pixels_chunked` and `score_tile_year` both open a single `pq.ParquetFile` and iterate
its row groups. To handle multiple strip files for one tile without merging them, we need an
abstraction that presents N parquets as a single row-group stream. The same abstraction also
lets future sources (e.g. a cloud STAC stream) drop in without touching the scorer.

### Interface

New file: `tam/core/pixel_source.py`

```python
class PixelSource(ABC):
    @property
    @abstractmethod
    def num_row_groups(self) -> int: ...

    @abstractmethod
    def read_row_group(self, i: int, columns: list[str]) -> pa.Table: ...

    @property
    @abstractmethod
    def schema(self) -> pa.Schema: ...
```

Two concrete implementations:

**`ParquetPixelSource(path: Path)`** — wraps a single `pq.ParquetFile`. Trivial one-liner
delegation; replaces the bare `pf = pq.ParquetFile(parquet)` in `score_pixels_chunked`.

**`StripPixelSource(paths: list[Path])`** — opens N `pq.ParquetFile`s, presents their row
groups concatenated in order (strips are already in northing order so no sort needed).
`num_row_groups` = sum of all strip row-group counts; `read_row_group(i)` maps global index
→ (strip_file, local_rg).

Both are cheap to construct (just open file handles, no data read).

### Wiring into the scorer

**`score_pixels_chunked`** — replace `parquet: Path` parameter with
`source: PixelSource | Path`. When a bare `Path` is received, wrap it in
`ParquetPixelSource` so all existing call sites are unchanged. Internally replace
`pf = pq.ParquetFile(parquet)` / `pf.read_row_group(rg)` / `pf.metadata.num_row_groups`
with `source.read_row_group` / `source.num_row_groups` / `source.schema`.

The pre-pass functions (`_compute_pixel_s1_stats`, `_compute_s1_despeckle_lookup`) also take
`parquet: Path` — give them the same `PixelSource | Path` treatment.

**`score_tile_year`** — replace `parquet: Path` with `source: PixelSource | Path`. Pass
through to `score_pixels_chunked`. No other changes needed.

**`_score_tile_worker` / `score_tiles_chunked`** — `tile_year_parquets` type stays
`dict[str, list[tuple[int, Path]]]` but becomes
`dict[str, list[tuple[int, PixelSource | Path]]]`. The construction of this dict in
`_cmd_score` is where strips are detected and wrapped (see below).

### Strip detection in `_cmd_score` (`tam/pipeline.py` line 629)

When building `tile_year_map`, detect strips by `_strip_` in the stem and group them:

```python
# group paths by real tile_id (strip prefix or plain stem)
for y, paths in sorted(tile_paths_by_year.items()):
    by_tile: dict[str, list[Path]] = {}
    for p in paths:
        tid = p.stem.split("_strip_")[0]  # "54LWH_strip_00" → "54LWH"
        by_tile.setdefault(tid, []).append(p)
    for tid, tile_paths in by_tile.items():
        strips = [p for p in tile_paths if "_strip_" in p.stem]
        non_strips = [p for p in tile_paths if "_strip_" not in p.stem]
        source = StripPixelSource(sorted(strips)) if strips else tile_paths[0]
        tile_year_map.setdefault(tid, []).append((y, source))
```

`parquet_tile_paths()` in `utils/location.py` needs no changes — strip files are not filtered
by its existing exclusions, so they pass through as-is.

`ensure_pixel_sorted()` is bypassed for `StripPixelSource` (strips are already in northing
order; pixel-sort is defined on a single file and doesn't apply here). Only wrap single-file
paths in `ensure_pixel_sorted`.

### Files to modify / create
- `tam/core/pixel_source.py` — new file (~60 lines)
- `tam/core/score.py` — `score_pixels_chunked`, `score_tile_year`, pre-pass helpers:
  `parquet: Path` → `source: PixelSource | Path` (~20 substitutions, mechanical)
- `tam/pipeline.py` — `_cmd_score` tile_year_map construction (~15 lines replaced)

---

## Component 2 — PMTiles output from the score pipeline

### Overview

After `score_tiles_chunked` writes a final `.scores.parquet` for each S2 tile (Phase 2),
a new optional step rasterises `(point_id, prob_tam)` → 256×256 PNG tiles at zoom 8–16
and appends them into a per-location `.pmtiles` archive. One S2 tile fits comfortably in
RAM (~96 MB uint8 before compression), so the write is strip-at-a-time.

### Rasterisation

`point_id` encodes `px_{xi:04d}_{yi:04d}` where xi/yi are 10 m grid offsets from the
tile's UTM origin. The origin and CRS are recoverable from the parquet's `lon`/`lat` columns
(snap to nearest 10 m in the tile's UTM zone).

Steps per S2 tile:
1. Read `.scores.parquet` (point_id + prob_tam uint8) + lon/lat from the year parquet
2. Reconstruct UTM grid → numpy `uint8[height, width]`
3. Warp to Web Mercator (rasterio) — needed for correct slippy-map tile alignment
4. Slice into 256×256 tiles at zoom levels 8–16; encode each as single-channel PNG
   (R channel = prob_tam value 0–100, scaled to 0–255 so MapLibre `raster-value` reads 0–255
   and the client expression rescales to 0–100)
5. Append tiles to the PMTiles archive via the `pmtiles` Python package

**New Python dependency:** `pmtiles` (pure Python, pip-installable).
**New Python dependency:** `rasterio` (for warp; likely already present or easy to add).

### Hook point

`score_tiles_chunked` gains an optional `pmtiles_out: Path | None = None` parameter.
After writing the final parquet (~line 2250), before staging cleanup, call:

```python
if pmtiles_out:
    rasterize_tile_to_pmtiles(final_path, coords_parquet, tile_id, pmtiles_out)
```

`tam/pipeline.py` `_cmd_score` gains `--pmtiles <path>` flag that sets `pmtiles_out`.

### New file

`utils/raster.py` — contains:
- `rasterize_tile_to_pmtiles(scores_path, coords_path, tile_id, out_pmtiles)` — top-level
  entry called from score pipeline
- `scores_to_grid(scores_df, lon_lat_df) -> (np.ndarray, Affine, CRS)` — fills uint8 grid
- `warp_to_mercator(grid, src_transform, src_crs) -> (np.ndarray, Affine)` — rasterio warp
- `iter_tiles(grid, transform, zoom_min, zoom_max) -> Iterator[(z,x,y,bytes)]` — PNG slices

### Files to modify / create
- `utils/raster.py` — new file (~150 lines)
- `tam/core/score.py` — `score_tiles_chunked`: add `pmtiles_out` param + 5-line hook
- `tam/pipeline.py` — `_cmd_score`: add `--pmtiles` flag

---

## Component 3 — Simplified UI rendering (replace .bin with PMTiles)

### What to delete from `ui/tile_renderer.ts`
- `buildBin()`, `loadBin()`, binary format constants, grid cache, `renderTile()`, PNG encoder
- The entire `.bin` build/load/render stack (~450 of 725 lines)

Keep: `listRankings()` (update to scan `.pmtiles` instead of `.csv`), S1 bin machinery
(separate concern, leave for now).

### Server changes (`ui/server.ts`)

Replace `/ranking-tile/{location}/{stem}/{z}/{x}/{y}` with a PMTiles byte-range handler:

```
GET /pmtiles/{location}/{stem}  →  stream file with Range header support
```

MapLibre's PMTiles JS plugin negotiates byte ranges client-side against this endpoint.
Deno's `Deno.open` + manual `Content-Range` response handles this in ~20 lines, or use
`serveDir` pointed at the scores output directory.

Update `listRankings()` to scan for `.pmtiles` files instead of `.csv`.

### Frontend changes (`ui/src/components/MapView.svelte`)

Replace the XYZ raster tile source + server-side colormap query params with a PMTiles
raster source + MapLibre `raster-color` paint expression:

```typescript
map.addSource('ranking', {
  type: 'raster',
  url: `pmtiles:///pmtiles/${location}/${stem}`,
  tileSize: 256,
});
map.addLayer({
  id: 'ranking-layer', type: 'raster', source: 'ranking',
  paint: {
    'raster-color': buildColormapExpression(cmap, cutoff),
    'raster-color-mix': [255, 0, 0, 0],  // R channel carries the value
    'raster-opacity': opacity,
  }
});
```

Changing `cmap`, `cutoff`, or `opacity` becomes a `setPaintProperty` call — no tile
re-request, no cache invalidation.

Add PMTiles JS protocol to `ui/src/index.html` (CDN script tag, or add
`pmtiles` to `deno.json` imports and import in `MapView.svelte`). Register once at map init:

```typescript
import { Protocol } from 'pmtiles';
maplibregl.addProtocol('pmtiles', new Protocol().tile);
```

Move colormap stop arrays from `tile_renderer.ts` → new `ui/src/lib/colormaps.ts`, returning
MapLibre `interpolate` expressions. `cutoff` maps to a `step` or `case` expression that
returns `rgba(0,0,0,0)` below threshold.

### Files to modify / create
- `ui/tile_renderer.ts` — gut bin/render stack, update `listRankings()`
- `ui/server.ts` — replace `/ranking-tile` route with PMTiles byte-range route
- `ui/src/components/MapView.svelte` — update ranking layer source + paint
- `ui/src/index.html` — add pmtiles protocol script
- `ui/deno.json` — add `pmtiles` npm dependency
- `ui/src/lib/colormaps.ts` — new file (~40 lines)

---

## Implementation order

1. **Component 1** — `PixelSource` abstraction + strip detection in pipeline. Unblocks
   scoring 54LWH immediately; self-contained with no new dependencies.
2. **Component 2** — `utils/raster.py` + score pipeline PMTiles hook. Produces the
   `.pmtiles` files Component 3 depends on.
3. **Component 3** — UI replacement. Can begin once one `.pmtiles` file exists to test
   against; `.bin` path stays live until Component 3 is complete.

---

## Verification

**Component 1:**
```bash
# Place strips in data/pixels/mitchell/2025/ and run scorer
python -m tam.pipeline score --checkpoint outputs/models/tam-v10-0530 \
  --location mitchell --years 2025 --out-parquet
# Confirm tile_id "54LWH" appears in staging, not "54LWH_strip_00" etc.
```

**Component 2:**
```bash
python -m tam.pipeline score --checkpoint outputs/models/tam-v10-0530 \
  --location mitchell --years 2025 --out-parquet \
  --pmtiles outputs/scores/mitchell/tam-v10-0530.pmtiles
pmtiles show outputs/scores/mitchell/tam-v10-0530.pmtiles
# Check: zoom range 8–16, bounds cover Mitchell catchment, tile count reasonable
```

**Component 3:**
- Load UI, select mitchell / tam-v10-0530, confirm tiles render at all zoom levels
- Change colormap — confirm instant client-side re-render, no network request
- Adjust cutoff slider — confirm threshold applies without tile re-request
- Confirm existing small locations (kowanyama etc.) still work once their CSVs are converted
