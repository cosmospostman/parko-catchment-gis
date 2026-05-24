# Pixel-level presence/absence training regions

## Context

Training data is currently defined exclusively as bounding boxes in `data/locations/training.yaml`. Each `TrainingRegion` has a `bbox` and is labelled by spatial containment inside that box. This works well for patches of land but is too coarse for isolated pixels — e.g. a single GPS-logged Parkinsonia stem that needs to contribute exactly one or two pixels to training without including surrounding background.

The goal is to allow defining training regions as an explicit list of individual S2 pixel centres (lon, lat), selectable interactively in the map UI via a "pixel picker" mode. The workflow mirrors the existing BBox tool: pick on the map, copy the YAML snippet, paste into `training.yaml`.

---

## Coordinate identity note

`point_id` strings (`px_0042_0031`) are **bbox-origin-relative** — the indices reset to 0 at the floor-snapped corner of each collection run. Two different collection bboxes produce different indices for the same physical pixel. Therefore pixel-list regions must store snapped (lon, lat) coordinates, not `point_id` strings. Matching in `label_pixels()` uses coordinate equality rounded to 7 decimal places (~1 cm, well within 10m).

---

## Design decisions

- **Single `TrainingRegion` class**, `bbox` and `pixels` both optional but mutually exclusive. A `__post_init__` guard enforces exactly one is provided.
- **YAML format**: new `pixels: [[lon, lat], ...]` key, mutually exclusive with `bbox`. Existing bbox entries are untouched.
- **`label_pixels()` extension**: after the existing bbox pass, a second pass joins pixel-list regions by rounded coordinate against the features DataFrame.
- **Server emits `MultiPoint`** GeoJSON for pixel-list regions so the existing colour expression still applies via `sub_role`.
- **UI: new `PixelPickerPanel`** accordion below BBoxPanel, sets `mapMode.current = 'pixel_pick'`, mutual-exclusion with bbox mode via the shared store.
- **Click snapping**: uses `Math.round` in UTM space (nearest centre), consistent with how the Python collector places pixel centres.
- **Export**: copy-to-clipboard YAML snippet only (no server write). Same pattern as BBoxPanel.

---

## Files to modify

| File | Change |
|------|--------|
| `utils/regions.py` | Make `bbox` optional, add `pixels`, add `__post_init__` validation, add `is_pixel_region` property |
| `tam/utils.py` | Extend `label_pixels()` with coordinate-join pass for pixel-list regions |
| `data/locations/training.yaml` | Add `pixels` to schema comment block |
| `ui/server.ts` | Emit `MultiPoint` for pixel-list regions; make `bbox` optional in TS interface |
| `ui/src/lib/s2grid.ts` | Add `snapToPixelCentre(lng, lat)` export |
| `ui/src/stores/mapMode.svelte.ts` | Add `'pixel_pick'` to the union type |
| `ui/src/stores/pixelPicker.svelte.ts` | New store: `pixels[]`, `label`, helpers |
| `ui/src/components/MapView.svelte` | `pixel_pick` mode: cursor, click handler, `picked-pixels` source+layer, `training-pixel-circles` layer for existing pixel-list regions |
| `ui/src/components/panels/PixelPickerPanel.svelte` | New panel: label selector, pixel list, count, copy/clear buttons |
| `ui/src/components/Sidebar.svelte` | Import and render `PixelPickerPanel` below `BBoxPanel` |
| `ui/src/components/panels/TrainingRegionsCard.svelte` | Guard `zoomTo` against null bbox (pixel-list features have no bbox property) |

---

## Step-by-step implementation

### 1 — `utils/regions.py`

```python
@dataclass(frozen=True)
class TrainingRegion:
    id: str
    name: str
    label: str                        # "presence" | "absence"
    bbox: list[float] | None          # [lon_min, lat_min, lon_max, lat_max]
    pixels: list[list[float]] | None  # [[lon, lat], ...] — mutually exclusive with bbox
    years: list[int]
    tags: list[str]
    notes: str | None

    def __post_init__(self):
        if (self.bbox is None) == (self.pixels is None):
            raise ValueError(
                f"Region {self.id!r}: exactly one of 'bbox' or 'pixels' must be set"
            )

    @property
    def is_presence(self) -> bool:
        return self.label == "presence"

    @property
    def is_pixel_region(self) -> bool:
        return self.pixels is not None

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        if self.bbox is None:
            raise ValueError(f"Region {self.id!r} is a pixel-list region — no bbox")
        return tuple(self.bbox)  # type: ignore[return-value]
```

In `load_regions()`, parse both fields:
```python
TrainingRegion(
    ...
    bbox=entry.get("bbox"),
    pixels=entry.get("pixels"),
    ...
)
```

### 2 — `tam/utils.py` — `label_pixels()`

After the existing bbox pass (the `pl.when` chain), add a pixel-list join. Keep the existing result expression for bbox regions, then overlay pixel-list labels:

```python
# --- pixel-list regions ---
if isinstance(train_loc, list):
    pixel_rows = []
    for region in train_loc:
        if region.pixels:
            val = region.label == "presence"
            for lon, lat in region.pixels:
                pixel_rows.append({
                    "_lon_r": round(lon, 7),
                    "_lat_r": round(lat, 7),
                    "pix_label": val,
                })
    if pixel_rows:
        pix_df = pl.DataFrame(pixel_rows)
        features_df = (
            features_df
            .with_columns([
                pl.col("lon").round(7).alias("_lon_r"),
                pl.col("lat").round(7).alias("_lat_r"),
            ])
            .join(pix_df, on=["_lon_r", "_lat_r"], how="left")
            .with_columns(
                pl.when(pl.col("pix_label").is_not_null())
                  .then(pl.col("pix_label"))
                  .otherwise(result)
                  .alias("is_presence")
            )
            .drop(["_lon_r", "_lat_r", "pix_label"])
        )
        return features_df
```

(For the no-pixel-list path, return unchanged — existing behaviour.)

### 3 — `data/locations/training.yaml` header comment

Add after the `notes` line:
```
# pixels     : list[[lon, lat]]  — optional; mutually exclusive with bbox.
#              Individual pixel centres (EPSG:4326) snapped to the S2 10 m UTM grid.
#              Use the UI pixel picker to generate this list.
#              Matching uses coordinate equality rounded to 7 decimal places.
```

### 4 — `ui/server.ts`

Update the `TrainingRegion` TS interface (around line 56):
```ts
interface TrainingRegion {
  id: string;
  name?: string;
  label: string;
  bbox?: [number, number, number, number];
  pixels?: [number, number][];
  year?: number;
  years?: number[];
  notes?: string;
}
```

In `loadRegionsYaml()`, branch on region type:
```ts
if (region.pixels && region.pixels.length > 0) {
  features.push({
    type: "Feature",
    geometry: { type: "MultiPoint", coordinates: region.pixels },
    properties: {
      id: region.id,
      name: region.name ?? region.id,
      label: region.name ?? region.id,
      role: "sub_bbox",
      sub_role: region.label,
      parent_id: parentId,
      region_type: "pixel_list",
      pixel_count: region.pixels.length,
      bbox: null,
      notes: region.notes ?? null,
    },
  });
} else if (region.bbox) {
  // existing polygon push — unchanged
}
```

### 5 — `ui/src/lib/s2grid.ts`

Add after `emptyS2Grid()`:
```ts
/**
 * Snap a WGS84 click to the nearest 10 m UTM pixel centre.
 * Uses Math.round (nearest centre), consistent with how pixel_collector places centres.
 * Returns lon/lat rounded to 7 decimal places to match label_pixels() join precision.
 */
export function snapToPixelCentre(lng: number, lat: number): { lon: number; lat: number } {
  const { zone, isSouth } = utmZone(lng, lat);
  const { e, n } = toUtm(lng, lat, zone, isSouth);
  const snappedE = Math.round(e / CELL) * CELL;
  const snappedN = Math.round(n / CELL) * CELL;
  const { lng: lon, lat: slat } = fromUtm(snappedE, snappedN, zone, isSouth);
  return {
    lon: Math.round(lon  * 1e7) / 1e7,
    lat: Math.round(slat * 1e7) / 1e7,
  };
}
```

### 6 — `ui/src/stores/mapMode.svelte.ts`

```ts
export const mapMode = $state({ current: 'locations' as 'locations' | 'bbox' | 'pixel_pick' });
```

### 7 — `ui/src/stores/pixelPicker.svelte.ts` (new file)

```ts
export interface PickedPixel { lon: number; lat: number; }

export const pixelPicker = $state({
  pixels: [] as PickedPixel[],
  label: 'presence' as 'presence' | 'absence',
});
```

YAML snippet generation lives in the panel component as a `$derived`.

### 8 — `ui/src/components/MapView.svelte`

- Import `snapToPixelCentre` and `pixelPicker`.
- On map load, add `picked-pixels` GeoJSON source and a circle layer:
  ```ts
  map.addSource('picked-pixels', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({
    id: 'picked-pixels-circle', type: 'circle', source: 'picked-pixels',
    paint: {
      'circle-radius': 6,
      'circle-color': ['match', ['get', 'label'], 'presence', '#22c55e', '#ef4444'],
      'circle-stroke-width': 2,
      'circle-stroke-color': '#fff',
    },
  });
  ```
- Add a `training-pixel-circles` layer on the `training-regions` source, filtered to `MultiPoint`:
  ```ts
  map.addLayer({
    id: 'training-pixel-circles', type: 'circle', source: 'training-regions',
    filter: ['==', ['geometry-type'], 'MultiPoint'],
    layout: { visibility: tv },
    paint: { 'circle-radius': 4, 'circle-color': COLOR_EXPR as any, 'circle-opacity': 0.8 },
  });
  ```
- Add `training-pixel-circles` to the training visibility `$effect` alongside `training-line`.
- Add cursor effect for `pixel_pick` mode.
- Add click handler inside the existing `$effect` that wires bbox handlers:
  ```ts
  const onClickPixelPick = (e: MapMouseEvent) => {
    if (mapMode.current !== 'pixel_pick') return;
    const { lon, lat } = snapToPixelCentre(e.lngLat.lng, e.lngLat.lat);
    const idx = pixelPicker.pixels.findIndex(p => p.lon === lon && p.lat === lat);
    if (idx >= 0) pixelPicker.pixels.splice(idx, 1);
    else pixelPicker.pixels.push({ lon, lat });
    syncPickedPixelsSource(m);
  };
  m.on('click', onClickPixelPick);
  // cleanup in return
  ```
- `syncPickedPixelsSource()` converts `pixelPicker.pixels` to GeoJSON Point features with a `label` property and calls `setData` on the source.
- `$effect` that clears the source when mode leaves `pixel_pick`.

### 9 — `ui/src/components/panels/PixelPickerPanel.svelte` (new file)

Mirrors BBoxPanel structure:
- `AccordionSection` with title "Pixel picker" and pixel count badge.
- `$effect`: open → `mapMode.current = 'pixel_pick'`; close → clear pixels + reset mode.
- Presence/absence toggle (two buttons, green/red styled to match the colour scheme).
- Scrollable pixel list: each item shows `lon, lat` with a remove (×) button.
- `$derived` YAML snippet:
  ```yaml
  id: <slug-placeholder>
  name: "<name placeholder>"
  label: presence
  pixels:
    - [141.539600, -15.808400]
    - [141.539700, -15.807500]
  years: [2020, 2021, 2022, 2023, 2024, 2025]
  ```
- Copy / Clear buttons (same styling as BBoxPanel).
- Warning text when pixel count > 50 suggesting a bbox instead.

### 10 — `ui/src/components/Sidebar.svelte`

```svelte
import PixelPickerPanel from './panels/PixelPickerPanel.svelte';
...
<BBoxPanel />
<PixelPickerPanel />
```

### 11 — `ui/src/components/panels/TrainingRegionsCard.svelte`

Guard `zoomTo` against null/missing bbox (pixel-list features have `bbox: null`):
```ts
function zoomTo(bboxRaw: any, feat?: LocationFeature) {
  const map = mapCtx.getMap();
  if (!map) return;
  if (bboxRaw) {
    const b = parseBbox(bboxRaw);
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 80, maxZoom: 16 });
    const { gridLines } = buildS2Grid(b, { sub_role: feat?.properties.sub_role ?? null });
    setGrid(gridLines);
  } else if (feat?.geometry.type === 'MultiPoint') {
    const coords = (feat.geometry as any).coordinates as [number, number][];
    const lons = coords.map(c => c[0]);
    const lats = coords.map(c => c[1]);
    map.fitBounds(
      [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
      { padding: 80, maxZoom: 16 },
    );
    setGrid(emptyS2Grid().gridLines);  // no pixel grid overlay for point regions
  }
}
```

---

## Verification

1. Add a test pixel-list region to `training.yaml`:
   ```yaml
   - id: test_pixel_presence
     name: "Test — presence pixels"
     label: presence
     pixels:
       - [145.371600, -22.519400]
       - [145.371700, -22.519300]
     years: [2022, 2023]
   ```
2. `python -c "from utils.regions import load_regions; r = load_regions(); print([x for x in r if x.id == 'test_pixel_presence'])"` — should show the region with `pixels` set and `bbox=None`.
3. Verify `label_pixels()` matches by running a minimal Polars test with two rows whose (lon, lat) match the pixels above.
4. Start the Deno dev server, open the app, toggle the training layer, confirm the test region appears as two circles at the expected coordinates.
5. Open "Pixel picker" in the sidebar, click two pixels on the map, confirm they appear as green/red circles and the YAML snippet is generated with snapped coordinates.
6. Click the same pixel again — confirms toggle/remove behaviour.
7. Copy YAML, verify coordinates round-trip cleanly through 7 d.p. precision.
8. Confirm BBoxPanel and PixelPickerPanel are mutually exclusive (opening one resets the other's mode).
