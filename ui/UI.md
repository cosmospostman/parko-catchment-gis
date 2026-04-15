# UI Implementation Plan

## Context

The project has 11 location YAML files (`data/locations/*.yaml`) defining georeferenced bboxes and sub-bboxes (presence/absence/survey regions). All current visualization is static matplotlib PNGs over Queensland Globe WMS tiles. This plan adds a minimal browser-based map UI: pan/zoom over QLD Globe aerial imagery with location bboxes as interactive overlays.

**Stack:** Hono + TypeScript (Node.js via `tsx`) · MapLibre GL JS · GeoJSON bbox layer

---

## File Structure

```
ui/
  server.ts          # Hono backend — /api/locations + static file serving
  package.json
  tsconfig.json
  public/
    index.html       # Full-viewport map page
    app.js           # MapLibre GL map, WMS layer, bbox overlay, sidebar
```

No frontend build step — vanilla JS in `public/` with MapLibre loaded from CDN. TypeScript only on the server, executed directly via `tsx`.

---

## Backend: `ui/server.ts`

**Dependencies:** `hono`, `@hono/node-server`, `js-yaml`

**`GET /api/locations`**

Reads all `../data/locations/*.yaml` and returns a GeoJSON `FeatureCollection`. Each YAML produces:

- One **location feature** — `Polygon` from `[lon_min, lat_min, lon_max, lat_max]`, properties: `id` (filename slug), `name`, `notes`, `role: "location"`
- One **sub-bbox feature** per entry in `sub_bboxes` — `Polygon`, properties: `parent_id`, `label`, `role: "sub_bbox"`, `sub_role` (presence/absence/survey)

**Static serving** — `serveStatic` middleware serves `ui/public/` for all other routes.

**Port** — 3000, configurable via `PORT` env var.

---

## Frontend: `ui/public/`

### `index.html`

- Full-viewport layout: left sidebar (~240px) + right map fills remaining space
- MapLibre GL JS + CSS from CDN
- `<script src="app.js">`

### `app.js`

**1. Map init**
```js
new maplibregl.Map({
  container: 'map',
  style: { version: 8, sources: {}, layers: [] },  // empty — no base style
  center: [144, -22],   // Queensland
  zoom: 6
})
```

**2. QLD Globe WMS layer**

Raster source using WMS `GetMap` URL template (EPSG:3857, MapLibre's native CRS):

```
https://spatial-img.information.qld.gov.au/arcgis/services/Basemaps/
LatestStateProgram_AllUsers/ImageServer/WMSServer
?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap
&LAYERS=LatestStateProgram_AllUsers&FORMAT=image/jpeg
&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}
```

`{bbox-epsg-3857}` is a MapLibre built-in placeholder — substituted automatically per tile.

> **Note:** Python code uses EPSG:4326 with WMS 1.3.0 axis-order; switching to EPSG:3857 avoids that quirk. ArcGIS ImageServer almost certainly supports it, but verify on first run.

**3. Location bbox layer**

```js
fetch('/api/locations')
  .then(r => r.json())
  .then(geojson => {
    map.addSource('locations', { type: 'geojson', data: geojson })
    map.addLayer({ id: 'loc-fill', type: 'fill', source: 'locations', paint: {
      'fill-color': ['match', ['get', 'sub_role'],
        'presence', '#22c55e',
        'absence',  '#ef4444',
        'survey',   '#eab308',
        /* location */ '#3b82f6'
      ],
      'fill-opacity': 0.2
    }})
    map.addLayer({ id: 'loc-line', type: 'line', source: 'locations', paint: {
      'line-color': /* same match expression */ ...,
      'line-width': 2
    }})
  })
```

**4. Click popup**

Click on any feature → `maplibregl.Popup` showing name, role, bbox coords, notes.

**5. Sidebar**

`<ul>` of location names populated from the API response. Click → `map.fitBounds(bbox, { padding: 40 })`.

---

## `ui/package.json`

```json
{
  "name": "parko-gis-ui",
  "type": "module",
  "scripts": { "start": "tsx server.ts" },
  "dependencies": {
    "hono": "^4",
    "@hono/node-server": "^1",
    "js-yaml": "^4"
  },
  "devDependencies": {
    "tsx": "^4",
    "typescript": "^5",
    "@types/node": "^20",
    "@types/js-yaml": "^4"
  }
}
```

## `ui/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "outDir": "dist"
  }
}
```

---

## Files to Create

| File | Notes |
|------|-------|
| `ui/package.json` | Dependencies + start script |
| `ui/tsconfig.json` | NodeNext module mode |
| `ui/server.ts` | Hono server, YAML→GeoJSON, static serving |
| `ui/public/index.html` | Sidebar + map layout, MapLibre CDN |
| `ui/public/app.js` | Map init, WMS layer, bbox layer, sidebar, popups |

No existing files modified.

---

## Verification

1. `cd ui && npm install && npm start`
2. Open `http://localhost:3000`
3. QLD Globe imagery loads and pans/zooms smoothly
4. All 11 location bboxes visible as blue rectangles
5. Sub-bboxes visible in green/red/yellow by role
6. Click a bbox → popup with name, notes, coords
7. Sidebar click → map flies to that location
8. Check browser console for WMS tile errors (EPSG:3857 support)
