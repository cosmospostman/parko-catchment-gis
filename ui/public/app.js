// ---------------------------------------------------------------------------
// Map initialisation
// ---------------------------------------------------------------------------

const map = new maplibregl.Map({
  container: 'map',
  style: { version: 8, sources: {}, layers: [] },
  center: [144, -22],
  zoom: 6,
});

// ---------------------------------------------------------------------------
// QLD Globe WMS base layer
// ---------------------------------------------------------------------------

const LAYERS = {
  LatestStateProgram_AllUsers:  'LatestStateProgram_AllUsers',
  LatestSatelliteWOS_AllUsers:  'LatestSatelliteWOS_AllUsers',
};

let activeLayer = 'LatestStateProgram_QGovSISPUsers';

const QGLOBE_LAYERS = new Set([
  'LatestStateProgram_QGovSISPUsers',
]);

function tileUrl(layerName) {
  return `/tile/${layerName}/{z}/{x}/{y}`;
}

function wmsUrl(layerName) {
  return '/wms/' + layerName +
    '?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap' +
    `&LAYERS=${layerName}&FORMAT=image/jpeg` +
    '&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}';
}

function layerUrl(layerName) {
  return QGLOBE_LAYERS.has(layerName) ? tileUrl(layerName) : wmsUrl(layerName);
}

map.on('load', () => {
  map.addSource('qld-globe', {
    type: 'raster',
    tiles: [layerUrl(activeLayer)],
    tileSize: 256,
    attribution: '© Queensland Globe',
  });

  map.addLayer({ id: 'qld-globe-layer', type: 'raster', source: 'qld-globe' });

  loadLocations();
  loadSightings();
});

document.getElementById('layer-select').addEventListener('change', (e) => {
  activeLayer = e.target.value;
  // Swap tile URL by removing and re-adding the source
  map.removeLayer('qld-globe-layer');
  map.removeSource('qld-globe');
  map.addSource('qld-globe', {
    type: 'raster',
    tiles: [layerUrl(activeLayer)],
    tileSize: 256,
    attribution: '© Queensland Globe',
  });
  map.addLayer({ id: 'qld-globe-layer', type: 'raster', source: 'qld-globe' }, 'loc-fill-location');
  // Clear stale imagery info
  idateCache.clear();
  iiDate.innerHTML = '<span class="dim">hover map</span>';
  iiName.innerHTML = '<span class="dim">—</span>';
});

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

const COLOR_EXPR = [
  'match', ['get', 'sub_role'],
  'presence', '#22c55e',
  'absence',  '#ef4444',
  'survey',   '#eab308',
  '#3b82f6',
];

const SCORE_BBOX_COLOR = '#a855f7';

// ---------------------------------------------------------------------------
// Location bbox layer
// ---------------------------------------------------------------------------

let geojsonData = null;

function loadLocations() {
  fetch('/api/locations')
    .then(r => r.json())
    .then(geojson => {
      geojsonData = geojson;

      map.addSource('locations', { type: 'geojson', data: geojson });

      map.addLayer({
        id: 'loc-fill-location',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'location'],
        paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.12 },
      });

      map.addLayer({
        id: 'loc-fill-sub',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'sub_bbox'],
        paint: { 'fill-color': COLOR_EXPR, 'fill-opacity': 0.25 },
      });

      map.addLayer({
        id: 'loc-fill-score',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'score_bbox'],
        paint: { 'fill-color': SCORE_BBOX_COLOR, 'fill-opacity': 0.15 },
      });

      map.addLayer({
        id: 'loc-line',
        type: 'line',
        source: 'locations',
        filter: ['!=', ['get', 'role'], 'score_bbox'],
        paint: {
          'line-color': [
            'match', ['get', 'role'],
            'location', '#93c5fd',
            COLOR_EXPR,
          ],
          'line-width': ['match', ['get', 'role'], 'location', 3, 2],
        },
      });

      map.addLayer({
        id: 'loc-line-score',
        type: 'line',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'score_bbox'],
        paint: {
          'line-color': SCORE_BBOX_COLOR,
          'line-width': 2,
          'line-dasharray': [4, 3],
        },
      });

      buildSidebar(geojson);
      attachPopups();
    })
    .catch(err => console.error('Failed to load locations:', err));
}

// ---------------------------------------------------------------------------
// Sightings layer
// ---------------------------------------------------------------------------

function loadSightings() {
  fetch('/api/sightings')
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(geojson => {
      map.addSource('sightings', { type: 'geojson', data: geojson });
      map.addLayer({
        id: 'sightings-layer',
        type: 'circle',
        source: 'sightings',
        layout: { visibility: 'none' },
        paint: {
          'circle-radius': 4,
          'circle-color': '#f97316',
          'circle-opacity': 0.75,
          'circle-stroke-width': 0.5,
          'circle-stroke-color': '#fff',
        },
      });

      const count = geojson.features?.length ?? 0;
      const el = document.getElementById('sightings-count');
      if (el) el.textContent = count.toLocaleString();

      map.on('click', 'sightings-layer', (e) => {
        const feat = e.features[0];
        if (!feat) return;
        popup.setLngLat(e.lngLat).setHTML(buildSightingPopupHtml(feat.properties)).addTo(map);
      });
      map.on('mouseenter', 'sightings-layer', () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', 'sightings-layer', () => { map.getCanvas().style.cursor = ''; });
    })
    .catch(err => console.error('Failed to load sightings:', err));
}

function buildSightingPopupHtml(p) {
  const date = p.eventDate || (p.year ? `${p.year}${p.month ? '-' + String(p.month).padStart(2, '0') : ''}` : null);
  const observer = p.recordedBy && p.recordedBy !== 'None' && p.recordedBy !== 'null' ? p.recordedBy : null;
  const source = p.dataResourceName || null;
  const basis = p.basisOfRecord
    ? p.basisOfRecord.replace(/_/g, ' ').toLowerCase().replace(/^\w/, c => c.toUpperCase())
    : null;
  const uncertainty = p.coordinateUncertaintyInMeters != null
    ? `±${Math.round(p.coordinateUncertaintyInMeters)} m`
    : null;
  const qaWarning = p.spatiallyValid === false || p.spatiallyValid === 'false'
    ? `<div class="popup-notes" style="color:#f87171;">⚠ Spatially invalid record</div>`
    : '';

  const row = (label, val) => val
    ? `<div class="popup-row"><span class="popup-label">${label}</span><span>${val}</span></div>`
    : '';

  return `
    <div class="popup-title">ALA sighting</div>
    ${row('date', date)}
    ${row('observer', observer)}
    ${row('source', source)}
    ${row('basis', basis)}
    ${row('uncertainty', uncertainty)}
    ${qaWarning}
  `;
}

// ---------------------------------------------------------------------------
// Sidebar — Locations panel
// ---------------------------------------------------------------------------

function buildSidebar(geojson) {
  const list = document.getElementById('location-list');
  const locations = geojson.features.filter(f => f.properties.role === 'location');
  locations.sort((a, b) => a.properties.name.localeCompare(b.properties.name));

  for (const feat of locations) {
    const { id, name } = feat.properties;
    const bbox = feat.properties.bbox;

    const li = document.createElement('li');
    li.dataset.id = id;
    li.innerHTML = `<div class="loc-name">${name}</div><div class="loc-id">${id}</div>`;

    li.addEventListener('click', () => {
      document.querySelectorAll('#location-list li').forEach(el => el.classList.remove('active'));
      li.classList.add('active');
      map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 60, maxZoom: 14 });
    });

    list.appendChild(li);
  }
}

// ---------------------------------------------------------------------------
// Click popups (locations mode)
// ---------------------------------------------------------------------------

const popup = new maplibregl.Popup({ closeButton: true, closeOnClick: false, maxWidth: '300px' });

function formatBbox(bbox) {
  if (!bbox) return '—';
  if (typeof bbox === 'string') bbox = JSON.parse(bbox);
  return `${bbox[0].toFixed(5)}, ${bbox[1].toFixed(5)}, ${bbox[2].toFixed(5)}, ${bbox[3].toFixed(5)}`;
}

function roleBadge(props) {
  if (props.role === 'sub_bbox') {
    const sub = props.sub_role ?? 'sub_bbox';
    return `<span class="role-badge role-${sub}">${sub}</span>`;
  }
  if (props.role === 'score_bbox') {
    return `<span class="role-badge role-score-bbox">score_bbox</span>`;
  }
  return `<span class="role-badge role-location">location</span>`;
}

function buildPopupHtml(props) {
  const title = props.name ?? props.label ?? props.id;
  const bbox = props.bbox;
  const notes = props.notes;
  const parentLabel = props.parent_id
    ? `<div class="popup-row"><span class="popup-label">parent</span><span>${props.parent_id}</span></div>`
    : '';
  return `
    <div class="popup-title">${title}</div>
    <div class="popup-row"><span class="popup-label">role</span>${roleBadge(props)}</div>
    ${parentLabel}
    <div class="popup-row"><span class="popup-label">bbox</span><span>${formatBbox(bbox)}</span></div>
    ${notes ? `<div class="popup-notes">${notes}</div>` : ''}
  `;
}

function attachPopups() {
  const clickableLayers = ['loc-fill-location', 'loc-fill-sub', 'loc-fill-score'];
  for (const layer of clickableLayers) {
    map.on('click', layer, (e) => {
      if (currentMode !== 'locations') return;
      const feat = e.features[0];
      if (!feat) return;
      popup.setLngLat(e.lngLat).setHTML(buildPopupHtml(feat.properties)).addTo(map);
    });
    map.on('mouseenter', layer, () => { if (currentMode === 'locations') map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = ''; });
  }
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

let currentMode = 'locations';

document.getElementById('mode-select').addEventListener('change', (e) => {
  setMode(e.target.value);
});

function setMode(mode) {
  currentMode = mode;
  document.getElementById('panel-locations').style.display = mode === 'locations' ? 'flex' : 'none';
  document.getElementById('panel-bbox').style.display      = mode === 'bbox'      ? 'flex' : 'none';

  if (mode === 'locations') {
    popup.remove();
    clearBboxDraw();
    map.getCanvas().style.cursor = '';
  } else {
    popup.remove();
    map.getCanvas().style.cursor = 'crosshair';
  }
}

// ---------------------------------------------------------------------------
// BBox draw mode
// ---------------------------------------------------------------------------

// Source + layer for the drawn rectangle
map.on('load', () => {
  map.addSource('drawn-bbox', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });
  map.addLayer({
    id: 'drawn-bbox-fill',
    type: 'fill',
    source: 'drawn-bbox',
    paint: { 'fill-color': '#f59e0b', 'fill-opacity': 0.2 },
  });
  map.addLayer({
    id: 'drawn-bbox-line',
    type: 'line',
    source: 'drawn-bbox',
    paint: { 'line-color': '#f59e0b', 'line-width': 2 },
  });
});

let drawStart = null;   // {lng, lat} of mousedown
let isDrawing = false;

function lngLatToBboxFeature(a, b) {
  const lon_min = Math.min(a.lng, b.lng);
  const lon_max = Math.max(a.lng, b.lng);
  const lat_min = Math.min(a.lat, b.lat);
  const lat_max = Math.max(a.lat, b.lat);
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
      ]],
    },
    properties: {},
  };
}

function updateDrawnBbox(a, b) {
  const src = map.getSource('drawn-bbox');
  if (!src) return;
  src.setData({ type: 'FeatureCollection', features: [lngLatToBboxFeature(a, b)] });
}

function clearBboxDraw() {
  drawStart = null;
  isDrawing = false;
  const src = map.getSource('drawn-bbox');
  if (src) src.setData({ type: 'FeatureCollection', features: [] });
  document.getElementById('bbox-coords').style.display = 'none';
}

function commitBbox(a, b) {
  const lon_min = Math.min(a.lng, b.lng);
  const lon_max = Math.max(a.lng, b.lng);
  const lat_min = Math.min(a.lat, b.lat);
  const lat_max = Math.max(a.lat, b.lat);

  const coords = [lon_min, lat_min, lon_max, lat_max];
  const raw = coords.map(v => v.toFixed(6)).join(', ');
  const yaml = `bbox: [${coords.map(v => v.toFixed(6)).join(', ')}]`;

  document.getElementById('bbox-raw').textContent = raw;
  document.getElementById('bbox-yaml').textContent = yaml;

  const box = document.getElementById('bbox-coords');
  box.style.display = 'flex';

  const btn = document.getElementById('btn-copy');
  btn.onclick = () => {
    const done = () => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy bbox'; btn.classList.remove('copied'); }, 1500);
    };
    if (navigator.clipboard) {
      navigator.clipboard.writeText(yaml).then(done);
    } else {
      const ta = document.createElement('textarea');
      ta.value = yaml;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      done();
    }
  };
}

// Mouse events — only active in bbox mode
map.on('mousedown', (e) => {
  if (currentMode !== 'bbox') return;
  e.preventDefault();
  drawStart = e.lngLat;
  isDrawing = true;
  // Clear previous result while drawing a new one
  document.getElementById('bbox-coords').style.display = 'none';
  const src = map.getSource('drawn-bbox');
  if (src) src.setData({ type: 'FeatureCollection', features: [] });
  map.dragPan.disable();
});

map.on('mousemove', (e) => {
  if (!isDrawing || currentMode !== 'bbox') return;
  updateDrawnBbox(drawStart, e.lngLat);
});

map.on('mouseup', (e) => {
  if (!isDrawing || currentMode !== 'bbox') return;
  isDrawing = false;
  map.dragPan.enable();
  updateDrawnBbox(drawStart, e.lngLat);
  commitBbox(drawStart, e.lngLat);
  drawStart = null;
});

document.getElementById('btn-clear').addEventListener('click', clearBboxDraw);

// ---------------------------------------------------------------------------
// Ranking overlay
// ---------------------------------------------------------------------------

const CMAP_GRADIENTS = {
  rdylgn:  'linear-gradient(to right, #a50026, #f46d43, #fee08b, #d9ef8b, #006837)',
  plasma:  'linear-gradient(to right, #0d0887, #cc4778, #f89540, #f0f921)',
  viridis: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
};

let currentRankingOpacity = 0.6;
let currentCmap = 'rdylgn';

document.getElementById('colormap-swatch').style.background = CMAP_GRADIENTS[currentCmap];

document.getElementById('colormap-select').addEventListener('change', (e) => {
  currentCmap = e.target.value;
  console.log('[cmap] changed to', currentCmap);
  document.getElementById('colormap-swatch').style.background = CMAP_GRADIENTS[currentCmap];
  const stem = document.getElementById('ranking-select').value;
  console.log('[cmap] current stem:', stem);
  if (stem) setRankingLayer(stem);
});

function setRankingLayer(stem) {
  if (map.getLayer('ranking-layer')) map.removeLayer('ranking-layer');
  if (map.getSource('ranking'))      map.removeSource('ranking');
  if (!stem) return;

  // bust MapLibre's internal tile cache when cmap changes
  const bust = Date.now();
  const tileUrl = `/ranking-tile/${stem}/{z}/{x}/{y}?cmap=${currentCmap}&v=${bust}`;
  console.log('[cmap] adding source with tile URL:', tileUrl);
  map.addSource('ranking', {
    type: 'raster',
    tiles: [tileUrl],
    tileSize: 256,
  });
  map.addLayer({
    id: 'ranking-layer',
    type: 'raster',
    source: 'ranking',
    paint: { 'raster-opacity': currentRankingOpacity },
  }, 'loc-fill-location');
}

fetch('/api/rankings')
  .then(r => r.json())
  .then(rankings => {
    const sel = document.getElementById('ranking-select');
    for (const { stem, label } of rankings) {
      const opt = document.createElement('option');
      opt.value = stem;
      opt.textContent = label;
      sel.appendChild(opt);
    }
  })
  .catch(err => console.error('Failed to load rankings:', err));

document.getElementById('ranking-select').addEventListener('change', (e) => {
  setRankingLayer(e.target.value);
});

document.getElementById('ranking-opacity').addEventListener('input', (e) => {
  currentRankingOpacity = Number(e.target.value) / 100;
  if (map.getLayer('ranking-layer')) {
    map.setPaintProperty('ranking-layer', 'raster-opacity', currentRankingOpacity);
  }
});

document.getElementById('sightings-toggle').addEventListener('change', (e) => {
  if (map.getLayer('sightings-layer')) {
    map.setLayoutProperty('sightings-layer', 'visibility', e.target.checked ? 'visible' : 'none');
  }
});

// ---------------------------------------------------------------------------
// Imagery info sidebar section
// ---------------------------------------------------------------------------

const iiDate = document.getElementById('ii-date');
const iiName = document.getElementById('ii-name');

let idateThrottle = null;
const idateCache = new Map(); // key → response JSON

function merc(lng, lat) {
  const x = lng * 20037508.34 / 180;
  const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) / (Math.PI / 180) * 20037508.34 / 180;
  return { x: Math.round(x), y: Math.round(y) };
}

function setImageryInfo(data) {
  if (data?.capturestart) {
    const dateLabel = data.capturestart === data.captureend || !data.captureend
      ? data.capturestart
      : `${data.capturestart} – ${data.captureend}`;
    iiDate.innerHTML = dateLabel;
    iiName.innerHTML = data.name ?? '—';
  } else {
    iiDate.innerHTML = '<span class="dim">no data</span>';
    iiName.innerHTML = '<span class="dim">—</span>';
  }
}

map.on('mousemove', (e) => {
  if (isDrawing) return;

  clearTimeout(idateThrottle);
  idateThrottle = setTimeout(async () => {
    const { x, y } = merc(e.lngLat.lng, e.lngLat.lat);
    // Snap to ~500 m grid to improve cache hit rate
    const gx = Math.round(x / 500) * 500;
    const gy = Math.round(y / 500) * 500;
    const key = `${gx},${gy}`;

    let data = idateCache.get(key);
    if (!data) {
      try {
        const resp = await fetch(`/api/imagery-date?x=${gx}&y=${gy}&layer=${activeLayer}`);
        data = await resp.json();
        idateCache.set(key, data);
        if (idateCache.size > 500) idateCache.delete(idateCache.keys().next().value);
      } catch {
        return;
      }
    }

    setImageryInfo(data);
  }, 300);
});
