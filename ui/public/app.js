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

const DIRECT_TILE_URLS = {
  EsriWorldImagery: 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
};

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
  if (DIRECT_TILE_URLS[layerName]) return DIRECT_TILE_URLS[layerName];
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

  loadLocations();
  loadSightings();
  loadCatchments();
});

document.getElementById('layer-select').addEventListener('change', (e) => {
  activeLayer = e.target.value;
  try {
    map.removeLayer('qld-globe-layer');
    map.removeSource('qld-globe');
  } catch (_) { /* in-flight tile decode may throw — safe to ignore */ }
  map.addSource('qld-globe', {
    type: 'raster',
    tiles: [layerUrl(activeLayer)],
    tileSize: 256,
    attribution: '© Queensland Globe',
  });
  map.addLayer({ id: 'qld-globe-layer', type: 'raster', source: 'qld-globe' }, 'loc-fill-location');
  // Refresh imagery info for new layer
  fetchImageryInfo();
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

const TRAINING_LAYERS = ['training-fill', 'training-line'];

function loadLocations() {
  Promise.all([
    fetch('/api/locations').then(r => r.json()),
    fetch('/api/rankings').then(r => r.json()).catch(() => ({})),
  ]).then(([geojson, rankings]) => {
      geojsonData = geojson;

      // Split training regions into their own source so they can be toggled.
      const trainingFeatures = geojson.features.filter(f => f.properties.parent_id === 'training');
      const locationFeatures = geojson.features.filter(f => f.properties.parent_id !== 'training');
      const locationGeojson = { type: 'FeatureCollection', features: locationFeatures };
      const trainingGeojson = { type: 'FeatureCollection', features: trainingFeatures };

      map.addSource('locations', { type: 'geojson', data: locationGeojson });
      map.addSource('training-regions', { type: 'geojson', data: trainingGeojson });

      map.addLayer({
        id: 'loc-fill-location',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'location'],
        paint: { 'fill-color': '#7c3aed', 'fill-opacity': 0 },
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
            'location', '#7c3aed',
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

      // Training region layers (toggleable) — initial visibility follows the checkbox
      const trainingVisibility = document.getElementById('training-toggle').checked ? 'visible' : 'none';
      map.addLayer({
        id: 'training-fill',
        type: 'fill',
        source: 'training-regions',
        layout: { visibility: trainingVisibility },
        paint: { 'fill-color': COLOR_EXPR, 'fill-opacity': 0.30 },
      });

      map.addLayer({
        id: 'training-line',
        type: 'line',
        source: 'training-regions',
        layout: { visibility: trainingVisibility },
        paint: { 'line-color': COLOR_EXPR, 'line-width': 2 },
      });

      buildSidebar(geojson, rankings);
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
        layout: { visibility: document.getElementById('sightings-toggle').checked ? 'visible' : 'none' },
        paint: {
          'circle-radius': 4,
          'circle-color': '#f97316',
          'circle-opacity': 0.75,
          'circle-stroke-width': 0.5,
          'circle-stroke-color': '#fff',
        },
      });

      sightingsTotalCount = geojson.features?.length ?? 0;
      const countEl = document.getElementById('sightings-count');
      if (countEl) countEl.textContent = sightingsTotalCount.toLocaleString();

      map.on('click', 'sightings-layer', (e) => {
        const feat = e.features[0];
        if (!feat) return;
        popup.setLngLat(e.lngLat).setHTML(buildSightingPopupHtml(feat.properties)).addTo(map);
      });
      map.on('mouseenter', 'sightings-layer', () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', 'sightings-layer', () => { if (currentMode === 'locations') map.getCanvas().style.cursor = ''; });

      sightingsFeatures = geojson.features ?? [];
      applySightingsYearFilter();
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
// Catchments layer
// ---------------------------------------------------------------------------

const CATCHMENT_LAYERS = ['catchments-fill', 'catchments-line'];

function loadCatchments() {
  fetch('/api/catchments')
    .then(r => r.json())
    .then(geojson => {
      map.addSource('catchments', { type: 'geojson', data: geojson });

      const vis = document.getElementById('catchments-toggle').checked ? 'visible' : 'none';

      map.addLayer({
        id: 'catchments-fill',
        type: 'fill',
        source: 'catchments',
        layout: { visibility: vis },
        paint: { 'fill-color': '#38bdf8', 'fill-opacity': 0.08 },
      });

      map.addLayer({
        id: 'catchments-line',
        type: 'line',
        source: 'catchments',
        layout: { visibility: vis },
        paint: { 'line-color': '#38bdf8', 'line-width': 2, 'line-dasharray': [4, 3] },
      });

      buildCatchmentsSidebar(geojson);
    })
    .catch(err => console.error('Failed to load catchments:', err));
}

function buildCatchmentsSidebar(geojson) {
  const body = document.getElementById('catchments-accordion-body');
  const header = document.getElementById('catchments-accordion-header');

  for (const feat of geojson.features) {
    const name = feat.properties?.name ?? feat.properties?.id ?? 'Catchment';
    const geom = feat.geometry;
    if (!geom) continue;
    const rings = geom.type === 'MultiPolygon'
      ? geom.coordinates.flat(1)
      : geom.coordinates;
    const allPts = rings.flat(1);
    const lngs = allPts.map(c => c[0]);
    const lats = allPts.map(c => c[1]);
    const bbox = [Math.min(...lngs), Math.min(...lats), Math.max(...lngs), Math.max(...lats)];

    const item = document.createElement('div');
    item.className = 'catchment-item';
    item.innerHTML = `<span class="catchment-dot"></span><span>${name}</span>`;
    item.addEventListener('click', () => {
      map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 60, maxZoom: 9 });
    });
    body.appendChild(item);
  }

  header.addEventListener('click', (e) => {
    if (e.target.id === 'catchments-toggle') return;
    header.classList.toggle('open');
    body.classList.toggle('open');
  });
}

document.getElementById('catchments-toggle').addEventListener('change', (e) => {
  const vis = e.target.checked ? 'visible' : 'none';
  for (const id of CATCHMENT_LAYERS) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
  }
});

// ---------------------------------------------------------------------------
// Sidebar — Locations panel
// ---------------------------------------------------------------------------

function buildSidebar(geojson, rankings) {
  // Training regions accordion
  const trainingFeatures = geojson.features.filter(f => f.properties.parent_id === 'training');
  document.querySelector('#training-accordion-header .accordion-label').textContent =
    `Training regions (${trainingFeatures.length})`;
  const body = document.getElementById('training-accordion-body');
  for (const feat of trainingFeatures) {
    const { name, sub_role } = feat.properties;
    let bbox = feat.properties.bbox;
    if (typeof bbox === 'string') bbox = JSON.parse(bbox);

    const item = document.createElement('div');
    item.className = 'training-region-item';
    item.innerHTML = `
      <span class="training-role-dot ${sub_role}"></span>
      <span>${name}</span>
    `;
    item.addEventListener('click', () => {
      map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 80, maxZoom: 16 });
    });
    body.appendChild(item);
  }

  const header = document.getElementById('training-accordion-header');
  header.addEventListener('click', (e) => {
    // Checkbox click handled separately — don't also toggle accordion
    if (e.target.id === 'training-toggle') return;
    header.classList.toggle('open');
    body.classList.toggle('open');
  });

  // Location list
  const list = document.getElementById('location-list');
  const locations = geojson.features.filter(f => f.properties.role === 'location');
  locations.sort((a, b) => a.properties.name.localeCompare(b.properties.name));

  for (const feat of locations) {
    const { id, name } = feat.properties;
    let bbox = feat.properties.bbox;
    if (typeof bbox === 'string') bbox = JSON.parse(bbox);

    const li = document.createElement('li');
    li.dataset.id = id;
    li.innerHTML = `<div class="loc-name">${name}</div>`;

    li.addEventListener('click', (e) => {
      if (e.target.classList.contains('loc-ranking-select')) return;
      document.querySelectorAll('#location-list li').forEach(el => el.classList.remove('active'));
      li.classList.add('active');
      map.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 60, maxZoom: 14 });
    });

    const runs = rankings[id];
    if (runs && runs.length > 0) {
      const sel = document.createElement('select');
      sel.className = 'loc-ranking-select';
      sel.dataset.location = id;
      sel.innerHTML = '<option value="">— ranking —</option>';
      for (const { stem, label } of runs) {
        const opt = document.createElement('option');
        opt.value = stem;
        opt.textContent = label;
        sel.appendChild(opt);
      }
      sel.addEventListener('change', (e) => {
        e.stopPropagation();
        setRankingLayer(id, e.target.value);
      });
      li.appendChild(sel);
    }

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
  const yearLabel = props.year
    ? `<div class="popup-row"><span class="popup-label">year</span><span>${props.year} (±5 yr window)</span></div>`
    : '';
  return `
    <div class="popup-title">${title}</div>
    <div class="popup-row"><span class="popup-label">role</span>${roleBadge(props)}</div>
    ${parentLabel}
    ${yearLabel}
    <div class="popup-row"><span class="popup-label">bbox</span><span>${formatBbox(bbox)}</span></div>
    ${notes ? `<div class="popup-notes">${notes}</div>` : ''}
  `;
}

function attachPopups() {
  const clickableLayers = ['loc-fill-location', 'loc-fill-sub', 'loc-fill-score', 'training-fill'];
  for (const layer of clickableLayers) {
    map.on('click', layer, (e) => {
      if (currentMode !== 'locations') return;
      const feat = e.features[0];
      if (!feat) return;
      popup.setLngLat(e.lngLat).setHTML(buildPopupHtml(feat.properties)).addTo(map);
    });
    map.on('mouseenter', layer, () => { if (currentMode === 'locations') map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layer, () => { if (currentMode === 'locations') map.getCanvas().style.cursor = ''; });
  }
}

// ---------------------------------------------------------------------------
// BBox accordion toggle
// ---------------------------------------------------------------------------

let currentMode = 'locations';

const bboxAccordionHeader = document.getElementById('bbox-accordion-header');
const bboxAccordionBody   = document.getElementById('bbox-accordion-body');

bboxAccordionHeader.addEventListener('click', () => {
  const opening = !bboxAccordionHeader.classList.contains('open');
  bboxAccordionHeader.classList.toggle('open');
  bboxAccordionBody.classList.toggle('open');

  if (opening) {
    currentMode = 'bbox';
    popup.remove();
    map.getCanvas().style.cursor = 'crosshair';
  } else {
    currentMode = 'locations';
    clearBboxDraw();
    map.getCanvas().style.cursor = '';
  }
});

document.getElementById('training-toggle').addEventListener('change', (e) => {
  const vis = e.target.checked ? 'visible' : 'none';
  for (const id of TRAINING_LAYERS) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
  }
});

// ---------------------------------------------------------------------------
// BBox draw mode
// ---------------------------------------------------------------------------

// drawn-bbox source + layers are added in the main map.on('load') handler above.

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
  const yaml = `bbox: [${coords.map(v => v.toFixed(6)).join(', ')}]`;

  document.getElementById('bbox-yaml').textContent = yaml;

  const box = document.getElementById('bbox-coords');
  box.style.display = 'flex';

  const btn = document.getElementById('btn-copy');
  btn.onclick = () => {
    const done = () => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
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

document.getElementById('btn-clear').addEventListener('click', () => {
  clearBboxDraw();
  bboxAccordionHeader.classList.remove('open');
  bboxAccordionBody.classList.remove('open');
  currentMode = 'locations';
  map.getCanvas().style.cursor = '';
});

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
let currentCutoff = 0;
let sightingsTotalCount = 0;
let sightingsFeatures = [];

document.getElementById('colormap-swatch').style.background = CMAP_GRADIENTS[currentCmap];

document.getElementById('colormap-select').addEventListener('change', (e) => {
  currentCmap = e.target.value;
  document.getElementById('colormap-swatch').style.background = CMAP_GRADIENTS[currentCmap];
  if (activeRankingLocation && activeRankingStem) setRankingLayer(activeRankingLocation, activeRankingStem);
});

let activeRankingLocation = null;
let activeRankingStem = null;

function setRankingLayer(location, stem) {
  if (map.getLayer('ranking-layer')) map.removeLayer('ranking-layer');
  if (map.getSource('ranking'))      map.removeSource('ranking');

  // Reset all per-location selects to none (except the one being set)
  document.querySelectorAll('.loc-ranking-select').forEach(sel => {
    if (sel.dataset.location !== location || !stem) sel.value = '';
  });

  activeRankingLocation = stem ? location : null;
  activeRankingStem = stem || null;
  if (!stem) return;

  const tileUrl = `/ranking-tile/${location}/${stem}/{z}/{x}/{y}?cmap=${currentCmap}&cutoff=${currentCutoff}`;
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

document.getElementById('ranking-opacity').addEventListener('input', (e) => {
  currentRankingOpacity = Number(e.target.value) / 100;
  if (map.getLayer('ranking-layer')) {
    map.setPaintProperty('ranking-layer', 'raster-opacity', currentRankingOpacity);
  }
});

document.getElementById('ranking-cutoff').addEventListener('input', (e) => {
  currentCutoff = Number(e.target.value) / 100;
  document.getElementById('ranking-cutoff-label').textContent = `${e.target.value}%`;
  if (activeRankingLocation && activeRankingStem) setRankingLayer(activeRankingLocation, activeRankingStem);
});

document.getElementById('sightings-toggle').addEventListener('change', (e) => {
  if (map.getLayer('sightings-layer')) {
    map.setLayoutProperty('sightings-layer', 'visibility', e.target.checked ? 'visible' : 'none');
  }
});

const sightingsAccordionHeader = document.getElementById('sightings-accordion-header');
const sightingsAccordionBody   = document.getElementById('sightings-accordion-body');
sightingsAccordionHeader.addEventListener('click', (e) => {
  if (e.target.closest('input[type="checkbox"]')) return;
  sightingsAccordionHeader.classList.toggle('open');
  sightingsAccordionBody.classList.toggle('open');
});

const sightingsYearSlider = document.getElementById('sightings-year-slider');
const sightingsYearLabel = document.getElementById('sightings-year-label');
const SIGHTINGS_MIN_YEAR = parseInt(sightingsYearSlider.min, 10);

function applySightingsYearFilter() {
  if (!map.getLayer('sightings-layer')) return;
  const since = parseInt(sightingsYearSlider.value, 10);
  if (since <= SIGHTINGS_MIN_YEAR) {
    map.setFilter('sightings-layer', null);
    sightingsYearLabel.textContent = 'all years';
    const countEl = document.getElementById('sightings-count');
    if (countEl) countEl.textContent = sightingsTotalCount.toLocaleString();
  } else {
    map.setFilter('sightings-layer', ['>=', ['to-number', ['get', 'year']], since]);
    sightingsYearLabel.textContent = `≥ ${since}`;
    const n = sightingsFeatures.filter(f => (f.properties.year ?? 0) >= since).length;
    const countEl = document.getElementById('sightings-count');
    if (countEl) countEl.textContent = n.toLocaleString();
  }
}

sightingsYearSlider.addEventListener('input', applySightingsYearFilter);

// ---------------------------------------------------------------------------
// Imagery info sidebar section
// ---------------------------------------------------------------------------

const iiDate = document.getElementById('ii-date');
const iiName = document.getElementById('ii-name');

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

async function fetchImageryInfo() {
  const center = map.getCenter();
  const { x, y } = merc(center.lng, center.lat);
  try {
    const zoom = Math.round(map.getZoom());
    const resp = await fetch(`/api/imagery-date?x=${x}&y=${y}&layer=${activeLayer}&zoom=${zoom}`);
    const data = await resp.json();
    setImageryInfo(data);
  } catch {
    // leave existing display intact
  }
}

let imageryInfoTimer = null;
map.on('moveend', () => {
  clearTimeout(imageryInfoTimer);
  imageryInfoTimer = setTimeout(fetchImageryInfo, 500);
});
