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

map.on('load', () => {
  const WMS_URL =
    '/wms' +
    '?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap' +
    '&LAYERS=LatestStateProgram_AllUsers&FORMAT=image/jpeg' +
    '&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}';

  map.addSource('qld-globe', {
    type: 'raster',
    tiles: [WMS_URL],
    tileSize: 256,
    attribution: '© Queensland Globe',
  });

  map.addLayer({
    id: 'qld-globe-layer',
    type: 'raster',
    source: 'qld-globe',
  });

  // Load location bboxes after base layer is ready
  loadLocations();
});

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/** MapLibre match expression: sub_role → colour (falls back on location colour) */
const COLOR_EXPR = [
  'match', ['get', 'sub_role'],
  'presence', '#22c55e',
  'absence',  '#ef4444',
  'survey',   '#eab308',
  /* location (sub_role is null) */ '#3b82f6',
];

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

      // Fill — location boxes below sub-bbox boxes
      map.addLayer({
        id: 'loc-fill-location',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'location'],
        paint: {
          'fill-color': '#3b82f6',
          'fill-opacity': 0.12,
        },
      });

      map.addLayer({
        id: 'loc-fill-sub',
        type: 'fill',
        source: 'locations',
        filter: ['==', ['get', 'role'], 'sub_bbox'],
        paint: {
          'fill-color': COLOR_EXPR,
          'fill-opacity': 0.25,
        },
      });

      // Outlines
      map.addLayer({
        id: 'loc-line',
        type: 'line',
        source: 'locations',
        paint: {
          'line-color': COLOR_EXPR,
          'line-width': [
            'match', ['get', 'role'],
            'location', 1.5,
            /* sub_bbox */ 2,
          ],
        },
      });

      buildSidebar(geojson);
      attachPopups();
    })
    .catch(err => console.error('Failed to load locations:', err));
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function buildSidebar(geojson) {
  const list = document.getElementById('location-list');
  const locations = geojson.features.filter(f => f.properties.role === 'location');

  // Sort alphabetically by name
  locations.sort((a, b) => a.properties.name.localeCompare(b.properties.name));

  for (const feat of locations) {
    const { id, name } = feat.properties;
    const bbox = feat.properties.bbox; // [lon_min, lat_min, lon_max, lat_max]

    const li = document.createElement('li');
    li.dataset.id = id;
    li.innerHTML = `<div class="loc-name">${name}</div><div class="loc-id">${id}</div>`;

    li.addEventListener('click', () => {
      document.querySelectorAll('#location-list li').forEach(el => el.classList.remove('active'));
      li.classList.add('active');
      map.fitBounds(
        [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        { padding: 60, maxZoom: 14 }
      );
    });

    list.appendChild(li);
  }
}

// ---------------------------------------------------------------------------
// Click popups
// ---------------------------------------------------------------------------

const popup = new maplibregl.Popup({
  closeButton: true,
  closeOnClick: false,
  maxWidth: '300px',
});

function formatBbox(bbox) {
  if (!bbox) return '—';
  return `${bbox[0].toFixed(5)}, ${bbox[1].toFixed(5)}, ${bbox[2].toFixed(5)}, ${bbox[3].toFixed(5)}`;
}

function roleBadge(props) {
  if (props.role === 'sub_bbox') {
    const sub = props.sub_role ?? 'sub_bbox';
    return `<span class="role-badge role-${sub}">${sub}</span>`;
  }
  return `<span class="role-badge role-location">location</span>`;
}

function buildPopupHtml(props) {
  const title = props.name ?? props.label ?? props.id;
  const bbox = props.bbox;
  const notes = props.notes;
  const parentLabel = props.parent_id ? `<div class="popup-row"><span class="popup-label">parent</span><span>${props.parent_id}</span></div>` : '';

  return `
    <div class="popup-title">${title}</div>
    <div class="popup-row"><span class="popup-label">role</span>${roleBadge(props)}</div>
    ${parentLabel}
    <div class="popup-row"><span class="popup-label">bbox</span><span>${formatBbox(bbox)}</span></div>
    ${notes ? `<div class="popup-notes">${notes}</div>` : ''}
  `;
}

function attachPopups() {
  const clickableLayers = ['loc-fill-location', 'loc-fill-sub'];

  for (const layer of clickableLayers) {
    map.on('click', layer, (e) => {
      const feat = e.features[0];
      if (!feat) return;
      popup
        .setLngLat(e.lngLat)
        .setHTML(buildPopupHtml(feat.properties))
        .addTo(map);
    });

    map.on('mouseenter', layer, () => {
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', layer, () => {
      map.getCanvas().style.cursor = '';
    });
  }
}
