<script lang="ts">
  import { getContext } from 'svelte';
  import type { Map as MapLibreMap, MapMouseEvent, Popup } from 'maplibre-gl';
  import { MAP_KEY } from '../lib/mapContext.ts';
  import type { MapContext } from '../lib/mapContext.ts';
  import { locationsStore } from '../stores/locations.svelte.ts';
  import { layerVisibility } from '../stores/layerVisibility.svelte.ts';
  import { mapMode } from '../stores/mapMode.svelte.ts';
  import { sightings } from '../stores/sightings.svelte.ts';
  import { bbox as bboxStore } from '../stores/bbox.svelte.ts';
  import { ranking } from '../stores/ranking.svelte.ts';
  import { imageryInfo } from '../stores/imageryInfo.svelte.ts';
  import { trainingSelection } from '../stores/trainingSelection.svelte.ts';
  import {
    fetchLocations, fetchRankings, fetchCatchments,
    fetchSightings, fetchImageryInfo,
  } from '../lib/api.ts';
  import {
    parseBbox, formatBbox, lngLatsToBbox, bboxToFeature,
    bboxToYaml, bboxPixelCount, merc,
  } from '../lib/geo.ts';
  import { buildS2Grid } from '../lib/s2grid.ts';

  declare const maplibregl: typeof import('maplibre-gl');

  interface Props {
    ontrainingclick?: (bbox: string, subRole: string | null) => void;
  }
  let { ontrainingclick }: Props = $props();

  let mapContainer: HTMLDivElement;
  let map: MapLibreMap | null = null;
  let popup: Popup | null = null;

  // Write our map instance into the context object provided by App.
  const mapCtx = getContext<MapContext>(MAP_KEY);
  $effect(() => { mapCtx.getMap = () => map; });

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
  // Layer URL helpers
  // ---------------------------------------------------------------------------
  const QGLOBE_LAYERS = new Set(['LatestStateProgram_QGovSISPUsers']);
  const DIRECT_TILE_URLS: Record<string, string> = {
    EsriWorldImagery: 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  };

  let activeLayer = 'LatestStateProgram_QGovSISPUsers';

  function tileUrl(name: string) { return `/tile/${name}/{z}/{x}/{y}`; }
  function wmsUrl(name: string) {
    return `/wms/${name}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=${name}&FORMAT=image/jpeg&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}`;
  }
  function layerUrl(name: string) {
    if (DIRECT_TILE_URLS[name]) return DIRECT_TILE_URLS[name];
    return QGLOBE_LAYERS.has(name) ? tileUrl(name) : wmsUrl(name);
  }

  // ---------------------------------------------------------------------------
  // Map init
  // ---------------------------------------------------------------------------
  $effect(() => {
    map = new maplibregl.Map({
      container: mapContainer,
      style: { version: 8, sources: {}, layers: [], glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf' },
      center: [144, -22],
      zoom: 6,
    });

    popup = new maplibregl.Popup({ closeButton: true, closeOnClick: false, maxWidth: '300px' });

    map.on('load', () => {
      if (!map) return;

      // Base imagery layer
      map.addSource('qld-globe', {
        type: 'raster',
        tiles: [layerUrl(activeLayer)],
        tileSize: 256,
        attribution: '© Queensland Globe',
      });
      map.addLayer({ id: 'qld-globe-layer', type: 'raster', source: 'qld-globe' });

      // Drawn bbox source
      map.addSource('drawn-bbox', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'drawn-bbox-fill', type: 'fill', source: 'drawn-bbox',
        paint: { 'fill-color': '#f59e0b', 'fill-opacity': 0.2 },
      });
      map.addLayer({
        id: 'drawn-bbox-line', type: 'line', source: 'drawn-bbox',
        paint: { 'line-color': '#f59e0b', 'line-width': 2 },
      });

      locationsStore.mapReady = true;

      Promise.all([fetchLocations(), fetchRankings()]).then(([geojson, rankings]) => {
        locationsStore.geojson = geojson;
        locationsStore.rankings = rankings;
        addLocationLayers(geojson);
        attachLocationPopups();
      }).catch(err => console.error('Failed to load locations:', err));

      loadSightings();
      loadCatchments();
      loadS2Tiles();
    });

    map.on('moveend', () => {
      clearTimeout(imageryInfoTimer);
      imageryInfoTimer = setTimeout(doFetchImageryInfo, 500);
    });

    return () => map?.remove();
  });

  // ---------------------------------------------------------------------------
  // Location layers
  // ---------------------------------------------------------------------------
  function addLocationLayers(geojson: typeof locationsStore.geojson) {
    if (!map || !geojson) return;
    const trainingFeatures = geojson.features
      .filter(f => f.properties.parent_id === 'training')
      .map(f => {
        const bbox = parseBbox(f.properties.bbox);
        const { pixelExtent } = buildS2Grid(bbox, f.properties as any);
        return pixelExtent.features[0] ?? f;
      });
    const locationFeatures = geojson.features.filter(f =>
      f.properties.parent_id !== 'training'
    );

    map.addSource('locations', { type: 'geojson', data: { type: 'FeatureCollection', features: locationFeatures } });
    map.addSource('training-regions', { type: 'geojson', data: { type: 'FeatureCollection', features: trainingFeatures } });

    map.addLayer({ id: 'loc-fill-location', type: 'fill', source: 'locations',
      filter: ['==', ['get', 'role'], 'location'],
      paint: { 'fill-color': '#7c3aed', 'fill-opacity': 0 } });
    map.addLayer({ id: 'loc-fill-sub', type: 'fill', source: 'locations',
      filter: ['==', ['get', 'role'], 'sub_bbox'],
      paint: { 'fill-color': COLOR_EXPR as any, 'fill-opacity': 0.25 } });
    map.addLayer({ id: 'loc-fill-score', type: 'fill', source: 'locations',
      filter: ['==', ['get', 'role'], 'score_bbox'],
      paint: { 'fill-color': SCORE_BBOX_COLOR, 'fill-opacity': 0.15 } });
    map.addLayer({ id: 'loc-line', type: 'line', source: 'locations',
      filter: ['!=', ['get', 'role'], 'score_bbox'],
      paint: {
        'line-color': ['match', ['get', 'role'], 'location', '#7c3aed', COLOR_EXPR as any],
        'line-width': ['match', ['get', 'role'], 'location', 3, 2],
      } });
    map.addLayer({ id: 'loc-line-score', type: 'line', source: 'locations',
      filter: ['==', ['get', 'role'], 'score_bbox'],
      paint: { 'line-color': SCORE_BBOX_COLOR, 'line-width': 2, 'line-dasharray': [4, 3] } });

    const tv = layerVisibility.training ? 'visible' : 'none';
    map.addLayer({ id: 'training-line', type: 'line', source: 'training-regions',
      layout: { visibility: tv },
      paint: { 'line-color': COLOR_EXPR as any, 'line-width': 2 } });

    map.addSource('training-grid', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'training-grid-lines', type: 'line', source: 'training-grid',
      paint: { 'line-color': COLOR_EXPR as any, 'line-width': 1, 'line-opacity': 0.6 } });


  }

  // ---------------------------------------------------------------------------
  // Reactive layer visibility
  // ---------------------------------------------------------------------------
  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    const vis = layerVisibility.training ? 'visible' : 'none';
    for (const id of ['training-line']) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
    }
  });

  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    const vis = layerVisibility.catchments ? 'visible' : 'none';
    for (const id of ['catchments-fill', 'catchments-line']) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
    }
  });

  $effect(() => {
    if (!map || !locationsStore.s2tilesReady) return;
    const vis = layerVisibility.s2tiles ? 'visible' : 'none';
    for (const id of ['s2tiles-fill', 's2tiles-line', 's2tiles-label']) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
    }
  });

  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    const vis = layerVisibility.sightings ? 'visible' : 'none';
    if (map.getLayer('sightings-layer')) map.setLayoutProperty('sightings-layer', 'visibility', vis);
  });

  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    const vis = layerVisibility.noise ? 'visible' : 'none';
    if (map.getLayer('noise-pixels')) map.setLayoutProperty('noise-pixels', 'visibility', vis);
  });

  // ---------------------------------------------------------------------------
  // Reactive ranking layer
  // ---------------------------------------------------------------------------
  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    const { location, stem, opacity, cmap, cutoff } = ranking;
    try { if (map.getLayer('ranking-layer')) map.removeLayer('ranking-layer'); } catch {}
    try { if (map.getSource('ranking')) map.removeSource('ranking'); } catch {}
    if (!location || !stem) return;
    const url = `/ranking-tile/${location}/${stem}/{z}/{x}/{y}?cmap=${cmap}&cutoff=${cutoff}`;
    map.addSource('ranking', { type: 'raster', tiles: [url], tileSize: 256 });
    map.addLayer({ id: 'ranking-layer', type: 'raster', source: 'ranking',
      paint: { 'raster-opacity': opacity } }, 'loc-fill-location');
  });

  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    if (map.getLayer('ranking-layer')) {
      map.setPaintProperty('ranking-layer', 'raster-opacity', ranking.opacity);
    }
  });

  // ---------------------------------------------------------------------------
  // Sightings layer
  // ---------------------------------------------------------------------------
  async function loadSightings() {
    try {
      const geojson = await fetchSightings();
      sightings.features = geojson.features ?? [];
      sightings.totalCount = sightings.features.length;

      map!.addSource('sightings', { type: 'geojson', data: geojson });
      map!.addLayer({
        id: 'sightings-layer', type: 'circle', source: 'sightings',
        layout: { visibility: layerVisibility.sightings ? 'visible' : 'none' },
        paint: {
          'circle-radius': 4, 'circle-color': '#f97316', 'circle-opacity': 0.75,
          'circle-stroke-width': 0.5, 'circle-stroke-color': '#fff',
        },
      });
      map!.moveLayer('sightings-layer');

      map!.on('click', 'sightings-layer', (e) => {
        const feat = e.features?.[0];
        if (!feat || !popup) return;
        popup.setLngLat(e.lngLat).setHTML(buildSightingPopupHtml(feat.properties)).addTo(map!);
      });
      map!.on('mouseenter', 'sightings-layer', () => { map!.getCanvas().style.cursor = 'pointer'; });
      map!.on('mouseleave', 'sightings-layer', () => { if (mapMode.current === 'locations') map!.getCanvas().style.cursor = ''; });
    } catch (err) { console.error('Failed to load sightings:', err); }
  }

  $effect(() => {
    if (!map || !locationsStore.mapReady) return;
    if (!map.getLayer('sightings-layer')) return;
    const since = sightings.yearFilter;
    const minYear = 1900;
    if (since <= minYear) {
      map.setFilter('sightings-layer', null);
    } else {
      map.setFilter('sightings-layer', ['>=', ['to-number', ['get', 'year']], since]);
    }
  });

  // ---------------------------------------------------------------------------
  // Catchments layer
  // ---------------------------------------------------------------------------
  async function loadCatchments() {
    try {
      const geojson = await fetchCatchments();
      map!.addSource('catchments', { type: 'geojson', data: geojson });
      const vis = layerVisibility.catchments ? 'visible' : 'none';
      map!.addLayer({ id: 'catchments-fill', type: 'fill', source: 'catchments',
        layout: { visibility: vis }, paint: { 'fill-color': '#38bdf8', 'fill-opacity': 0.08 } });
      map!.addLayer({ id: 'catchments-line', type: 'line', source: 'catchments',
        layout: { visibility: vis }, paint: { 'line-color': '#38bdf8', 'line-width': 2, 'line-dasharray': [4, 3] } });
    } catch (err) { console.error('Failed to load catchments:', err); }
  }

  // ---------------------------------------------------------------------------
  // Sentinel-2 tiles layer
  // ---------------------------------------------------------------------------
  async function loadS2Tiles() {
    try {
      const fetchJson = async (url: string) => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url} returned HTTP ${r.status}`);
        return r.json();
      };
      const [tiles, labels] = await Promise.all([
        fetchJson('/sentinel2_tiles.geojson'),
        fetchJson('/sentinel2_tile_labels.geojson'),
      ]);
      const vis = layerVisibility.s2tiles ? 'visible' : 'none';
      map!.addSource('s2tiles', { type: 'geojson', data: tiles });
      map!.addSource('s2tiles-centroids', { type: 'geojson', data: labels });
      map!.addLayer({ id: 's2tiles-fill', type: 'fill', source: 's2tiles',
        layout: { visibility: vis }, paint: { 'fill-color': '#a78bfa', 'fill-opacity': 0.04 } });
      map!.addLayer({ id: 's2tiles-line', type: 'line', source: 's2tiles',
        layout: { visibility: vis }, paint: { 'line-color': '#a78bfa', 'line-width': 1, 'line-opacity': 0.6 } });
      map!.addLayer({ id: 's2tiles-label', type: 'symbol', source: 's2tiles-centroids',
        minzoom: 6,
        layout: {
          visibility: vis,
          'text-field': ['get', 'name'], 'text-size': 11,
          'text-font': ['Noto Sans Regular'], 'text-anchor': 'center', 'text-allow-overlap': false,
        },
        paint: { 'text-color': '#a78bfa', 'text-opacity': 0.8, 'text-halo-color': '#111', 'text-halo-width': 1 } });
      locationsStore.s2tilesReady = true;
    } catch (err) { console.error('Failed to load S2 tiles:', err); }
  }

  // ---------------------------------------------------------------------------
  // BBox draw mode
  // ---------------------------------------------------------------------------
  let drawStart: { lng: number; lat: number } | null = null;
  let isDrawing = false;

  $effect(() => {
    if (!map) return;
    if (mapMode.current === 'bbox') {
      map.getCanvas().style.cursor = 'crosshair';
    } else {
      map.getCanvas().style.cursor = '';
      clearDraw();
    }
  });

  function clearDraw() {
    drawStart = null;
    isDrawing = false;
    const src = map?.getSource('drawn-bbox') as any;
    src?.setData({ type: 'FeatureCollection', features: [] });
    bboxStore.visible = false;
  }

  $effect(() => {
    if (!map) return;
    const m = map;

    const onMousedown = (e: MapMouseEvent) => {
      if (mapMode.current !== 'bbox') return;
      e.preventDefault();
      drawStart = { lng: e.lngLat.lng, lat: e.lngLat.lat };
      isDrawing = true;
      bboxStore.visible = false;
      (m.getSource('drawn-bbox') as any)?.setData({ type: 'FeatureCollection', features: [] });
      m.dragPan.disable();
    };
    const onMousemove = (e: MapMouseEvent) => {
      if (!isDrawing || mapMode.current !== 'bbox' || !drawStart) return;
      const b = lngLatsToBbox(drawStart as any, e.lngLat);
      (m.getSource('drawn-bbox') as any)?.setData({ type: 'FeatureCollection', features: [bboxToFeature(b)] });
      bboxStore.yaml = bboxToYaml(b);
      bboxStore.pixelCount = bboxPixelCount(b);
      bboxStore.visible = true;
    };
    const onMouseup = (e: MapMouseEvent) => {
      if (!isDrawing || mapMode.current !== 'bbox' || !drawStart) return;
      isDrawing = false;
      m.dragPan.enable();
      const b = lngLatsToBbox(drawStart as any, e.lngLat);
      (m.getSource('drawn-bbox') as any)?.setData({ type: 'FeatureCollection', features: [bboxToFeature(b)] });
      bboxStore.yaml = bboxToYaml(b);
      bboxStore.pixelCount = bboxPixelCount(b);
      bboxStore.visible = true;
      drawStart = null;
    };

    m.on('mousedown', onMousedown);
    m.on('mousemove', onMousemove);
    m.on('mouseup', onMouseup);
    return () => {
      m.off('mousedown', onMousedown);
      m.off('mousemove', onMousemove);
      m.off('mouseup', onMouseup);
    };
  });

  // ---------------------------------------------------------------------------
  // Imagery info
  // ---------------------------------------------------------------------------
  let imageryInfoTimer: ReturnType<typeof setTimeout> | null = null;

  async function doFetchImageryInfo() {
    if (!map) return;
    const center = map.getCenter();
    const { x, y } = merc(center.lng, center.lat);
    try {
      const data = await fetchImageryInfo(x, y, activeLayer, Math.round(map.getZoom()));
      if (data?.capturestart) {
        imageryInfo.date = data.capturestart === data.captureend || !data.captureend
          ? data.capturestart
          : `${data.capturestart} – ${data.captureend}`;
        imageryInfo.name = data.name ?? '—';
      }
    } catch {}
  }

  export function setActiveLayer(name: string) {
    if (!map) return;
    activeLayer = name;
    imageryInfo.activeLayer = name;
    try { map.removeLayer('qld-globe-layer'); map.removeSource('qld-globe'); } catch {}
    map.addSource('qld-globe', { type: 'raster', tiles: [layerUrl(name)], tileSize: 256, attribution: '© Queensland Globe' });
    const first = map.getStyle().layers[0]?.id;
    map.addLayer({ id: 'qld-globe-layer', type: 'raster', source: 'qld-globe' }, first);
    doFetchImageryInfo();
  }

  // ---------------------------------------------------------------------------
  // Popup helpers
  // ---------------------------------------------------------------------------
  function attachLocationPopups() {
    if (!map || !popup) return;
    const layers = ['loc-fill-location', 'loc-fill-sub', 'loc-fill-score', 'training-line'];
    for (const layer of layers) {
      map.on('click', layer, (e) => {
        if (mapMode.current !== 'locations') return;
        const feat = e.features?.[0];
        if (!feat) return;
        if (layer === 'training-line' && feat.properties.bbox) {
          ontrainingclick?.(feat.properties.bbox, feat.properties.sub_role ?? null);
          return;
        }
        popup!.setLngLat(e.lngLat).setHTML(buildLocationPopupHtml(feat.properties)).addTo(map!);
      });
      map.on('mouseenter', layer, () => { if (mapMode.current === 'locations') map!.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', layer, () => { if (mapMode.current === 'locations') map!.getCanvas().style.cursor = ''; });
    }
  }

  function roleBadge(props: Record<string, any>): string {
    if (props.role === 'sub_bbox') {
      const sub = props.sub_role ?? 'sub_bbox';
      return `<span class="role-badge role-${sub}">${sub}</span>`;
    }
    if (props.role === 'score_bbox') return `<span class="role-badge role-score-bbox">score_bbox</span>`;
    return `<span class="role-badge role-location">location</span>`;
  }

  function buildLocationPopupHtml(props: Record<string, any>): string {
    const title = props.name ?? props.label ?? props.id;
    const row = (label: string, val: string) =>
      `<div class="popup-row"><span class="popup-label">${label}</span><span>${val}</span></div>`;
    return `
      <div class="popup-title">${title}</div>
      ${row('role', roleBadge(props))}
      ${props.parent_id ? row('parent', props.parent_id) : ''}
      ${props.year ? row('year', `${props.year} (±5 yr window)`) : ''}
      ${row('bbox', formatBbox(props.bbox))}
      ${props.notes ? `<div class="popup-notes">${props.notes}</div>` : ''}
    `;
  }

  function buildSightingPopupHtml(p: Record<string, any>): string {
    const date = p.eventDate || (p.year ? `${p.year}${p.month ? '-' + String(p.month).padStart(2, '0') : ''}` : null);
    const observer = p.recordedBy && p.recordedBy !== 'None' && p.recordedBy !== 'null' ? p.recordedBy : null;
    const source = p.dataResourceName || null;
    const basis = p.basisOfRecord
      ? p.basisOfRecord.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c: string) => c.toUpperCase()) : null;
    const uncertainty = p.coordinateUncertaintyInMeters != null
      ? `±${Math.round(p.coordinateUncertaintyInMeters)} m` : null;
    const qaWarning = p.spatiallyValid === false || p.spatiallyValid === 'false'
      ? `<div class="popup-notes" style="color:#f87171;">⚠ Spatially invalid record</div>` : '';
    const row = (label: string, val: string | null) => val
      ? `<div class="popup-row"><span class="popup-label">${label}</span><span>${val}</span></div>` : '';
    return `
      <div class="popup-title">ALA sighting</div>
      ${row('date', date)} ${row('observer', observer)} ${row('source', source)}
      ${row('basis', basis)} ${row('uncertainty', uncertainty)} ${qaWarning}
    `;
  }
</script>

<div id="map" bind:this={mapContainer}></div>

<style>
  #map {
    flex: 1;
    height: 100%;
  }
</style>
