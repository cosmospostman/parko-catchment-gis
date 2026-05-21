<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../../lib/mapContext.ts';
  import type { MapContext } from '../../lib/mapContext.ts';
  import { noise } from '../../stores/noiseFilter.svelte.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';
  import { fetchHeuristicRegions, fetchHeuristicScores } from '../../lib/api.ts';
  import type { HeuristicRegion, NoisePixel } from '../../lib/api.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  let regions = $state<HeuristicRegion[]>([]);
  const presence = $derived(regions.filter(r => r.label === 'presence').sort((a,b) => a.id.localeCompare(b.id)));
  const absence  = $derived(regions.filter(r => r.label === 'absence').sort((a,b) => a.id.localeCompare(b.id)));

  $effect(() => {
    if (!locationsStore.mapReady) return;
    fetchHeuristicRegions().then(r => (regions = r)).catch(() => {});
  });

  $effect(() => {
    const regionId = noise.regionId;
    if (!regionId) {
      noise.pixelData = null;
      noise.pollState = 'idle';
      clearNoiseLayer();
      return;
    }
    let active = true;
    noise.pollState = 'scoring';

    async function poll() {
      if (!active) return;
      try {
        const { status, data } = await fetchHeuristicScores(regionId);
        if (!active) return;
        if (status === 202) { noise.pollState = 'scoring'; return; }
        if (status !== 200) { noise.pollState = 'error'; return; }
        if (data.missing_data) { noise.pollState = 'error'; return; }
        noise.pixelData = data.pixels ?? null;
        noise.pollState = 'ready';
        ensureNoiseLayer();
        reclassify();
      } catch {
        if (!active) return;
        noise.pollState = 'error';
      }
    }

    poll();
    const timer = setInterval(poll, 3000);
    return () => { active = false; clearInterval(timer); };
  });

  $effect(() => {
    if (noise.pollState === 'ready' && noise.pixelData) reclassify();
  });

  function classify(p: NoisePixel, threshold: number) {
    const prob = p.prob_woody;
    if (prob === null || prob === undefined) return 'kept';
    return Number(prob) >= threshold ? 'kept' : 'dropped';
  }

  function buildGeoJSON(pixels: NoisePixel[], threshold: number) {
    let kept = 0, dropped = 0;
    const features = pixels.map(p => {
      const status = classify(p, threshold);
      if (status === 'kept') kept++; else dropped++;
      return { type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: [p.lon, p.lat] },
        properties: { id: p.id, status, prob_woody: p.prob_woody } };
    });
    return { geojson: { type: 'FeatureCollection' as const, features }, kept, dropped };
  }

  let keptCount = $state(0);
  let droppedCount = $state(0);

  function reclassify() {
    const map = mapCtx.getMap();
    if (!noise.pixelData || !map?.getSource('noise-pixels')) return;
    const { geojson, kept, dropped } = buildGeoJSON(noise.pixelData, noise.threshold);
    (map.getSource('noise-pixels') as any).setData(geojson);
    keptCount = kept;
    droppedCount = dropped;
  }

  function clearNoiseLayer() {
    const map = mapCtx.getMap();
    if (!map) return;
    if (map.getSource('noise-pixels')) {
      (map.getSource('noise-pixels') as any).setData({ type: 'FeatureCollection', features: [] });
    }
  }

  function ensureNoiseLayer() {
    const map = mapCtx.getMap();
    if (!map || map.getSource('noise-pixels')) return;
    map.addSource('noise-pixels', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({
      id: 'noise-pixels', type: 'circle', source: 'noise-pixels',
      layout: { visibility: 'visible' },
      paint: {
        'circle-radius': 3, 'circle-opacity': 0.75,
        'circle-color': ['match', ['get', 'status'], 'dropped', '#ef4444', '#22c55e'],
      },
    });
    layerVisibility.noise = true;
  }

  const statusText = $derived(
    noise.pollState === 'scoring' ? 'scoring…' :
    noise.pollState === 'error' ? 'no data for this region' :
    noise.pollState === 'ready' ? `${keptCount} kept · ${droppedCount} dropped` : ''
  );

  const statusClass = $derived(
    noise.pollState === 'scoring' ? 'scoring' :
    noise.pollState === 'error' ? 'error' : ''
  );
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.noise} />
  </label>
{/snippet}

<AccordionSection title="Noise filter pixels (heuristic)" {headerExtra}>
  <div class="irow">
    <span class="ikey">region</span>
    <select bind:value={noise.regionId} class="sel">
      <option value="">— select —</option>
      <optgroup label="Presence">
        {#each presence as r}
          <option value={r.id}>{r.id}</option>
        {/each}
      </optgroup>
      <optgroup label="Absence">
        {#each absence as r}
          <option value={r.id}>{r.id}</option>
        {/each}
      </optgroup>
    </select>
  </div>
  <div class="irow">
    <span class="ikey">min score ≥</span>
    <input type="range" min="0" max="1" step="0.01" bind:value={noise.threshold} />
    <span class="noise-val">{noise.threshold.toFixed(2)}</span>
  </div>
  <div class="noise-status {statusClass}">{statusText}</div>
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #facc15; cursor: pointer; flex-shrink: 0; }

  .irow {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #b0b0b0;
    padding: 2px 16px;
  }

  .ikey { font-size: 11px; color: #555; min-width: 60px; flex-shrink: 0; }

  .sel {
    flex: 1;
    background: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 4px;
    font-size: 11px;
    cursor: pointer;
    min-width: 0;
  }

  input[type="range"] { flex: 1; accent-color: #facc15; }

  .noise-val { font-size: 11px; color: #888; min-width: 28px; text-align: right; }

  .noise-status {
    font-size: 11px;
    color: #555;
    padding: 2px 16px;
    min-height: 16px;
  }
  .noise-status.scoring { color: #facc15; }
  .noise-status.error { color: #f87171; }
</style>
