<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../../lib/mapContext.ts';
  import type { MapContext } from '../../lib/mapContext.ts';
  import { s1 } from '../../stores/s1.svelte.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';
  import { fetchS1Locations, fetchS1Dates, fetchS1Status } from '../../lib/api.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  let locations = $state<string[]>([]);
  let dates = $state<string[]>([]);

  $effect(() => {
    if (!locationsStore.mapReady) return;
    fetchS1Locations().then(locs => (locations = locs)).catch(() => {});
  });

  $effect(() => {
    if (!s1.location) { dates = []; return; }
    fetchS1Dates(s1.location).then(d => {
      dates = d;
      if (d.length && !s1.date) s1.date = d[0];
    }).catch(() => { dates = []; });
  });

  let dots = 0;
  const dotStr = $derived('.'.repeat(dots));

  $effect(() => {
    const { location, band, date } = s1;
    if (!location || !date) return;
    let active = true;
    s1.pollState = 'idle';
    clearS1Layer();

    async function poll() {
      if (!active) return;
      try {
        const { state } = await fetchS1Status(location, band, date);
        if (!active) return;
        if (state === 'ready') {
          s1.pollState = 'ready';
          addS1Layer(location, band, date);
        } else {
          s1.pollState = 'building';
          dots = (dots + 1) % 4;
        }
      } catch {
        if (!active) return;
        s1.pollState = 'error';
      }
    }

    poll();
    const timer = setInterval(poll, 3000);
    return () => { active = false; clearInterval(timer); };
  });

  $effect(() => {
    if (!locationsStore.mapReady) return;
    const map = mapCtx.getMap();
    if (!map || !map.getLayer('s1-layer')) return;
    map.setPaintProperty('s1-layer', 'raster-opacity', s1.opacity);
  });

  function clearS1Layer() {
    const map = mapCtx.getMap();
    if (!map) return;
    try { if (map.getLayer('s1-layer')) map.removeLayer('s1-layer'); } catch {}
    try { if (map.getSource('s1-source')) map.removeSource('s1-source'); } catch {}
  }

  function addS1Layer(location: string, band: string, date: string) {
    const map = mapCtx.getMap();
    if (!map) return;
    clearS1Layer();
    try {
      map.addSource('s1-source', {
        type: 'raster',
        tiles: [`/s1-tile/${location}/${band}/${date}/{z}/{x}/{y}?cmap=plasma`],
        tileSize: 256, minzoom: 6, maxzoom: 14,
      });
      map.addLayer({
        id: 's1-layer', type: 'raster', source: 's1-source',
        paint: { 'raster-opacity': s1.opacity },
        layout: { visibility: layerVisibility.s1 ? 'visible' : 'none' },
      });
    } catch (err) {
      console.error('s1AddLayer failed:', err);
      s1.pollState = 'error';
    }
  }

  const statusText = $derived(
    s1.pollState === 'building' ? `building${dotStr}` :
    s1.pollState === 'ready' ? `${s1.location} · ${s1.band.toUpperCase()} · ${s1.date}` :
    s1.pollState === 'error' ? 'error' : ''
  );
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.s1} />
  </label>
{/snippet}

<AccordionSection title="Sentinel 1 Renderer" {headerExtra}>
  <div class="irow">
    <span class="ikey">parquet</span>
    <select bind:value={s1.location} class="sel">
      <option value="">— location —</option>
      {#each locations as loc}
        <option value={loc}>{loc}</option>
      {/each}
    </select>
  </div>
  <div class="irow">
    <span class="ikey">band</span>
    <select bind:value={s1.band} class="sel">
      <option value="vh">VH</option>
      <option value="vv">VV</option>
    </select>
  </div>
  <div class="irow">
    <span class="ikey">date</span>
    <select bind:value={s1.date} class="sel" disabled={dates.length === 0}>
      {#if dates.length === 0}
        <option value="">—</option>
      {:else}
        {#each dates as d}
          <option value={d}>{d}</option>
        {/each}
      {/if}
    </select>
  </div>
  <div class="irow">
    <span class="ikey">opacity</span>
    <input type="range" min="0" max="100" step="1"
      value={Math.round(s1.opacity * 100)}
      oninput={(e) => { s1.opacity = Number((e.target as HTMLInputElement).value) / 100; }}
    />
  </div>
  {#if statusText}
    <div id="s1-status" class:building={s1.pollState === 'building'} class:error={s1.pollState === 'error'}>
      {statusText}
    </div>
  {/if}
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #38bdf8; cursor: pointer; flex-shrink: 0; }

  .irow {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #b0b0b0;
    padding: 2px 14px;
  }

  .ikey { font-size: 11px; color: #555; min-width: 44px; flex-shrink: 0; }

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

  input[type="range"] { flex: 1; accent-color: #38bdf8; }

  #s1-status {
    font-size: 11px;
    color: #555;
    padding: 2px 14px;
    min-height: 16px;
  }
  #s1-status.building { color: #38bdf8; }
  #s1-status.error { color: #f87171; }
</style>
