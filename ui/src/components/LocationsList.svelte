<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../lib/mapContext.ts';
  import type { MapContext } from '../lib/mapContext.ts';
  import { locationsStore } from '../stores/locations.svelte.ts';
  import { ranking } from '../stores/ranking.svelte.ts';
  import { parseBbox } from '../lib/geo.ts';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  const locations = $derived(
    (locationsStore.geojson?.features.filter(f => f.properties.role === 'location') ?? [])
      .slice()
      .sort((a, b) => a.properties.name.localeCompare(b.properties.name))
  );

  let activeId = $state<string | null>(null);

  function zoomTo(id: string, bboxRaw: any) {
    activeId = id;
    const map = mapCtx.getMap();
    if (!map) return;
    const b = parseBbox(bboxRaw);
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 60, maxZoom: 14 });
  }

  function setRanking(locId: string, stem: string) {
    ranking.location = stem ? locId : null;
    ranking.stem = stem || null;
  }
</script>

<ul class="location-list">
  {#each locations as feat}
    {@const id = feat.properties.id}
    {@const runs = locationsStore.rankings[id] ?? []}
    <li class="loc-item" class:active={activeId === id}>
      <button
        class="loc-name-btn"
        type="button"
        onclick={() => zoomTo(id, feat.properties.bbox)}
      >{feat.properties.name}</button>
      {#if runs.length > 0}
        <select
          class="loc-ranking-select"
          value={ranking.location === id ? (ranking.stem ?? '') : ''}
          onchange={(e) => { setRanking(id, (e.target as HTMLSelectElement).value); }}
          autocomplete="off"
        >
          <option value="">— ranking —</option>
          {#each runs as { stem, label }}
            <option value={stem}>{label}</option>
          {/each}
        </select>
      {/if}
    </li>
  {/each}
</ul>

<style>
  .location-list {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .loc-item {
    padding: 10px 16px;
    border-bottom: 1px solid #2d2d2d;
    line-height: 1.4;
    transition: background 0.1s;
  }

  .loc-item:hover { background: #2e2e2e; }
  .loc-item.active { background: var(--primary-dark); }

  .loc-name-btn {
    display: block;
    width: 100%;
    background: none;
    border: none;
    padding: 0;
    font-size: 13px;
    font-weight: 500;
    color: #d0d0d0;
    cursor: pointer;
    text-align: left;
  }

  .loc-ranking-select {
    width: 100%;
    margin-top: 5px;
    background: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 11px;
    cursor: pointer;
  }
</style>
