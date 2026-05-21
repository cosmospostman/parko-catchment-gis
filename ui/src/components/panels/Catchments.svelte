<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../../lib/mapContext.ts';
  import type { MapContext } from '../../lib/mapContext.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';
  import RegionListItem from '../shared/RegionListItem.svelte';
  import { fetchCatchments } from '../../lib/api.ts';
  import type { FeatureCollection } from 'geojson';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  let catchments = $state<FeatureCollection | null>(null);

  $effect(() => {
    if (!locationsStore.mapReady) return;
    fetchCatchments().then(g => (catchments = g)).catch(() => {});
  });

  function getBbox(geom: any): [number, number, number, number] | null {
    if (!geom) return null;
    const rings = geom.type === 'MultiPolygon' ? geom.coordinates.flat(1) : geom.coordinates;
    const pts = rings.flat(1) as [number, number][];
    const lngs = pts.map(c => c[0]);
    const lats = pts.map(c => c[1]);
    return [Math.min(...lngs), Math.min(...lats), Math.max(...lngs), Math.max(...lats)];
  }

  function zoomTo(geom: any) {
    const b = getBbox(geom);
    const map = mapCtx.getMap();
    if (!b || !map) return;
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 60, maxZoom: 9 });
  }
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.catchments} />
  </label>
{/snippet}

<AccordionSection title="Catchments" {headerExtra}>
  {#if catchments}
    {#each catchments.features as feat}
      <RegionListItem
        name={(feat.properties?.name ?? feat.properties?.id ?? 'Catchment') as string}
        dotColor="#38bdf8"
        onclick={() => zoomTo(feat.geometry)}
      />
    {/each}
  {:else}
    <p class="empty">Loading…</p>
  {/if}
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #38bdf8; cursor: pointer; flex-shrink: 0; }
  .empty { padding: 8px 16px; color: #555; font-size: 12px; }
  :global(.accordion-body) { max-height: 180px; overflow-y: auto; }
</style>
