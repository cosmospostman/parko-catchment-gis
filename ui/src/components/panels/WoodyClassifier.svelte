<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../../lib/mapContext.ts';
  import type { MapContext } from '../../lib/mapContext.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import { parseBbox } from '../../lib/geo.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';
  import RegionListItem from '../shared/RegionListItem.svelte';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  const SUB_ROLE_COLORS: Record<string, string> = {
    presence: '#22c55e',
    absence:  '#ef4444',
    survey:   '#eab308',
  };

  const features = $derived(
    locationsStore.geojson?.features.filter(f => f.properties.parent_id === 'woody-classifier') ?? []
  );

  function zoomTo(bboxRaw: any) {
    const map = mapCtx.getMap();
    if (!map) return;
    const b = parseBbox(bboxRaw);
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 80, maxZoom: 16 });
  }
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.woody} />
  </label>
{/snippet}

<AccordionSection title="Woody classifier" count={features.length} {headerExtra}>
  {#each features as feat}
    <RegionListItem
      name={feat.properties.name}
      dotColor={SUB_ROLE_COLORS[feat.properties.sub_role ?? ''] ?? '#3b82f6'}
      dotStyle="dashed"
      onclick={() => zoomTo(feat.properties.bbox)}
    />
  {/each}
  {#if features.length === 0}
    <p class="empty">No regions loaded</p>
  {/if}
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #a78bfa; cursor: pointer; flex-shrink: 0; }
  .empty { padding: 8px 16px; color: #555; font-size: 12px; }
  :global(.accordion-body) { max-height: 220px; overflow-y: auto; }
</style>
