<script lang="ts">
  import { getContext } from 'svelte';
  import { createTreeView } from '@melt-ui/svelte';
  import { MAP_KEY } from '../lib/mapContext.ts';
  import type { MapContext } from '../lib/mapContext.ts';
  import { locationsStore } from '../stores/locations.svelte.ts';
  import { ranking } from '../stores/ranking.svelte.ts';
  import { parseBbox } from '../lib/geo.ts';

  const mapCtx = getContext<MapContext>(MAP_KEY);

  const { elements: { tree, item, group }, helpers: { isExpanded } } = createTreeView();

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

  function selectRun(locId: string, stem: string) {
    ranking.location = locId;
    ranking.stem = stem;
  }
</script>

<ul class="location-tree" use:tree>
  {#each locations as feat}
    {@const id = feat.properties.id}
    {@const runs = locationsStore.rankings[id] ?? []}
    {@const hasChildren = runs.length > 0}
    {@const expanded = $isExpanded(id)}

    <li
      class="loc-item"
      class:active={activeId === id}
      {...$item({ id, hasChildren })}
      use:item
    >
      <button
        class="loc-row"
        type="button"
        onclick={() => zoomTo(id, feat.properties.bbox)}
      >
        {#if hasChildren}
          <span class="chevron" class:open={expanded}>&#9658;</span>
        {:else}
          <span class="chevron-spacer"></span>
        {/if}
        <span class="loc-name">{feat.properties.name}</span>
      </button>

      {#if hasChildren}
        <ul class="run-list" {...$group({ id })} use:group>
          {#each runs as { stem, label }}
            {@const childId = `${id}::${stem}`}
            {@const active = ranking.location === id && ranking.stem === stem}
            <li
              class="run-item"
              class:run-active={active}
              {...$item({ id: childId })}
              use:item
            >
              <button
                class="run-chip"
                type="button"
                onclick={() => selectRun(id, stem)}
              >{label}</button>
            </li>
          {/each}
        </ul>
      {/if}
    </li>
  {/each}
</ul>

<style>
  .location-tree {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .loc-item {
    border-bottom: 1px solid #2d2d2d;
  }

  .loc-item:hover { background: #2a2a2a; }
  .loc-item.active > .loc-row { color: var(--primary-accent, #4ade80); }

  .loc-row {
    display: flex;
    align-items: center;
    gap: 5px;
    width: 100%;
    padding: 9px 12px 9px 10px;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
  }

  .chevron {
    font-size: 9px;
    color: #555;
    transition: transform 0.12s;
    flex-shrink: 0;
  }

  .chevron.open {
    transform: rotate(90deg);
  }

  .chevron-spacer {
    display: inline-block;
    width: 9px;
    flex-shrink: 0;
  }

  .loc-name {
    font-size: 13px;
    font-weight: 500;
    color: #d0d0d0;
    user-select: none;
  }

  .loc-row:hover .loc-name { color: #eee; }

  .run-list {
    list-style: none;
    margin: 0;
    padding: 2px 0 6px 24px;
  }

  .run-item {
    padding: 1px 0;
  }

  .run-chip {
    background: #2a2a2a;
    border: none;
    border-radius: 3px;
    color: #aaa;
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
    text-align: left;
    width: 100%;
  }

  .run-chip:hover {
    background: #333;
    color: #ddd;
  }

  .run-item.run-active .run-chip {
    background: #1a3a2a;
    color: var(--primary-accent, #4ade80);
  }
</style>
