<script lang="ts">
  import { getContext } from 'svelte';
  import { MAP_KEY } from '../../lib/mapContext.ts';
  import type { MapContext } from '../../lib/mapContext.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';
  import type { LocationFeature } from '../../lib/api.ts';
  import { parseBbox } from '../../lib/geo.ts';
  import { buildS2Grid, emptyS2Grid } from '../../lib/s2grid.ts';
  import { trainingSelection } from '../../stores/trainingSelection.svelte.ts';
  import MapCard from '../shared/MapCard.svelte';

  interface Props {
    onclose: () => void;
  }

  let { onclose }: Props = $props();

  function setGrid(gridLines: object) {
    const map = mapCtx.getMap();
    if (!map) return;
    (map.getSource('training-grid') as any)?.setData(gridLines);
  }

  function handleClose() {
    const { gridLines } = emptyS2Grid();
    setGrid(gridLines);
    onclose();
  }

  const mapCtx = getContext<MapContext>(MAP_KEY);

  const features = $derived(
    locationsStore.geojson?.features.filter(f => f.properties.parent_id === 'training') ?? []
  );

  // Group features by location (part of name before ' — ')
  interface Group {
    location: string;
    presence: LocationFeature[];
    absence: LocationFeature[];
    survey: LocationFeature[];
  }

  const groups = $derived(() => {
    const map = new Map<string, Group>();
    for (const feat of features) {
      const [loc] = feat.properties.name.split(' — ');
      const key = loc.trim();
      if (!map.has(key)) map.set(key, { location: key, presence: [], absence: [], survey: [] });
      const g = map.get(key)!;
      const role = feat.properties.sub_role ?? feat.properties.label ?? '';
      if (role === 'presence') g.presence.push(feat as LocationFeature);
      else if (role === 'absence') g.absence.push(feat as LocationFeature);
      else g.survey.push(feat as LocationFeature);
    }
    return [...map.values()];
  });

  // Track which accordions are open (all open by default)
  let openGroups = $state<Record<string, boolean>>({});

  $effect(() => {
    const g = groups();
    for (const { location } of g) {
      if (!(location in openGroups)) openGroups[location] = false;
    }
  });

  function zoomTo(bboxRaw: any, subRole?: string) {
    const map = mapCtx.getMap();
    if (!map) return;
    const b = parseBbox(bboxRaw);
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 80, maxZoom: 16 });
    const { gridLines } = buildS2Grid(b, { sub_role: subRole ?? null });
    setGrid(gridLines);
  }

  $effect(() => {
    if (trainingSelection.bbox) zoomTo(trainingSelection.bbox, trainingSelection.sub_role ?? undefined);
  });

  // Strip the location prefix from a name to get a short label
  function shortLabel(name: string): string {
    const idx = name.indexOf(' — ');
    return idx >= 0 ? name.slice(idx + 3).trim() : name;
  }
</script>

<MapCard title="Training regions ({features.length})" onclose={handleClose}>
  <div class="body">
    {#if features.length === 0}
      <p class="empty">No regions loaded</p>
    {:else}
      {#each groups() as group (group.location)}
        <div class="group">
          <button
            class="group-header"
            class:open={openGroups[group.location]}
            onclick={() => { openGroups[group.location] = !openGroups[group.location]; }}
            type="button"
          >
            <span class="group-name">{group.location}</span>
            <span class="chevron">&#9658;</span>
          </button>
          {#if openGroups[group.location]}
            <div class="group-body">
              <div class="columns">
                <div class="col">
                  <div class="col-header presence">presence</div>
                  {#each group.presence as feat (feat.properties.id)}
                    <button class="chip" onclick={() => zoomTo(feat.properties.bbox, feat.properties.sub_role)} type="button">
                      {shortLabel(feat.properties.name)}
                    </button>
                  {/each}
                </div>
                <div class="col">
                  <div class="col-header absence">absence</div>
                  {#each group.absence as feat (feat.properties.id)}
                    <button class="chip" onclick={() => zoomTo(feat.properties.bbox, feat.properties.sub_role)} type="button">
                      {shortLabel(feat.properties.name)}
                    </button>
                  {/each}
                  {#each group.survey as feat (feat.properties.id)}
                    <button class="chip survey" onclick={() => zoomTo(feat.properties.bbox, feat.properties.sub_role)} type="button">
                      {shortLabel(feat.properties.name)}
                    </button>
                  {/each}
                </div>
              </div>
            </div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>
</MapCard>

<style>
  .body {
    background: #1e1e1e;
    border-radius: 0 0 6px 6px;
    max-height: 420px;
    overflow-y: auto;
  }

  .empty {
    padding: 8px 16px;
    color: #555;
    font-size: 12px;
  }

  .group {
    border-bottom: 1px solid #2a2a2a;
  }

  .group:last-child {
    border-bottom: none;
  }

  .group-header {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    background: none;
    border: none;
    color: #aaa;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: left;
    user-select: none;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .group-header:hover { background: #252525; }

  .group-name { flex: 1; }

  .chevron {
    font-size: 9px;
    color: #555;
    transition: transform 0.12s;
  }

  .group-header.open .chevron {
    transform: rotate(90deg);
  }

  .group-body {
    padding: 4px 8px 8px;
  }

  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .col {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .col-header {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 4px 4px;
    user-select: none;
  }

  .col-header.presence { color: #22c55e; }
  .col-header.absence  { color: #ef4444; }

  .chip {
    background: #2a2a2a;
    border: none;
    border-radius: 3px;
    color: #bbb;
    font-size: 11px;
    padding: 3px 6px;
    text-align: left;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .chip:hover {
    background: #333;
    color: #eee;
  }

  .chip.survey {
    color: #eab308;
  }
</style>
