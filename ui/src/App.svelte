<script lang="ts">
  import { setContext } from 'svelte';
  import type { Map as MapLibreMap } from 'maplibre-gl';
  import MapView from './components/MapView.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import TrainingRegionsCard from './components/panels/TrainingRegionsCard.svelte';
  import ScoreCard from './components/shared/ScoreCard.svelte';
  import SightingsCard from './components/panels/SightingsCard.svelte';
  import BBoxCard from './components/panels/BBoxCard.svelte';
  import { MAP_KEY } from './lib/mapContext.ts';
  import { layerVisibility } from './stores/layerVisibility.svelte.ts';
  import { trainingSelection } from './stores/trainingSelection.svelte.ts';
  import { ranking } from './stores/ranking.svelte.ts';
  import { locationsStore } from './stores/locations.svelte.ts';

  // Shared mutable object placed in context so Sidebar's descendants can read the map.
  // MapView writes map into this object after construction.
  const mapRef = { getMap: () => null as MapLibreMap | null };
  setContext(MAP_KEY, mapRef);

  let mapView: MapView | null = $state(null);
  let trainingCardOpen = $state(false);
  let bboxOpen = $state(false);

  function handleLayerChange(layer: string) {
    mapView?.setActiveLayer(layer);
  }

  function handleTrainingClick(bboxRaw: string, subRole: string | null) {
    trainingCardOpen = true;
    trainingSelection.bbox = JSON.parse(bboxRaw);
    trainingSelection.sub_role = subRole;
  }

  function closeTrainingCard() {
    trainingCardOpen = false;
    layerVisibility.training = false;
    trainingSelection.bbox = null;
    trainingSelection.sub_role = null;
  }

  $effect(() => {
    if (trainingCardOpen) layerVisibility.training = true;
  });
</script>

<div class="layout">
  <Sidebar onLayerChange={handleLayerChange} bind:trainingCardOpen bind:bboxOpen />
  <div class="map-wrapper">
    <MapView bind:this={mapView} ontrainingclick={handleTrainingClick} />
    <div class="map-cards">
      {#if bboxOpen}
        <BBoxCard onclose={() => { bboxOpen = false; }} />
      {/if}
      {#if trainingCardOpen}
        <TrainingRegionsCard onclose={closeTrainingCard} />
      {/if}
      {#if layerVisibility.sightings}
        <SightingsCard />
      {/if}
      {#if ranking.location && ranking.stem}
        {@const runs = locationsStore.rankings[ranking.location] ?? []}
        {@const run = runs.find(r => r.stem === ranking.stem)}
        {#if run}
          <ScoreCard
            locId={ranking.location}
            stem={ranking.stem}
            label={run.label}
            onclose={() => { ranking.location = null; ranking.stem = null; }}
          />
        {/if}
      {/if}
    </div>
  </div>
</div>

<style>
  .layout {
    display: flex;
    width: 100%;
    height: 100%;
  }

  .map-wrapper {
    position: relative;
    flex: 1;
    overflow: hidden;
  }

  .map-cards {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 340px;
  }
</style>
