<script lang="ts">
  import { setContext } from 'svelte';
  import type { Map as MapLibreMap } from 'maplibre-gl';
  import MapView from './components/MapView.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import { MAP_KEY } from './lib/mapContext.ts';

  // Shared mutable object placed in context so Sidebar's descendants can read the map.
  // MapView writes map into this object after construction.
  const mapRef = { getMap: () => null as MapLibreMap | null };
  setContext(MAP_KEY, mapRef);

  let mapView: MapView | null = $state(null);

  function handleLayerChange(layer: string) {
    mapView?.setActiveLayer(layer);
  }
</script>

<div class="layout">
  <Sidebar onLayerChange={handleLayerChange} />
  <MapView bind:this={mapView} />
</div>

<style>
  .layout {
    display: flex;
    width: 100%;
    height: 100%;
  }
</style>
