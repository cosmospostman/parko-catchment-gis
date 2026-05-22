<script lang="ts">
  import BBoxPanel from './panels/BBoxPanel.svelte';
  import Catchments from './panels/Catchments.svelte';
  import ALASightings from './panels/ALASightings.svelte';
  import S1Renderer from './panels/S1Renderer.svelte';
  import NoiseFilter from './panels/NoiseFilter.svelte';
  import RankingsOverlay from './panels/RankingsOverlay.svelte';
  import ImageryInfo from './panels/ImageryInfo.svelte';
  import LocationsList from './LocationsList.svelte';
  import { locationsStore } from '../stores/locations.svelte.ts';

  interface Props {
    onLayerChange: (layer: string) => void;
    trainingCardOpen: boolean;
  }
  let { onLayerChange, trainingCardOpen = $bindable(false) }: Props = $props();

  const trainingCount = $derived(
    locationsStore.geojson?.features.filter(f => f.properties.parent_id === 'training').length ?? 0
  );
</script>

<aside class="sidebar">
  <div class="sidebar-header">
    <h1>🌿 Parko GIS</h1>
  </div>
  <div class="scroll-area">
    <BBoxPanel />
    <button
      class="sidebar-row"
      class:active={trainingCardOpen}
      onclick={() => { trainingCardOpen = true; }}
      type="button"
    >
      Training regions{trainingCount > 0 ? ` (${trainingCount})` : ''}
    </button>
    <Catchments />
    <ALASightings />
    <S1Renderer />
    <NoiseFilter />
    <LocationsList />
  </div>
  <div class="bottom-panels">
    <RankingsOverlay />
    <ImageryInfo {onLayerChange} />
  </div>
</aside>

<style>
  .sidebar {
    width: 280px;
    height: 100%;
    background: #242424;
    border-right: 1px solid #333;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 10px 14px;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .sidebar-header h1 {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #f0f0f0;
  }

  .scroll-area {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
  }

  .bottom-panels {
    flex-shrink: 0;
  }

  .sidebar-row {
    width: 100%;
    display: flex;
    align-items: center;
    padding: 10px 16px;
    background: none;
    border: none;
    border-bottom: 1px solid #333;
    color: #c0c0c0;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
    user-select: none;
  }

  .sidebar-row:hover { background: #2e2e2e; }
  .sidebar-row.active { color: var(--primary-accent, #4ade80); }
</style>
