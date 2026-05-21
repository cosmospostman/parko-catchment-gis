<script lang="ts">
  import BBoxPanel from './panels/BBoxPanel.svelte';
  import TrainingRegions from './panels/TrainingRegions.svelte';
  import WoodyClassifier from './panels/WoodyClassifier.svelte';
  import Catchments from './panels/Catchments.svelte';
  import S2Tiles from './panels/S2Tiles.svelte';
  import ALASightings from './panels/ALASightings.svelte';
  import S1Renderer from './panels/S1Renderer.svelte';
  import NoiseFilter from './panels/NoiseFilter.svelte';
  import RankingsOverlay from './panels/RankingsOverlay.svelte';
  import ImageryInfo from './panels/ImageryInfo.svelte';
  import LocationsList from './LocationsList.svelte';

  interface Props {
    onLayerChange: (layer: string) => void;
  }
  let { onLayerChange }: Props = $props();
</script>

<aside class="sidebar">
  <div class="sidebar-header">
    <h1>🌿 Parko GIS</h1>
  </div>
  <div class="scroll-area">
    <BBoxPanel />
    <TrainingRegions />
    <WoodyClassifier />
    <Catchments />
    <S2Tiles />
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
</style>
