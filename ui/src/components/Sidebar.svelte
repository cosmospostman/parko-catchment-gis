<script lang="ts">
  import ImageryInfo from './panels/ImageryInfo.svelte';
  import LocationsList from './LocationsList.svelte';
  import { locationsStore } from '../stores/locations.svelte.ts';

  interface Props {
    onLayerChange: (layer: string) => void;
    trainingCardOpen: boolean;
    bboxOpen: boolean;
  }
  let { onLayerChange, trainingCardOpen = $bindable(false), bboxOpen = $bindable(false) }: Props = $props();

  const trainingCount = $derived(
    locationsStore.geojson?.features.filter(f => f.properties.parent_id === 'training').length ?? 0
  );
</script>

<aside class="sidebar">
  <div class="sidebar-header">
    <img src="/logo-560.png" alt="Parkinsonia Navigator 2026" class="logo" />
  </div>
  <div class="scroll-area">
    <button
      class="sidebar-row"
      class:active={trainingCardOpen}
      onclick={() => { trainingCardOpen = true; }}
      type="button"
    >
      Training regions{trainingCount > 0 ? ` (${trainingCount})` : ''}
    </button>
    <LocationsList />
  </div>
  <div class="bottom-panels">
    <ImageryInfo {onLayerChange} bind:bboxOpen />
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
    border-bottom: 1px solid #333;
    flex-shrink: 0;
  }

  .logo {
    width: 100%;
    display: block;
    user-select: none;
    pointer-events: none;
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
  .sidebar-row.active { color: var(--primary-accent); }
</style>
