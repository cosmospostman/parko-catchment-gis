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
    <div class="tools-section">
      <div class="section-label">Tools</div>
      <div class="tools-row">
        <button
          class="icon-btn"
          class:active={bboxOpen}
          title="BBox tool"
          type="button"
          onclick={() => { bboxOpen = !bboxOpen; }}
          aria-label="Toggle BBox tool"
        >
          <i class="ph ph-selection-plus"></i>
        </button>
      </div>
    </div>
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

  .tools-section {
    border-top: 1px solid #333;
    padding: 10px 16px;
  }

  .section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 8px;
  }

  .tools-row {
    display: flex;
    gap: 6px;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    flex-shrink: 0;
    background: none;
    border: 1px solid #444;
    border-radius: 4px;
    color: #666;
    font-size: 15px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s, border-color 0.1s;
  }
  .icon-btn:hover { border-color: #666; color: #aaa; }
  .icon-btn.active {
    background: var(--primary-dark);
    border-color: var(--primary-border);
    color: var(--primary-accent);
  }
</style>
