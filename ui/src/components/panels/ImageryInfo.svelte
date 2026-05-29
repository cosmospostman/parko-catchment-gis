<script lang="ts">
  import { imageryInfo } from '../../stores/imageryInfo.svelte.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import ToggleButton from '../shared/ToggleButton.svelte';

  interface Props {
    onLayerChange: (layer: string) => void;
    bboxOpen: boolean;
  }
  let { onLayerChange, bboxOpen = $bindable(false) }: Props = $props();

  const LAYERS = [
    { value: 'LatestStateProgram_QGovSISPUsers', label: 'Latest aerial (QGlobe)' },
    { value: 'LatestStateProgram_AllUsers', label: 'Latest aerial (public)' },
    { value: 'LatestSatelliteWOS_AllUsers', label: 'Latest satellite (Planet)' },
    { value: 'EsriWorldImagery', label: 'Recent satellite (Esri)' },
  ];
</script>

<div class="imagery-info">
  <div class="section-label">Imagery</div>
  <div class="irow">
    <span class="ikey">layer</span>
    <select bind:value={imageryInfo.activeLayer} class="sel" onchange={() => onLayerChange(imageryInfo.activeLayer)}>
      {#each LAYERS as { value, label }}
        <option {value}>{label}</option>
      {/each}
    </select>
  </div>
  <div class="irow">
    <span class="ikey">captured</span>
    <span class="ival">{imageryInfo.date || ''}{#if !imageryInfo.date}<span class="dim">loading…</span>{/if}</span>
  </div>
  <div class="irow">
    <span class="ikey">dataset</span>
    <span class="ival mono">{imageryInfo.name || ''}{#if !imageryInfo.name}<span class="dim">—</span>{/if}</span>
  </div>
  <div class="toggle-row">
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
    <ToggleButton
      label="S2 tiles"
      palette="secondary"
      active={layerVisibility.s2tiles}
      onclick={() => { layerVisibility.s2tiles = !layerVisibility.s2tiles; }}
    />
    <ToggleButton
      label="Catchments"
      palette="blue"
      active={layerVisibility.catchments}
      onclick={() => { layerVisibility.catchments = !layerVisibility.catchments; }}
    />
    <ToggleButton
      label="Sightings"
      palette="orange"
      active={layerVisibility.sightings}
      onclick={() => { layerVisibility.sightings = !layerVisibility.sightings; }}
    />
  </div>
</div>

<style>
  .imagery-info {
    border-top: 1px solid #333;
    padding: 12px 16px;
    flex-shrink: 0;
    min-height: 0;
  }

  .section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 6px;
  }

  .irow {
    display: flex;
    gap: 6px;
    margin-bottom: 4px;
    align-items: baseline;
  }

  .ikey {
    font-size: 11px;
    color: #555;
    min-width: 44px;
    flex-shrink: 0;
  }

  .sel {
    flex: 1;
    background: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 4px;
    font-size: 11px;
    cursor: pointer;
    min-width: 0;
  }

  .ival { font-size: 12px; color: #b0b0b0; line-height: 1.3; word-break: break-word; }
  .ival.mono { font-family: monospace; font-size: 11px; }
  .dim { color: #444; font-style: italic; }
  .toggle-row {
    display: flex;
    gap: 6px;
    margin-top: 8px;
    align-items: stretch;
  }
  .toggle-row :global(.toggle-btn) { flex: 1; }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
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
