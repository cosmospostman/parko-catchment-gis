<script lang="ts">
  import { imageryInfo } from '../../stores/imageryInfo.svelte.ts';

  interface Props {
    onLayerChange: (layer: string) => void;
  }
  let { onLayerChange }: Props = $props();

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
</style>
