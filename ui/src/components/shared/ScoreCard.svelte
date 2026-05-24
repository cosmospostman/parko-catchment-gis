<script lang="ts">
  import MapCard from './MapCard.svelte';
  import { ranking, CMAP_GRADIENTS, type Cmap } from '../../stores/ranking.svelte.ts';

  interface Props {
    locId: string;
    stem: string;
    label: string;
    onclose: () => void;
  }

  let { locId, stem, label, onclose }: Props = $props();

  const swatchGradient = $derived(CMAP_GRADIENTS[ranking.cmap]);
</script>

<MapCard title="{locId} — {label}" {onclose}>
  <div class="body">
    <div class="irow meta">
      <span class="ikey">stem</span>
      <span class="meta-val">{stem}</span>
    </div>

    <div class="irow">
      <span class="ikey">opacity</span>
      <input type="range" min="0" max="100" step="1"
        value={Math.round(ranking.opacity * 100)}
        oninput={(e) => { ranking.opacity = Number((e.target as HTMLInputElement).value) / 100; }}
      />
    </div>

    <div class="irow">
      <span class="ikey">cutoff</span>
      <input type="range" min="0" max="100" step="1"
        value={Math.round(ranking.cutoff * 100)}
        oninput={(e) => { ranking.cutoff = Number((e.target as HTMLInputElement).value) / 100; }}
      />
      <span class="cutoff-label">{Math.round(ranking.cutoff * 100)}%</span>
    </div>

    <div class="irow">
      <span class="ikey">color</span>
      <select bind:value={ranking.cmap} class="sel">
        <option value="rdylgn">RdYlGn</option>
        <option value="plasma">Plasma</option>
        <option value="viridis">Viridis</option>
      </select>
      <span class="colormap-swatch" style="background: {swatchGradient}"></span>
    </div>
  </div>
</MapCard>

<style>
  .body {
    padding: 8px 12px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .irow {
    display: flex;
    gap: 6px;
    align-items: baseline;
  }

  .ikey {
    font-size: 11px;
    color: #555;
    min-width: 44px;
    flex-shrink: 0;
  }

  .meta-val {
    font-size: 11px;
    color: #666;
    font-family: monospace;
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

  input[type="range"] { flex: 1; accent-color: var(--primary-accent); }

  .cutoff-label { min-width: 32px; text-align: right; font-size: 11px; color: #888; }

  .colormap-swatch {
    display: inline-block;
    width: 52px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
    align-self: center;
  }
</style>
