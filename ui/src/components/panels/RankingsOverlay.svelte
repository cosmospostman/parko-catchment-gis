<script lang="ts">
  import { ranking, CMAP_GRADIENTS, type Cmap } from '../../stores/ranking.svelte.ts';
  import { locationsStore } from '../../stores/locations.svelte.ts';

  const locations = $derived(
    Object.entries(locationsStore.rankings)
      .filter(([, runs]) => runs.length > 0)
      .map(([id]) => id)
      .sort()
  );

  const stems = $derived(
    ranking.location ? (locationsStore.rankings[ranking.location] ?? []) : []
  );

  const swatchGradient = $derived(CMAP_GRADIENTS[ranking.cmap]);
</script>

<div class="ranking-section">
  <div class="section-label">Active ranking</div>

  {#if locations.length > 0}
    <div class="irow">
      <span class="ikey">location</span>
      <select id="rank-loc" bind:value={ranking.location} class="sel" onchange={() => { ranking.stem = null; }}>
        <option value={null}>— location —</option>
        {#each locations as loc}
          <option value={loc}>{loc}</option>
        {/each}
      </select>
    </div>
    {#if stems.length > 0}
      <div class="irow">
        <span class="ikey">run</span>
        <select id="rank-stem" bind:value={ranking.stem} class="sel">
          <option value={null}>— ranking —</option>
          {#each stems as { stem, label }}
            <option value={stem}>{label}</option>
          {/each}
        </select>
      </div>
    {/if}
  {/if}

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
    <select id="rank-cmap" bind:value={ranking.cmap} class="sel">
      <option value="rdylgn">RdYlGn</option>
      <option value="plasma">Plasma</option>
      <option value="viridis">Viridis</option>
    </select>
    <span class="colormap-swatch" style="background: {swatchGradient}"></span>
  </div>
</div>

<style>
  .ranking-section {
    border-top: 1px solid #333;
    padding: 12px 16px;
    flex-shrink: 0;
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
