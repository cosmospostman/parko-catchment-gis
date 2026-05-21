<script lang="ts">
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.s2tiles} />
  </label>
{/snippet}

<AccordionSection title="Sentinel-2 tiles" {headerExtra}>
  <p class="info">543 MGRS tiles. Labels appear at zoom ≥ 6.</p>
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #a78bfa; cursor: pointer; flex-shrink: 0; }
  .info { padding: 10px 16px; color: #666; font-size: 11px; }
</style>
