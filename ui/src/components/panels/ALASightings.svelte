<script lang="ts">
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';
  import { sightings } from '../../stores/sightings.svelte.ts';
  import AccordionSection from '../shared/AccordionSection.svelte';

  const MIN_YEAR = 1900;
  const MAX_YEAR = 2026;

  const filteredCount = $derived(
    sightings.yearFilter <= MIN_YEAR
      ? sightings.totalCount
      : sightings.features.filter(f => (f.properties.year ?? 0) >= sightings.yearFilter).length
  );

  const yearLabel = $derived(sightings.yearFilter <= MIN_YEAR ? 'all years' : `≥ ${sightings.yearFilter}`);
</script>

{#snippet headerExtra()}
  <label class="toggle-label">
    <input type="checkbox" onclick={(e) => e.stopPropagation()} bind:checked={layerVisibility.sightings} />
  </label>
{/snippet}

<AccordionSection title="ALA sightings ({filteredCount.toLocaleString()})" {headerExtra}>
  <div class="sightings-body">
    <span class="ikey">since</span>
    <div class="slider-wrap">
      <input
        type="range"
        min={MIN_YEAR}
        max={MAX_YEAR}
        step="1"
        bind:value={sightings.yearFilter}
      />
      <span class="year-label">{yearLabel}</span>
    </div>
  </div>
</AccordionSection>

<style>
  .toggle-label { display: flex; align-items: center; cursor: pointer; }
  .toggle-label input { accent-color: #f97316; cursor: pointer; flex-shrink: 0; }

  .sightings-body {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px 10px 30px;
    font-size: 12px;
    color: #b0b0b0;
  }

  .ikey { font-size: 11px; color: #555; min-width: 36px; flex-shrink: 0; }

  .slider-wrap { flex: 1; display: flex; flex-direction: column; gap: 2px; }

  input[type="range"] { width: 100%; accent-color: #f97316; }

  .year-label { font-size: 11px; color: #888; }
</style>
