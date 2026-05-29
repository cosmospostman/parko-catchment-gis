<script lang="ts">
  import MapCard from '../shared/MapCard.svelte';
  import { sightings } from '../../stores/sightings.svelte.ts';
  import { layerVisibility } from '../../stores/layerVisibility.svelte.ts';

  const MIN_YEAR = 1900;
  const MAX_YEAR = 2026;
  const RANGE = MAX_YEAR - MIN_YEAR;

  const filteredCount = $derived(
    sightings.features.filter(f => {
      const y = f.properties.year ?? 0;
      return y >= sightings.yearMin && y <= sightings.yearMax;
    }).length
  );

  const loFrac = $derived((sightings.yearMin - MIN_YEAR) / RANGE);
  const hiFrac = $derived((sightings.yearMax - MIN_YEAR) / RANGE);

  let trackEl: HTMLDivElement;
  let dragging = $state<'lo' | 'hi' | null>(null);

  function fracFromEvent(e: MouseEvent | TouchEvent): number {
    const rect = trackEl.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }

  function yearFromFrac(frac: number): number {
    return Math.round(MIN_YEAR + frac * RANGE);
  }

  function onTrackPointerDown(e: MouseEvent | TouchEvent) {
    const frac = fracFromEvent(e);
    const year = yearFromFrac(frac);
    const distLo = Math.abs(year - sightings.yearMin);
    const distHi = Math.abs(year - sightings.yearMax);
    dragging = distLo <= distHi ? 'lo' : 'hi';
    onMove(e);
    e.preventDefault();
  }

  function onMove(e: MouseEvent | TouchEvent) {
    if (!dragging) return;
    const year = yearFromFrac(fracFromEvent(e));
    if (dragging === 'lo') {
      sightings.yearMin = Math.min(year, sightings.yearMax);
    } else {
      sightings.yearMax = Math.max(year, sightings.yearMin);
    }
  }

  function onUp() { dragging = null; }

  const rangeLabel = $derived(`${sightings.yearMin} – ${sightings.yearMax}`);
</script>

<svelte:window
  onmousemove={onMove}
  ontouchmove={onMove}
  onmouseup={onUp}
  ontouchend={onUp}
/>

<MapCard
  title="ALA sightings — {filteredCount.toLocaleString()}"
  onclose={() => { layerVisibility.sightings = false; }}
>
  <div class="body">
    <div class="irow">
      <span class="ikey">DATE FILTER</span>
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="track-outer">
        <div
          class="track-wrap"
          bind:this={trackEl}
          onmousedown={onTrackPointerDown}
          ontouchstart={onTrackPointerDown}
          role="none"
        >
          <div class="track-bg"></div>
          <div
            class="track-fill"
            style="left:{loFrac * 100}%; width:{(hiFrac - loFrac) * 100}%;"
          ></div>

          <!-- Low knob -->
          <div
            class="knob"
            style="left:{loFrac * 100}%;"
            class:active={dragging === 'lo'}
            role="slider"
            tabindex="0"
            aria-valuemin={MIN_YEAR}
            aria-valuemax={sightings.yearMax}
            aria-valuenow={sightings.yearMin}
            aria-label="Start year"
            onkeydown={(e) => {
              if (e.key === 'ArrowLeft') sightings.yearMin = Math.max(MIN_YEAR, sightings.yearMin - 1);
              if (e.key === 'ArrowRight') sightings.yearMin = Math.min(sightings.yearMax, sightings.yearMin + 1);
            }}
          ></div>

          <!-- High knob -->
          <div
            class="knob"
            style="left:{hiFrac * 100}%;"
            class:active={dragging === 'hi'}
            role="slider"
            tabindex="0"
            aria-valuemin={sightings.yearMin}
            aria-valuemax={MAX_YEAR}
            aria-valuenow={sightings.yearMax}
            aria-label="End year"
            onkeydown={(e) => {
              if (e.key === 'ArrowLeft') sightings.yearMax = Math.max(sightings.yearMin, sightings.yearMax - 1);
              if (e.key === 'ArrowRight') sightings.yearMax = Math.min(MAX_YEAR, sightings.yearMax + 1);
            }}
          ></div>
        </div>
      </div>
      <span class="range-label">{rangeLabel}</span>
    </div>
  </div>
</MapCard>

<style>
  .body {
    padding: 8px 16px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .irow {
    display: flex;
    gap: 14px;
    align-items: center;
  }

  .ikey {
    font-size: 10px;
    font-weight: 600;
    color: #666;
    letter-spacing: 0.05em;
    flex-shrink: 0;
    white-space: nowrap;
  }

  .track-outer {
    flex: 1;
  }

  .track-wrap {
    position: relative;
    height: 18px;
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }

  .track-bg {
    position: absolute;
    left: 0; right: 0;
    height: 4px;
    border-radius: 2px;
    background: #444;
  }

  .track-fill {
    position: absolute;
    height: 4px;
    border-radius: 2px;
    background: var(--orange-accent);
    pointer-events: none;
  }

  .knob {
    position: absolute;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: var(--orange-accent);
    border: 2px solid #fff;
    transform: translateX(-50%);
    cursor: grab;
    transition: transform 0.08s, box-shadow 0.08s;
    z-index: 1;
  }

  .knob:hover, .knob.active {
    transform: translateX(-50%) scale(1.25);
    box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.35);
    cursor: grabbing;
  }

  .knob:focus-visible {
    outline: 2px solid var(--orange-accent);
    outline-offset: 2px;
  }

  .range-label {
    font-size: 11px;
    color: #888;
    white-space: nowrap;
  }
</style>
