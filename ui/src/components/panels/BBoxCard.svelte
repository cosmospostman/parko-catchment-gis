<script lang="ts">
  import MapCard from '../shared/MapCard.svelte';
  import { mapMode } from '../../stores/mapMode.svelte.ts';
  import { bbox as bboxStore } from '../../stores/bbox.svelte.ts';
  import { fetchChunkCoverage } from '../../lib/api.ts';

  interface Props {
    onclose: () => void;
  }
  let { onclose }: Props = $props();

  let copied = $state(false);
  let coverageYears = $state<number[] | null>(null);
  let inspecting = $state(false);

  $effect(() => {
    mapMode.current = 'bbox';
    return () => { mapMode.current = 'locations'; };
  });

  $effect(() => {
    const coords = bboxStore.coords;
    if (!coords) { coverageYears = null; inspecting = false; return; }
    fetchChunkCoverage(coords).then(years => { coverageYears = years; });
  });

  async function copyToClipboard() {
    try {
      await navigator.clipboard.writeText(bboxStore.yaml);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = bboxStore.yaml;
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    copied = true;
    setTimeout(() => (copied = false), 1500);
  }

  function clear() {
    bboxStore.yaml = '';
    bboxStore.pixelCount = 0;
    bboxStore.visible = false;
    bboxStore.coords = null;
    inspecting = false;
  }
</script>

<MapCard title="BBox tool" {onclose}>
  <div class="body">
    <div class="bbox-result">
      {#if bboxStore.visible}
        <div class="yaml-snippet">{bboxStore.yaml}</div>
        <div class="pixel-estimate">{bboxStore.pixelCount.toLocaleString()} S2 pixels (10 m)</div>
      {:else}
        <div class="empty-hint">Click and drag to select</div>
      {/if}
      <div class="btn-row">
        <button class="btn-copy" class:copied disabled={!bboxStore.visible} onclick={copyToClipboard} type="button">
          {copied ? 'Copied!' : 'Copy'}
        </button>
        <button class="btn-inspect" class:active={inspecting} disabled={!bboxStore.visible} onclick={() => { inspecting = !inspecting; }} type="button">
          Inspect
        </button>
        <button class="btn-clear" disabled={!bboxStore.visible} onclick={clear} type="button">Clear</button>
      </div>
    </div>

    {#if inspecting && bboxStore.visible}
      <div class="inspect-result">
        <div class="inspect-chips">
          {#if coverageYears === null}
            <span class="coverage-checking">checking…</span>
          {:else if coverageYears.length === 0}
            <span class="coverage-none">no pixel data cached</span>
          {:else}
            {#each coverageYears as year}
              <span class="coverage-year">{year}</span>
            {/each}
          {/if}
        </div>
      </div>
    {/if}
  </div>
</MapCard>

<style>
  .body {
    padding: 10px 12px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .bbox-result {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .yaml-snippet {
    background: #111;
    border-radius: 3px;
    padding: 7px 8px;
    font-family: monospace;
    font-size: 11px;
    color: #e0e0e0;
    white-space: pre;
    overflow-x: auto;
  }

  .pixel-estimate { font-size: 11px; color: #aaa; }

  .empty-hint { font-size: 12px; color: #555; font-style: italic; }

  .btn-row { display: flex; gap: 6px; margin-top: 2px; }

  .btn-copy {
    background: var(--primary-dark);
    color: var(--primary-accent);
    border: 1px solid var(--primary-border);
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s;
  }
  .btn-copy:hover:not(:disabled) { background: var(--primary-mid); }
  .btn-copy.copied { background: #14532d; color: #86efac; border-color: #1a6b35; }
  .btn-copy:disabled { opacity: 0.35; cursor: default; }

  .btn-inspect {
    background: none;
    color: #888;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s, border-color 0.1s;
  }
  .btn-inspect:hover:not(:disabled) { border-color: #666; color: #bbb; }
  .btn-inspect.active {
    background: #1a2a1a;
    border-color: #1a6b35;
    color: #86efac;
  }
  .btn-inspect:disabled { opacity: 0.35; cursor: default; }

  .btn-clear {
    background: none;
    color: #666;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s;
    margin-left: auto;
  }
  .btn-clear:hover:not(:disabled) { background: #2e2e2e; color: #aaa; }
  .btn-clear:disabled { opacity: 0.35; cursor: default; }

  .inspect-result {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .inspect-chips {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    align-items: center;
  }

  .coverage-year {
    font-size: 11px;
    font-weight: 600;
    color: #86efac;
    background: #14532d;
    border: 1px solid #1a6b35;
    border-radius: 3px;
    padding: 1px 5px;
  }

  .coverage-none { font-size: 11px; color: #555; font-style: italic; }
  .coverage-checking { font-size: 11px; color: #444; font-style: italic; }
</style>
