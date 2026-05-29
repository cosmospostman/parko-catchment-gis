<script lang="ts">
  import MapCard from '../shared/MapCard.svelte';
  import { mapMode } from '../../stores/mapMode.svelte.ts';
  import { bbox as bboxStore } from '../../stores/bbox.svelte.ts';

  interface Props {
    onclose: () => void;
  }
  let { onclose }: Props = $props();

  let copied = $state(false);

  $effect(() => {
    mapMode.current = 'bbox';
    return () => { mapMode.current = 'locations'; };
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
  }
</script>

<MapCard title="BBox tool" {onclose}>
  <div class="body">
    <p class="hint">Click and drag on the map to draw a bounding box.</p>
    {#if bboxStore.visible}
      <div class="bbox-result">
        <div class="yaml-snippet">{bboxStore.yaml}</div>
        <div class="pixel-estimate">{bboxStore.pixelCount.toLocaleString()} S2 pixels (10 m)</div>
        <div class="btn-row">
          <button class="btn-copy" class:copied onclick={copyToClipboard} type="button">
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button class="btn-clear" onclick={clear} type="button">Clear</button>
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

  .hint { color: #888; font-size: 12px; line-height: 1.5; }

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
  .btn-copy:hover { background: var(--primary-mid); }
  .btn-copy.copied { background: #14532d; color: #86efac; border-color: #1a6b35; }

  .btn-clear {
    background: none;
    color: #666;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s;
  }
  .btn-clear:hover { background: #2e2e2e; color: #aaa; }
</style>
