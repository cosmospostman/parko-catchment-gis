<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    count?: number | null;
    open?: boolean;
    headerExtra?: Snippet;
    children: Snippet;
  }

  let { title, count = null, open = $bindable(false), headerExtra, children }: Props = $props();
</script>

<div class="accordion">
  <button
    class="accordion-header"
    class:open
    onclick={() => (open = !open)}
    type="button"
  >
    {#if headerExtra}{@render headerExtra()}{/if}
    <span class="accordion-label">
      {title}{count != null ? ` (${count})` : ''}
    </span>
    <span class="accordion-chevron">&#9658;</span>
  </button>
  {#if open}
    <div class="accordion-body">
      {@render children()}
    </div>
  {/if}
</div>

<style>
  .accordion {
    border-bottom: 1px solid #333;
    flex-shrink: 0;
  }

  .accordion-header {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 10px 16px;
    background: none;
    border: none;
    color: #aaa;
    font-size: 12px;
    cursor: pointer;
    text-align: left;
    user-select: none;
  }

  .accordion-header:hover { background: #2e2e2e; }

  .accordion-label {
    flex: 1;
    font-weight: 500;
    color: #c0c0c0;
  }

  .accordion-chevron {
    font-size: 10px;
    color: #555;
    transition: transform 0.15s;
  }

  .accordion-header.open .accordion-chevron {
    transform: rotate(90deg);
  }

  .accordion-body {
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
  }
</style>
