<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    onclose: () => void;
    children: Snippet;
  }

  let { title, onclose, children }: Props = $props();

  let collapsed = $state(false);
</script>

<div class="map-card">
  <div class="card-header" class:collapsed>
    <button class="title-btn" onclick={() => { collapsed = !collapsed; }} type="button">
      <span class="chevron" class:open={!collapsed}>&#9658;</span>
      <span class="card-title">{title}</span>
    </button>
    <button class="close-btn" onclick={onclose} type="button" aria-label="Close">×</button>
  </div>
  {#if !collapsed}
    <div class="card-body">
      {@render children()}
    </div>
  {/if}
</div>

<style>
  .map-card {
    background: #242424;
    border: 1px solid #333;
    border-radius: 6px;
    width: 340px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px 8px 10px;
    border-bottom: 1px solid #333;
  }

  .card-header.collapsed {
    border-bottom: none;
  }

  .title-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    flex: 1;
    text-align: left;
  }

  .chevron {
    font-size: 9px;
    color: #555;
    transition: transform 0.12s;
    transform: rotate(0deg);
  }

  .chevron.open {
    transform: rotate(90deg);
  }

  .card-title {
    font-size: 12px;
    font-weight: 600;
    color: #c0c0c0;
    letter-spacing: 0.02em;
    user-select: none;
  }

  .title-btn:hover .card-title {
    color: #e0e0e0;
  }

  .close-btn {
    background: none;
    border: none;
    color: #666;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    padding: 0 2px;
    border-radius: 3px;
  }

  .close-btn:hover {
    color: #ccc;
    background: #333;
  }

  .card-body {
    display: flex;
    flex-direction: column;
  }
</style>
