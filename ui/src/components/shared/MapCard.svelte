<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    onclose: () => void;
    children: Snippet;
  }

  let { title, onclose, children }: Props = $props();
</script>

<div class="map-card">
  <div class="card-header">
    <span class="card-title">{title}</span>
    <button class="close-btn" onclick={onclose} type="button" aria-label="Close">×</button>
  </div>
  <div class="card-body">
    {@render children()}
  </div>
</div>

<style>
  .map-card {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 10;
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
    padding: 8px 10px 8px 14px;
    border-bottom: 1px solid #333;
  }

  .card-title {
    font-size: 12px;
    font-weight: 600;
    color: #c0c0c0;
    letter-spacing: 0.02em;
    user-select: none;
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
