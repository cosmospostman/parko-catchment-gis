<script lang="ts">
  interface Props {
    name: string;
    dotColor?: string;
    dotStyle?: string;
    onclick: () => void;
  }
  let { name, dotColor = '#888', dotStyle = 'solid', onclick }: Props = $props();
</script>

<button class="region-item" type="button" {onclick}>
  <span class="dot" style="background:{dotColor}; {dotStyle === 'dashed' ? 'border: 2px dashed ' + dotColor + '; background: transparent;' : ''}"></span>
  <span class="name">{name}</span>
</button>

<style>
  .region-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 7px 16px 7px 30px;
    background: none;
    border: none;
    border-top: 1px solid #2a2a2a;
    color: #b0b0b0;
    font-size: 12px;
    cursor: pointer;
    text-align: left;
    transition: background 0.1s;
  }
  .region-item:hover { background: #262626; }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
