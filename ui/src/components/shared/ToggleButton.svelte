<script lang="ts">
  interface Props {
    active: boolean;
    onclick: () => void;
    label: string;
    palette?: 'primary' | 'secondary' | 'blue' | 'orange';
  }
  let { active, onclick, label, palette = 'primary' }: Props = $props();
</script>

<button class="toggle-btn" class:active data-palette={palette} {onclick} type="button">{label}</button>

<style>
  .toggle-btn {
    width: 100%;
    padding: 5px 0;
    background: none;
    border: 1px solid #444;
    border-radius: 4px;
    color: #666;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s, border-color 0.1s;
  }
  .toggle-btn:hover { border-color: #666; color: #aaa; }

  .toggle-btn[data-palette="primary"].active {
    background: var(--primary-dark);
    border-color: var(--primary-border);
    color: var(--primary-accent);
  }
  .toggle-btn[data-palette="secondary"].active {
    background: var(--secondary-dark);
    border-color: var(--secondary-muted);
    color: var(--secondary-muted);
  }
  .toggle-btn[data-palette="blue"].active {
    background: var(--blue-dark);
    border-color: var(--blue-border);
    color: var(--blue-accent);
  }
  .toggle-btn[data-palette="orange"].active {
    background: var(--orange-mid);
    border-color: var(--orange-accent);
    color: var(--orange-accent);
  }
</style>
