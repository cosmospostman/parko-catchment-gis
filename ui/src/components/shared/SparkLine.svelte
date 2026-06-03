<script lang="ts">
  import { scaleLinear, scaleTime } from 'd3-scale';
  import { line, area, curveMonotoneX } from 'd3-shape';

  interface DataPoint {
    date: string;
    value: number | null;
    p25: number | null;
    p75: number | null;
  }

  interface Props {
    data: DataPoint[];
    label: string;
    color: string;
    refBand?: [number, number];
    unit?: string;
  }

  let { data, label, color, refBand, unit = '' }: Props = $props();

  const H = 64;
  const PAD = { top: 6, right: 8, bottom: 20, left: 36 };
  const IH = H - PAD.top - PAD.bottom;

  let containerW = $state(0);
  const IW = $derived(Math.max(0, containerW - PAD.left - PAD.right));

  // Parse dates once — only non-null value points are plotted.
  const parsed = $derived(
    data
      .map(d => ({ ...d, t: new Date(d.date) }))
      .filter(d => d.value !== null)
  );

  // X domain is always full calendar year derived from the data's year.
  const xScale = $derived.by(() => {
    if (parsed.length < 1 || IW <= 0) return null;
    const year = parsed[0].t.getFullYear();
    return scaleTime()
      .domain([new Date(year, 0, 1), new Date(year, 11, 31)])
      .range([0, IW]);
  });

  const allValues = $derived(
    parsed.flatMap(d => [d.value!, d.p25 ?? d.value!, d.p75 ?? d.value!])
  );

  const yScale = $derived(
    parsed.length < 1
      ? null
      : (() => {
          const lo = Math.min(...allValues, ...(refBand ?? []));
          const hi = Math.max(...allValues, ...(refBand ?? []));
          const pad = (hi - lo) * 0.12 || 0.05;
          return scaleLinear().domain([lo - pad, hi + pad]).range([IH, 0]);
        })()
  );

  // SVG path for the mean line.
  const linePath = $derived.by(() => {
    if (!xScale || !yScale || parsed.length < 2) return null;
    const gen = line<typeof parsed[0]>()
      .defined(d => d.value !== null)
      .x(d => xScale(d.t))
      .y(d => yScale(d.value!))
      .curve(curveMonotoneX);
    return gen(parsed);
  });

  // SVG path for the p25–p75 band.
  const bandPath = $derived.by(() => {
    if (!xScale || !yScale) return null;
    const withBand = parsed.filter(d => d.p25 !== null && d.p75 !== null);
    if (withBand.length < 2) return null;
    const gen = area<typeof withBand[0]>()
      .defined(d => d.p25 !== null && d.p75 !== null)
      .x(d => xScale(d.t))
      .y0(d => yScale(d.p25!))
      .y1(d => yScale(d.p75!))
      .curve(curveMonotoneX);
    return gen(withBand);
  });

  // X-axis ticks: all 12 months of the year (fixed domain).
  const xTicks = $derived.by(() => {
    if (!xScale || parsed.length < 1) return [] as Date[];
    const year = parsed[0].t.getFullYear();
    return Array.from({ length: 12 }, (_, i) => new Date(year, i, 1));
  });

  // Y-axis ticks: 3 evenly spaced.
  const yTicks = $derived.by(() => {
    if (!yScale) return [] as number[];
    return yScale.ticks(3);
  });

  const monthAbbr = ['J','F','M','A','M','J','J','A','S','O','N','D'];

  // Band color: color at ~20% opacity.
  function bandFill(hex: string): string {
    // Convert #rrggbb to rgba.
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},0.18)`;
  }
</script>

<div class="sparkline-wrap" bind:clientWidth={containerW}>
  <div class="spark-label">{label}</div>
  {#if parsed.length < 1}
    <div class="spark-empty">no data</div>
  {:else}
    <svg width="100%" height={H} viewBox={`0 0 ${containerW} ${H}`} class="spark-svg">
      <g transform={`translate(${PAD.left},${PAD.top})`}>

        <!-- Band p25–p75 -->
        {#if bandPath}
          <path d={bandPath} fill={bandFill(color)} stroke="none" />
        {/if}

        <!-- Mean line -->
        {#if linePath}
          <path d={linePath} fill="none" stroke={color} stroke-width="1.5" stroke-linejoin="round" />
        {/if}

        {#if xScale && yScale}
          <!-- Parkinsonia reference band -->
          {#if refBand}
            {@const ry = yScale(refBand[1])}
            {@const rh = yScale(refBand[0]) - ry}
            <rect x={0} y={ry} width={IW} height={rh} fill="rgba(200,200,200,0.08)" stroke="rgba(200,200,200,0.18)" stroke-width="0.5" />
          {/if}

          <!-- Observation dots -->
          {#each parsed as pt}
            <circle
              cx={xScale(pt.t)}
              cy={yScale(pt.value!)}
              r="2.5"
              fill={color}
              stroke="#1a1a1a"
              stroke-width="0.8"
            />
          {/each}

          <!-- X axis -->
          <line x1={0} y1={IH} x2={IW} y2={IH} stroke="#333" stroke-width="0.5" />
          {#each xTicks as t}
            {@const tx = xScale(t)}
            <line x1={tx} y1={IH} x2={tx} y2={IH + 3} stroke="#444" stroke-width="0.5" />
            <text x={tx} y={IH + 10} text-anchor="middle" class="tick-label">
              {monthAbbr[t.getMonth()]}
            </text>
          {/each}

          <!-- Y axis ticks -->
          {#each yTicks as v}
            {@const ty = yScale(v)}
            <line x1={-3} y1={ty} x2={0} y2={ty} stroke="#444" stroke-width="0.5" />
            <text x={-5} y={ty + 3.5} text-anchor="end" class="tick-label">
              {v.toFixed(2)}
            </text>
          {/each}
        {/if}

      </g>
    </svg>
  {/if}
</div>

<style>
  .sparkline-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .spark-label {
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .spark-empty {
    font-size: 11px;
    color: #444;
    font-style: italic;
    height: 64px;
    display: flex;
    align-items: center;
  }

  .spark-svg {
    display: block;
    overflow: visible;
  }

  :global(.tick-label) {
    font-size: 8px;
    fill: #555;
    font-family: monospace;
  }
</style>
