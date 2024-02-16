<script lang="ts">
  import { format } from 'd3-format';
  import { schemeTableau10 } from 'd3-scale-chromatic';
  import { LayerCake, Svg, Html } from 'layercake';
  import { scaleOrdinal, scaleLinear } from 'd3-scale';
  import { range } from 'd3-array';

  import BarTooltip from './BarTooltip.svelte';
  import BarSegment from './BarSegment.svelte';

  export let width: number | string = 100;

  export let counts: { [key: string]: number } | null = null;

  export let order: Array<string> = [];

  interface Datum {
    name: string;
    start: number;
    end: number;
    index: number;
    count: number;
    share: number;
  }
  let data: Array<Datum> = [];

  $: if (!!counts && order.length > 0) {
    let totalCount = Object.values(counts).reduce((curr, val) => curr + val, 0);
    let runningCount = 0;
    data = order.map((d, i) => {
      let curr = runningCount;
      runningCount += counts![d] || 0;
      return {
        start: curr / totalCount,
        end: runningCount / totalCount,
        index: i,
        name: d,
        count: counts![d] || 0,
        share: counts![d] / totalCount,
      };
    });
  } else {
    data = [];
  }

  let hoveredIndex: number;

  let countFormat = format(',');
  let percentFormat = format('.1~%');

  function makeTooltipText(d: Datum) {
    return `<strong>${percentFormat(d.share)}</strong> ${d.name}`;
  }

  let mostCommonDatum: Datum | null = null;
  $: if (data.length > 0)
    mostCommonDatum = data.reduce(
      (prev, curr) => (prev.count > curr.count ? prev : curr),
      data[0]
    );
  else mostCommonDatum = null;
</script>

{#if !!counts}
  <div
    style="width: {`${width}`.includes('%')
      ? width
      : `${width}px`}; height: 6px;"
    class="relative rounded overflow-hidden mb-1"
  >
    <LayerCake
      padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
      x="start"
      y="index"
      z="end"
      xScale={scaleLinear()}
      xDomain={[0, 1]}
      yScale={scaleOrdinal()}
      yDomain={range(counts.length)}
      yRange={schemeTableau10}
      {data}
      custom={{
        hoveredGet: (d) => d.index == hoveredIndex,
      }}
    >
      <Html>
        <BarSegment
          on:hover={(e) => (hoveredIndex = e.detail ? e.detail.index : null)}
        />
      </Html>
    </LayerCake>
  </div>
  <div
    class="text-xs text-slate-800 dark:text-slate-100"
    style="width: {`${width}`.includes('%') ? width : `${width}px`};"
  >
    {#if $$slots.caption}
      <slot name="caption" />
    {:else if hoveredIndex != null}
      {@html makeTooltipText(data[hoveredIndex])}
    {:else if !!mostCommonDatum}
      {@html makeTooltipText(mostCommonDatum)}
    {/if}
  </div>
{/if}
