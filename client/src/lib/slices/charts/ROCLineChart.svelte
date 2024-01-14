<svelte:options accessors />

<script lang="ts">
  import { LayerCake, Svg } from 'layercake';

  import Line from './Line.svelte';
  import AxisX from './AxisX.svelte';
  import AxisY from './AxisY.svelte';
  import * as d3 from 'd3';
  import VLine from './VLine.svelte';

  type Datum = { x: number; y: number; threshold: number };
  export let roc: {
    tpr: number[];
    fpr: number[];
    thresholds: number[];
  } | null = null;
  export let selectedThreshold: number | null;
  export let hoveredThreshold: number | null;

  let chartData: Datum[] = [];

  $: if (!!roc) {
    chartData = roc.tpr.map((_, i) => ({
      x: roc!.fpr[i],
      y: roc!.tpr[i],
      threshold: roc!.thresholds[i],
    }));
  } else chartData = [];

  let thresholdFormat = d3.format('.3~');
</script>

{#if chartData.length > 0}
  <div class="chart-container">
    <LayerCake
      padding={{ top: 20, bottom: 60, left: 20 }}
      x="x"
      y="y"
      yNice={4}
      yDomain={[0, 1]}
      xDomain={[0, 1]}
      data={chartData}
      custom={{ hoveredGet: (d) => d.threshold == hoveredThreshold }}
    >
      <Svg>
        <AxisX ticks={4} baseline label="FPR" />
        <AxisY ticks={4} textAnchor="end" label="TPR" />
        <Line
          stroke="steelblue"
          allowHover
          allowSelect
          on:hover={(e) => (hoveredThreshold = e.detail?.threshold)}
          on:click={(e) => (selectedThreshold = e.detail?.threshold)}
        />
        {#if hoveredThreshold !== null}
          {@const hoveredDatum = chartData.find(
            (d) => Math.abs(d.threshold - hoveredThreshold) < 0.001
          )}
          {#if !!hoveredDatum}
            <VLine
              xValue={hoveredDatum.x}
              title={thresholdFormat(hoveredDatum.threshold)}
              color="#94a3b8"
            />
          {/if}
        {/if}
        {#if selectedThreshold !== null}
          {@const selectedDatum = chartData.find(
            (d) => Math.abs(d.threshold - selectedThreshold) < 0.001
          )}
          {#if !!selectedDatum}
            <VLine
              xValue={selectedDatum.x}
              title={thresholdFormat(selectedDatum.threshold)}
              color="#475569"
            />
          {/if}
        {/if}
      </Svg>
    </LayerCake>
  </div>
{:else}
  <div class="flex w-100 h-100 justify-center items-center">
    <p class="f5 tc dark-gray">No data to display.</p>
  </div>
{/if}

<style>
  /*
    The wrapper div needs to have an explicit width and height in CSS.
    It can also be a flexbox child or CSS grid element.
    The point being it needs dimensions since the <LayerCake> element will
    expand to fill it.
  */
  .chart-container {
    width: 100%;
    height: 100%;
  }
</style>
