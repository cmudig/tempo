<script lang="ts">
  import { Html, LayerCake, Svg } from 'layercake';
  import { scaleLinear } from 'd3-scale';
  import * as d3 from 'd3';
  import AxisX from './AxisX.svelte';
  import AxisY from './AxisY.svelte';
  import Rect from './Rect.svelte';
  import { interpolateRdBu } from 'd3-scale-chromatic';
  import Colorbar from './Colorbar.svelte';
  import Tooltip from './Tooltip.svelte';
  import { createEventDispatcher } from 'svelte';
  import { areObjectsEqual } from '../utils/utils';

  const dispatch = createEventDispatcher();

  type Datum = { binX: any; binY: any; value: number };

  export let data: { values: number[][]; bins: any[] };
  // function that takes a data point and returns a string describing it
  export let formatTooltip: ((datum: Datum) => string) | null = null;

  export let valueDomain: [number, number] | null = null; // range of values should be mapped to the ends of the color map
  export let colorMap: (v: number) => string = interpolateRdBu;
  export let nullColor = 'black';
  export let invertY = false;

  let hoveredDatum: Datum | null = null;

  let visData: (Datum & { x: number; y: number })[] = [];
  let numBins: number = 0;

  $: if (!!data) {
    visData = data.values
      .map((row, i) =>
        row.map((d, j) => ({
          binX: j,
          binY: i,
          value: d,
          x: j - 0.5,
          y: i - 0.5,
        }))
      )
      .flat();
    numBins = data.values.length;
    valueDomain = [
      visData.reduce((prev, curr) => Math.min(prev, curr.value), 1e10),
      visData.reduce((prev, curr) => Math.max(prev, curr.value), -1e10),
    ];
  }

  const tickFormat = d3.format(',.3~s');

  function formatTick(b: number) {
    return typeof data.bins[b] === 'number'
      ? tickFormat(data.bins[b])
      : data.bins[b];
  }

  function formatTickY(b: number) {
    b = invertY ? b + 1 : b;
    return typeof data.bins[b] === 'number'
      ? tickFormat(data.bins[b])
      : data.bins[b];
  }
</script>

<div class="w-full h-full flex mb-3">
  {#if !!visData && !!valueDomain}
    <div class="heatmap-plot h-full flex-auto ml-4">
      <LayerCake
        padding={{ top: 10, right: 10, bottom: 90, left: 35 }}
        x="x"
        y="y"
        z="value"
        yReverse={!invertY}
        extents={{
          x: [-0.5, numBins - 0.5],
          y: invertY ? [-1.5, numBins - 1.5] : [-0.5, numBins - 0.5],
        }}
        zScale={d3.scaleSymlog()}
        zDomain={valueDomain}
        zRange={[0, 1]}
        data={visData}
        custom={{
          hoveredGet: (d) => areObjectsEqual(d, hoveredDatum),
        }}
      >
        <Svg>
          <AxisX
            angle
            gridlines={false}
            ticks={d3.range(numBins)}
            {formatTick}
            label="Predicted"
            labelLeft
          />
          <AxisY
            gridlines={false}
            dyTick={4}
            ticks={invertY ? d3.range(-1, numBins - 1) : d3.range(numBins)}
            formatTick={formatTickY}
            label="True"
            textAnchor="end"
          />
          <Rect
            {colorMap}
            {nullColor}
            on:hover={(e) => (hoveredDatum = e.detail ? e.detail : null)}
          />
        </Svg>
        <Html pointerEvents={false}>
          <Tooltip
            formatText={formatTooltip ??
              ((d) =>
                `True ${formatTick(d.binY)}, predicted ${formatTick(d.binX)}: ${
                  d.value
                } instances`)}
            dx={-0.5}
            horizontalAlign="middle"
          />
        </Html>
      </LayerCake>
    </div>
    <div class="legend-container mr-4 h-full overflow-visible">
      <Colorbar
        valueScale={d3.scaleSymlog()}
        logTicks
        width={80}
        {colorMap}
        {valueDomain}
        numTicks={4}
        margin={{ top: 10, bottom: 90 }}
      />
    </div>
  {/if}
</div>

<style>
  .heatmap-plot {
    z-index: 10;
  }

  .legend-container {
    width: 60px;
  }
</style>
