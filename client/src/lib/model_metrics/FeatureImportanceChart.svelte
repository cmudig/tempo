<script lang="ts">
  import { Html, LayerCake, Svg } from 'layercake';
  import Bar from './Bar.svelte';
  import { scaleBand, scaleOrdinal, scaleLinear } from 'd3-scale';
  import { schemeTableau10 } from 'd3-scale-chromatic';
  import BarTooltip from './BarTooltip.svelte';
  import * as d3 from 'd3';
  import { createEventDispatcher } from 'svelte';
  import AxisX from '../slices/charts/AxisX.svelte';
  import AxisY from '../slices/charts/AxisY.svelte';

  const dispatch = createEventDispatcher();

  export let importances: { feature: string; mean: number; std: number }[] = [];
  export let numKeys = 10;
  export let xAxisTicks = 11;
  export let xAxisLimit = null;
  export let featureOrder: string[] | null = null;

  let featureNames: string[] = [];
  $: if (!!importances) {
    let keys;
    if (featureOrder != null) keys = featureOrder;
    else
      keys = importances
        .sort(
          // (a, b) => Math.abs(importances[b]) - Math.abs(importances[a])
          (a, b) => Math.abs(b.mean) - Math.abs(a.mean)
        )
        .map((x) => x.feature);
    featureNames = keys.slice(0, Math.min(numKeys, keys.length));
    // featureNames = [
    //   ...keys.slice(0, numKeys / 2),
    //   ...keys.slice(keys.length - numKeys / 2),
    // ];
  }

  let barData: any[] | null = null;
  $: if (featureNames.length > 0) {
    barData = featureNames.map((feature, i) => {
      let importance = importances.find((v) => v.feature == feature)!;
      return {
        i,
        name: feature,
        feature,
        absImportance: Math.abs(importance.mean),
        importance: importance.mean,
        std: importance.std,
      };
    });
  } else barData = null;

  let maxImportance = 0;
  $: if (xAxisLimit != null) maxImportance = xAxisLimit;
  else if (!!barData) {
    maxImportance = barData.reduce(
      (curr, p) => Math.max(curr, p.absImportance + 0),
      0
    );
    maxImportance = Math.ceil(maxImportance / 0.01) * 0.01;
  }

  let hoveredFeature: string | null = null;
  const weightFormat = d3.format('.2~');
  function makeTooltipText(d) {
    // On bar hover, show information about how much the value deviates from
    // the average
    return `<strong>${d.feature}</strong> has an average feature importance weight of ${weightFormat(d.absImportance)} &plusmn; ${weightFormat(d.std)}`;
  }
</script>

<div class="w-full">
  {#if !!barData}
    <div class="chart-container" style="height: {36 * barData.length + 48}px;">
      <LayerCake
        padding={{ top: 0, bottom: 0, left: 120 }}
        x="importance"
        y="name"
        yScale={scaleBand().paddingInner([0.05]).round(true)}
        xDomain={[0, maxImportance]}
        yDomain={featureNames}
        data={barData}
        custom={{
          hoveredGet: (d) => d.feature == hoveredFeature,
          stdGet: (d) => d.std,
        }}
      >
        <Svg>
          <AxisX
            gridlines={true}
            baseline={true}
            snapTicks={false}
            ticks={xAxisTicks}
            label="Feature Importance"
            formatTick={d3.format('.2')}
          />
          <AxisY label="" gridlines={false} />
          <Bar
            fill="#2563eb"
            on:hover={(e) =>
              (hoveredFeature = e.detail != null ? e.detail.feature : null)}
          />
          <!-- <CategoricalLegend scale={colorScale} inset={{ x: 20, y: 8 }} /> -->
          <!-- interpolateRdBu((d.importance + maxImportance) / (2 * maxImportance)) -->
        </Svg>
        <Html pointerEvents={false}>
          <BarTooltip formatText={makeTooltipText} />
        </Html>
      </LayerCake>
    </div>
  {/if}
  <div class="flex items-center w-full gap-8 mt-2">
    <div class="flex-auto text-slate-700 text-sm">
      Feature Importances show the average importance of each input variable in
      the model, computed using Shapley Additive Explanations (SHAP).
    </div>
    <div class="shrink-0">
      Show <select class="flat-select mx-1" bind:value={numKeys}>
        <option value={10}>10</option>
        <option value={20}>20</option>
        <option value={50}>50</option>
        <option value={1e9}>all</option>
      </select> variables
    </div>
  </div>
</div>

<style>
  .chart-container {
    box-sizing: border-box;
    width: calc(100% - 120px);
    margin-left: 96px;
    margin-right: 24px;
    margin-bottom: 48px;
    padding-top: 18px;
    padding-left: 24px;
  }
</style>
