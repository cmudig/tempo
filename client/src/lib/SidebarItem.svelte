<script lang="ts">
  import type { ModelMetrics, ModelSummary } from './model';
  import * as d3 from 'd3';
  import { SidebarTableWidths } from './utils/sidebarwidths';
  import Checkbox from './utils/Checkbox.svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import type { SliceMetric } from './slices/utils/slice.type';

  export let model: ModelSummary;
  export let modelName: string;
  export let isActive: boolean;
  export let metricToShow: string;
  export let customMetrics: { [key: string]: SliceMetric } | undefined =
    undefined;

  export let metricScales: { [key: string]: (v: number) => number } = {};

  const accuracyFormat = d3.format('.1~%');
  const countFormat = d3.format(',');

  let metricValues: { [key: string]: number } | undefined;
  $: if (!!customMetrics)
    metricValues = {
      Timesteps: customMetrics['Timesteps'].count ?? 0,
      Trajectories: customMetrics['Trajectories'].count ?? 0,
      [metricToShow]: customMetrics[metricToShow].mean ?? 0,
      'Positive Rate': customMetrics['Positive Rate'].mean ?? 0,
    };
  else if (!!model)
    metricValues = {
      Timesteps: model.metrics?.n_val.instances ?? 0,
      Trajectories: model.metrics?.n_val.trajectories ?? 0,
      [metricToShow]: model.metrics?.performance[metricToShow] ?? 0,
      'Positive Rate': model.metrics?.positive_rate ?? 0,
    };
  else metricValues = undefined;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  on:click
  class="inline-flex items-center gap-2 px-4 py-4 cursor-pointer {isActive
    ? 'bg-blue-100'
    : 'hover:bg-slate-100'} "
>
  <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Checkbox}px;">
    {#if model.training && !!model.status}
      <div
        role="status"
        title={model.status.message}
        class="w-full h-full flex items-center justify-center"
      >
        <svg
          aria-hidden="true"
          class="text-gray-200 animate-spin stroke-blue-600 w-4 h-4 align-middle"
          viewBox="-0.5 -0.5 99.5 99.5"
          xmlns="http://www.w3.org/2000/svg"
        >
          <ellipse
            cx="50"
            cy="50"
            rx="45"
            ry="45"
            fill="none"
            stroke="currentColor"
            stroke-width="10"
          />
          <path
            d="M 50 5 A 45 45 0 0 1 95 50"
            stroke-width="10"
            stroke-linecap="round"
            fill="none"
          />
        </svg>
      </div>
    {:else}
      <Checkbox />
    {/if}
  </div>
  <div
    class="font-mono grow-0 shrink-0"
    style="width: {SidebarTableWidths.ModelName}px;"
  >
    {modelName}
  </div>
  {#if !!metricValues}
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      <SliceMetricBar
        value={metricValues['Timesteps'] ?? 0}
        scale={metricScales['Timesteps'] ?? ((v) => v)}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Timesteps'] ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      <SliceMetricBar
        value={metricValues['Trajectories'] ?? 0}
        scale={metricScales['Trajectories'] ?? ((v) => v)}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Trajectories'] ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      <SliceMetricBar
        value={metricValues[metricToShow] ?? 0}
        scale={metricScales[metricToShow] ?? ((v) => v)}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{accuracyFormat(metricValues[metricToShow] ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      <SliceMetricBar
        value={metricValues['Positive Rate'] ?? 0}
        scale={metricScales['Positive Rate'] ?? ((v) => v)}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{accuracyFormat(metricValues['Positive Rate'] ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
  {/if}
</div>
