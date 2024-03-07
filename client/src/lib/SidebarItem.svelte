<script lang="ts">
  import {
    type ModelSummary,
    decimalMetrics,
    metricsHaveWarnings,
  } from './model';
  import * as d3 from 'd3';
  import { SidebarTableWidths } from './utils/sidebarwidths';
  import Checkbox from './utils/Checkbox.svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import type { SliceMetric } from './slices/utils/slice.type';
  import { faWarning } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher } from 'svelte';
  import { MetricColors } from './colors';
  import SliceMetricHistogram from './slices/metric_charts/SliceMetricHistogram.svelte';
  import SliceMetricCategoryBar from './slices/metric_charts/SliceMetricCategoryBar.svelte';
  import Tooltip from './utils/Tooltip.svelte';

  const dispatch = createEventDispatcher();

  export let model: ModelSummary;
  export let modelName: string;
  export let isActive: boolean;
  export let isChecked: boolean;
  export let metricToShow: string;
  export let customMetrics: { [key: string]: SliceMetric } | undefined =
    undefined;
  export let showCheckbox: boolean = true;
  export let allowCheck: boolean = true;
  export let checkDisabledReason: string | null = null;
  export let differences: string[] = [];

  export let metricScales: { [key: string]: (v: number) => number } = {};

  const accuracyFormat = d3.format('.1~%');
  const decimalFormat = d3.format(',.2~');
  const countFormat = d3.format(',');

  let metricValues: { [key: string]: SliceMetric | null } | undefined;
  $: if (!!customMetrics)
    metricValues = {
      Timesteps: {
        type: 'binary',
        mean: customMetrics['Timesteps']?.count ?? 0,
      },
      Trajectories: {
        type: 'binary',
        mean: customMetrics['Trajectories']?.count ?? 0,
      },
      [metricToShow]: customMetrics[metricToShow] ?? null,
      Labels: customMetrics['Labels'] ?? null,
      Predictions: customMetrics['Predictions'] ?? null,
    };
  else if (!!model && !!model.metrics)
    metricValues = {
      Timesteps: {
        type: 'binary',
        mean: model.metrics?.n_test.instances ?? 0,
      },
      Trajectories: {
        type: 'binary',
        mean: model.metrics?.n_test.trajectories ?? 0,
      },
      [metricToShow]:
        model.metrics?.performance[metricToShow] !== undefined
          ? {
              type: 'numeric',
              value: model.metrics?.performance[metricToShow],
            }
          : null,
      Labels: model.metrics?.labels ?? null,
      Predictions: model.metrics?.predictions ?? null,
    };
  else metricValues = undefined;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  on:click
  class="inline-flex slice-row items-center py-4 cursor-pointer {isActive
    ? 'bg-blue-100'
    : 'hover:bg-slate-100'} "
>
  {#if showCheckbox}
    <div
      class="grow-0 shrink-0"
      style="width: {SidebarTableWidths.Checkbox}px;"
    >
      {#if !allowCheck && !!checkDisabledReason}
        <Tooltip title={checkDisabledReason} position="right">
          <Checkbox disabled={isActive || !allowCheck} checked={isChecked} />
        </Tooltip>
      {:else}
        <Checkbox
          disabled={isActive || !allowCheck}
          checked={isChecked}
          on:change={(e) => dispatch('toggle')}
        />
      {/if}
    </div>
  {/if}
  <div
    class="font-mono p-2 grow-0 shrink-0 text-sm flex items-center"
    style="width: {SidebarTableWidths.ModelName}px;"
  >
    {#if model.training && !!model.status && model.status.state != 'error'}
      <div
        role="status"
        title={model.status.message}
        class="grow-0 shrink-0 flex items-center justify-center mr-2"
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
    {/if}
    <div class="flex-auto min-w-0">
      <div class="whitespace-nowrap truncate">
        {#if !!model && !!model.metrics && metricsHaveWarnings(model.metrics)}
          <Fa class="text-orange-300 inline" icon={faWarning} />
        {/if}
        {modelName}
      </div>
      {#if differences.length > 0}
        <div class="text-xs text-slate-500 font-sans">
          <strong>&Delta;:</strong>
          {differences.join(', ')}
        </div>
      {/if}
    </div>
  </div>
  {#if !!metricValues}
    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      <SliceMetricBar
        value={metricValues['Timesteps']?.mean ?? 0}
        scale={metricScales['Timesteps'] ?? ((v) => v)}
        color={MetricColors.Timesteps}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Timesteps']?.mean ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      <SliceMetricBar
        value={metricValues['Trajectories']?.mean ?? 0}
        scale={metricScales['Trajectories'] ?? ((v) => v)}
        color={MetricColors.Trajectories}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Trajectories']?.mean ?? 0)}</strong
          >
        </span>
      </SliceMetricBar>
    </div>
    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !metricValues[metricToShow] && !metricValues.Predictions}
        <Tooltip title="Not enough data"><span>&mdash;</span></Tooltip>
      {:else if !!metricValues[metricToShow]}
        {@const metric = metricValues[metricToShow]}
        <SliceMetricBar
          value={metric?.value ?? 0}
          scale={metricScales[metricToShow] ?? ((v) => v)}
          color={MetricColors.Accuracy}
          width={SidebarTableWidths.Metric - 20}
        >
          <span slot="caption">
            <strong
              >{decimalMetrics.includes(metricToShow)
                ? decimalFormat(metric?.value ?? 0)
                : accuracyFormat(metric?.value ?? 0)}</strong
            >
          </span>
        </SliceMetricBar>
      {:else}
        &mdash;
      {/if}
    </div>
    <div
      class="p-2 grow-0 shrink-0 whitespace-nowrap"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !!metricValues.Labels}
        {@const metric = metricValues.Labels}
        {#if metric.type == 'binary'}
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'numeric'}
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            mean={metric.mean}
            histValues={metric.hist}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
            width={SidebarTableWidths.Metric - 20}
          />
        {/if}
      {/if}
    </div>
    <div
      class="p-2 grow-0 shrink-0 whitespace-nowrap"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !!metricValues.Predictions}
        {@const metric = metricValues.Predictions}
        {#if metric.type == 'binary'}
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'numeric'}
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            mean={metric.mean}
            histValues={metric.hist}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
            width={SidebarTableWidths.Metric - 20}
          />
        {/if}
      {/if}
    </div>
  {:else}
    {#each ['Timesteps', 'Trajectories', metricToShow, 'Labels', 'Predictions'] as label}
      <div
        class="p-2 grow-0 shrink-0 text-slate-500"
        style="width: {SidebarTableWidths.Metric}px;"
      >
        &mdash;
      </div>
    {/each}
  {/if}
</div>

<style>
  .slice-row {
    min-width: 100%;
  }
  .slice-row > * {
    flex: 0 0 auto;
  }
</style>
