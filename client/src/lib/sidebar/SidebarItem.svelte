<script lang="ts">
  import {
    type ModelSummary,
    decimalMetrics,
    metricsHaveWarnings,
  } from '../model';
  import * as d3 from 'd3';
  import { SidebarTableWidths } from '../utils/sidebarwidths';
  import Checkbox from '../utils/Checkbox.svelte';
  import SliceMetricBar from '../slices/metric_charts/SliceMetricBar.svelte';
  import type { SliceMetric } from '../slices/utils/slice.type';
  import { faWarning } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher } from 'svelte';
  import { MetricColors, makeCategoricalColorScale } from '../colors';
  import SliceMetricHistogram from '../slices/metric_charts/SliceMetricHistogram.svelte';
  import SliceMetricCategoryBar from '../slices/metric_charts/SliceMetricCategoryBar.svelte';
  import Tooltip from '../utils/Tooltip.svelte';

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
  class="w-full inline-flex slice-row items-center justify-center flex-wrap py-4 cursor-pointer {isActive
    ? 'bg-blue-100'
    : 'hover:bg-slate-100'} "
>
  <div
    class="font-mono p-2 text-sm flex grow shrink items-center gap-2"
    style="flex-basis: 0; min-width: 240px;"
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
      class="p-2 whitespace-nowrap grow-0 shrink-0 grid auto-rows-max text-xs gap-x-2 gap-y-0 items-center"
      style="width: 40%; min-width: 300px; max-width: {SidebarTableWidths.AllMetrics}px; grid-template-columns: max-content auto 96px;"
    >
      <div class="font-bold text-right">Timesteps</div>
      <SliceMetricBar
        value={metricValues['Timesteps']?.mean ?? 0}
        scale={metricScales['Timesteps'] ?? ((v) => v)}
        showFullBar
        horizontalLayout
        showTooltip={false}
        color={MetricColors.Timesteps}
        width={null}
      />
      <div>
        <strong>{countFormat(metricValues['Timesteps']?.mean ?? 0)}</strong>
      </div>

      <div class="font-bold text-right">Trajectories</div>
      <SliceMetricBar
        value={metricValues['Trajectories']?.mean ?? 0}
        scale={metricScales['Trajectories'] ?? ((v) => v)}
        horizontalLayout
        showTooltip={false}
        showFullBar
        color={MetricColors.Trajectories}
        width={null}
      />
      <div>
        <strong>{countFormat(metricValues['Trajectories']?.mean ?? 0)}</strong>
      </div>

      {#if !metricValues[metricToShow] && !metricValues.Predictions}
        <div class="col-span-full">
          <Tooltip title="Not enough data"><span>&mdash;</span></Tooltip>
        </div>
      {:else if !!metricValues[metricToShow]}
        {@const metric = metricValues[metricToShow]}
        <div class="font-bold text-right">{metricToShow}</div>
        <SliceMetricBar
          value={metric?.value ?? 0}
          scale={metricScales[metricToShow] ?? ((v) => v)}
          color={MetricColors.Accuracy}
          width={null}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong
            >{decimalMetrics.includes(metricToShow)
              ? decimalFormat(metric?.value ?? 0)
              : accuracyFormat(metric?.value ?? 0)}</strong
          >
        </div>
      {:else}
        <div class="col-span-full">&mdash;</div>
      {/if}

      {#if !!metricValues.Labels}
        {@const metric = metricValues.Labels}
        {#if metric.type == 'binary'}
          <div class="font-bold text-right">Labels</div>
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Labels}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
          </div>
        {:else if metric.type == 'numeric'}
          <div class="font-bold text-right">Labels</div>
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Labels}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
          </div>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            noParent
            title={'Labels'}
            width={null}
            horizontalLayout
            mean={metric.mean}
            color={MetricColors.Labels}
            histValues={metric.hist ?? {}}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            noParent
            width={null}
            title={'Labels'}
            horizontalLayout
            colorScale={makeCategoricalColorScale(MetricColors.Labels)}
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
          />
        {/if}
      {/if}

      {#if !!metricValues.Predictions}
        {@const metric = metricValues.Predictions}
        {#if metric.type == 'binary'}
          <div class="font-bold text-right">Predictions</div>
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Predictions}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
          </div>
        {:else if metric.type == 'numeric'}
          <div class="font-bold text-right">Predictions</div>
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Predictions}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
          </div>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            noParent
            title={'Predictions'}
            width={null}
            horizontalLayout
            mean={metric.mean}
            color={MetricColors.Predictions}
            histValues={metric.hist ?? {}}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            noParent
            width={null}
            title={'Predictions'}
            horizontalLayout
            colorScale={makeCategoricalColorScale(MetricColors.Predictions)}
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
          />
        {/if}
      {/if}

      <!-- {#each metricNames as name, i}
    {@const metric = sliceForScores.metrics[name]}

    {#if !!metricInfo[name] && metricInfo[name].visible}
      {#if metric.type == 'binary'}
        <div class="font-bold text-right">{name}</div>
        <SliceMetricBar
          value={metric.mean}
          color={ColorWheel[i]}
          width={null}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong>{format('.1%')(metric.mean)}</strong>
        </div>
      {:else if metric.type == 'count'}
        <div class="font-bold text-right">{name}</div>
        <SliceMetricBar
          value={metric.share}
          width={null}
          color={ColorWheel[i]}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong>{format(',')(metric.count)}</strong>
          <span style="font-size: 0.7rem;" class="italic text-gray-700"
            >({format('.1%')(metric.share)})</span
          >
        </div>
      {:else if metric.type == 'continuous'}
        <SliceMetricHistogram
          noParent
          title={name}
          width={null}
          horizontalLayout
          mean={metric.mean}
          color={ColorWheel[i]}
          histValues={metric.hist}
        />
      {:else if metric.type == 'categorical'}
        <SliceMetricCategoryBar
          noParent
          width={null}
          title={name}
          horizontalLayout
          colorScale={makeCategoricalColorScale(ColorWheel[i])}
          order={metricInfo[name].order}
          counts={metric.counts}
        />
      {/if}
    {/if}
  {/each}
</div>

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
      {/if}-->
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
</style>
