<script lang="ts">
  import {
    decimalMetrics,
    ModelTypeStrings,
    type ModelMetrics,
    type ModelSummary,
  } from './model';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import { checkTrainingStatus } from './training';
  import { faWarning, faXmarkCircle } from '@fortawesome/free-solid-svg-icons';
  import RocLineChart from './slices/charts/ROCLineChart.svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import { MetricColors } from './colors';
  import Histogram2D from './slices/charts/Histogram2D.svelte';
  import Tooltip from './utils/Tooltip.svelte';

  const dispatch = createEventDispatcher();

  export let currentDataset: string | null = null;
  export let modelName: string | null = null;
  export let modelSummary: ModelSummary | null = null;
  let metrics: ModelMetrics | null = null;

  let selectedThreshold: number | null = null;

  $: if (!!modelName) {
    loadModelResults();
  }

  async function loadModelResults() {
    if (!modelName) return;
    try {
      let result = await fetch(
        `/datasets/${currentDataset}/models/${modelName}/metrics`
      );
      metrics = await result.json();
      selectedThreshold = null;
      console.log(metrics);
    } catch (e) {
      console.error('error loading model metrics:', e);
    }
  }

  const percentageFormat = d3.format('.1~%');
  const decimalFormat = d3.format(',.3~');
  const thresholdFormat = d3.format('.3~');
  const nFormat = d3.format(',');

  let performanceMetrics: { [key: string]: number } | undefined;
  $: if (
    selectedThreshold != metrics?.threshold &&
    !!metrics &&
    !!metrics.roc
  ) {
    let idx = metrics.roc.thresholds.findIndex((t) => t === selectedThreshold);
    if (idx !== undefined)
      performanceMetrics = {
        ...(metrics.performance ?? {}),
        ...metrics?.roc.performance[idx],
      };
  } else {
    performanceMetrics = metrics?.performance;
  }

  $: if (!!metrics && selectedThreshold == null) {
    selectedThreshold = metrics.threshold ?? null;
  }
</script>

<div class="w-full pt-4 px-4 flex flex-col">
  {#if !!metrics}
    <div class="text-lg font-bold mb-3 w-full flex items-center gap-2">
      <div class="font-mono">{modelName}</div>
      {#if !!modelSummary && !!modelSummary.model_type}
        <div class="rounded text-xs font-normal bg-slate-200 px-2 py-1">
          {ModelTypeStrings[modelSummary.model_type]}
        </div>
      {/if}
    </div>
    {#if !!metrics.trivial_solution_warning}
      <div class="mb-2 p-4 bg-orange-100 rounded-lg">
        <h4 class="font-bold text-orange-700/80 mb-2">
          <Fa class="inline text-orange-300" icon={faWarning} /> Model may be approximated
          with fewer variables
        </h4>
        <div class="text-gray-800 text-sm mb-2">
          The target variable can be predicted with {metrics
            .trivial_solution_warning.metric} of {decimalMetrics.includes(
            metrics.trivial_solution_warning.metric
          )
            ? decimalFormat(metrics.trivial_solution_warning.metric_value)
            : percentageFormat(metrics.trivial_solution_warning.metric_value)} using
          only the input variables:
        </div>
        <ul>
          {#each metrics.trivial_solution_warning.variables as varName}
            <li class="font-mono">{varName}</li>
          {/each}
        </ul>
      </div>
    {/if}
    {#if !!metrics.class_not_predicted_warnings}
      {#each metrics.class_not_predicted_warnings as warning}
        {@const className = !!modelSummary?.output_values
          ? modelSummary?.output_values[warning.class]
          : warning.class}
        <div class="mb-2 p-4 bg-orange-100 rounded-lg">
          <h4 class="font-bold text-orange-700/80 mb-2">
            <Fa class="inline text-orange-300" icon={faWarning} /> Class {className}
            rarely predicted
          </h4>
          <div class="text-gray-800 text-sm">
            The class {className} is only correctly predicted in {percentageFormat(
              warning.true_positive_fraction
            )} of timesteps that have a true label of {className}.
          </div>
        </div>
      {/each}
    {/if}
    <div class="mb-2 rounded bg-slate-100 p-4 flex w-full gap-8">
      <div class="flex-auto">
        {#if metrics.threshold !== undefined || selectedThreshold !== null}
          <div class="font-bold text-slate-600 text-sm mb-4">
            {#if Math.abs((selectedThreshold ?? 1e9) - (metrics?.threshold ?? 1e9)) > 0.001}Selected
              prediction threshold{:else}Optimal prediction threshold{/if}:
            <span class="font-mono font-normal text-normal"
              >{thresholdFormat(
                selectedThreshold ?? metrics.threshold ?? 0
              )}</span
            >
            {#if selectedThreshold != metrics.threshold}
              <button
                class="hover:opacity-50 ml-3"
                on:click={() =>
                  (selectedThreshold = metrics?.threshold ?? null)}
                ><Fa class="inline" icon={faXmarkCircle} /></button
              >
            {/if}
          </div>
        {/if}
        {#if !!performanceMetrics}
          <div class="flex flex-wrap gap-4 mb-4">
            {#each Object.entries(performanceMetrics) as [metricName, value] (metricName)}
              <div class="w-32">
                <div class="font-bold text-slate-600 text-sm mb-2">
                  {metricName}
                </div>
                <SliceMetricBar
                  width={96}
                  {value}
                  color={MetricColors.Accuracy}
                  showFullBar
                >
                  <span slot="caption" class="font-mono text-base">
                    {decimalMetrics.includes(metricName)
                      ? decimalFormat(value)
                      : percentageFormat(value)}
                  </span>
                </SliceMetricBar>
              </div>
            {/each}
          </div>
        {/if}
        <div class="flex flex-wrap gap-8 mb-4">
          <div class="w-64">
            <div class="font-bold text-slate-600 text-sm mb-2">
              Training <Tooltip
                position="right"
                title="Trajectories used as input during model training only."
              />
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_train.instances)}
              <span class="text-xs font-sans">instances</span>
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_train.trajectories)}
              <span class="text-xs font-sans">trajectories</span>
            </div>
          </div>
          <div class="w-64">
            <div class="font-bold text-slate-600 text-sm mb-2">
              Validation <Tooltip
                position="right"
                title="Trajectories used to determine when to stop training the model, and to discover slices."
              />
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_val.instances)}
              <span class="text-xs font-sans">instances</span>
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_val.trajectories)}
              <span class="text-xs font-sans">trajectories</span>
            </div>
          </div>
          <div class="w-64">
            <div class="font-bold text-slate-600 text-sm mb-2">
              Testing <Tooltip
                position="right"
                title="Trajectories used to score and rank slices and calculate metrics."
              />
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_test.instances)}
              <span class="text-xs font-sans">instances</span>
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_test.trajectories)}
              <span class="text-xs font-sans">trajectories</span>
            </div>
          </div>
        </div>
      </div>
      <div
        class="aspect-square {!!metrics.roc ? 'h-64' : 'h-72'} shrink-0 grow-0"
        style="min-width: 200px; min-height: 300px;"
      >
        {#if !!metrics.roc}
          <RocLineChart roc={metrics.roc} bind:selectedThreshold />
        {:else if !!metrics.confusion_matrix}
          <Histogram2D
            invertY
            colorMap={d3.interpolateBlues}
            data={{
              values: metrics.confusion_matrix,
              bins:
                modelSummary?.output_values ??
                d3.range(metrics.confusion_matrix.length),
            }}
          />
        {:else if !!metrics.hist}
          <Histogram2D colorMap={d3.interpolateBlues} data={metrics.hist} />
        {/if}
      </div>
    </div>
  {:else}
    <div class="w-full h-full flex flex-column items-center justify-center">
      <div class="text-slate-500">No metrics available for this model yet.</div>
    </div>
  {/if}
</div>
