<script lang="ts">
  import {
    decimalMetrics,
    ModelArchitectureType,
    ModelTypeStrings,
    type ModelMetrics,
    type ModelSummary,
  } from '../model';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, getContext, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import ModelTrainingView from '../ModelTrainingView.svelte';
  import { checkTrainingStatus } from '../training';
  import { faWarning, faXmarkCircle } from '@fortawesome/free-solid-svg-icons';
  import RocLineChart from '../slices/charts/ROCLineChart.svelte';
  import SliceMetricBar from '../slices/metric_charts/SliceMetricBar.svelte';
  import { MetricColors } from '../colors';
  import Histogram2D from '../slices/charts/Histogram2D.svelte';
  import Tooltip from '../utils/Tooltip.svelte';
  import type { Writable } from 'svelte/store';
  import FeatureImportanceChart from './FeatureImportanceChart.svelte';

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  const dispatch = createEventDispatcher();

  export let modelName: string | null = null;
  export let modelSummary: ModelSummary | null = null;
  let metrics: ModelMetrics | null = null;
  let isLoadingMetrics: boolean = false;

  enum MetricsTab {
    performance = 'Performance',
    featureImportance = 'Feature Importance',
    hyperparameters = 'Hyperparameters',
  }

  export let currentView: MetricsTab = MetricsTab.performance;

  let selectedThreshold: number | null = null;

  $: if (!!modelName) {
    loadModelResults();
  }

  async function loadModelResults() {
    if (!modelName) return;
    try {
      isLoadingMetrics = true;
      let result = await fetch(
        import.meta.env.BASE_URL +
          `/datasets/${$currentDataset}/models/${modelName}/metrics`
      );
      metrics = await result.json();
      selectedThreshold = null;
      console.log(metrics);
    } catch (e) {
      console.error('error loading model metrics:', e);
    }
    isLoadingMetrics = false;
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

{#if isLoadingMetrics}
  <div class="w-full h-full flex flex-col items-center justify-center">
    <div class="text-center mb-4">Loading metrics...</div>
    <div role="status">
      <svg
        aria-hidden="true"
        class="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600"
        viewBox="0 0 100 101"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
          fill="currentColor"
        />
        <path
          d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
          fill="currentFill"
        />
      </svg>
    </div>
  </div>
{:else}
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
              : percentageFormat(metrics.trivial_solution_warning.metric_value)}
            using only the input variables:
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
      <div class="mb-2 rounded bg-slate-100 p-4">
        <div class="w-full mb-4 flex gap-3">
          {#each [MetricsTab.performance, MetricsTab.featureImportance, MetricsTab.hyperparameters] as view}
            <button
              class="rounded text-xs py-1 text-center w-48 {currentView == view
                ? 'bg-slate-500 text-white font-bold hover:bg-slate-400'
                : 'text-slate-700 hover:bg-slate-200'}"
              on:click={() => (currentView = view)}>{view}</button
            >
          {/each}
        </div>
        {#if currentView == MetricsTab.performance}
          <div class="flex w-full gap-8">
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
              class="aspect-square {!!metrics.roc
                ? 'h-64'
                : 'h-72'} shrink-0 grow-0"
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
                <Histogram2D
                  colorMap={d3.interpolateBlues}
                  data={metrics.hist}
                />
              {/if}
            </div>
          </div>
        {:else if currentView == MetricsTab.featureImportance}
          {#if !!metrics.feature_importances}
            <FeatureImportanceChart importances={metrics.feature_importances} />
          {:else}
            <div
              class="w-full flex justify-center items-center h-64 text-slate-500"
            >
              No feature importances available.
            </div>
          {/if}
        {:else if currentView == MetricsTab.hyperparameters}
          {#if !!metrics.model_architecture}
            <div
              style="max-width: 600px;"
              class="w-full grid grid-cols-2 gap-2 items-baseline"
            >
              <div class="text-right font-bold text-sm text-slate-600">
                Architecture
              </div>
              <div class="text-left font-mono ml-2">
                {ModelArchitectureType[metrics.model_architecture.type]}
              </div>
              <div class="text-right font-bold text-sm text-slate-600">
                Number of Training Runs
              </div>
              <div class="text-left font-mono ml-2">
                {metrics.model_architecture.num_samples}
              </div>
              {#each Object.entries(metrics.model_architecture.hyperparameters) as [hyperparamName, hyperparamValue] (hyperparamName)}
                <div class="text-right font-bold text-sm text-slate-600">
                  {hyperparamName}
                </div>
                <div class="text-left font-mono ml-2">{hyperparamValue}</div>
              {/each}
            </div>
          {:else}
            <div
              class="w-full flex justify-center items-center h-64 text-slate-500"
            >
              No feature importances available.
            </div>
          {/if}
        {/if}
      </div>
    {:else}
      <div class="w-full h-full flex flex-column items-center justify-center">
        <div class="text-slate-500">
          No metrics available for this model yet.
        </div>
      </div>
    {/if}
  </div>
{/if}
