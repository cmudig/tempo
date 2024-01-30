<script lang="ts">
  import {
    AllCategories,
    VariableCategory,
    type ModelMetrics,
    type VariableDefinition,
  } from './model';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import { checkTrainingStatus } from './training';
  import { faWarning, faXmarkCircle } from '@fortawesome/free-solid-svg-icons';
  import RocLineChart from './slices/charts/ROCLineChart.svelte';
  import TableCellBar from './slices/metric_charts/TableCellBar.svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import ModelDataSummaryElement from './ModelDataSummaryElement.svelte';

  const dispatch = createEventDispatcher();

  export let modelName = 'vasopressor_8h';
  let metrics: ModelMetrics | null = null;

  let selectedThreshold: number | null = null;

  let isTraining: boolean = false;

  $: if (!!modelName) {
    loadModelResults();
  }

  async function loadModelResults() {
    try {
      let trainingStatus = await checkTrainingStatus(modelName);
      if (!!trainingStatus && trainingStatus.state != 'error') {
        isTraining = true;
        return;
      }
      isTraining = false;
      let result = await fetch(`/models/${modelName}/metrics`);
      metrics = await result.json();
      selectedThreshold = null;
      console.log(metrics);
    } catch (e) {
      console.error('error loading model metrics:', e);
      trainingStatusTimer = setTimeout(checkTrainingStatus, 1000);
    }
  }

  let trainingStatusTimer: any | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
  });

  const percentageFormat = d3.format('.1~%');
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

<div class="w-full py-2 px-4 h-full flex flex-col">
  {#if isTraining}
    <ModelTrainingView {modelName} on:finish={loadModelResults} />
  {:else if !!metrics}
    <h2 class="text-lg font-bold mb-3">
      Model Metrics: <span class="font-mono">{modelName}</span>
    </h2>
    {#if !!metrics.trivial_solution_warning}
      <div class="mb-2 p-4 bg-orange-100 rounded-lg">
        <h4 class="font-bold text-orange-700/80 mb-2">
          <Fa class="inline text-orange-300" icon={faWarning} /> Predictive task
          may be trivially solvable
        </h4>
        <div class="text-gray-800 text-sm mb-2">
          The target variable can be predicted with an AUROC of {percentageFormat(
            metrics.trivial_solution_warning.auc
          )} using the following input variables:
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
        <div class="mb-2 p-4 bg-orange-100 rounded-lg">
          <h4 class="font-bold text-orange-700/80 mb-2">
            <Fa class="inline text-orange-300" icon={faWarning} /> Class {warning.class}
            rarely predicted
          </h4>
          <div class="text-gray-800 text-sm">
            The class {warning.class} is only correctly predicted in {percentageFormat(
              warning.true_positive_fraction
            )} of timesteps with a true label of {warning.class}.
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
            {#each Object.entries(performanceMetrics) as [metricName, value]}
              <div class="w-32">
                <div class="font-bold text-slate-600 text-sm mb-2">
                  {metricName}
                </div>
                <SliceMetricBar width={96} {value}>
                  <span slot="caption" class="font-mono text-base">
                    {percentageFormat(value)}
                  </span>
                </SliceMetricBar>
              </div>
            {/each}
          </div>
        {/if}
        <div class="flex flex-wrap gap-8 mb-4">
          <div class="w-64">
            <div class="font-bold text-slate-600 text-sm mb-2">Training</div>
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
            <div class="font-bold text-slate-600 text-sm mb-2">Validation</div>
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
              Slice Evaluation (from validation set)
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_slice_eval.instances)}
              <span class="text-xs font-sans">instances</span>
            </div>
            <div class="font-mono text-base mb-2">
              {nFormat(metrics.n_slice_eval.trajectories)}
              <span class="text-xs font-sans">trajectories</span>
            </div>
          </div>
        </div>
      </div>
      <div class="aspect-square h-64 shrink-0 grow-0" style="min-width: 200px;">
        <RocLineChart roc={metrics.roc} bind:selectedThreshold />
      </div>
    </div>
    {#if !!metrics.data_summary}
      <div
        class="mb-2 rounded bg-slate-100 p-4 w-full flex-auto min-h-0 overflow-y-auto"
        style="min-height: 300px;"
      >
        <div style="max-width: 500px;">
          <div class="font-bold mb-4">Training Set Overview</div>
          {#each metrics.data_summary.fields as field}
            <ModelDataSummaryElement element={field} />
          {/each}
        </div>
      </div>
    {/if}
  {/if}
</div>
