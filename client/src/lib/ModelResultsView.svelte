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

  const dispatch = createEventDispatcher();

  export let modelName = 'vasopressor_8h';
  let metrics: ModelMetrics | null = null;

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
      console.log(metrics);
    } catch (e) {
      console.error('error loading model metrics:', e);
      setTimeout(checkTrainingStatus, 1000);
    }
  }

  let trainingStatusTimer: NodeJS.Timeout | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
  });

  const rocFormat = d3.format('.3~');
  const percentageFormat = d3.format('.1~%');
  const nFormat = d3.format(',');
</script>

<div class="w-full py-2 px-4">
  {#if isTraining}
    <ModelTrainingView {modelName} on:finish={loadModelResults} />
  {:else if !!metrics}
    <h2 class="text-lg font-bold mb-3">
      Model Metrics: <span class="font-mono">{modelName}</span>
    </h2>
    <div class="mb-2">
      <span class="font-bold text-slate-700 mr-2">Instances</span><span
        class="font-mono">{nFormat(metrics.n_train)}</span
      >
      training, <span class="font-mono">{nFormat(metrics.n_val)}</span> validation
    </div>
    {#if !!metrics.roc_auc}
      <div class="mb-2">
        <span class="font-bold text-slate-700 mr-2">AUROC</span><span
          class="font-mono">{rocFormat(metrics.roc_auc)}</span
        >
      </div>
    {/if}
    {#if !!metrics.acc}
      <div class="mb-2">
        <span class="font-bold text-slate-700 mr-2">Accuracy</span><span
          class="font-mono">{percentageFormat(metrics.acc)}</span
        >
      </div>
    {/if}
    {#if !!metrics.sensitivity}
      <div class="mb-2">
        <span class="font-bold text-slate-700 mr-2">Sensitivity</span><span
          class="font-mono">{percentageFormat(metrics.sensitivity)}</span
        >
      </div>
    {/if}
    {#if !!metrics.specificity}
      <div class="mb-2">
        <span class="font-bold text-slate-700 mr-2">Specificity</span><span
          class="font-mono">{percentageFormat(metrics.specificity)}</span
        >
      </div>
    {/if}
  {/if}
</div>
