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
  import { faWarning } from '@fortawesome/free-solid-svg-icons';

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
      trainingStatusTimer = setTimeout(checkTrainingStatus, 1000);
    }
  }

  let trainingStatusTimer: any | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
  });

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
    <div class="mb-2">
      <span class="font-bold text-slate-700 mr-2">Instances</span><span
        class="font-mono"
        >{nFormat(metrics.n_train.instances)} ({nFormat(
          metrics.n_train.trajectories
        )} trajectories)</span
      >
      training,
      <span class="font-mono"
        >{nFormat(metrics.n_val.instances)} ({nFormat(
          metrics.n_val.trajectories
        )} trajectories)</span
      > validation
    </div>
    {#each Object.entries(metrics.performance) as [metricName, value]}
      <div class="mb-2">
        <span class="font-bold text-slate-700 mr-2">{metricName}</span><span
          class="font-mono">{percentageFormat(value)}</span
        >
      </div>
    {/each}
  {/if}
</div>
