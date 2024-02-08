<script lang="ts">
  import {
    faCheck,
    faPencil,
    faPlus,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import { type VariableDefinition } from './model';
  import Checkbox from './utils/Checkbox.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, onDestroy } from 'svelte';
  import VariableEditor from './VariableEditor.svelte';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import { checkTrainingStatus } from './training';
  import VariableEditorPanel from './VariableEditorPanel.svelte';

  const dispatch = createEventDispatcher();

  export let inputVariables: { [key: string]: VariableDefinition } = {};
  export let outcomeVariable = '';
  export let patientCohort = '';
  export let timestepDefinition = 'every 1 hour between {intime} and {outtime}';

  export let modelName = 'vasopressor_8h';

  let saveError: string | null = null;

  let newModelName: string = modelName;

  let isTraining: boolean = false;

  $: if (!!modelName) {
    newModelName = modelName;
    loadModelSpec();
  }

  async function loadModelSpec() {
    try {
      saveError = null;
      let result = await fetch(`/models/${modelName}/spec`);
      let spec = await result.json();
      isTraining = false;
      if (spec.training) {
        let status = await checkTrainingStatus(modelName);
        if (!!status) {
          if (status.state == 'error') saveError = status.message;
          else {
            saveError = null;
            isTraining = true;
          }
        }
      }
      inputVariables = spec.variables;
      outcomeVariable = spec.outcome;
      patientCohort = spec.cohort;
      timestepDefinition = spec.timestep_definition;
    } catch (e) {
      console.error('error loading models:', e);
    }
  }

  function reset() {
    loadModelSpec();
  }

  async function trainModel() {
    dispatch('train', newModelName);

    try {
      let result = await fetch('/models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newModelName,
          meta: {
            variables: inputVariables,
            outcome: outcomeVariable,
            cohort: patientCohort,
            timestep_definition: timestepDefinition,
          },
        }),
      });
      saveError = null;
      isTraining = true;
    } catch (e) {
      console.error('error saving model:', e);
      saveError = `${e}`;
    }
  }

  async function saveAsNewModel() {
    let newName = prompt('Choose a new model name.');
    if (!newName) {
      saveError = 'The model name cannot be empty.';
      return;
    }
    try {
      let result = await fetch(`/models/${newName}/metrics`);
      if (result.status == 200)
        saveError = 'A model with that name already exists.';
    } catch (e) {}
    newModelName = newName!;
    trainModel();
  }
</script>

<div class="w-full py-2 px-4">
  {#if isTraining}
    <ModelTrainingView {modelName} on:finish={reset} />
  {/if}
  <h2 class="text-lg font-bold mb-3">
    Edit Model <span class="font-mono">{modelName}</span>
  </h2>
  {#if !!saveError}
    <div class="rounded mt-2 p-3 text-red-500 bg-red-50">
      Training error: <span class="font-mono">{saveError}</span>
    </div>
  {/if}
  <h3 class="font-bold mt-3 mb-1">Timestep Definition</h3>
  <textarea
    class="w-full font-mono flat-text-input"
    bind:value={timestepDefinition}
  />

  <h3 class="font-bold mt-2 mb-1">Input Variables</h3>
  <div class="w-full" style="height: 368px;">
    <VariableEditorPanel {timestepDefinition} bind:inputVariables />
  </div>
  <h3 class="font-bold mt-3 mb-1">Outcome Variable</h3>
  <textarea
    class="flat-text-input w-full font-mono"
    bind:value={outcomeVariable}
  />
  <h3 class="font-bold mt-3 mb-1">Timestep Filter</h3>
  <textarea
    class="flat-text-input w-full font-mono"
    bind:value={patientCohort}
  />
  <div class="mt-2 flex gap-2">
    <button
      class="my-1 py-1.5 text-sm px-4 rounded text-slate-800 bg-red-200 hover:bg-red-300 font-bold"
      on:click={reset}
    >
      Reset
    </button>
    <button class="my-1 btn btn-blue" on:click={trainModel}>
      Overwrite Model
    </button>
    <button class="my-1 btn btn-blue" on:click={saveAsNewModel}>
      Save as New Model...
    </button>
  </div>
</div>
