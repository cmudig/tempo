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

  export let modelName: string | null = null;

  let saveError: string | null = null;

  let newModelName: string = modelName;

  let isTraining: boolean = false;

  $: if (!!modelName) {
    newModelName = modelName;
    loadModelSpec();
  }

  async function loadModelSpec() {
    if (!modelName) return;
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
      outcomeVariable = spec.outcome ?? '';
      patientCohort = spec.cohort ?? '';
      timestepDefinition = spec.timestep_definition ?? '';
    } catch (e) {
      console.error('error loading models:', e);
    }
  }

  function reset() {
    if (!modelName) return;
    newModelName = modelName;
    loadModelSpec();
  }

  async function deleteModel() {
    try {
      await fetch(`/models/${modelName}`, { method: 'DELETE' });
    } catch (e) {
      console.error('error deleting model:', e);
    }
  }

  async function trainModel() {
    if (newModelName.length == 0) {
      saveError = 'Model must have a name.';
      return;
    }
    saveError = null;
    if (newModelName != modelName) {
      // Delete the old version of the model
      await deleteModel();
    }

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
      if (result.status == 200) {
        saveError = null;
        isTraining = true;
      } else {
        saveError = await result.text();
        isTraining = false;
      }
    } catch (e) {
      console.error('error saving model:', e);
      saveError = `${e}`;
    }
    dispatch('train', newModelName);
  }
</script>

<div class="w-full py-2 px-4">
  {#if isTraining && !!modelName}
    <ModelTrainingView {modelName} on:finish />
  {/if}
  {#if !!saveError}
    <div class="rounded my-2 p-3 text-red-500 bg-red-50">
      Training error: <span class="font-mono">{saveError}</span>
    </div>
  {/if}

  <div class="mb-3 flex items-center">
    <h2 class="text-lg font-bold">Edit Model</h2>
    <input
      type="text"
      placeholder="Model Name"
      class="flex-auto font-mono ml-2 flat-text-input"
      bind:value={newModelName}
    />
  </div>
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
  <VariableEditor
    varName="outcome"
    varInfo={{ query: outcomeVariable, category: '', enabled: true }}
    {timestepDefinition}
    showCheckbox={false}
    showButtons={false}
    autosave
    showName={false}
    editing
    on:save={(e) => (outcomeVariable = e.detail.query)}
  />
  <!-- <textarea
    class="flat-text-input w-full font-mono"
    bind:value={outcomeVariable}
  /> -->
  <h3 class="font-bold mt-3 mb-1">Timestep Filter</h3>
  <VariableEditor
    varName="cohort"
    varInfo={{ query: patientCohort, category: '', enabled: true }}
    {timestepDefinition}
    showCheckbox={false}
    showButtons={false}
    autosave
    showName={false}
    editing
    on:save={(e) => (patientCohort = e.detail.query)}
  />
  <!-- <textarea
    class="flat-text-input w-full font-mono"
    bind:value={patientCohort}
  /> -->
  <div class="mt-2 flex gap-2">
    <button class="my-1 btn btn-blue" on:click={trainModel}>
      Save and Train
    </button>
    <button class="my-1 btn btn-slate" on:click={reset}> Reset </button>
    <button
      class="my-1 btn text-slate-800 bg-red-200 hover:bg-red-300"
      on:click={async () => {
        await deleteModel();
        dispatch('delete', modelName);
      }}
    >
      Delete Model
    </button>
  </div>
</div>
