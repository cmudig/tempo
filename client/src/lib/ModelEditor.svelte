<script lang="ts">
  import {
    faCheck,
    faPencil,
    faPlus,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import { type ModelSummary, type VariableDefinition } from './model';
  import Checkbox from './utils/Checkbox.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import VariableEditor from './VariableEditor.svelte';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import { checkTrainingStatus } from './training';
  import VariableEditorPanel from './VariableEditorPanel.svelte';
  import Tooltip from './utils/Tooltip.svelte';
  import { areObjectsEqual } from './slices/utils/utils';

  const dispatch = createEventDispatcher();

  export let inputVariables: { [key: string]: VariableDefinition } | null =
    null;
  export let outcomeVariable: string | null = null;
  export let patientCohort: string | null = null;
  export let timestepDefinition: string | null = null;

  export let modelName: string | null = null;
  export let otherModels: string[] = [];

  let allSpecs: ModelSummary[] = [];

  let saveError: string | null = null;

  let newModelName: string | null = modelName;

  let isLoadingSpecs: boolean = false;
  let isTraining: boolean = false;

  $: if (!!modelName) {
    setupModels(modelName, otherModels);
  } else {
    allSpecs = [];
  }

  function setupModels(active: string, others: string[]) {
    newModelName = active;
    loadAllModelSpecs([active, ...others]);
  }

  async function loadAllModelSpecs(allModels: string[]) {
    saveError = null;

    if (!modelName) return;

    isTraining = false;
    isLoadingSpecs = true;
    let loadedSpecs = await Promise.all(
      allModels.map(async (model) => {
        let spec = await loadModelSpec(model);
        if (!spec) return null;

        if (spec.training) {
          let status = await checkTrainingStatus(model);
          if (!!status) {
            if (status.state == 'error') saveError = status.message;
            else {
              saveError = null;
              isTraining = true;
            }
          }
        }
        return spec;
      })
    );
    isLoadingSpecs = false;
    if (loadedSpecs.some((s) => !s)) return;
    allSpecs = loadedSpecs as ModelSummary[];

    if (
      allSpecs.every((s) => areObjectsEqual(s.variables, allSpecs[0].variables))
    )
      inputVariables = allSpecs[0].variables;
    else inputVariables = null;
    if (allSpecs.every((s) => (s.cohort ?? '') == (allSpecs[0].cohort ?? '')))
      patientCohort = allSpecs[0].cohort ?? '';
    else patientCohort = null;
    if (
      allSpecs.every(
        (s) =>
          (s.timestep_definition ?? '') ==
          (allSpecs[0].timestep_definition ?? '')
      )
    )
      timestepDefinition = allSpecs[0].timestep_definition ?? '';
    else timestepDefinition = null;
    if (allSpecs.every((s) => (s.outcome ?? '') == (allSpecs[0].outcome ?? '')))
      outcomeVariable = allSpecs[0].outcome ?? '';
    else outcomeVariable = null;
    console.log(timestepDefinition, outcomeVariable, patientCohort);
  }

  async function loadModelSpec(model: string): Promise<ModelSummary | null> {
    if (!model) return null;
    try {
      saveError = null;
      let result = await fetch(`/models/${model}/spec`);
      let spec = await result.json();
      return spec;
    } catch (e) {
      console.error('error loading models:', e);
    }
    return null;
  }

  function reset() {
    if (!modelName) return;
    newModelName = modelName;
    loadAllModelSpecs([modelName, ...otherModels]);
  }

  async function deleteModel() {
    try {
      await fetch(`/models/${modelName}`, { method: 'DELETE' });
    } catch (e) {
      console.error('error deleting model:', e);
    }
  }

  async function trainModel() {
    if (otherModels.length == 0) {
      if (!newModelName || newModelName.length == 0) {
        saveError = 'Model must have a name.';
        return;
      }
      saveError = null;
      if (newModelName != modelName) {
        // Delete the old version of the model
        await deleteModel();
      }
    }

    try {
      let modelsToSave = [newModelName, ...otherModels];
      for (let i = 0; i < modelsToSave.length; i++) {
        let result = await fetch('/models', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: modelsToSave[i],
            meta: {
              variables: inputVariables ?? allSpecs[i].variables,
              outcome: outcomeVariable ?? allSpecs[i].outcome,
              cohort: patientCohort ?? allSpecs[i].cohort,
              timestep_definition:
                timestepDefinition ?? allSpecs[i].timestep_definition,
            },
          }),
        });
        if (result.status == 200) {
          saveError = null;
          isTraining = true;
        } else {
          saveError = await result.text();
          isTraining = false;
          break;
        }
      }
    } catch (e) {
      console.error('error saving model:', e);
      saveError = `${e}`;
    }
    dispatch('train', newModelName);
  }

  async function saveAsNewModel() {
    let newName = prompt('Choose a new model name.');
    if (newName == null) return;
    if (!newName) {
      saveError = 'The model name cannot be empty.';
      return;
    }
    try {
      let result = await fetch(`/models/${newName}/metrics`);
      if (result.status == 200) {
        saveError = 'A model with that name already exists.';
        return;
      }
    } catch (e) {}
    newModelName = newName!;
    await trainModel();
  }

  let dataFields: string[] = [];
  onMount(
    async () =>
      (dataFields = (await (await fetch('/data/fields')).json()).fields)
  );
</script>

{#if isLoadingSpecs}
  <div class="w-full h-full flex flex-col items-center justify-center">
    <div class="text-center mb-4">Loading model specification...</div>
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
{:else if allSpecs.length > 0}
  <div class="w-full py-4 px-4">
    {#if isTraining && !!modelName}
      <ModelTrainingView {modelName} on:finish />
    {/if}
    {#if !!saveError}
      <div class="rounded my-2 p-3 text-red-500 bg-red-50">
        Training error: <span class="font-mono">{saveError}</span>
      </div>
    {/if}

    <div class="mb-3 flex items-center">
      {#if otherModels.length == 0}
        <h2 class="text-lg font-bold">Edit Model</h2>
        <input
          type="text"
          placeholder="Model Name"
          class="flex-auto font-mono ml-2 flat-text-input"
          bind:value={newModelName}
        />
      {:else}
        <h2 class="text-lg font-bold">Edit {1 + otherModels.length} Models</h2>
      {/if}
    </div>
    <h3 class="font-bold mt-3 mb-1">
      Timestep Definition &nbsp;<Tooltip
        title="Defines the points in time within each trajectory at which the model will be run."
        position="right"
      />
    </h3>
    {#if timestepDefinition !== null}
      <textarea
        class="w-full font-mono flat-text-input"
        bind:value={timestepDefinition}
      />
    {:else}
      <div class="text-sm text-slate-600 mb-1">
        The selected models have multiple values. To modify all models, choose a
        variant to use:
      </div>
      <select
        class="flat-select font-mono mb-2"
        on:change={(e) => {
          if (!!e.target && e.target.value >= 0)
            timestepDefinition = allSpecs[e.target.value].timestep_definition;
        }}
      >
        <option value={-1}></option>
        {#each [modelName, ...otherModels] as model, i}
          <option value={i}>{model} | {allSpecs[i].timestep_definition}</option>
        {/each}
      </select>
    {/if}

    <h3 class="font-bold mt-2 mb-1">
      Input Variables &nbsp;<Tooltip
        title="Variables that are computed at each timestep defined by the Timestep Definition, to be used as inputs to the model."
        position="right"
      />
    </h3>
    {#if !!inputVariables}
      <div class="w-full" style="height: 368px;">
        <VariableEditorPanel
          {timestepDefinition}
          {dataFields}
          bind:inputVariables
        />
      </div>
    {:else}
      <div class="text-sm text-slate-600 mb-1">
        The selected models have multiple values. To modify all models, choose a
        variant to use:
      </div>
      <select
        class="flat-select font-mono mb-2"
        on:change={(e) => {
          if (!!e.target && e.target.value >= 0)
            inputVariables = allSpecs[e.target.value].variables;
        }}
      >
        <option value={-1}></option>
        {#each [modelName, ...otherModels] as model, i}
          <option value={i}
            >{model} | {Object.keys(allSpecs[i].variables).length} variables</option
          >
        {/each}
      </select>
    {/if}
    <h3 class="font-bold mt-3 mb-1">
      Target Variable &nbsp;<Tooltip
        title="The variable that the model will attempt to predict at each timestep defined by the Timestep Definition."
        position="right"
      />
    </h3>
    {#if !!outcomeVariable}
      <VariableEditor
        varName="outcome"
        varInfo={{ query: outcomeVariable, category: '', enabled: true }}
        {timestepDefinition}
        showCheckbox={false}
        showButtons={false}
        autosave
        showName={false}
        {dataFields}
        editing
        on:save={(e) => (outcomeVariable = e.detail.query)}
      />
    {:else}
      <div class="text-sm text-slate-600 mb-1">
        The selected models have multiple values. To modify all models, choose a
        variant to use:
      </div>
      <select
        class="flat-select font-mono mb-2"
        on:change={(e) => {
          if (!!e.target && e.target.value >= 0)
            outcomeVariable = allSpecs[e.target.value].outcome;
        }}
      >
        <option value={-1}></option>
        {#each [modelName, ...otherModels] as model, i}
          <option value={i}>{model} | {allSpecs[i].outcome}</option>
        {/each}
      </select>
    {/if}
    <!-- <textarea
    class="flat-text-input w-full font-mono"
    bind:value={outcomeVariable}
  /> -->
    <h3 class="font-bold mt-3 mb-1">
      Timestep Filter &nbsp;<Tooltip
        title="A predicate that returns true for any timestep that should be evaluated by the model."
        position="right"
      />
    </h3>
    {#if !!patientCohort}
      <VariableEditor
        varName="cohort"
        varInfo={{ query: patientCohort, category: '', enabled: true }}
        {timestepDefinition}
        showCheckbox={false}
        showButtons={false}
        autosave
        showName={false}
        {dataFields}
        editing
        on:save={(e) => (patientCohort = e.detail.query)}
      />
    {:else}
      <div class="text-sm text-slate-600 mb-1">
        The selected models have multiple values. To modify all models, choose a
        variant to use:
      </div>
      <select
        class="flat-select font-mono mb-2"
        on:change={(e) => {
          if (!!e.target && e.target.value >= 0)
            patientCohort = allSpecs[e.target.value].cohort;
        }}
      >
        <option value={-1}></option>
        {#each [modelName, ...otherModels] as model, i}
          <option value={i}>{model} | {allSpecs[i].cohort}</option>
        {/each}
      </select>
    {/if}
    <!-- <textarea
    class="flat-text-input w-full font-mono"
    bind:value={patientCohort}
  /> -->
    <div class="mt-2 flex gap-2">
      <button class="my-1 btn btn-blue" on:click={trainModel}>
        {#if otherModels.length > 0}Save All{:else}Save and Train{/if}
      </button>
      {#if otherModels.length == 0}
        <button class="my-1 btn btn-slate" on:click={saveAsNewModel}>
          Save As...
        </button>
      {/if}
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
{/if}
