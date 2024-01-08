<script lang="ts">
  import {
    faCheck,
    faPencil,
    faPlus,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import {
    AllCategories,
    VariableCategory,
    type VariableDefinition,
  } from './model';
  import Checkbox from './utils/Checkbox.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher, onDestroy } from 'svelte';
  import VariableEditor from './VariableEditor.svelte';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import { checkTrainingStatus } from './training';

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
      if (spec.training) {
        let status = await checkTrainingStatus(modelName);
        if (!!status) {
          if (status.state == 'error') saveError = status.message;
          else {
            saveError = null;
            isTraining = true;
            return;
          }
        }
      }
      isTraining = false;
      inputVariables = spec.variables;
      outcomeVariable = spec.outcome;
      patientCohort = spec.cohort;
      timestepDefinition = spec.timestep_definition;
    } catch (e) {
      console.error('error loading models:', e);
    }
  }

  let visibleInputVariableCategory: VariableCategory =
    VariableCategory.Demographics;

  let currentEditingVariableName: string | null = null;

  function defineNewVariable() {
    let varName = 'Unnamed';
    if (!!inputVariables[varName]) {
      let num = 1;
      while (!!inputVariables[`Unnamed ${num}`]) num++;
      varName = `Unnamed ${num}`;
    }
    inputVariables = {
      ...inputVariables,
      [varName]: {
        category: visibleInputVariableCategory,
        query: '',
        enabled: true,
      },
    };
    currentEditingVariableName = varName;
  }

  function saveVariableEdits(
    newVariableName: string,
    newVariableQuery: string
  ) {
    inputVariables = Object.fromEntries([
      ...Object.entries(inputVariables).filter(
        (item) => item[0] != currentEditingVariableName
      ),
      [
        newVariableName,
        {
          category: visibleInputVariableCategory,
          query: newVariableQuery!,
          enabled: true,
        },
      ],
    ]);
    currentEditingVariableName = null;
  }

  $: visibleInputVariableCategory,
    (() => (currentEditingVariableName = null))();

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
  {:else}
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
      class="bg-slate-200 appearance-none border-2 border-slate-200 w-full rounded text-slate-700 font-mono text-xs p-2 leading-tight focus:outline-none focus:border-blue-600 focus:bg-white"
      bind:value={timestepDefinition}
    />

    <h3 class="font-bold mt-2 mb-1">Input Variables</h3>
    <div
      class="w-full rounded bg-slate-100 flex gap-1"
      style="max-height: 368px;"
    >
      <div
        class="w-1/5 pt-2 px-3 max-h-full overflow-y-scroll"
        style="min-width: 200px; max-width: 400px;"
      >
        {#each AllCategories as cat}
          <button
            class="w-full my-1 py-1 text-sm px-4 rounded {visibleInputVariableCategory ==
            cat
              ? 'bg-slate-600 text-white hover:bg-slate-700 font-bold'
              : 'text-slate-800 hover:bg-slate-200'}"
            on:click={() => (visibleInputVariableCategory = cat)}
          >
            {cat}
          </button>
        {/each}
      </div>
      <div class="flex-auto max-h-full overflow-y-scroll pr-3 pl-2 py-4">
        {#each Object.entries(inputVariables)
          .filter((c) => c[1].category == visibleInputVariableCategory)
          .sort((a, b) => a[0].localeCompare(b[0])) as [varName, varInfo]}
          <VariableEditor
            {varName}
            {varInfo}
            {timestepDefinition}
            editing={currentEditingVariableName == varName}
            on:cancel={() => (currentEditingVariableName = null)}
            on:edit={() => (currentEditingVariableName = varName)}
            on:save={(e) => saveVariableEdits(e.detail.name, e.detail.query)}
            on:toggle={(e) => (inputVariables[varName].enabled = e.detail)}
          />
        {/each}
        <button
          class="my-1 py-1 text-sm px-3 rounded text-slate-800 bg-slate-200 hover:bg-slate-300 font-bold"
          on:click={defineNewVariable}
        >
          <Fa class="inline mr-2" icon={faPlus} /> New Variable
        </button>
      </div>
    </div>
    <h3 class="font-bold mt-3 mb-1">Outcome Variable</h3>
    <textarea
      class="bg-slate-200 appearance-none border-2 border-slate-200 w-full rounded text-slate-700 font-mono text-xs p-2 leading-tight focus:outline-none focus:border-blue-600 focus:bg-white"
      bind:value={outcomeVariable}
    />
    <h3 class="font-bold mt-3 mb-1">Timestep Filter</h3>
    <textarea
      class="bg-slate-200 appearance-none border-2 border-slate-200 w-full rounded text-slate-700 font-mono text-xs p-2 leading-tight focus:outline-none focus:border-blue-600 focus:bg-white"
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
  {/if}
</div>
