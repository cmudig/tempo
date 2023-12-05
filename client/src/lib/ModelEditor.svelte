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

  const dispatch = createEventDispatcher();

  export let inputVariables: { [key: string]: VariableDefinition } = {};
  export let outcomeVariable = '';
  export let patientCohort = '';
  export let timestepDefinition = 'every 1 hour between {intime} and {outtime}';

  export let modelName = 'vasopressor_8h';

  let saveError: string | null = null;

  let newModelName: string = modelName;

  let trainingStatus: {
    state: string;
    message: string;
  } | null = null;

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
        pollTrainingStatus();
        return;
      }
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
      pollTrainingStatus();
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

  let trainingStatusTimer: NodeJS.Timeout | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
  });

  async function checkTrainingStatus() {
    trainingStatus = await (
      await fetch(`/training_status/${modelName}`)
    ).json();
    if (trainingStatus!.state == 'error') saveError = trainingStatus!.message;
    else saveError = null;
    if (
      trainingStatus!.state == 'none' ||
      trainingStatus!.state == 'complete' ||
      !!saveError
    )
      trainingStatus = null;
  }

  function pollTrainingStatus() {
    let wasTraining = !!trainingStatus;
    checkTrainingStatus();
    if (!!trainingStatus) {
      if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
      trainingStatusTimer = setTimeout(pollTrainingStatus, 2000);
    } else {
      trainingStatusTimer = null;
      if (wasTraining) loadModelSpec();
    }
  }
</script>

<div class="w-full py-2 px-4">
  {#if !!trainingStatus}
    <div class="my-6 flex flex-col items-center justify-center">
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
      <div class="text-center mt-4">{trainingStatus.message}</div>
    </div>
  {:else}
    <h2 class="text-lg font-bold mb-3">
      Edit Model <span class="font-mono">{modelName}</span>
    </h2>
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
      <button
        class="my-1 py-1.5 text-sm px-4 rounded text-slate-800 bg-blue-100 hover:bg-blue-200 font-bold"
        on:click={trainModel}
      >
        Overwrite Model
      </button>
      <button
        class="my-1 py-1.5 text-sm px-4 rounded text-slate-800 bg-blue-100 hover:bg-blue-200 font-bold"
        on:click={saveAsNewModel}
      >
        Save as New Model...
      </button>
    </div>
    {#if !!saveError}
      <div class="mt-2 text-red-500">{saveError}</div>
    {/if}
  {/if}
</div>
