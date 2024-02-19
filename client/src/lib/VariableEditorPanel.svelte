<script lang="ts">
  import { faPlus } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import VariableEditor from './VariableEditor.svelte';
  import { type VariableDefinition } from './model';
  import Checkbox from './utils/Checkbox.svelte';

  export let timestepDefinition: string;
  export let inputVariables: { [key: string]: VariableDefinition } = {};

  export let fillHeight: boolean = true;

  let allCategories: string[] = [];
  $: allCategories = Array.from(
    new Set(Object.values(inputVariables).map((v) => v.category))
  ).sort();

  let visibleInputVariableCategory: string | null = null;
  $: if (allCategories.length > 0 && visibleInputVariableCategory == null)
    visibleInputVariableCategory = allCategories[0];

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
        category: visibleInputVariableCategory!,
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

  let categoryVariables: [string, VariableDefinition][] = [];
  $: categoryVariables = Object.entries(inputVariables)
    .filter((c) => c[1].category == visibleInputVariableCategory)
    .sort((a, b) => a[0].localeCompare(b[0]));
</script>

<div class="w-full rounded bg-slate-100 flex gap-1" class:h-full={fillHeight}>
  <div
    class="w-1/5 pt-2 px-3 max-h-full overflow-y-scroll"
    style="min-width: 200px; max-width: 400px;"
  >
    {#each allCategories as cat}
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
    <div
      class="ml-2 pb-2 mb-2 flex items-center gap-1 border-b border-slate-300"
    >
      <Checkbox
        checked={categoryVariables.every((item) => item[1].enabled)}
        on:change={(e) => {
          categoryVariables.forEach(
            (item) => (inputVariables[item[0]].enabled = e.detail)
          );
        }}
      />
      <div class="w-2" />
      <div class="text-slate-500 flex-auto text-left px-2 py-1">
        {categoryVariables.length} variable{categoryVariables.length != 1
          ? 's'
          : ''}
      </div>
    </div>
    {#each categoryVariables as [varName, varInfo]}
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
