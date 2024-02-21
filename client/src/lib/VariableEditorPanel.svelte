<script lang="ts">
  import {
    faCode,
    faListDots,
    faPlus,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import VariableEditor from './VariableEditor.svelte';
  import { type VariableDefinition } from './model';
  import Checkbox from './utils/Checkbox.svelte';
  import TextareaAutocomplete from './slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from './utils/query_autocomplete';

  export let timestepDefinition: string;
  export let inputVariables: { [key: string]: VariableDefinition } = {};
  export let dataFields: string[] = [];

  export let fillHeight: boolean = true;

  let showRaw: boolean = false;

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

  function deleteVariable() {
    inputVariables = Object.fromEntries(
      Object.entries(inputVariables).filter(
        (item) => item[0] != currentEditingVariableName
      )
    );
    currentEditingVariableName = null;
  }

  $: visibleInputVariableCategory,
    (() => (currentEditingVariableName = null))();

  let categoryVariables: [string, VariableDefinition][] = [];
  $: categoryVariables = Object.entries(inputVariables)
    .filter((c) => c[1].category == visibleInputVariableCategory)
    .sort((a, b) => a[0].localeCompare(b[0]));

  let rawCategory: string | null = null;
  let rawRepresentation: string | null = null;
  let rawParseError: string | null = null;
  let rawParseWaiting: boolean = false;
  let rawParseSuccess: boolean = false;
  let rawParseTimer: NodeJS.Timeout | null = null;
  let rawInput: HTMLElement;

  $: if (
    showRaw &&
    (rawRepresentation == null || rawCategory != visibleInputVariableCategory)
  ) {
    rawRepresentation = categoryVariables
      .filter((v) => v[1].enabled ?? true)
      .map(([name, varInfo]) => `${name}: ${varInfo.query}`)
      .join(',\n');
    rawCategory = visibleInputVariableCategory;
  } else if (!showRaw && rawRepresentation !== null) {
    rawRepresentation = null;
    rawParseSuccess = false;
    rawParseWaiting = false;
    rawParseError = null;
  }

  function updateRawRepresentation() {
    rawParseWaiting = true;
    rawParseSuccess = false;
    if (!!rawParseTimer) clearTimeout(rawParseTimer);
    rawParseTimer = setTimeout(validateRawRepresentation, 2000);
  }

  async function validateRawRepresentation() {
    rawParseError = null;
    try {
      let query = `(${rawRepresentation}) ${timestepDefinition}`;
      let result = await (
        await fetch(`/data/validate_syntax`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query }),
        })
      ).json();
      if (result.success) {
        rawParseError = null;
        rawParseSuccess = true;
        rawParseWaiting = false;
        inputVariables = Object.fromEntries([
          ...Object.entries(inputVariables).filter(
            (item) => item[1].category != visibleInputVariableCategory
          ),
          ...Object.entries(
            result.variables as {
              [key: string]: { query: string; enabled: boolean };
            }
          ).map(([name, varInfo]) => [
            name,
            Object.assign(varInfo, { category: visibleInputVariableCategory }),
          ]),
        ]);
      } else {
        rawParseError = result.error;
        rawParseSuccess = false;
        rawParseWaiting = false;
      }
    } catch (e) {
      rawParseError = `${e}`;
    }
  }
</script>

<div class="w-full rounded bg-slate-100 flex gap-1" class:h-full={fillHeight}>
  <div
    class="w-1/5 pt-2 px-3 h-full flex flex-col shrink-0"
    style="min-width: 200px; max-width: 400px;"
  >
    <div class="flex-auto min-h-0 overflow-auto">
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
    <div class="my-2 flex w-full justify-center gap-2">
      <button
        on:click={() => (showRaw = false)}
        disabled={!showRaw}
        class="btn {!showRaw ? 'bg-slate-400' : 'hover:bg-slate-200'} rounded"
        ><Fa icon={faListDots} class="inline mr-2" />Structured</button
      >
      <button
        on:click={() => (showRaw = true)}
        disabled={showRaw}
        class="btn {showRaw ? 'bg-slate-400' : 'hover:bg-slate-200'} rounded"
        ><Fa icon={faCode} class="inline mr-2" />Raw</button
      >
    </div>
  </div>
  {#if showRaw && rawRepresentation != null}
    <div class="flex-auto flex flex-col h-full pr-3 pl-2 py-4">
      <div class="relative flex-auto w-full">
        <textarea
          class="flat-text-input-large w-full h-full font-mono"
          bind:this={rawInput}
          bind:value={rawRepresentation}
          on:input={updateRawRepresentation}
        />
        <TextareaAutocomplete
          ref={rawInput}
          resolveFn={(query, prefix) =>
            getAutocompleteOptions(dataFields, query, prefix)}
          replaceFn={performAutocomplete}
          triggers={['{', '#']}
          delimiterPattern={/[\s\(\[\]\)](?=[\{#])/}
          menuItemTextFn={(v) => v.value}
          maxItems={3}
          menuItemClass="p-2"
          on:replace={(e) => {
            rawRepresentation = e.detail;
            updateRawRepresentation();
          }}
        />
      </div>
      {#if !!rawParseError}
        <div class="w-full mt-1 text-red-600 text-sm">
          <strong>Changes not saved:</strong>
          {rawParseError}
        </div>
      {:else}
        <div class="w-full mt-1 text-slate-600 text-sm">
          {#if rawParseWaiting}Parsing...{:else if rawParseSuccess}Changes
            saved.{:else}
            Edit the variables above or copy and paste from your favorite
            editor.
          {/if}
        </div>
      {/if}
    </div>
  {:else}
    <div class="flex-auto max-h-full overflow-y-scroll pr-3 pl-2 py-4">
      <div
        class="ml-2 pb-2 mb-2 flex items-center gap-1 border-b border-slate-300"
      >
        <Checkbox
          checked={categoryVariables.every((item) => item[1].enabled ?? true)}
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
          {dataFields}
          {timestepDefinition}
          editing={currentEditingVariableName == varName}
          on:cancel={() => (currentEditingVariableName = null)}
          on:edit={() => (currentEditingVariableName = varName)}
          on:save={(e) => saveVariableEdits(e.detail.name, e.detail.query)}
          on:toggle={(e) => (inputVariables[varName].enabled = e.detail)}
          on:delete={deleteVariable}
        />
      {/each}
      <button
        class="my-1 py-1 text-sm px-3 rounded text-slate-800 bg-slate-200 hover:bg-slate-300 font-bold"
        on:click={defineNewVariable}
      >
        <Fa class="inline mr-2" icon={faPlus} /> New Variable
      </button>
    </div>
  {/if}
</div>
