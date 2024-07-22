<script lang="ts">
  import {
    faCode,
    faListDots,
    faPlus,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import VariableEditor from './VariableEditor.svelte';
  import { type VariableDefinition } from '../model';
  import Checkbox from '../utils/Checkbox.svelte';
  import TextareaAutocomplete from '../slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from '../utils/query_autocomplete';
  import type { Writable } from 'svelte/store';
  import { getContext } from 'svelte';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';

  let {
    currentDataset,
    dataFields,
  }: {
    currentDataset: Writable<string | null>;
    dataFields: Writable<string[]>;
  } = getContext('dataset');

  export let timestepDefinition: string;
  export let inputVariables: { [key: string]: VariableDefinition } = {};

  export let fillHeight: boolean = true;

  let showRaw: boolean = false;

  let allCategories: string[] = [];
  $: allCategories = Array.from(
    new Set(Object.values(inputVariables).map((v) => v.category))
  )
    .filter((c) => !!c)
    .sort();

  let visibleInputVariableCategory: string | null = null;
  $: if (allCategories.length > 0 && visibleInputVariableCategory == null)
    visibleInputVariableCategory = allCategories[0];

  let currentEditingVariableName: string | null = null;

  let selectedVariables: string[] = [];

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

  function toggleVariables(variableNames: string[], enabled: boolean) {
    inputVariables = Object.fromEntries([
      ...Object.entries(inputVariables).filter(
        (item) => !variableNames.includes(item[0])
      ),
      ...variableNames.map((name) => [
        name,
        { ...inputVariables[name], enabled },
      ]),
    ]);
    currentEditingVariableName = null;
  }

  function deleteVariables(names: string[]) {
    inputVariables = Object.fromEntries(
      Object.entries(inputVariables).filter((item) => !names.includes(item[0]))
    );
    selectedVariables = [];
    if (
      !!currentEditingVariableName &&
      names.includes(currentEditingVariableName)
    ) {
      currentEditingVariableName = null;
    }
  }

  $: visibleInputVariableCategory,
    (() => (currentEditingVariableName = null))();

  let categoryVariables: [string, VariableDefinition][] = [];
  $: categoryVariables = Object.entries(inputVariables)
    .filter((c) => c[1].category == visibleInputVariableCategory)
    .sort((a, b) => a[0].localeCompare(b[0]));

  let searchText: string = '';

  let visibleVariables: [string, VariableDefinition][] = [];

  $: if (searchText.length > 0 && categoryVariables.length > 0) {
    visibleVariables = categoryVariables
      .filter((v) =>
        v[0].toLocaleLowerCase().includes(searchText.toLocaleLowerCase())
      )
      .sort((a, b) => a.length - b.length);
  } else visibleVariables = categoryVariables;

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
        await fetch(`/datasets/${$currentDataset}/data/validate_syntax`, {
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

  let oldEditingVariable: string | null = null;
  $: if (oldEditingVariable != currentEditingVariableName) {
    if (currentEditingVariableName != null) {
      selectedVariables = [currentEditingVariableName];
    } else if (
      currentEditingVariableName == null &&
      oldEditingVariable != null &&
      selectedVariables.length == 1 &&
      selectedVariables[0] == oldEditingVariable
    )
      selectedVariables = [];
    oldEditingVariable = currentEditingVariableName;
  }

  function toggleSelection(name: string, toggleValue: boolean) {
    if (toggleValue) selectedVariables = [...selectedVariables, name];
    else {
      let idx = selectedVariables.indexOf(name);
      selectedVariables = [
        ...selectedVariables.slice(0, idx),
        ...selectedVariables.slice(idx + 1),
      ];
    }
  }

  function duplicateVariables(names: string[]) {
    selectedVariables = [];

    let newNames = names.map((name) => {
      let varName = name + ' 2';
      if (!!inputVariables[varName]) {
        let num = 3;
        while (!!inputVariables[`${name} ${num}`]) num++;
        varName = `${name} ${num}`;
      }
      inputVariables = {
        ...inputVariables,
        [varName]: {
          category: visibleInputVariableCategory!,
          query: inputVariables[name].query,
          enabled: inputVariables[name].enabled ?? true,
        },
      };
      return varName;
    });

    if (newNames.length == 1) currentEditingVariableName = newNames[0];
  }
</script>

<div
  class="flex flex-col w-full rounded border-slate-300 border overflow-hidden"
  style={fillHeight ? '' : 'max-height: 400px;'}
  class:h-full={fillHeight}
>
  <div class="w-full py-3 px-3 flex items-center bg-slate-100">
    {#if allCategories.length > 0}
      <select bind:value={visibleInputVariableCategory} class="flat-select">
        {#each allCategories as cat}
          <option value={cat}>
            {cat}
          </option>
        {/each}
      </select>
    {/if}

    <div class="flex justify-center gap-2">
      <button
        on:click={() => (showRaw = false)}
        disabled={!showRaw}
        class="btn {!showRaw ? 'bg-slate-400' : 'hover:bg-slate-200'} rounded"
        title="Use the graphical interface to enter and update variables"
        ><Fa icon={faListDots} class="inline mr-2" />Structured</button
      >
      <button
        on:click={() => (showRaw = true)}
        disabled={showRaw}
        class="btn {showRaw ? 'bg-slate-400' : 'hover:bg-slate-200'} rounded"
        title="Use a text editor to enter and update variables"
        ><Fa icon={faCode} class="inline mr-2" />Raw</button
      >
    </div>
    <div class="flex-auto" />
    {#if !showRaw}
      <input
        type="search"
        bind:value={searchText}
        class="w-48 shrink-1 flat-text-input"
        placeholder="Find variable..."
      />
    {/if}
  </div>
  {#if showRaw && rawRepresentation != null}
    <div class="flex flex-col flex-auto p-4">
      <div class="relative flex-auto w-full">
        <textarea
          class="flat-text-input-large w-full h-full font-mono"
          spellcheck={false}
          bind:this={rawInput}
          bind:value={rawRepresentation}
          on:input={updateRawRepresentation}
        />
        <TextareaAutocomplete
          ref={rawInput}
          resolveFn={(query, prefix) =>
            getAutocompleteOptions($dataFields, query, prefix)}
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
    <div class="overflow-y-auto flex-auto min-h-0">
      <div
        class="px-4 py-2 mb-2 flex items-center gap-1 bg-slate-100 sticky top-0 z-10"
      >
        <Checkbox
          checked={selectedVariables.length >= visibleVariables.length &&
            visibleVariables.every((item) =>
              selectedVariables.includes(item[0])
            )}
          indeterminate={selectedVariables.length > 0 &&
            selectedVariables.length < visibleVariables.length}
          on:change={(e) => {
            if (e.detail) selectedVariables = visibleVariables.map((x) => x[0]);
            else selectedVariables = [];
          }}
        />
        <div class="text-slate-500 flex-auto text-left px-2 py-1">
          {#if selectedVariables.length > 0}
            {selectedVariables.length} of
          {/if}
          {visibleVariables.length} variable{visibleVariables.length != 1
            ? 's'
            : ''}
        </div>
        <div class="flex items-center gap-2">
          <button
            class="mr-1 my-1 py-1 text-sm px-3 rounded text-slate-800 bg-slate-200 hover:bg-slate-300 font-bold"
            on:click={defineNewVariable}
          >
            <Fa class="inline mr-2" icon={faPlus} /> New Variable
          </button>
          <label
            class="relative inline-flex items-center cursor-pointer {selectedVariables.length ==
            0
              ? 'opacity-40'
              : ''}"
          >
            <input
              type="checkbox"
              value=""
              class="sr-only peer"
              disabled={selectedVariables.length == 0}
              checked={visibleVariables.every(
                (item) => item[1].enabled ?? true
              )}
              on:change={(e) =>
                toggleVariables(selectedVariables, e.target?.checked)}
            />
            <div
              title="Enable or disable this feature from the model"
              class="relative w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-500"
            ></div>
          </label>
          <ActionMenuButton
            buttonClass="bg-transparent px-3 py-1 hover:opacity-40 text-slate-600 {selectedVariables.length ==
            0
              ? 'opacity-40'
              : ''}"
            disabled={selectedVariables.length == 0}
            align="right"
          >
            <div slot="options">
              <a
                href="#"
                tabindex="0"
                role="menuitem"
                title="Create a copy of these variables"
                on:click={() => duplicateVariables(selectedVariables)}
                >Duplicate</a
              >
              <a
                href="#"
                tabindex="0"
                role="menuitem"
                title="Permanently delete this variable"
                on:click={() => deleteVariables(selectedVariables)}>Delete</a
              >
            </div>
          </ActionMenuButton>
        </div>
      </div>
      {#each visibleVariables as [varName, varInfo] (varName)}
        <VariableEditor
          class={currentEditingVariableName == null ||
          currentEditingVariableName == varName
            ? ''
            : 'hidden'}
          {varName}
          {varInfo}
          {timestepDefinition}
          editing={currentEditingVariableName == varName}
          isChecked={selectedVariables.includes(varName)}
          on:cancel={() => {
            currentEditingVariableName = null;
            if (!varInfo.query) deleteVariables([varName]);
          }}
          on:edit={() => (currentEditingVariableName = varName)}
          on:save={(e) => saveVariableEdits(e.detail.name, e.detail.query)}
          on:toggle={(e) => toggleVariables([varName], e.detail)}
          on:select={(e) => toggleSelection(varName, e.detail)}
          on:duplicate={(e) => duplicateVariables([varName])}
          on:delete={(e) => deleteVariables([varName])}
        />
      {/each}
    </div>
  {/if}
</div>
