<script lang="ts">
  import { type VariableDefinition } from '../model';
  import Checkbox from '../utils/Checkbox.svelte';
  import { createEventDispatcher } from 'svelte';
  import QueryResultView from '../QueryResultView.svelte';
  import { areObjectsEqual } from '../slices/utils/utils';
  import TextareaAutocomplete from '../slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from '../utils/query_autocomplete';

  const dispatch = createEventDispatcher();

  export let varName: string = '';
  export let varInfo: VariableDefinition | null = null;
  export let editing = false;
  export let showName = true;
  export let showButtons = true;
  export let showCheckbox = true;
  export let autosave = false;

  export let dataFields: string[] = [];

  export let timestepDefinition: string = '';

  let queryInput: HTMLElement;

  let newVariableName: string | null = null;
  let newVariableQuery: string | null = null;

  let oldVarInfo: VariableDefinition | null = null;
  $: if (editing) {
    if (!areObjectsEqual(varInfo, oldVarInfo)) {
      newVariableName = varName;
      newVariableQuery = varInfo!.query;
      oldVarInfo = varInfo;
    }
  } else {
    newVariableName = null;
    newVariableQuery = null;
  }

  let evaluationError: string | null = null;
</script>

{#if !!varInfo && !!varName}
  <div class:ml-2={showCheckbox} class="mb-1 flex items-center gap-1">
    {#if showCheckbox}
      <Checkbox
        checked={varInfo.enabled ?? true}
        on:change={(e) => {
          dispatch('toggle', e.detail);
        }}
      />
      <div class="w-2" />
    {/if}
    {#if editing}
      <div class="flex-auto flex flex-col gap-2 h-full">
        {#if showName}
          <input
            type="text"
            class="flat-text-input w-full font-mono text-sm"
            placeholder="Variable Name"
            bind:value={newVariableName}
          />
        {/if}
        <div class="flex flex-auto w-full">
          {#if !!timestepDefinition}
            <div class="flex-auto">
              {#if showName}
                <div class="mb-1 text-slate-500 text-xs w-32">Query</div>
              {/if}
              <div class="relative w-full {showName ? 'h-24' : ''}">
                <textarea
                  spellcheck={false}
                  class="flat-text-input w-full h-full font-mono"
                  style="field-sizing: content; {!showName
                    ? 'min-height: 84px;'
                    : ''}"
                  bind:this={queryInput}
                  bind:value={newVariableQuery}
                  on:input={() => {
                    if (autosave) {
                      dispatch('save', {
                        name: newVariableName,
                        query: newVariableQuery,
                      });
                    }
                  }}
                />
                <TextareaAutocomplete
                  ref={queryInput}
                  resolveFn={(query, prefix) =>
                    getAutocompleteOptions(dataFields, query, prefix)}
                  replaceFn={performAutocomplete}
                  triggers={['{', '#']}
                  delimiterPattern={/[\s\(\[\]\)](?=[\{#])/}
                  menuItemTextFn={(v) => v.value}
                  maxItems={3}
                  menuItemClass="p-2"
                  on:replace={(e) => (newVariableQuery = e.detail)}
                />
              </div>
              {#if showButtons}
                <div class="mt-2 flex justify-end gap-1">
                  <button
                    class="my-1 py-1 btn text-sm text-slate-800 bg-red-200 hover:bg-red-300"
                    on:click={() => dispatch('delete')}>Delete</button
                  >
                  <button
                    class="my-1 py-1 btn btn-slate text-sm"
                    on:click={() => dispatch('cancel')}>Cancel</button
                  >
                  <button
                    class="my-1 py-1 btn btn-blue text-sm"
                    class:opacity-30={newVariableQuery == varInfo.query}
                    disabled={newVariableQuery == varInfo.query ||
                      !!evaluationError}
                    on:click={() =>
                      dispatch('save', {
                        name: newVariableName,
                        query: newVariableQuery,
                      })}>Save</button
                  >
                </div>
              {/if}
            </div>

            <div class="text-sm w-48 shrink-0 grow-0 self-stretch ml-2 p-2">
              <QueryResultView
                delayEvaluation
                bind:evaluationError
                query={!!newVariableQuery
                  ? `${newVariableQuery} ${timestepDefinition}`
                  : ''}
              />
            </div>
          {/if}
        </div>
      </div>
    {:else}
      <button
        class="font-mono hover:bg-slate-200 rounded flex-auto text-left px-2 py-1"
        on:click={() => dispatch('edit')}>{varName}</button
      >
    {/if}
  </div>
{/if}
