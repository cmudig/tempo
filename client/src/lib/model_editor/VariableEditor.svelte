<script lang="ts">
  import { type VariableDefinition } from '../model';
  import Checkbox from '../utils/Checkbox.svelte';
  import { createEventDispatcher } from 'svelte';
  import QueryResultView from '../QueryResultView.svelte';
  import { areObjectsEqual } from '../slices/utils/utils';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';
  import QueryEditorTextarea from './QueryEditorTextarea.svelte';
  import highlight from 'custom-syntax-highlighter';
  import { highlightPatterns } from './syntaxhighlight';

  const dispatch = createEventDispatcher();

  export let varName: string = '';
  export let varInfo: VariableDefinition | null = null;
  export let editing = false;
  export let showName = true;
  export let showButtons = true;
  export let showTableControls = true;
  export let autosave = false;
  export let isChecked = false;
  export let templates: {
    title: string;
    children: { name: string; query: string }[];
  }[] = [];

  export let timestepDefinition: string = '';

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
    oldVarInfo = null;
  }

  let evaluationError: string | null = null;

  let timestepDefLabel: HTMLElement;
  let timestepDefLabelID: string =
    'timestepDefLabel-' +
    new Array(20)
      .fill(0)
      .map(() => Math.floor(Math.random() * 10))
      .join('');

  $: if (!!timestepDefLabel) {
    timestepDefLabel.innerText = timestepDefinition;
    highlight({
      selector: `#${timestepDefLabelID}`,
      patterns: highlightPatterns,
    });
  }
</script>

{#if !!varInfo && !!varName}
  <div
    class:mx-4={showTableControls}
    class="mb-1 grid items-center {$$props.class ?? ''}"
    class:gap-2={showTableControls}
    style="grid-template-columns: max-content auto max-content;"
  >
    {#if showTableControls}
      <Checkbox
        checked={isChecked}
        on:change={(e) => {
          dispatch('select', e.detail);
        }}
      />
    {/if}
    {#if showName || showButtons}
      <div class="flex items-center gap-2">
        {#if editing}
          {#if showName}
            <input
              type="text"
              class="flat-text-input flex-auto font-mono text-sm"
              style="padding-top: 0.4rem; padding-bottom: 0.4rem;"
              placeholder="Variable Name"
              bind:value={newVariableName}
            />
          {/if}
        {:else}
          <button
            class="font-mono hover:bg-slate-100 rounded flex-auto text-left px-2 py-1 {varInfo.enabled ??
            true
              ? ''
              : 'line-through text-slate-400'}"
            on:click={() => dispatch('edit')}>{varName}</button
          >
          <div class="text-sm w-48 shrink-0 grow-0">
            <QueryResultView
              delayEvaluation
              compact
              query={!!varInfo.query
                ? `${varInfo.query} ${timestepDefinition}`
                : ''}
            />
          </div>
        {/if}
        {#if editing && showButtons}
          <button
            class="shrink-0 py-1 btn btn-slate text-sm"
            on:click={() => dispatch('cancel')}>Cancel</button
          >
          <button
            class="shrink-0 py-1 btn btn-blue text-sm disabled:opacity-30"
            disabled={(newVariableQuery == varInfo.query &&
              newVariableName == varName) ||
              !!evaluationError}
            on:click={() =>
              dispatch('save', {
                name: newVariableName,
                query: newVariableQuery,
              })}>Save</button
          >
        {/if}
      </div>
    {/if}

    {#if showTableControls}
      <div class="flex items-center gap-2">
        <label class="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            value=""
            class="sr-only peer"
            checked={varInfo.enabled ?? true}
            on:change={(e) => {
              dispatch('toggle', !(varInfo.enabled ?? true));
            }}
          />
          <div
            title="Enable or disable this feature from the model"
            class="relative w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-500"
          ></div>
        </label>
        <ActionMenuButton
          buttonClass="bg-transparent px-3 py-1 hover:opacity-40 text-slate-600"
          align="right"
        >
          <div slot="options">
            {#if !editing}
              <a
                href="#"
                tabindex="0"
                role="menuitem"
                title="Edit this variable"
                on:click={() => dispatch('edit')}>Edit</a
              >
            {/if}
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Create a copy of this variable"
              on:click={() => dispatch('duplicate')}>Duplicate</a
            >
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Permanently delete this variable"
              on:click={() => dispatch('delete')}>Delete</a
            >
          </div>
        </ActionMenuButton>
      </div>
    {/if}
    {#if editing}
      <div></div>
      <div class="flex flex-auto w-full">
        <div class="flex-auto">
          {#if showName}
            <div class="mb-1 text-slate-500 text-xs w-32">Query</div>
          {/if}
          <div class="relative w-full {showName ? 'h-24' : ''}">
            <QueryEditorTextarea
              style="field-sizing: content; {!showName
                ? 'min-height: 84px;'
                : ''}"
              bind:value={newVariableQuery}
              {templates}
              on:input={() => {
                if (autosave) {
                  dispatch('save', {
                    name: newVariableName,
                    query: newVariableQuery,
                  });
                }
              }}
            >
              <div class="text-xs" slot="buttons">
                {#if !!timestepDefinition}
                  <span class="text-slate-500">Evaluated at timesteps:</span>
                  <span
                    class="font-mono"
                    id={timestepDefLabelID}
                    bind:this={timestepDefLabel}
                  ></span>
                {:else}
                  <span class="text-slate-500"
                    >Variables require a Timestep Definition to be evaluated.</span
                  >
                {/if}
              </div>
            </QueryEditorTextarea>
          </div>
        </div>

        {#if !!timestepDefinition}
          <div class="text-sm w-48 shrink-0 grow-0 self-stretch ml-4">
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
      <div></div>
    {/if}
  </div>
{/if}
