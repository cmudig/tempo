<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import type { SliceFilter, SliceSpec, VariableDefinition } from './model';
  import VariableEditorPanel from './VariableEditorPanel.svelte';
  import { areObjectsEqual, deepCopy } from './slices/utils/utils';
  import { faRotateRight } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';

  const dispatch = createEventDispatcher();
  export let sliceSpec: string = 'default';
  export let timestepDefinition: string = '';

  let specs: { [key: string]: SliceSpec } = {};
  let loadingSpecs: boolean = false;
  let savingSpecs: boolean = false;
  let saveError: string | null = null;

  async function loadSpecs() {
    try {
      loadingSpecs = true;
      let response = await fetch('/slices/specs');
      specs = await response.json();
      loadingSpecs = false;
    } catch (e) {
      console.error('Error loading specs:', e);
      loadingSpecs = false;
    }
  }

  onMount(loadSpecs);

  let specVariables: { [key: string]: VariableDefinition } | null = null;
  let sliceFilter: SliceFilter | null = null;
  $: !!specs[sliceSpec] && resetSpec();

  let specChanged = false;
  $: {
    specChanged = !areObjectsEqual(specs[sliceSpec], {
      variables: specVariables,
      slice_filter: sliceFilter,
    });
    console.log(
      'changed:',
      specChanged,
      specs[sliceSpec],
      specVariables,
      sliceFilter
    );
  }

  async function saveSpec(overwrite: boolean) {
    if (!specVariables || !sliceFilter) return;
    let newSpec: SliceSpec = {
      variables: specVariables,
      slice_filter: sliceFilter,
    };
    let newSpecName: string = sliceSpec;
    if (!overwrite)
      newSpecName =
        prompt(
          'Choose a name for the new slice specification.',
          new Date().toLocaleString('en-US')
        ) ?? '';
    if (!newSpecName) {
      saveError = 'The specification name cannot be empty.';
      return;
    }
    saveError = null;
    savingSpecs = true;

    try {
      let status = await fetch(`/slices/specs/${newSpecName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newSpec),
      });
      savingSpecs = false;
      if (status.status == 200) dispatch('dismiss', newSpecName);
      else {
        saveError = await status.text();
      }
    } catch (e) {
      saveError = `${e}`;
      savingSpecs = false;
    }
  }

  function resetSpec() {
    if (!!sliceSpec) {
      specVariables = deepCopy(specs[sliceSpec].variables);
      sliceFilter = deepCopy(specs[sliceSpec].slice_filter);
    } else {
      specVariables = null;
      sliceFilter = null;
    }
  }
</script>

{#if loadingSpecs || savingSpecs}
  <div class="w-full h-full flex flex-col items-center justify-center">
    <div class="text-center mb-4">
      {#if loadingSpecs}Loading slice specifications{:else if savingSpecs}Saving
        specification{/if}
    </div>
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
{:else}
  <div class="w-full h-full flex flex-col">
    <div class="w-full py-4 px-4 flex justify-between">
      <div class="font-bold">Slice Specifications</div>
      <div class="text-sm">
        Saved specifications: <select
          class="ml-1 flat-select"
          bind:value={sliceSpec}
        >
          {#each Object.keys(specs) as specName}
            <option value={specName}>{specName}</option>
          {/each}
        </select>
        {#if specChanged}
          <button class="ml-2 my-1 hover:opacity-50" on:click={resetSpec}>
            <Fa icon={faRotateRight} class="inline" />
          </button>
        {/if}
      </div>
    </div>
    {#if !!saveError}
      <div class="rounded mx-4 my-2 p-3 text-red-500 bg-red-50">
        Error: <span class="font-mono">{saveError}</span>
      </div>
    {/if}
    <h3 class="px-4 font-bold mt-2 mb-1">Slicing Variables</h3>
    <div class="w-full flex-auto min-h-0 px-4">
      {#if !!specVariables}
        <VariableEditorPanel
          {timestepDefinition}
          bind:inputVariables={specVariables}
        />
      {/if}
    </div>
    <div class="mt-2 flex gap-2 justify-end items-center mb-4 mx-4">
      {#if specChanged && sliceSpec != 'default'}
        <div class="text-slate-500 text-xs mr-2">
          Overwriting a specification will clear slices found using that
          specification.
        </div>
      {/if}
      <button
        class="my-1 btn btn-slate"
        on:click={() => dispatch('dismiss', null)}
      >
        Cancel
      </button>
      {#if specChanged}
        {#if sliceSpec != 'default'}
          <button class="my-1 btn btn-blue" on:click={() => saveSpec(true)}>
            Overwrite
          </button>
        {/if}
        <button class="my-1 btn btn-blue" on:click={() => saveSpec(false)}>
          Save as New Specification...
        </button>
      {:else}
        <button
          class="my-1 btn btn-blue"
          on:click={() => dispatch('dismiss', sliceSpec)}
        >
          Use Specification
        </button>
      {/if}
    </div>
  </div>
{/if}
