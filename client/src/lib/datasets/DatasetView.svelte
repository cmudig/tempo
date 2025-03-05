<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Dataset } from '../dataset';
  import ResizablePanel from '../utils/ResizablePanel.svelte';
  import DatasetInfoView from './DatasetInfoView.svelte';
  import DatasetSidebar from './DatasetSidebar.svelte';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';

  const dispatch = createEventDispatcher();

  export let showSidebar: boolean = true;
  export let datasets: { [key: string]: { spec: Dataset; models: string[] } } =
    {};
  export let currentDataset: string | null = null;
  export let selectedDatasets: string[] = [];
  export let isLoadingDatasets: boolean = false;

  let sidebar: DatasetSidebar;
  let recentRenamedDataset: string | null = null;

  enum View {
    specification = 'Specification',
    info = 'Info',
    utilities = 'Utilities',
  }
  let currentView: View = View.specification;

  let clearingCache: boolean = false;
  let clearCacheMessage: string | null = null;
  async function clearCache(target: string) {
    try {
      clearCacheMessage = null;
      clearingCache = true;
      for (let dataset of new Set([...selectedDatasets, currentDataset])) {
        let result = await fetch(
          import.meta.env.BASE_URL + `/datasets/${dataset}/clear_cache`,
          {
            body: JSON.stringify({ target }),
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );
        if (result.status != 200)
          console.error('error clearing cache:', result);
        else clearCacheMessage = `(${dataset}) ` + (await result.text());
      }
      clearingCache = false;
      setTimeout(() => (clearCacheMessage = null), 10000);
    } catch (e) {
      console.error('error clearing cache:', e);
      clearingCache = false;
    }
  }

  async function createDataset(reference: string) {
    try {
      let newDataset = await (
        await fetch(
          import.meta.env.BASE_URL +
            (!!reference ? `/datasets/new/${reference}` : '/datasets/new'),
          {
            method: 'POST',
          }
        )
      ).json();
      currentDataset = newDataset.name;
      selectedDatasets = [];
      setTimeout(() => {
        if (!!sidebar && !!currentDataset)
          sidebar?.editDatasetName(currentDataset);
      }, 100);
    } catch (e) {
      console.error('error creating new model:', e);
    }
    dispatch('refresh');
  }

  async function renameDataset(datasetName: string, newName: string) {
    console.log('renaming dataset', datasetName, newName);
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${newName}`
      );
      if (result.status == 200) {
        alert('A dataset with that name already exists.');
        return;
      }
    } catch (e) {}

    try {
      await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/rename`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: newName,
          }),
        }
      );
      recentRenamedDataset = newName;
      dispatch('refresh');
    } catch (e) {
      console.error('error renaming dataset:', e);
    }
  }

  $: if (!!recentRenamedDataset && datasets[recentRenamedDataset]) {
    currentDataset = recentRenamedDataset;
    recentRenamedDataset = null;
  }

  async function deleteDatasets(datasetNames: string[]) {
    if (
      !confirm(
        `Are you sure you want to permanently delete the following dataset(s): ${datasetNames.join(', ')}?`
      )
    )
      return;
    try {
      await Promise.all(
        datasetNames.map((n) =>
          fetch(import.meta.env.BASE_URL + `/datasets/${n}`, {
            method: 'DELETE',
          })
        )
      );
    } catch (e) {
      console.error('error deleting dataset:', e);
    }
    dispatch('refresh');
    selectedDatasets = [];
  }
</script>

<div class="w-full h-full flex">
  {#if showSidebar}
    <ResizablePanel
      rightResizable
      width={300}
      minWidth={360}
      maxWidth="50%"
      collapsible={false}
      height="100%"
    >
      <div class="mt-1">
        <DatasetSidebar
          {datasets}
          {isLoadingDatasets}
          bind:currentDataset
          bind:selectedDatasets
          bind:this={sidebar}
          on:new={(e) => createDataset(e.detail)}
          on:rename={(e) => renameDataset(e.detail.old, e.detail.new)}
          on:delete={(e) => deleteDatasets(e.detail)}
        />
      </div>
    </ResizablePanel>
  {/if}
  <div class="flex-auto h-full flex flex-col">
    <div class="w-full px-4 py-2 flex gap-3 bg-slate-200">
      {#each [View.specification, View.info, View.utilities] as view}
        <button
          class="rounded my-2 py-1 text-center w-36 {currentView == view
            ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
            : 'text-slate-700 hover:bg-slate-300'}"
          on:click={() => (currentView = view)}>{view}</button
        >
      {/each}
      <div class="flex-auto" />
      <button
        class="text-slate-600 px-2 hover:opacity-50"
        on:click={() => dispatch('close')}
        ><Fa icon={faXmark} class="inline" /></button
      >
    </div>

    <div
      class="text-lg font-bold whitespace-nowrap flex-auto truncate font-mono p-4 grow-0 shrink-0"
    >
      {#if currentView == View.specification}Specification{:else if currentView == View.info}Dataset
        Info{:else}Utilities{/if}
      for
      {#if selectedDatasets.length > 0}
        {@const numDatasets = new Set([...selectedDatasets, currentDataset])
          .size}
        {numDatasets} dataset{numDatasets > 1 ? 's' : ''}
      {:else}{currentDataset}{/if}
    </div>
    <div class="flex-auto min-h-0 overflow-auto w-full">
      {#if currentView == View.info}
        {#if selectedDatasets.length == 0}
          <div class="py-4">
            <DatasetInfoView
              showHeader={false}
              showCloseButton={false}
              scroll={false}
            />
          </div>
        {:else}
          <div class="w-full h-full flex flex-col items-center justify-center">
            <div class="text-slate-500">Multiple models selected</div>
          </div>
        {/if}
      {:else if currentView == View.specification}{:else if currentView == View.utilities}
        <div
          class="mb-4 px-4 grid gap-4 items-center"
          style="grid-template-columns: max-content auto;"
        >
          <button
            class="btn btn-red"
            disabled={clearingCache}
            on:click={() => clearCache('variables')}
          >
            Clear Variable Caches
          </button>
          <div class="flex-auto text-sm text-slate-500">
            Remove all computed variables for model training, slicing, and
            queries.
          </div>
          <button
            class="btn btn-red"
            disabled={clearingCache}
            on:click={() => clearCache('models')}
          >
            Clear Trained Models
          </button>
          <div class="flex-auto text-sm text-slate-500">
            Remove all model training results (without removing the
            specifications).
          </div>
          <button
            class="btn btn-red"
            disabled={clearingCache}
            on:click={() => clearCache('slices')}
          >
            Clear Subgroup Results
          </button>
          <div class="flex-auto text-sm text-slate-500">
            Remove discovered subgroups.
          </div>
          <button
            class="btn btn-red"
            disabled={clearingCache}
            on:click={() => clearCache('all')}
          >
            Clear All Caches
          </button>
          <div class="flex-auto text-sm text-slate-500">
            Remove all of the above cached data for this dataset, including the
            train/val/test split.
          </div>
        </div>
        {#if !!clearCacheMessage}
          <div class="px-4 mb-4 text-blue-500 text-sm">
            {clearCacheMessage}
          </div>
        {/if}
      {/if}
    </div>
  </div>
</div>
