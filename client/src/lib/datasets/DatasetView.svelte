<script lang="ts">
  import { createEventDispatcher, getContext, onDestroy } from 'svelte';
  import type { Dataset } from '../dataset';
  import ResizablePanel from '../utils/ResizablePanel.svelte';
  import DatasetInfoView from './DatasetInfoView.svelte';
  import DatasetSidebar from './DatasetSidebar.svelte';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';
  import DatasetSpecificationView from './DatasetSpecificationView.svelte';
  import type { Writable } from 'svelte/store';
  import {
    checkDatasetBuildStatus,
    taskSuccessful,
    type TrainingStatus,
  } from '../training';

  let csrf: Writable<string> = getContext('csrf');

  const dispatch = createEventDispatcher();

  export let showSidebar: boolean = true;
  export let datasets: { [key: string]: { spec: Dataset; models: string[] } } =
    {};
  export let currentDataset: string | null = null;
  export let selectedDatasets: string[] = [];
  export let isLoadingDatasets: boolean = false;

  let sidebar: DatasetSidebar;
  let recentRenamedDataset: string | null = null;

  $: console.log('datasets:', datasets);
  enum View {
    specification = 'Specification',
    info = 'Info',
    utilities = 'Utilities',
  }
  let currentView: View = View.info;

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
              'X-CSRF-Token': $csrf,
            },
            credentials: 'same-origin',
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
      let response = await fetch(
        import.meta.env.BASE_URL +
          (!!reference ? `/datasets/new/${reference}` : '/datasets/new'),
        {
          method: 'POST',
          headers: {
            'X-CSRF-Token': $csrf,
          },
          credentials: 'same-origin',
        }
      );
      if (response.status != 200) {
        alert('Error creating new model: ' + (await response.text()));
        return;
      }
      let newDataset = await response.json();
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
    if (datasetName == newName || newName.length == 0) return;
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
            'X-CSRF-Token': $csrf,
          },
          credentials: 'same-origin',
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
            headers: {
              'X-CSRF-Token': $csrf,
            },
            credentials: 'same-origin',
          })
        )
      );
    } catch (e) {
      console.error('error deleting dataset:', e);
    }
    dispatch('refresh');
    selectedDatasets = [];
  }

  let buildStatus: TrainingStatus | null = null;
  let buildTaskID: string | null = null;
  let buildStatusTimer: NodeJS.Timeout | null = null;

  $: if (!!currentDataset) refreshBuildStatus();
  else buildStatus = null;

  onDestroy(() => {
    if (!!buildStatusTimer) clearTimeout(buildStatusTimer);
  });

  async function refreshBuildStatus() {
    if (!currentDataset) return;

    let wasBuilding: boolean = !!buildStatus;
    buildStatus = await checkDatasetBuildStatus(currentDataset, buildTaskID);
    if (wasBuilding && !buildStatus) {
      console.log('refreshing datasets');
      dispatch('refresh');
      if (!!buildTaskID && (await taskSuccessful(buildTaskID)) == true)
        currentView = View.info;
      buildTaskID = null;
    }
    console.log('build status:', buildStatus);

    if (!!buildStatusTimer) clearTimeout(buildStatusTimer);
    buildStatusTimer = setTimeout(
      refreshBuildStatus,
      !!buildStatus ? 1000 : 5000
    );
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
  <div class="flex-auto h-full flex flex-col overflow-x-scroll">
    {#if !!buildStatus}
      <div class="w-full h-full flex flex-col items-center justify-center">
        <div class="text-center mb-4">
          {buildStatus.status_info?.message ?? 'Building dataset'}
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

      {#if !!currentDataset}
        {#if currentView == View.info}
          <div
            class="text-lg font-bold whitespace-nowrap flex-auto truncate font-mono p-4 grow-0 shrink-0"
          >
            Dataset Info for
            {#if selectedDatasets.length > 0}
              {@const numDatasets = new Set([
                ...selectedDatasets,
                currentDataset,
              ]).size}
              {numDatasets} dataset{numDatasets > 1 ? 's' : ''}
            {:else}{currentDataset}{/if}
          </div>
          <div class="flex-auto min-h-0 overflow-auto w-full">
            {#if selectedDatasets.length == 0}
              <div class="py-4">
                <DatasetInfoView
                  showHeader={false}
                  showCloseButton={false}
                  scroll={false}
                  spec={datasets[currentDataset]?.spec ?? null}
                />
              </div>
            {:else}
              <div
                class="w-full h-full flex flex-col items-center justify-center"
              >
                <div class="text-slate-500">Multiple models selected</div>
              </div>
            {/if}
          </div>
        {:else if currentView == View.specification}
          <DatasetSpecificationView
            datasetName={currentDataset}
            spec={datasets[currentDataset]?.spec}
            hasModels={(datasets[currentDataset]?.models ?? []).length > 0}
            on:refresh
            on:build={(e) => {
              console.log('BUILDING');
              buildTaskID = e.detail;
              buildStatus = {
                id: `${e.detail}`,
                info: {},
                status: 'waiting',
                status_info: { message: 'Building dataset' },
              };
              refreshBuildStatus();
            }}
          />
        {:else if currentView == View.utilities}
          <div
            class="text-lg font-bold whitespace-nowrap flex-auto truncate font-mono p-4 grow-0 shrink-0"
          >
            Utilities for
            {#if selectedDatasets.length > 0}
              {@const numDatasets = new Set([
                ...selectedDatasets,
                currentDataset,
              ]).size}
              {numDatasets} dataset{numDatasets > 1 ? 's' : ''}
            {:else}{currentDataset}{/if}
          </div>
          <div class="flex-auto min-h-0 overflow-auto w-full">
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
                Remove all of the above cached data for this dataset, including
                the train/val/test split.
              </div>
            </div>
            {#if !!clearCacheMessage}
              <div class="px-4 mb-4 text-blue-500 text-sm">
                {clearCacheMessage}
              </div>
            {/if}
          </div>
        {/if}
      {:else}
        <div class="w-full h-full flex flex-col items-center justify-center">
          <div class="text-slate-500">No datasets yet!</div>
        </div>
      {/if}
    {/if}
  </div>
</div>
