<script lang="ts">
  import {
    createEventDispatcher,
    getContext,
    onDestroy,
    onMount,
  } from 'svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import type { QueryResult } from '../model';
  import QueryResultView from '../QueryResultView.svelte';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';
  import TextareaAutocomplete from '../slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from '../utils/query_autocomplete';
  import type { Writable } from 'svelte/store';

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  const dispatch = createEventDispatcher();
  type DatasetInfo = {
    attributes: { [key: string]: QueryResult };
    events: { [key: string]: QueryResult };
    intervals: { [key: string]: QueryResult };
  };
  export let datasetInfo: DatasetInfo | null = null;
  export let showHeader: boolean = true;
  let loadingInfo: boolean = false;
  let infoLoadStatus: {
    id: string;
    status: string;
    status_info?: { message: string };
  } | null = null;
  export let showCloseButton: boolean = true;

  let tabNames: (keyof DatasetInfo)[] = ['attributes', 'events', 'intervals'];
  let tabExplanatoryText: { [key in keyof DatasetInfo]: string } = {
    attributes: 'Attributes remain constant throughout a trajectory.',
    events:
      'Events are observations that occur at a specific timepoint within a trajectory.',
    intervals:
      'Intervals occur between a start and an end time during a trajectory.',
  };
  let selectedTab: keyof DatasetInfo = 'attributes';

  let oldTab: keyof DatasetInfo = 'attributes';
  let startIndex: number = 0;
  let pageSize: number = 20;
  let variableInfos: [string, QueryResult][] = [];
  $: if (selectedTab !== oldTab) {
    startIndex = 0;
    oldTab = selectedTab;
  }

  $: if (!!$currentDataset) loadDatasetInfo($currentDataset);
  else datasetInfo = null;

  $: if (!!datasetInfo)
    variableInfos = Object.entries(datasetInfo[selectedTab]).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
  else variableInfos = [];

  async function loadDatasetInfo(datasetName: string | null) {
    if (!datasetName) return;
    try {
      loadingInfo = true;
      let response = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/data/summary`
      );
      let result = await response.json();
      if (!!result.attributes) {
        datasetInfo = result;

        loadingInfo = false;
        if (Object.keys(datasetInfo!.attributes).length > 0)
          selectedTab = 'attributes';
        else if (Object.keys(datasetInfo!.events).length > 0)
          selectedTab = 'events';
        else if (Object.keys(datasetInfo!.intervals).length > 0)
          selectedTab = 'intervals';
        else selectedTab = 'attributes';
      } else {
        infoLoadStatus = result;
        setTimeout(() => loadDatasetInfo($currentDataset), 1000);
      }
    } catch (e) {
      console.error('Error loading dataset info:', e);
      loadingInfo = false;
    }
  }
</script>

<div class="flex flex-col w-full h-full">
  {#if showHeader}
    <div class="w-full py-4 px-4 flex justify-between">
      <div class="font-bold">
        Dataset Info for <span class="font-mono"
          >{$currentDataset ?? '(none)'}</span
        >
      </div>
      {#if showCloseButton}
        <button
          class="text-slate-600 px-2 hover:opacity-50"
          on:click={() => dispatch('close')}
          ><Fa icon={faXmark} class="inline" /></button
        >
      {/if}
    </div>
  {/if}
  <div class="w-full flex-auto overflow-y-auto min-h-0 relative">
    {#if loadingInfo}
      <div class="w-full flex-auto flex flex-col items-center justify-center">
        <div class="text-center mb-4">
          {infoLoadStatus?.status_info?.message ??
            'Loading dataset info' +
              (!!infoLoadStatus?.status ? ` (${infoLoadStatus?.status})` : '')}
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
    {:else if !!datasetInfo}
      <div class="w-full px-4 py-2 flex gap-3 sticky top-0 bg-white z-10">
        {#each tabNames as tab}
          <button
            class="rounded my-2 py-1 px-6 text-center w-32 {selectedTab == tab
              ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
              : 'text-slate-700 hover:bg-slate-200'}"
            on:click={() => (selectedTab = tab)}
            >{tab.slice(0, 1).toUpperCase() + tab.slice(1)}</button
          >
        {/each}
      </div>
      <div class="mb-2 px-4 text-sm text-slate-500">
        {tabExplanatoryText[selectedTab]}
      </div>
      {#if variableInfos.length == 0}
        <div
          class="my-4 flex items-center justify-center text-base text-slate-500"
        >
          No variables to show
        </div>
      {:else}
        <div class="flex flex-wrap px-2">
          {#each variableInfos.slice(startIndex, startIndex + pageSize) as [field, values]}
            <div class="p-2 w-1/4">
              <div class="p-2 rounded bg-slate-100">
                <QueryResultView
                  evaluationSummary={values}
                  evaluateQuery={false}
                  showName
                />
              </div>
            </div>
          {/each}
        </div>
        {#if variableInfos.length > pageSize}
          <div class="flex items-center justify-center mt-2 py-2 px-2 gap-3">
            <button
              class="btn btn-slate"
              on:click={() => (startIndex -= pageSize)}
              disabled={startIndex == 0}>Previous</button
            >
            <div class="text-sm text-slate-600">
              Showing {startIndex + 1} - {Math.min(
                variableInfos.length,
                startIndex + pageSize
              )} of {variableInfos.length}
            </div>
            <button
              class="btn btn-slate"
              on:click={() => (startIndex += pageSize)}
              disabled={startIndex + pageSize >= variableInfos.length}
              >Next</button
            >
          </div>
        {/if}
      {/if}
    {/if}
  </div>
</div>

<style>
  textarea::placeholder {
    @apply text-slate-500;
  }
</style>
