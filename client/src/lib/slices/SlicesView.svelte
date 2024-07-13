<script lang="ts">
  import { type ModelMetrics, type VariableDefinition } from '../model';
  import { createEventDispatcher, getContext, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import ModelTrainingView from '../ModelTrainingView.svelte';
  import {
    checkSlicingStatus,
    checkTrainingStatus,
    type SliceFindingStatus,
    type TrainingStatus,
  } from '../training';
  import SliceSearchView from './SliceSearchView.svelte';
  import {
    SliceSearchControl,
    type Slice,
    type SliceFeatureBase,
  } from '../slices/utils/slice.type';
  import { areObjectsEqual } from '../slices/utils/utils';
  import SliceDetailsView from '../slice_details/SliceDetailsView.svelte';
  import ResizablePanel from '../utils/ResizablePanel.svelte';
  import type { Writable } from 'svelte/store';

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  const dispatch = createEventDispatcher();

  export let modelName: string | null = null;
  export let modelsToShow: string[];
  export let metricToShow = 'AUROC';
  export let selectedSlice: SliceFeatureBase | null = null;
  export let timestepDefinition: string = '';
  export let sliceSpec: string = 'default';

  let scoreFunctionSpec: any[] = [];

  let isTraining: boolean = false;
  let searchTaskID: string | null = null;
  let searchStatus: TrainingStatus | null = null;
  let wasSearching: boolean = false;
  let sliceSearchError: string | null = null;

  let selectedSlices: SliceFeatureBase[] = [];
  export let savedSlices: {
    [key: string]: { [key: string]: SliceFeatureBase };
  } = {};

  let oldSelectedSlices: SliceFeatureBase[] = [];
  $: if (oldSelectedSlices !== selectedSlices) {
    if (selectedSlices.length > 0) selectedSlice = selectedSlices[0];
    else selectedSlice = null;
    oldSelectedSlices = selectedSlices;
  } else if (!selectedSlice && selectedSlices.length > 0) {
    selectedSlices = [];
  } else if (
    !!selectedSlice &&
    (selectedSlices.length != 1 || selectedSlices[0] !== selectedSlice)
  ) {
    selectedSlices = [selectedSlice];
  }

  let loadingSliceStatus = false;
  let retrievingSlices = false;

  let slices: Slice[] | null = null;
  let baseSlice: Slice | null = null;
  let valueNames: { [key: string]: [any, { [key: string]: any }] } | null =
    null;

  let numSlicesToLoad: number = 20;

  let oldModels: string[] = [];
  $: if (oldModels !== modelsToShow) {
    searchStatus = null;
    numSlicesToLoad = 20;
    scoreFunctionSpec = [
      { model_name: modelName, criterion: 'positive_label' },
    ];
    searchTaskID = null;
    if (modelsToShow.length > 0) {
      if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
      getSlicesIfAvailable(modelsToShow);
    }
    oldModels = modelsToShow;
  }

  async function pollSliceStatus() {
    if (!searchTaskID) return;
    try {
      loadingSliceStatus = true;
      searchStatus = await (await fetch(`/tasks/${searchTaskID}`)).json();
    } catch (e) {
      console.log('error getting slice status');
      searchStatus = null;
    }
    loadingSliceStatus = false;
    console.log(searchStatus);
    if (searchStatus?.status == 'complete' || searchStatus?.status == 'error') {
      getSlicesIfAvailable(modelsToShow);
    } else {
      if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
      slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
    }
  }

  async function getSlicesIfAvailable(models: string[]) {
    if (!$currentDataset || !sliceSpec || !scoreFunctionSpec) return;

    try {
      console.log('Fetching slices', slices);
      retrievingSlices = true;
      let response = await fetch(
        `/datasets/${$currentDataset}/slices/${modelName}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            variable_spec_name: sliceSpec,
            score_function_spec: scoreFunctionSpec,
            model_names: models,
          }),
        }
      );
      if (response.status == 400) {
        sliceSearchError = await response.text();
        retrievingSlices = false;
        return;
      }
      let result = await response.json();
      retrievingSlices = false;
      searchStatus = null;
      console.log('results:', result);
      if (!!result.slices && areObjectsEqual(models, modelsToShow)) {
        sliceSearchError = null;
        slices = result.slices;
        baseSlice = result.base_slice;
        valueNames = result.value_names;
        selectedSlices = selectedSlices.filter(
          (sf) =>
            slices!.find((other) => areObjectsEqual(sf, other.feature)) ||
            Object.values(savedSlices[sliceSpec]).find((other) =>
              areObjectsEqual(sf, other)
            )
        );
      } else if (!!result.error) {
        baseSlice = null;
        slices = null;
        selectedSlice = null;
        valueNames = null;
        sliceSearchError = result.error;
      } else {
        baseSlice = null;
        slices = null;
        selectedSlice = null;
        valueNames = null;
        if (!!result.status) {
          searchStatus = result;
          searchTaskID = result.id;
          if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
          slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
        }
      }
      console.log('slices:', slices);
    } catch (e) {
      console.error('error:', e);
      retrievingSlices = false;
    }
  }

  async function loadSlices() {
    if (!modelName || !$currentDataset) return;
    try {
      let trainingStatuses = await checkTrainingStatus(
        $currentDataset,
        modelsToShow
      );
      if (!!trainingStatuses && trainingStatuses.length > 0) {
        isTraining = true;
        return;
      }
      isTraining = false;

      console.log('STARTING slice finding');
      let result = await (
        await fetch(`/datasets/${$currentDataset}/slices/${modelName}/find`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            variable_spec_name: sliceSpec,
            score_function_spec: scoreFunctionSpec,
          }),
        })
      ).json();

      if (!!result.id) {
        searchTaskID = result.id;
        if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
        slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
      } else {
        console.log('result unexpected format:', result);
        sliceSearchError = 'Unable to start finding slices.';
      }
    } catch (e) {
      console.error('error loading slices:', e);
      trainingStatusTimer = setTimeout(checkTrainingStatus, 1000);
    }
  }

  async function stopFindingSlices() {
    if (!searchTaskID) return;
    try {
      await fetch(`/tasks/${searchTaskID}/stop`, { method: 'POST' });
      pollSliceStatus();
    } catch (e) {
      console.error("couldn't stop slice finding:", e);
    }
  }

  function saveSlice(slice: Slice) {
    let saved = savedSlices[sliceSpec] ?? {};
    if (!!saved[slice.stringRep]) delete saved[slice.stringRep];
    else saved[slice.stringRep] = slice.feature;
    savedSlices = { ...savedSlices, [sliceSpec]: saved };
    console.log('saved slices:', savedSlices);
  }

  let trainingStatusTimer: any | null = null;
  let slicesStatusTimer: any | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
    if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
  });
</script>

<div class="w-full pt-4 flex flex-col h-full">
  {#if !!sliceSearchError}
    <div class="px-4">
      <div class="rounded p-3 mb-2 text-red-500 bg-red-50">
        Slice search error: <span class="font-mono">{sliceSearchError}</span>
      </div>
    </div>
  {/if}
  <div class="px-4 text-lg font-bold mb-3 w-full flex items-center gap-2">
    <div>
      Slices for <span class="font-mono">{modelName}</span
      >{#if modelsToShow.length > 1}&nbsp; vs. <span class="font-mono"
          >{modelsToShow.filter((m) => m != modelName).join(', ')}</span
        >{/if}
    </div>
  </div>
  <div class="px-4 flex-auto h-0 overflow-auto" style="width: 100% !important;">
    <SliceSearchView
      {modelName}
      {modelsToShow}
      metricsToShow={[
        'Timesteps',
        'Trajectories',
        metricToShow,
        'Labels',
        'Predictions',
      ]}
      slices={slices ?? []}
      {baseSlice}
      {timestepDefinition}
      savedSlices={savedSlices[sliceSpec] ?? []}
      bind:selectedSlices
      bind:sliceSpec
      {valueNames}
      runningSampler={!!searchStatus || loadingSliceStatus}
      {retrievingSlices}
      samplerProgressMessage={!!searchStatus && !!searchStatus.status
        ? searchStatus.status_info?.message
        : 'Loading'}
      samplerRunProgress={!!searchStatus && !!searchStatus.status
        ? searchStatus.status_info?.progress ?? null
        : null}
      on:load={loadSlices}
      on:loadmore={() => (numSlicesToLoad += 20)}
      on:cancel={stopFindingSlices}
      on:saveslice={(e) => saveSlice(e.detail)}
    />
  </div>
  <!-- <div
    class={selectedSlice != null ? 'mt-4 rounded bg-slate-100' : ''}
    style={selectedSlice != null ? 'min-height: 300px; height: 30vh;' : ''}
  >
    <SliceDetailsView slice={selectedSlice} modelNames={modelsToShow} />
  </div> -->
  {#if !!selectedSlice}
    <ResizablePanel
      class="bg-slate-50"
      topResizable
      height={300}
      minHeight={200}
      maxHeight="80%"
      width="100%"
    >
      <SliceDetailsView
        slice={selectedSlice}
        modelNames={modelsToShow}
        {sliceSpec}
        on:close={() => (selectedSlices = [])}
      />
    </ResizablePanel>
  {/if}
</div>
