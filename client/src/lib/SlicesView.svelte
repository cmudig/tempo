<script lang="ts">
  import {
    AllCategories,
    VariableCategory,
    type ModelMetrics,
    type VariableDefinition,
  } from './model';
  import { createEventDispatcher, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import ModelTrainingView from './ModelTrainingView.svelte';
  import {
    checkSlicingStatus,
    checkTrainingStatus,
    type SliceFindingStatus,
    type TrainingStatus,
  } from './training';
  import SliceSearchView from './SliceSearchView.svelte';
  import type { Slice, SliceFeatureBase } from './slices/utils/slice.type';
  import { areObjectsEqual } from './slices/utils/utils';

  const dispatch = createEventDispatcher();

  export let modelName = 'vasopressor_8h';
  export let metricToShow = 'AUROC';
  export let selectedSlice: SliceFeatureBase | null = null;

  let isTraining: boolean = false;
  let searchStatus: SliceFindingStatus | null = null;
  let wasSearching: boolean = false;
  let sliceSearchError: string | null = null;

  let selectedSlices: SliceFeatureBase[] = [];

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
  let retrievedScoreWeights: { [key: string]: number } | null = null;
  let scoreWeights: { [key: string]: number } | null = null;
  let valueNames: { [key: string]: [any, { [key: string]: any }] } | null =
    null;

  $: if (modelName) {
    searchStatus = null;
    scoreWeights = null;
    getSlicesIfAvailable();
    pollSliceStatus();
  }

  async function pollSliceStatus() {
    try {
      loadingSliceStatus = true;
      searchStatus = await checkSlicingStatus();
    } catch (e) {
      console.log('error getting slice status');
      searchStatus = null;
    }
    loadingSliceStatus = false;
    console.log(searchStatus);
    if (!!searchStatus) {
      if (!!searchStatus.errors && !!searchStatus.errors[modelName])
        sliceSearchError = searchStatus.errors[modelName];
      else sliceSearchError = null;

      if (searchStatus.status?.state == 'none') {
        searchStatus = null;
        if (wasSearching) {
          wasSearching = false;
          getSlicesIfAvailable();
        }
      } else {
        wasSearching = true;
        sliceSearchError = null;
        if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
        slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
      }
    }
  }

  async function getSlicesIfAvailable() {
    try {
      console.log('Fetching slices');
      retrievingSlices = true;
      let response = await fetch(`/slices/${modelName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: !!scoreWeights
          ? JSON.stringify({
              score_weights: scoreWeights,
            })
          : '',
      });
      if (response.status == 400) {
        sliceSearchError = await response.text();
        retrievingSlices = false;
        return;
      }
      let result = await response.json();
      if (!!result.results.slices) {
        retrievingSlices = false;
        sliceSearchError = null;
        slices = result.results.slices;
        baseSlice = result.results.base_slice;
        scoreWeights = result.results.score_weights;
        retrievedScoreWeights = result.results.score_weights;
        valueNames = result.results.value_names;
        console.log(slices, baseSlice, scoreWeights, valueNames);
      } else {
        retrievingSlices = false;
        pollSliceStatus();
      }
    } catch (e) {
      console.error('error:', e);
      retrievingSlices = false;
    }
  }

  async function loadSlices() {
    try {
      let trainingStatus = await checkTrainingStatus(modelName);
      if (!!trainingStatus && trainingStatus.state != 'error') {
        isTraining = true;
        return;
      }
      isTraining = false;

      console.log('STARTING slice finding');
      let result = await (
        await fetch(`/slices/${modelName}/start`, {
          method: 'POST',
        })
      ).json();
      if (result.searching) {
        searchStatus = result;
        pollSliceStatus();
      } else {
        console.log('not searching for some reason');
        if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
        slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
      }
      console.log(result);
    } catch (e) {
      console.error('error loading slices:', e);
      trainingStatusTimer = setTimeout(checkTrainingStatus, 1000);
    }
  }

  async function stopFindingSlices() {
    try {
      await fetch(`/slices/stop_finding`, { method: 'POST' });
      pollSliceStatus();
    } catch (e) {
      console.error("couldn't stop slice finding:", e);
    }
  }

  let trainingStatusTimer: any | null = null;
  let slicesStatusTimer: any | null = null;

  onDestroy(() => {
    if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
    if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
  });

  $: if (!areObjectsEqual(retrievedScoreWeights, scoreWeights)) {
    getSlicesIfAvailable();
  }
</script>

<div class="p-4 w-full flex flex-col h-full">
  {#if !!sliceSearchError}
    <div class="p-3 mb-2 text-red-500 bg-red-50">
      Slice search error: <span class="font-mono">{sliceSearchError}</span>
    </div>
  {/if}
  <div class="flex-auto h-0 overflow-auto" style="width: 100% !important;">
    <SliceSearchView
      modelNames={[modelName]}
      metricsToShow={[
        'Timesteps',
        'Trajectories',
        metricToShow,
        'Positive Rate',
      ]}
      slices={slices ?? []}
      {baseSlice}
      bind:scoreWeights
      bind:selectedSlices
      {valueNames}
      runningSampler={!!searchStatus || loadingSliceStatus}
      {retrievingSlices}
      samplerProgressMessage={!!searchStatus && !!searchStatus.status
        ? searchStatus.status.message
        : 'Loading'}
      samplerRunProgress={!!searchStatus && !!searchStatus.status
        ? searchStatus.status.progress
        : null}
      on:load={loadSlices}
      on:cancel={stopFindingSlices}
    />
  </div>
  <div class="mt-4 rounded bg-slate-100 h-64">Slice Details</div>
</div>
