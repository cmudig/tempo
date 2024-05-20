<script lang="ts">
  import { type ModelMetrics, type VariableDefinition } from '../model';
  import { createEventDispatcher, onDestroy } from 'svelte';
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

  const dispatch = createEventDispatcher();

  export let modelName: string | null = null;
  export let modelsToShow: string[];
  export let metricToShow = 'AUROC';
  export let selectedSlice: SliceFeatureBase | null = null;
  export let timestepDefinition: string = '';
  export let sliceSpec: string = 'default';

  let isTraining: boolean = false;
  let searchStatus: SliceFindingStatus | null = null;
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

  let samplingStatusOverview: string | null = null;
  $: if (!!searchStatus && !!modelName && (slices?.length ?? 0) > 0) {
    if (searchStatus.models.includes(modelName))
      samplingStatusOverview = `Showing ${slices?.length ?? 0} of ${
        searchStatus.n_results
      } slices from ${searchStatus.n_runs} sampled timesteps`;
    else
      samplingStatusOverview = `Showing ${slices?.length ?? 0} of ${
        searchStatus.n_results
      } slices from ${
        searchStatus.n_runs
      } timesteps (sampled from other models)`;
  } else {
    samplingStatusOverview = null;
  }

  let slices: Slice[] | null = null;
  let baseSlice: Slice | null = null;
  // the slice controls used to generate the results that are displayed
  type Controls = { [key in SliceSearchControl]?: SliceFeatureBase } & {
    slice_spec_name?: string;
  };
  let resultControls: Controls = {};
  let retrievedScoreWeights: { [key: string]: number } | null = null;
  let scoreWeights: { [key: string]: number } | null = null;
  let valueNames: { [key: string]: [any, { [key: string]: any }] } | null =
    null;

  let enabledSliceControls: { [key in SliceSearchControl]?: boolean } = {};
  let containsSlice: any = {};
  let containedInSlice: any = {};
  let similarToSlice: any = {};
  let subsliceOfSlice: any = {};
  let queryControls: Controls = { slice_spec_name: sliceSpec };
  let numSlicesToLoad: number = 20;

  let oldModels: string[] = [];
  $: if (oldModels !== modelsToShow) {
    searchStatus = null;
    scoreWeights = null;
    numSlicesToLoad = 20;
    oldNumSlices = 20;
    if (modelsToShow.length > 0) {
      getSlicesIfAvailable(modelsToShow);
      pollSliceStatus();
    }
    oldModels = modelsToShow;
  }

  async function pollSliceStatus() {
    if (!modelName) return;
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
        if (wasSearching && !sliceSearchError) {
          wasSearching = false;
          getSlicesIfAvailable(modelsToShow);
        }
      } else {
        wasSearching = true;
        sliceSearchError = null;
        if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
        slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
      }
    }
  }

  async function getSlicesIfAvailable(models: string[]) {
    try {
      console.log('Fetching slices', slices);
      retrievingSlices = true;
      retrievedScoreWeights = null;
      resultControls = {};
      let response = await fetch(`/slices/${models.join(',')}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...(!!scoreWeights ? { score_weights: scoreWeights } : {}),
          controls: queryControls,
          num_slices: numSlicesToLoad,
        }),
      });
      if (response.status == 400) {
        sliceSearchError = await response.text();
        retrievingSlices = false;
        resultControls = queryControls;
        return;
      }
      let result = await response.json();
      resultControls = result.controls;
      retrievingSlices = false;
      if (!!result.results.slices && areObjectsEqual(models, modelsToShow)) {
        sliceSearchError = null;
        slices = result.results.slices;
        baseSlice = result.results.base_slice;
        scoreWeights = result.results.score_weights;
        retrievedScoreWeights = result.results.score_weights;
        valueNames = result.results.value_names;
        selectedSlices = selectedSlices.filter(
          (sf) =>
            slices!.find((other) => areObjectsEqual(sf, other.feature)) ||
            Object.values(savedSlices[sliceSpec]).find((other) =>
              areObjectsEqual(sf, other)
            )
        );
      } else {
        baseSlice = null;
        slices = null;
        selectedSlice = null;
        scoreWeights = null;
        retrievedScoreWeights = null;
        valueNames = null;
        pollSliceStatus();
      }
      console.log('slices:', slices);
    } catch (e) {
      console.error('error:', e);
      retrievingSlices = false;
      resultControls = queryControls;
    }
  }

  async function loadSlices() {
    if (!modelName) return;
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
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            controls: queryControls,
          }),
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

  let oldNumSlices: number = numSlicesToLoad;
  $: if (
    ((retrievedScoreWeights != null &&
      !areObjectsEqual(retrievedScoreWeights, scoreWeights)) ||
      !areObjectsEqual(resultControls, queryControls) ||
      numSlicesToLoad != oldNumSlices) &&
    !retrievingSlices
  ) {
    console.log(
      'changing bc parameters changed',
      resultControls,
      queryControls,
      numSlicesToLoad
    );
    getSlicesIfAvailable(modelsToShow);
    oldNumSlices = numSlicesToLoad;
  }

  $: {
    queryControls = {
      ...(enabledSliceControls[SliceSearchControl.containsSlice]
        ? { [SliceSearchControl.containsSlice]: containsSlice }
        : {}),
      ...(enabledSliceControls[SliceSearchControl.containedInSlice]
        ? { [SliceSearchControl.containedInSlice]: containedInSlice }
        : {}),
      ...(enabledSliceControls[SliceSearchControl.similarToSlice]
        ? { [SliceSearchControl.similarToSlice]: similarToSlice }
        : {}),
      ...(enabledSliceControls[SliceSearchControl.subsliceOfSlice]
        ? { [SliceSearchControl.subsliceOfSlice]: subsliceOfSlice }
        : {}),
      slice_spec_name: sliceSpec,
    };
  }
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
      modelNames={modelsToShow}
      metricsToShow={[
        'Timesteps',
        'Trajectories',
        metricToShow,
        'Labels',
        'Predictions',
      ]}
      slices={areObjectsEqual(queryControls, resultControls) || retrievingSlices
        ? slices ?? []
        : []}
      {baseSlice}
      {timestepDefinition}
      {samplingStatusOverview}
      savedSlices={savedSlices[sliceSpec] ?? []}
      bind:scoreWeights
      bind:selectedSlices
      bind:enabledSliceControls
      bind:sliceSpec
      bind:containsSlice
      bind:containedInSlice
      bind:similarToSlice
      bind:subsliceOfSlice
      {valueNames}
      runningSampler={(!!searchStatus && searchStatus.status.state != 'none') ||
        loadingSliceStatus}
      {retrievingSlices}
      samplerProgressMessage={!!searchStatus && !!searchStatus.status
        ? searchStatus.status.message
        : 'Loading'}
      samplerRunProgress={!!searchStatus && !!searchStatus.status
        ? searchStatus.status.progress
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
