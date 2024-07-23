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
  import { faChevronLeft, faWrench } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';
  import SliceSpecEditor from './SliceSpecEditor.svelte';
  import { scoreFunctionToString, type ScoreFunction } from './scorefunctions';
  import ScoreFunctionPanel from './ScoreFunctionPanel.svelte';

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  const dispatch = createEventDispatcher();

  export let modelName: string | null = null;
  export let modelsToShow: string[];
  export let metricToShow = 'AUROC';
  export let selectedSlice: SliceFeatureBase | null = null;
  export let timestepDefinition: string = '';
  export let sliceSpec: string = 'default';

  enum View {
    slices = 0,
    specEditor = 1,
    scoreFunctionEditor = 2,
  }
  let visibleView: View = View.slices;

  let scoreFunctionSpec: ScoreFunction[] = [];

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

  let oldModelName: string | null = null;
  $: if (oldModelName !== modelName) {
    loadSliceSearchSettings();
  }

  let oldScoreSpec: any[] = [];
  let oldSliceSpec: string = 'default';
  let oldModels: string[] = [];
  $: if (
    !areObjectsEqual(oldScoreSpec, scoreFunctionSpec) ||
    oldSliceSpec !== sliceSpec ||
    oldModels !== modelsToShow
  ) {
    console.log('different score or slice spec');
    oldScoreSpec = scoreFunctionSpec;
    oldSliceSpec = sliceSpec;
    oldModels = modelsToShow;
    searchStatus = null;
    numSlicesToLoad = 20;
    searchTaskID = null;
    saveSliceSearchSettings();
    if (modelsToShow.length > 0) initiateSliceLookup();
  }

  function loadSliceSearchSettings() {
    let slicingSettingsString = window.localStorage.getItem('slicingSettings');
    let slicingSettings: {
      [key: string]: { scoreFns: ScoreFunction[]; spec: string };
    } = !!slicingSettingsString ? JSON.parse(slicingSettingsString) : {};
    if (!!modelName && !!slicingSettings[modelName]) {
      scoreFunctionSpec = slicingSettings[modelName].scoreFns;
      sliceSpec = slicingSettings[modelName].spec;
    } else {
      scoreFunctionSpec = [
        {
          type: 'relation',
          relation: '=',
          lhs: {
            type: 'model_property',
            model_name: modelName ?? '',
            property: 'label',
          },
          rhs: {
            type: 'constant',
            value: 1,
          },
        },
      ];
      sliceSpec = `${modelName} (Default)`;
    }
  }

  function saveSliceSearchSettings() {
    let slicingSettingsString = window.localStorage.getItem('slicingSettings');
    let slicingSettings: {
      [key: string]: { scoreFns: ScoreFunction[]; spec: string };
    } = !!slicingSettingsString ? JSON.parse(slicingSettingsString) : {};
    if (!!modelName)
      slicingSettings[modelName] = {
        scoreFns: scoreFunctionSpec,
        spec: sliceSpec,
      };
    window.localStorage.setItem(
      'slicingSettings',
      JSON.stringify(slicingSettings)
    );
  }

  function initiateSliceLookup() {
    pollTrainingStatus().then(() => {
      if (!isTraining) {
        if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
        getSlicesIfAvailable(modelsToShow);
      }
    });
  }

  async function pollSliceStatus() {
    if (!searchTaskID) return;
    try {
      loadingSliceStatus = true;
      searchStatus = await (await fetch(import.meta.env.BASE_URL + `/tasks/${searchTaskID}`)).json();
    } catch (e) {
      console.log('error getting slice status');
      searchStatus = null;
    }
    loadingSliceStatus = false;
    console.log(searchStatus);
    if (
      ['complete', 'error', 'canceling', 'canceled'].includes(
        searchStatus?.status ?? ''
      )
    ) {
      getSlicesIfAvailable(modelsToShow);
    } else {
      if (!!slicesStatusTimer) clearTimeout(slicesStatusTimer);
      slicesStatusTimer = setTimeout(pollSliceStatus, 1000);
    }
  }

  async function pollTrainingStatus() {
    if (!$currentDataset) return;
    try {
      let trainingStatuses = await checkTrainingStatus(
        $currentDataset,
        modelsToShow
      );
      if (!!trainingStatuses && trainingStatuses.length > 0) {
        isTraining = true;
        if (!!trainingStatusTimer) clearTimeout(trainingStatusTimer);
        trainingStatusTimer = setTimeout(pollTrainingStatus, 1000);
        return;
      }
      isTraining = false;
    } catch (e) {
      console.error('error getting training status:', e);
    }
  }

  async function getSlicesIfAvailable(models: string[]) {
    if (!$currentDataset || !sliceSpec || !scoreFunctionSpec) return;

    try {
      console.log('Fetching slices', slices);
      retrievingSlices = true;
      sliceSearchError = null;
      let response = await fetch(
        import.meta.env.BASE_URL + `/datasets/${$currentDataset}/slices/${modelName}`,
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
      if (response.status != 200) {
        sliceSearchError = await response.text();
        retrievingSlices = false;
        searchStatus = null;
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
    visibleView = View.slices;
    loadingSliceStatus = true;
    try {
      await pollTrainingStatus();
      if (isTraining) return;

      console.log('STARTING slice finding');
      let result = await (
        await fetch(import.meta.env.BASE_URL + `/datasets/${$currentDataset}/slices/${modelName}/find`, {
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
      trainingStatusTimer = setTimeout(pollTrainingStatus, 1000);
    }
  }

  async function stopFindingSlices() {
    if (!searchTaskID) return;
    try {
      await fetch(import.meta.env.BASE_URL + `/tasks/${searchTaskID}/stop`, { method: 'POST' });
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

  let specChanged = false;
  let scoreFunctionsChanged = false;
  function dismissSpecEditor() {
    if (
      specChanged &&
      !confirm(
        'Are you sure you want to cancel? Your changes to the slicing variables will not be saved.'
      )
    )
      return;

    visibleView = View.slices;
  }

  function dismissScoreFunctionEditor() {
    if (
      scoreFunctionsChanged &&
      !confirm(
        'Are you sure you want to cancel? Your changes to the score functions will not be saved.'
      )
    )
      return;

    visibleView = View.slices;
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
  <div class="flex-auto h-0 overflow-y-auto" style="width: 100% !important;">
    <div class="w-full h-full flex flex-col items-stretch">
      <div class="mx-4 mb-3">
        <div class="rounded bg-slate-100 flex items-stretch w-full">
          <div class="p-3 border-r border-slate-200 flex gap-2 items-center">
            {#if visibleView != View.slices}
              <button
                class="btn text-slate-600 px-1 py-0.5 text-xs font-bold disabled:opacity-50"
                on:click={visibleView == View.specEditor
                  ? dismissSpecEditor
                  : dismissScoreFunctionEditor}
                ><Fa icon={faChevronLeft} class="inline mr-1" /> Back</button
              >
            {:else}
              <button
                class="btn btn-blue disabled:opacity-50"
                on:click={loadSlices}
                disabled={retrievingSlices || isTraining}
                >{retrievingSlices ? 'Loading...' : 'Find Slices'}</button
              >
            {/if}
          </div>
          {#if visibleView == View.slices && (!!searchStatus || loadingSliceStatus)}
            {@const samplerProgressMessage =
              !!searchStatus && !!searchStatus.status
                ? searchStatus.status_info?.message
                : 'Loading'}
            {@const samplerRunProgress =
              !!searchStatus && !!searchStatus.status
                ? searchStatus.status_info?.progress ?? null
                : null}
            <div
              role="status"
              class="ml-3 my-3 w-8 h-8 grow-0 shrink-0 self-center"
            >
              <svg
                aria-hidden="true"
                class="text-gray-200 animate-spin stroke-blue-600 w-8 h-8 align-middle"
                viewBox="-0.5 -0.5 99.5 99.5"
                xmlns="http://www.w3.org/2000/svg"
              >
                <ellipse
                  cx="50"
                  cy="50"
                  rx="45"
                  ry="45"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="10"
                />
                <path
                  d="M 50 5 A 45 45 0 0 1 95 50"
                  stroke-width="10"
                  stroke-linecap="round"
                  fill="none"
                />
              </svg>
            </div>
            <div class="mx-3 flex-auto whitespace-nowrap self-center">
              <div class="text-sm">
                {samplerProgressMessage ?? 'Waiting to load slices...'}
              </div>
              {#if samplerRunProgress != null}
                <div
                  class="w-full bg-slate-300 rounded-full h-1.5 mt-1 indigo:bg-slate-700"
                >
                  <div
                    class="bg-blue-600 h-1.5 rounded-full indigo:bg-indigo-200 duration-100"
                    style="width: {(samplerRunProgress * 100).toFixed(1)}%"
                  />
                </div>
              {/if}
            </div>
            <div class="py-3 mr-3 self-center">
              <button
                class="btn btn-blue disabled:opacity-50"
                on:click={stopFindingSlices}>Stop</button
              >
            </div>
          {:else}
            <div
              class="p-3 flex gap-4 items-center flex-auto whitespace-nowrap"
            >
              <button
                disabled={retrievingSlices || isTraining}
                class="btn {visibleView == View.specEditor
                  ? 'btn-dark-slate dark'
                  : 'hover:bg-slate-200'} px-0.5 py-0.5 disabled:opacity-50 text-left"
                style="max-width: 50%;"
                on:click={visibleView == View.specEditor
                  ? dismissSpecEditor
                  : () => (visibleView = View.specEditor)}
                ><div class="text-xs w-full">
                  <div class="text-slate-600 dark:text-slate-100 font-normal">
                    Slicing variables
                  </div>
                  <div class="font-bold truncate">{sliceSpec}</div>
                </div></button
              >

              {#if scoreFunctionSpec.length == 1}
                <button
                  disabled={retrievingSlices || isTraining}
                  class="btn {visibleView == View.scoreFunctionEditor
                    ? 'btn-dark-slate dark'
                    : 'hover:bg-slate-200'} px-0.5 py-0.5 disabled:opacity-50 text-left flex-auto w-min"
                  style="max-width: 50%;"
                  on:click={visibleView == View.scoreFunctionEditor
                    ? dismissScoreFunctionEditor
                    : () => (visibleView = View.scoreFunctionEditor)}
                  ><div class="text-xs w-full">
                    <div class="text-slate-600 dark:text-slate-100 font-normal">
                      Search criteria
                    </div>
                    <div class="font-bold truncate">
                      {scoreFunctionToString(scoreFunctionSpec[0])}
                    </div>
                  </div></button
                >
              {/if}
            </div>
          {/if}
        </div>
      </div>

      {#if visibleView == View.specEditor}
        <div class="mx-4 w-full">
          <SliceSpecEditor
            bind:sliceSpec
            {timestepDefinition}
            on:dismiss={dismissSpecEditor}
          />
        </div>
      {:else if visibleView == View.scoreFunctionEditor}
        <div class="w-full pb-4 mx-4">
          <ScoreFunctionPanel
            bind:scoreFunctionSpec
            bind:changesPending={scoreFunctionsChanged}
          />
        </div>
      {:else}
        <SliceSearchView
          {modelName}
          {modelsToShow}
          {scoreFunctionSpec}
          metricsToShow={[
            'Timesteps',
            'Trajectories',
            metricToShow,
            'Labels',
            'Predictions',
          ]}
          slices={slices ?? []}
          {baseSlice}
          savedSlices={savedSlices[sliceSpec] ?? []}
          bind:selectedSlices
          bind:sliceSpec
          {valueNames}
          runningSampler={!!searchStatus || loadingSliceStatus}
          {retrievingSlices}
          on:loadmore={() => (numSlicesToLoad += 20)}
          on:saveslice={(e) => saveSlice(e.detail)}
        />
      {/if}
    </div>
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
