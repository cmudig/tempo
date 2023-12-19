<script lang="ts">
  import {
    SliceSearchControl,
    type Slice,
    type SliceFeatureBase,
    type SliceMetric,
    SliceControlStrings,
  } from './slices/utils/slice.type';
  import SliceRow from './slices/slice_table/SliceRow.svelte';
  import Hoverable from './slices/utils/Hoverable.svelte';
  import {
    faAngleLeft,
    faAngleRight,
    faEye,
    faEyeSlash,
    faGripLinesVertical,
    faMinus,
    faPencil,
    faPlus,
    faScaleBalanced,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { areObjectsEqual, areSetsEqual } from './slices/utils/utils';
  import { TableWidths } from './slices/slice_table/tablewidths';
  import SliceTable from './slices/slice_table/SliceTable.svelte';
  import SliceFeatureEditor from './slices/slice_table/SliceFeatureEditor.svelte';
  import { featureToString, parseFeature } from './slices/utils/slice_parsing';
  import SliceFeature from './slices/slice_table/SliceFeature.svelte';
  import ActionMenuButton from './slices/utils/ActionMenuButton.svelte';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  export let modelNames: string[] = [];

  export let runningSampler = false;
  export let numSamples = 10;
  export let shouldCancel = false;
  export let samplerRunProgress: number | null = null;
  export let samplerProgressMessage: string | null = null;
  export let retrievingSlices: boolean = false;

  export let slices: Array<Slice> = [];

  export let baseSlice: Slice | null = null;
  export let sliceRequests: { [key: string]: SliceFeature } = {};
  export let sliceRequestResults: { [key: string]: Slice } = {};

  export let scoreWeights: any = {};

  export let fixedFeatureOrder: Array<any> = [];
  export let searchBaseSlice: any = null;

  export let showScores = false;
  export let positiveOnly = false;

  export let valueNames: {
    [key: string]: [any, { [key: string]: any }];
  } | null = {};

  export let enabledSliceControls: { [key in SliceSearchControl]?: boolean } =
    {};
  export let containsSlice: any = {};
  export let containedInSlice: any = {};
  export let similarToSlice: any = {};
  export let subsliceOfSlice: any = {};

  export let selectedSlices: Slice[] = [];
  export let savedSlices: Slice[] = [];

  let controlFeatures: { [key in SliceSearchControl]?: any };
  $: controlFeatures = {
    [SliceSearchControl.containsSlice]: containsSlice,
    [SliceSearchControl.containedInSlice]: containedInSlice,
    [SliceSearchControl.similarToSlice]: similarToSlice,
    [SliceSearchControl.subsliceOfSlice]: subsliceOfSlice,
  };

  let metricNames: string[] = [];
  let metricInfo = {};
  let scoreNames: string[] = [];
  let scoreWidthScalers: { [key: string]: (v: number) => number } = {};

  let allSlices: Array<Slice> = [];
  $: allSlices = [...(!!baseSlice ? [baseSlice] : []), ...slices];

  $: if (allSlices.length > 0) {
    let testSlice = allSlices.find((s) => !s.isEmpty);
    if (!testSlice) testSlice = allSlices[0];

    // tabulate score names and normalize
    let newScoreNames = Object.keys(testSlice.scoreValues);
    if (!areSetsEqual(new Set(scoreNames), new Set(newScoreNames))) {
      scoreNames = newScoreNames;
      scoreNames.sort();
    }

    scoreWidthScalers = {};
    scoreNames.forEach((n) => {
      let maxScore =
        allSlices.reduce(
          (curr, next) => Math.max(curr, next.scoreValues[n]),
          -1e9
        ) + 0.01;
      let minScore =
        allSlices.reduce(
          (curr, next) => Math.min(curr, next.scoreValues[n]),
          1e9
        ) - 0.01;
      scoreWidthScalers[n] = (v: number) =>
        (v - minScore) / (maxScore - minScore);
    });
    console.log('score names:', testSlice, scoreNames);

    // tabulate metric names and normalize
    if (!!testSlice.metrics) {
      let newMetricNames = Object.keys(testSlice.metrics);
      if (!areSetsEqual(new Set(metricNames), new Set(newMetricNames))) {
        metricNames = newMetricNames;
        metricNames.sort();
      }
      updateMetricInfo(testSlice.metrics);
    }
  } else {
    scoreNames = [];
    scoreWidthScalers = {};
    metricNames = [];
    metricInfo = {};
  }

  $: requestSliceScores(sliceRequests);

  async function requestSliceScores(requests: { [key: string]: SliceFeature }) {
    try {
      let results = await (
        await fetch(`/slices/${modelNames.join(',')}/score`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sliceRequests: requests }),
        })
      ).json();
      sliceRequestResults = results.sliceRequestResults;
    } catch (e) {
      console.log('error calculating slice requests:', e);
    }
  }

  let allowedValues: { [key: string]: (string | number)[] } | null = null;
  $: if (!!valueNames) {
    allowedValues = {};
    Object.entries(valueNames).forEach((item) => {
      allowedValues![item[1][0]] = Object.values(item[1][1]);
    });
  } else {
    allowedValues = null;
  }

  function updateMetricInfo(testMetrics: { [key: string]: SliceMetric }) {
    let oldMetricInfo = metricInfo;
    metricInfo = {};
    metricNames.forEach((n) => {
      if (testMetrics[n].type == 'binary' || testMetrics[n].type == 'count') {
        let maxScore =
          testMetrics[n].type == 'count'
            ? allSlices.reduce(
                (curr, next) => Math.max(curr, next.metrics[n].mean),
                -1e9
              ) + 0.01
            : 1;
        let minScore =
          allSlices.reduce(
            (curr, next) => Math.min(curr, next.metrics[n].mean),
            1e9
          ) - 0.01;
        metricInfo[n] = { scale: (v: number) => v / maxScore };
      } else if (testMetrics[n].type == 'categorical') {
        let uniqueKeys: Set<string> = new Set();
        allSlices.forEach((s) =>
          Object.keys(s.metrics[n].counts).forEach((v) => uniqueKeys.add(v))
        );
        let order = Array.from(uniqueKeys);
        order.sort(
          (a, b) => testMetrics[n].counts[b] - testMetrics[n].counts[a]
        );
        metricInfo[n] = { order };
      } else {
        metricInfo[n] = {};
      }
      metricInfo[n].visible = (oldMetricInfo[n] || { visible: true }).visible;
    });
    console.log('metric info:', metricInfo, testMetrics);
  }

  let searchViewHeader: HTMLElement;
  let samplerPanel: HTMLElement;

  $: if (!!searchViewHeader && !!samplerPanel) {
    samplerPanel.style.top = `${searchViewHeader.clientHeight}px`;
    searchViewHeader.addEventListener('resize', () => {
      samplerPanel.style.top = `${searchViewHeader.clientHeight}px`;
    });
  }

  function toggleSliceControl(
    flagField: SliceSearchControl,
    value: boolean | null = null
  ) {
    let newControls = Object.assign({}, enabledSliceControls);
    if (value == null) newControls[flagField] = !newControls[flagField];
    else newControls[flagField] = value;
    enabledSliceControls = newControls;
    if (newControls[flagField] && controlFeatures[flagField].type == 'base')
      editingControl = flagField;
  }

  let editingControl: SliceSearchControl | null = null;

  function updateEditingControl(
    control: SliceSearchControl,
    feature: SliceFeatureBase
  ) {
    if (control == SliceSearchControl.containsSlice) containsSlice = feature;
    else if (control == SliceSearchControl.containedInSlice)
      containedInSlice = feature;
    else if (control == SliceSearchControl.similarToSlice)
      similarToSlice = feature;
    else if (control == SliceSearchControl.subsliceOfSlice)
      subsliceOfSlice = feature;
    controlFeatures[control] = feature;
  }
</script>

<div class="w-full h-full flex flex-col">
  <div class="mb-2">
    {#if runningSampler}
      <div class="flex items-center py-3">
        <button
          class="ml-2 mr-4 btn btn-blue disabled:opacity-50"
          on:click={() => dispatch('cancel')}>Stop</button
        >
        {#if samplerRunProgress == null}
          <div role="status" class="w-8 h-8 grow-0 shrink-0 mr-2">
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
        {/if}
        <div class="flex-auto">
          <div class="text-sm">
            {samplerProgressMessage ?? 'Loading slices...'}
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
      </div>
    {:else}
      <div class="flex pl-3 items-stretch">
        <div class="flex-1 w-0 pr-3">
          <div class="flex items-center whitespace-nowrap py-3 gap-3">
            <button
              class="btn btn-blue"
              on:click={() => dispatch('load')}
              disabled={retrievingSlices}
              >{retrievingSlices ? 'Loading...' : 'Find Slices'}</button
            >
            <ActionMenuButton
              buttonClass="btn btn-slate"
              buttonStyle="padding-left: 1rem;"
              buttonTitle="Add a filter option"
            >
              <span slot="button-content"
                ><Fa icon={faPlus} class="inline mr-1" />
                Filter</span
              >
              <div slot="options">
                {#each Object.values(SliceSearchControl) as control}
                  {#if !enabledSliceControls[control]}
                    <a
                      href="#"
                      tabindex="0"
                      role="menuitem"
                      on:click={() => toggleSliceControl(control)}
                      >{SliceControlStrings[control]} Slice</a
                    >
                  {/if}
                {/each}
              </div>
            </ActionMenuButton>
          </div>
        </div>
      </div>
    {/if}
  </div>
  {#if !!slices && slices.length > 0}
    <div class="flex-auto min-h-0 overflow-auto">
      <div class="search-view-header bg-white" bind:this={searchViewHeader}>
        <SliceTable
          slices={[]}
          {savedSlices}
          bind:selectedSlices
          {baseSlice}
          bind:sliceRequests
          bind:sliceRequestResults
          {positiveOnly}
          {valueNames}
          {allowedValues}
          bind:metricInfo
          bind:metricNames
          bind:scoreNames
          bind:scoreWidthScalers
          bind:showScores
          on:newsearch={(e) => {
            updateEditingControl(e.detail.type, e.detail.base_slice);
            toggleSliceControl(e.detail.type, true);
          }}
          on:saveslice
        />
      </div>
      {#if Object.values(enabledSliceControls).some((v) => !!v)}
        <div
          class="sampler-panel w-full mb-2 bg-white"
          bind:this={samplerPanel}
        >
          <div
            class="mx-2 pt-3 rounded transition-colors duration-300 bg-slate-200 text-gray-700 border-slate-200 border-2 box-border"
          >
            <div class="flex pl-3 items-stretch">
              <div class="flex-1 w-0 pr-3">
                {#each Object.values(SliceSearchControl) as control}
                  {#if enabledSliceControls[control]}
                    <div class="flex items-center pb-3 w-full">
                      <button
                        style="padding-left: 1rem;"
                        class="ml-1 btn btn-dark-slate flex-0 mr-3 whitespace-nowrap"
                        on:click={() => toggleSliceControl(control)}
                        ><Fa icon={faMinus} class="inline mr-1" />
                        {SliceControlStrings[control]}</button
                      >
                      {#if editingControl == control}
                        <SliceFeatureEditor
                          featureText={featureToString(
                            controlFeatures[control],
                            false,
                            positiveOnly
                          )}
                          {positiveOnly}
                          {allowedValues}
                          on:cancel={(e) => {
                            editingControl = null;
                          }}
                          on:save={(e) => {
                            let newFeature = parseFeature(
                              e.detail,
                              allowedValues
                            );
                            updateEditingControl(control, newFeature);
                            editingControl = null;
                          }}
                        />
                      {:else}
                        <div
                          class="overflow-x-auto whitespace-nowrap"
                          style="flex: 0 1 auto;"
                        >
                          <SliceFeature
                            feature={controlFeatures[control]}
                            currentFeature={controlFeatures[control]}
                            canToggle={false}
                            {positiveOnly}
                          />
                        </div>
                        <button
                          class="bg-transparent hover:opacity-60 pr-1 pl-2 py-3 text-slate-600"
                          on:click={() => {
                            editingControl = control;
                          }}
                          title="Modify the slice definition"
                          ><Fa icon={faPencil} /></button
                        >
                      {/if}
                    </div>
                  {/if}
                {/each}
              </div>
            </div>
          </div>
        </div>
      {/if}
      <div class="flex-1 min-h-0" class:disable-div={runningSampler}>
        <SliceTable
          {slices}
          {savedSlices}
          bind:selectedSlices
          bind:sliceRequests
          bind:sliceRequestResults
          {positiveOnly}
          {valueNames}
          {allowedValues}
          showHeader={false}
          bind:metricInfo
          bind:metricNames
          bind:scoreNames
          bind:scoreWidthScalers
          bind:showScores
          on:newsearch={(e) => {
            updateEditingControl(e.detail.type, e.detail.base_slice);
            toggleSliceControl(e.detail.type, true);
          }}
          on:saveslice
        />

        {#if slices.length > 0}
          <div class="mt-2">
            <button
              class="btn btn-blue disabled:opacity-50"
              on:click={() => dispatch('loadmore')}>Load More</button
            >
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <div
      class="w-full flex-auto min-h-0 flex flex-col items-center justify-center text-slate-500"
    >
      {#if retrievingSlices}
        <div>Retrieving slices...</div>
        <div role="status" class="w-8 h-8 grow-0 shrink-0 mt-2">
          <svg
            aria-hidden="true"
            class="text-gray-200 animate-spin stroke-gray-600 w-8 h-8 align-middle"
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
      {:else}
        <div>No slices yet!</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .search-view-header {
    position: sticky;
    top: 0;
    z-index: 1;
  }

  .sampler-panel {
    position: sticky;
    left: 0;
    z-index: 1;
  }
</style>
