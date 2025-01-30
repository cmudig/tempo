<script lang="ts">
  import {
    type Slice,
    type SliceFeatureBase,
    type SliceMetricInfo,
  } from './utils/slice.type';
  import { areSetsEqual, sortMetrics } from './utils/utils';
  import SliceTable from './slice_table/SliceTable.svelte';
  import { createEventDispatcher, getContext } from 'svelte';
  import { makeCategoricalColorScale, MetricColors } from '../colors';
  import type { Writable } from 'svelte/store';
  import { type ScoreFunction, scoreFunctionToString } from './scorefunctions';

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  const dispatch = createEventDispatcher();

  export let modelName: string | null = null;
  export let modelsToShow: string[] = [];

  export let scoreFunctionSpec: ScoreFunction[] = [];

  export let runningSampler = false;
  export let numSamples = 10;
  export let shouldCancel = false;
  export let retrievingSlices: boolean = false;

  export let sliceSpec: string = 'default';
  export let slices: Array<Slice> = [];

  export let baseSlice: Slice | null = null;
  export let sliceRequests: { [key: string]: SliceFeatureBase } = {};
  export let sliceRequestResults: { [key: string]: Slice } = {};
  let savedSliceResults: { [key: string]: Slice } = {};
  let savedSliceRequests: { [key: string]: SliceFeatureBase } = {};
  let savedSliceRequestResults: { [key: string]: Slice } = {};
  let customSliceResults: { [key: string]: Slice } = {};
  let customSliceRequests: { [key: string]: SliceFeatureBase } = {};
  let customSliceRequestResults: { [key: string]: Slice } = {};

  export let fixedFeatureOrder: Array<any> = [];

  export let showScores = false;
  export let positiveOnly = false;
  export let showSavedSlices = false;

  export let valueNames: {
    [key: string]: [any, { [key: string]: any }];
  } | null = {};

  export let customSlices: { [key: string]: SliceFeatureBase } = {};
  export let selectedSlices: SliceFeatureBase[] = [];
  export let savedSlices: { [key: string]: SliceFeatureBase } = {};

  export let groupedMetrics: boolean = false;
  export let metricsToShow: string[] | null = null;

  let metricNames: any[] = [];
  let metricGroups: string[] | null = null;
  let metricInfo: {
    [key: string]: SliceMetricInfo | { [key: string]: SliceMetricInfo };
  } = {};
  const SearchCriteriaMetricGroup = 'Search Criteria';

  let allSlices: Array<Slice> = [];
  $: allSlices = [...(!!baseSlice ? [baseSlice] : []), ...slices];

  $: if (allSlices.length > 0) {
    let testSlice = allSlices.find((s) => !s.isEmpty);
    if (!testSlice) testSlice = allSlices[0];
    updateMetricInfo(testSlice, metricsToShow);
  } else {
    groupedMetrics = false;
    metricGroups = null;
    metricNames = [];
    metricInfo = {};
  }

  $: requestSliceScores(sliceRequests, modelName, modelsToShow).then(
    (r) => (sliceRequestResults = r)
  );
  $: requestSliceScores(savedSlices, modelName, modelsToShow).then(
    (r) => (savedSliceResults = r)
  );
  $: requestSliceScores(savedSliceRequests, modelName, modelsToShow).then(
    (r) => (savedSliceRequestResults = r)
  );
  $: requestSliceScores(customSlices, modelName, modelsToShow).then(
    (r) => (customSliceResults = r)
  );
  $: requestSliceScores(customSliceRequests, modelName, modelsToShow).then(
    (r) => (customSliceRequestResults = r)
  );

  let oldSliceSpec = sliceSpec;
  $: if (sliceSpec !== oldSliceSpec) {
    sliceRequests = {};
    savedSliceRequests = {};
    savedSlices = {};
  }

  async function requestSliceScores(
    requests: {
      [key: string]: SliceFeatureBase;
    },
    baseModel: string | null,
    models: string[]
  ): Promise<{ [key: string]: Slice }> {
    if (Object.keys(requests).length == 0 || baseModel == null) {
      return {};
    }
    try {
      let results = await (
        await fetch(
          import.meta.env.BASE_URL +
            `/datasets/${$currentDataset}/slices/${baseModel}/score`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              slices: requests,
              score_function_spec: scoreFunctionSpec,
              variable_spec_name: sliceSpec,
              model_names: models,
            }),
          }
        )
      ).json();
      return results.slices;
    } catch (e) {
      console.log('error calculating slice requests:', e);
    }
    return {};
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

  function getMetric<T>(
    metrics: { [key: string]: T | { [key: string]: T } },
    key: string | [string, string]
  ): T | undefined {
    if (Array.isArray(key)) {
      if (!metrics[key[0]]) return undefined;
      return (metrics[key[0]] as { [key: string]: T })[key[1]]! as T;
    } else return metrics[key]! as T;
  }

  function updateMetricInfo(testSlice: Slice, showingMetrics: string[] | null) {
    // tabulate metric names and normalize
    if (!!testSlice.metrics) {
      let newMetricNames = Object.keys(testSlice.metrics).filter(
        (n) => n != SearchCriteriaMetricGroup
      );
      if (
        newMetricNames.length > 0 &&
        !testSlice.metrics[newMetricNames[0]].hasOwnProperty('type')
      ) {
        // grouped metrics
        groupedMetrics = true;
        if (
          (metricGroups !== null &&
            !areSetsEqual(new Set(metricGroups), new Set(newMetricNames))) ||
          metricGroups == null
        ) {
          newMetricNames.sort();
          metricGroups = newMetricNames;
        }
        metricNames = metricGroups
          .map((g) =>
            Object.keys(testSlice!.metrics![g])
              .filter(
                (m) =>
                  g == SearchCriteriaMetricGroup ||
                  !showingMetrics ||
                  showingMetrics.includes(m)
              )
              .sort(sortMetrics)
              .map((m) => [g, m])
          )
          .flat();
      } else {
        groupedMetrics = false;
        metricGroups = null;
        if (!areSetsEqual(new Set(metricNames), new Set(newMetricNames))) {
          metricNames = newMetricNames.filter(
            (m) => !showingMetrics || showingMetrics.includes(m)
          );
          metricNames.sort(sortMetrics);
        }
      }
    }

    let oldMetricInfo = metricInfo;
    let testMetrics = testSlice.metrics;
    metricInfo = {};
    metricNames.forEach((n) => {
      let met = getMetric(testMetrics, n);
      if (!met) return;

      let newInfo: SliceMetricInfo = { visible: true };

      if (met.type == 'binary' || met.type == 'count') {
        let maxScore =
          met.type == 'count'
            ? allSlices.reduce(
                (curr, next) =>
                  Math.max(curr, getMetric(next.metrics!, n!)?.mean ?? -1e9),
                -1e9
              ) + 0.01
            : 1;
        let minScore =
          allSlices.reduce(
            (curr, next) =>
              Math.min(curr, getMetric(next.metrics!, n)?.mean ?? 1e9),
            1e9
          ) - 0.01;
        newInfo.scale = (v: number) => v / maxScore;
      } else if (met.type == 'numeric') {
        let maxScore =
          allSlices.reduce(
            (curr, next) =>
              Math.max(curr, getMetric(next.metrics!, n)?.value ?? -1e9),
            -1e9
          ) + 0.01;
        newInfo.scale = (v: number) => v / Math.max(1, maxScore);
      } else if (met.type == 'categorical') {
        let uniqueKeys: Set<string> = new Set();
        allSlices.forEach((s) =>
          Object.keys(getMetric(s.metrics!, n)?.counts ?? {}).forEach((v) =>
            uniqueKeys.add(v)
          )
        );
        let order = Array.from(uniqueKeys).sort();
        // order.sort((a, b) => met.counts![b] - met.counts![a]);
        newInfo.order = order;
        newInfo.colorScale = makeCategoricalColorScale(
          MetricColors[groupedMetrics ? n[1] : n]
        );
      }
      newInfo.visible = (
        getMetric<SliceMetricInfo>(oldMetricInfo, n) || { visible: true }
      ).visible;
      newInfo.color =
        MetricColors[groupedMetrics ? n[1] : n] ?? MetricColors.Accuracy;

      if (groupedMetrics) {
        if (!metricInfo[n[0]]) metricInfo[n[0]] = {};
        (metricInfo[n[0]] as { [key: string]: SliceMetricInfo })[n[1]] =
          newInfo;
      } else metricInfo[n] = newInfo;
    });
  }

  let searchViewHeader: HTMLElement;
  let samplerPanel: HTMLElement;

  $: if (!!searchViewHeader && !!samplerPanel) {
    samplerPanel.style.top = `${searchViewHeader.clientHeight}px`;
    searchViewHeader.addEventListener('resize', () => {
      samplerPanel.style.top = `${searchViewHeader.clientHeight}px`;
    });
  }

  let showDetailLoadingMessage: boolean = false;
  let showDetailLoadingTimeout: NodeJS.Timeout | null = null;
  $: if (retrievingSlices) {
    if (!showDetailLoadingTimeout)
      showDetailLoadingTimeout = setTimeout(() => {
        showDetailLoadingMessage = true;
        showDetailLoadingTimeout = null;
      }, 10000);
  } else showDetailLoadingMessage = false;
</script>

<div
  class="flex-auto min-h-0 overflow-auto relative"
  style="min-height: 400px;"
>
  <div class="inline-block min-w-full">
    {#if !!baseSlice}
      <div class="bg-white sticky top-0 z-10 px-4" bind:this={searchViewHeader}>
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
          bind:metricGroups
          searchCriteriaName={scoreFunctionSpec.length > 0
            ? scoreFunctionToString(scoreFunctionSpec[0])
            : null}
          allowFavorite={true}
          allowMultiselect={false}
          metricInfo={(n) => getMetric(metricInfo, n)}
          metricGetter={(s, name) => getMetric(s.metrics, name)}
          bind:metricNames
          allowShowScores={false}
          showCheckboxes={false}
          allowSearch={false}
          on:saveslice
        />
      </div>
      {#if showSavedSlices}
        <div class="bg-white px-4">
          <SliceTable
            slices={Object.keys(savedSlices).map((sr) =>
              Object.assign(
                savedSliceResults[sr] ?? {
                  feature: savedSlices[sr],
                  scoreValues: {},
                  metrics: {},
                  stringRep: sr,
                },
                { stringRep: 'saved_' + sr }
              )
            )}
            {savedSlices}
            bind:selectedSlices
            showHeader={false}
            bind:sliceRequests={savedSliceRequests}
            bind:sliceRequestResults={savedSliceRequestResults}
            searchCriteriaName={scoreFunctionSpec.length > 0
              ? scoreFunctionToString(scoreFunctionSpec[0])
              : null}
            {positiveOnly}
            {valueNames}
            {allowedValues}
            bind:metricGroups
            allowFavorite={true}
            allowMultiselect={false}
            metricInfo={(n) => getMetric(metricInfo, n)}
            metricGetter={(s, name) => getMetric(s.metrics, name)}
            bind:metricNames
            allowShowScores={false}
            showCheckboxes={false}
            allowSearch={false}
            on:saveslice={(e) => {
              dispatch(
                'saveslice',
                Object.assign(e.detail, {
                  stringRep: e.detail.stringRep.replace('saved_', ''),
                })
              );
            }}
          />
        </div>
      {:else}
        {#if Object.keys(customSlices).length > 0}
          <div class="bg-white px-4">
            <SliceTable
              slices={Object.keys(customSlices).map(
                (sr) =>
                  customSliceResults[sr] ?? {
                    feature: customSlices[sr],
                    scoreValues: {},
                    metrics: {},
                    stringRep: sr,
                  }
              )}
              {savedSlices}
              bind:selectedSlices
              showHeader={false}
              bind:sliceRequests={customSliceRequests}
              bind:sliceRequestResults={customSliceRequestResults}
              searchCriteriaName={scoreFunctionSpec.length > 0
                ? scoreFunctionToString(scoreFunctionSpec[0])
                : null}
              {positiveOnly}
              {valueNames}
              {allowedValues}
              custom
              bind:metricGroups
              allowFavorite={true}
              allowMultiselect={false}
              metricInfo={(n) => getMetric(metricInfo, n)}
              metricGetter={(s, name) => getMetric(s.metrics, name)}
              bind:metricNames
              allowShowScores={false}
              showCheckboxes={false}
              allowSearch={false}
              on:edit={(e) => {
                customSlices = {
                  ...customSlices,
                  [e.detail.stringRep]: e.detail.feature,
                };
                console.log('custom slices:', customSlices);
              }}
              on:saveslice={(e) => {
                dispatch(
                  'saveslice',
                  Object.assign(e.detail, {
                    stringRep: e.detail.stringRep.replace('saved_', ''),
                  })
                );
              }}
              on:delete={(e) => {
                customSlices = Object.fromEntries(
                  Object.entries(customSlices).filter(([k, v]) => k != e.detail)
                );
                console.log('custom slices', customSlices, e.detail);
              }}
            />
          </div>
        {/if}
        {#if !!slices && slices.length > 0}
          <div class="px-4 mb-2 w-full">
            <div
              class="w-full px-4 py-2 bg-slate-100 text-slate-700 text-sm rounded z-10"
            >
              Search Results
            </div>
          </div>
        {/if}
        <div class="flex-auto relative w-full px-4">
          {#if !!slices && slices.length > 0}
            <div class="w-full min-h-0" class:disable-div={runningSampler}>
              <SliceTable
                {slices}
                {savedSlices}
                bind:selectedSlices
                bind:sliceRequests
                bind:sliceRequestResults
                searchCriteriaName={scoreFunctionSpec.length > 0
                  ? scoreFunctionToString(scoreFunctionSpec[0])
                  : null}
                {positiveOnly}
                {valueNames}
                {allowedValues}
                showHeader={false}
                bind:metricGroups
                allowFavorite={true}
                allowMultiselect={false}
                metricInfo={(n) => getMetric(metricInfo, n)}
                metricGetter={(s, name) => getMetric(s.metrics, name)}
                bind:metricNames
                allowShowScores={false}
                showCheckboxes={false}
                allowSearch={false}
                on:saveslice
              />
            </div>
          {:else}
            <div
              class="w-full mt-6 flex-auto min-h-0 flex flex-col items-center justify-center text-slate-500"
            >
              <div>No subgroups yet!</div>
            </div>
          {/if}
        </div>
      {/if}
    {:else if !retrievingSlices}
      <div
        class="w-full flex-auto min-h-0 mt-6 flex flex-col items-center justify-center text-slate-500"
      >
        <div>No subgroups yet!</div>
      </div>
    {/if}
    {#if retrievingSlices}
      <div
        class="absolute top-0 left-0 bg-white/80 w-full h-full flex flex-col items-center justify-center z-20"
      >
        <div>Retrieving subgroups...</div>
        {#if showDetailLoadingMessage}
          <div class="mt-2 text-sm text-slate-500">
            It may take several minutes to load subgrouping variables the first
            time you visit.
          </div>
        {/if}
        <div role="status" class="w-8 h-8 grow-0 shrink-0 mt-4">
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
      </div>
    {/if}
  </div>
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
