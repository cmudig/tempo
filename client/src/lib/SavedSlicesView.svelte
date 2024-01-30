<script lang="ts">
  import { onMount } from 'svelte';
  import SliceTable from './slices/slice_table/SliceTable.svelte';
  import type {
    Slice,
    SliceFeatureBase,
    SliceMetricInfo,
  } from './slices/utils/slice.type';
  import { areSetsEqual, sortMetrics } from './slices/utils/utils';

  export let savedSlices: { [key: string]: SliceFeatureBase[] } = {};
  export let sliceSpec: string = 'default';

  export let selectedSlices: SliceFeatureBase[] = [];
  export let sliceRequests: { [key: string]: SliceFeatureBase } = {};
  export let sliceRequestResults: { [key: string]: Slice } = {};

  export let scoreWeights: any = {};

  export let fixedFeatureOrder: Array<any> = [];
  export let searchBaseSlice: any = null;

  export let showScores = false;
  export let positiveOnly = false;
  export let metricToShow: string | null = null;

  export let valueNames: {
    [key: string]: [any, { [key: string]: any }];
  } | null = {};

  let baseSlice: Slice | null = null;
  let slices: Slice[] = [];
  let loadingSlices: boolean = false;

  $: if (!!sliceSpec) loadSlices(sliceSpec);

  async function loadSlices(spec: string) {
    try {
      loadingSlices = true;
      let response = await fetch('/slices/score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sliceSpec: spec,
          sliceRequests: {
            base: { type: 'base' },
            ...Object.fromEntries(
              (savedSlices[spec] ?? []).map((s, i) => [`${i}`, s])
            ),
          },
        }),
      });
      let result = await response.json();
      baseSlice = result.sliceRequestResults.base;
      slices = savedSlices[spec].map(
        (_: any, i: number) => result.sliceRequestResults[`${i}`]
      );
    } catch (e) {
      console.error('error loading saved slices:', e);
    }
    loadingSlices = false;
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

  let groupedMetrics = false;
  let metricNames: any[] = [];
  let metricGroups: string[] | null = null;
  let metricInfo: {
    [key: string]: SliceMetricInfo | { [key: string]: SliceMetricInfo };
  } = {};
  let scoreNames: string[] = [];
  let scoreWidthScalers: { [key: string]: (v: number) => number } = {};

  let allSlices: Array<Slice> = [];
  $: allSlices = [...(!!baseSlice ? [baseSlice] : []), ...slices];

  $: if (allSlices.length > 0) {
    let testSlice = allSlices.find((s) => !s.isEmpty);
    if (!testSlice) testSlice = allSlices[0];
    updateMetricInfo(testSlice);
  } else {
    scoreNames = [];
    scoreWidthScalers = {};
    groupedMetrics = false;
    metricGroups = null;
    metricNames = [];
    metricInfo = {};
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

  function updateMetricInfo(testSlice: Slice) {
    if (!!scoreWeights) scoreNames = Object.keys(scoreWeights).sort();
    else scoreNames = Object.keys(testSlice.scoreValues).sort();

    // tabulate metric names and normalize
    if (!!testSlice.metrics) {
      let newMetricNames = Object.keys(testSlice.metrics);
      if (
        newMetricNames.length > 0 &&
        !testSlice.metrics[newMetricNames[0]].hasOwnProperty('type')
      ) {
        // grouped metrics
        groupedMetrics = true;
        if (
          metricGroups !== null &&
          !areSetsEqual(new Set(metricGroups), new Set(newMetricNames))
        ) {
          metricGroups = newMetricNames;
          metricGroups.sort();
        } else if (metricGroups === null) metricGroups = newMetricNames;
        metricNames = metricGroups
          .map((g) =>
            Object.keys(testSlice!.metrics![g])
              .filter((m) => !metricToShow || metricToShow == m)
              .sort(sortMetrics)
              .map((m) => [g, m])
          )
          .flat();
      } else {
        groupedMetrics = false;
        metricGroups = null;
        if (!areSetsEqual(new Set(metricNames), new Set(newMetricNames))) {
          metricNames = newMetricNames.filter(
            (m) => !metricToShow || metricToShow == m
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
                  Math.max(curr, getMetric(next.metrics!, n)!.mean!),
                -1e9
              ) + 0.01
            : 1;
        let minScore =
          allSlices.reduce(
            (curr, next) => Math.min(curr, getMetric(next.metrics!, n)!.mean!),
            1e9
          ) - 0.01;
        newInfo.scale = (v: number) => v / maxScore;
      } else if (met.type == 'categorical') {
        let uniqueKeys: Set<string> = new Set();
        allSlices.forEach((s) =>
          Object.keys(s.metrics![n].counts!).forEach((v) => uniqueKeys.add(v))
        );
        let order = Array.from(uniqueKeys);
        order.sort((a, b) => met.counts![b] - met.counts![a]);
        newInfo.order = order;
      }
      newInfo.visible = (
        getMetric<SliceMetricInfo>(oldMetricInfo, n) || { visible: true }
      ).visible;

      if (groupedMetrics) {
        if (!metricInfo[n[0]]) metricInfo[n[0]] = {};
        (metricInfo[n[0]] as { [key: string]: SliceMetricInfo })[n[1]] =
          newInfo;
      } else metricInfo[n] = newInfo;
    });
  }
</script>

{#if loadingSlices}
  <div class="w-full h-full flex flex-col items-center justify-center">
    <div>Retrieving slices...</div>
    <div role="status" class="w-8 h-8 grow-0 shrink-0 mt-4">
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
  </div>
{:else}
  <div class="w-full h-full overflow-y-auto p-2">
    <div class="search-view-header bg-white">
      <SliceTable
        slices={[]}
        savedSlices={savedSlices[sliceSpec]}
        bind:selectedSlices
        {baseSlice}
        bind:sliceRequests
        bind:sliceRequestResults
        {positiveOnly}
        {valueNames}
        {allowedValues}
        bind:metricGroups
        allowFavorite={true}
        allowMultiselect={false}
        metricInfo={(n) => getMetric(metricInfo, n)}
        metricGetter={(s, name) => getMetric(s.metrics, name)}
        bind:metricNames
        bind:scoreNames
        bind:scoreWidthScalers
        allowShowScores={false}
        showCheckboxes={false}
        on:saveslice
      />
    </div>
    <SliceTable
      {slices}
      savedSlices={savedSlices[sliceSpec]}
      bind:selectedSlices
      bind:sliceRequests
      bind:sliceRequestResults
      {positiveOnly}
      {valueNames}
      {allowedValues}
      bind:metricGroups
      showHeader={false}
      allowFavorite={true}
      allowMultiselect={false}
      metricInfo={(n) => getMetric(metricInfo, n)}
      metricGetter={(s, name) => getMetric(s.metrics, name)}
      bind:metricNames
      bind:scoreNames
      bind:scoreWidthScalers
      allowShowScores={false}
      showCheckboxes={false}
      on:saveslice
    />
  </div>
{/if}

<style>
  .search-view-header {
    position: sticky;
    top: 0;
    z-index: 1;
  }
</style>
