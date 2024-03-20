<script lang="ts">
  import {
    SliceSearchControl,
    type Slice,
    type SliceFeatureBase,
    type SliceMetric,
    SliceControlStrings,
    type SliceMetricInfo,
  } from './slices/utils/slice.type';
  import SliceRow from './slices/slice_table/SliceRow.svelte';
  import Hoverable from './slices/utils/Hoverable.svelte';
  import {
    faFilter,
    faMinus,
    faPencil,
    faScaleBalanced,
    faSearch,
    faWrench,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import {
    areObjectsEqual,
    areSetsEqual,
    sortMetrics,
    sortScoreNames,
  } from './slices/utils/utils';
  import { TableWidths } from './slices/slice_table/tablewidths';
  import SliceTable from './slices/slice_table/SliceTable.svelte';
  import SliceFeatureEditor from './slices/slice_table/SliceFeatureEditor.svelte';
  import { featureToString, parseFeature } from './slices/utils/slice_parsing';
  import SliceFeature from './slices/slice_table/SliceFeature.svelte';
  import ActionMenuButton from './slices/utils/ActionMenuButton.svelte';
  import { createEventDispatcher } from 'svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import ScoreWeightMenu from './slices/utils/ScoreWeightMenu.svelte';
  import SliceSpecEditor from './SliceSpecEditor.svelte';
  import { MetricColors } from './colors';

  const dispatch = createEventDispatcher();

  export let modelNames: string[] = [];

  export let runningSampler = false;
  export let numSamples = 10;
  export let shouldCancel = false;
  export let samplingStatusOverview: string | null = null;
  export let samplerRunProgress: number | null = null;
  export let samplerProgressMessage: string | null = null;
  export let retrievingSlices: boolean = false;

  export let timestepDefinition: string = '';
  export let sliceSpec: string = 'default';
  export let slices: Array<Slice> = [];

  export let baseSlice: Slice | null = null;
  export let sliceRequests: { [key: string]: SliceFeatureBase } = {};
  export let sliceRequestResults: { [key: string]: Slice } = {};
  let savedSliceResults: { [key: string]: Slice } = {};
  let savedSliceRequests: { [key: string]: SliceFeatureBase } = {};
  let savedSliceRequestResults: { [key: string]: Slice } = {};

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

  export let selectedSlices: SliceFeatureBase[] = [];
  export let savedSlices: { [key: string]: SliceFeatureBase } = {};

  export let groupedMetrics: boolean = false;
  export let metricsToShow: string[] | null = null;

  let controlFeatures: { [key in SliceSearchControl]?: any };
  $: controlFeatures = {
    [SliceSearchControl.containsSlice]: containsSlice,
    [SliceSearchControl.containedInSlice]: containedInSlice,
    [SliceSearchControl.similarToSlice]: similarToSlice,
    [SliceSearchControl.subsliceOfSlice]: subsliceOfSlice,
  };

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
    updateMetricInfo(testSlice, metricsToShow);
  } else {
    scoreNames = [];
    scoreWidthScalers = {};
    groupedMetrics = false;
    metricGroups = null;
    metricNames = [];
    metricInfo = {};
  }

  $: requestSliceScores(sliceRequests, modelNames).then(
    (r) => (sliceRequestResults = r)
  );
  $: requestSliceScores(savedSlices, modelNames).then(
    (r) => (savedSliceResults = r)
  );
  $: requestSliceScores(savedSliceRequests, modelNames).then(
    (r) => (savedSliceRequestResults = r)
  );

  async function requestSliceScores(
    requests: {
      [key: string]: SliceFeatureBase;
    },
    models: string[]
  ) {
    try {
      let results = await (
        await fetch(`/slices/${models.join(',')}/score`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sliceRequests: requests, sliceSpec }),
        })
      ).json();
      return results.sliceRequestResults;
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
    if (!!scoreWeights)
      scoreNames = Object.keys(scoreWeights).sort(sortScoreNames);
    else scoreNames = Object.keys(testSlice.scoreValues).sort(sortScoreNames);

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
              .filter((m) => !showingMetrics || showingMetrics.includes(m))
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
        newInfo.scale = (v: number) => v / maxScore;
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

  let specEditorVisible: boolean = false;

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

<div class="w-full h-full flex flex-col">
  <div class="mb-3">
    <div class="rounded bg-slate-100 flex items-center items-stretch">
      <div class="p-3 border-r border-slate-200 flex gap-2 items-center">
        {#if !!samplingStatusOverview}<div
            class="text-xs w-max"
            style="max-width: 240px;"
          >
            {samplingStatusOverview}
          </div>{/if}
        {#if runningSampler}
          <button
            class="btn btn-blue disabled:opacity-50"
            on:click={() => dispatch('cancel')}>Stop</button
          >
        {:else}
          <button
            class="btn btn-blue"
            on:click={() => dispatch('load')}
            disabled={retrievingSlices}
            >{retrievingSlices ? 'Loading...' : 'Find Slices'}</button
          >
        {/if}
      </div>
      <div class="p-3 border-r border-slate-200 flex gap-2 items-center">
        {#if !!scoreWeights}
          {@const sortedNames = Object.keys(scoreWeights)
            .filter((n) => scoreWeights[n] > 0)
            .sort((a, b) => scoreWeights[b] - scoreWeights[a])}
          <div class="text-xs" style="max-width: 240px;">
            Sorting by <strong>{sortedNames.slice(0, 2).join(', ')}</strong>
            {#if sortedNames.length > 2}
              and {sortedNames.length - 2} other{sortedNames.length - 2 > 1
                ? 's'
                : ''}{/if}
          </div>
        {/if}
        <ActionMenuButton
          buttonClass="btn btn-slate"
          buttonTitle="Adjust weights for how slices are ranked"
          disabled={retrievingSlices}
          menuWidth={400}
          singleClick={false}
        >
          <span slot="button-content"
            ><Fa icon={faScaleBalanced} class="inline mr-1" />
            Sort</span
          >
          <div
            slot="options"
            let:dismiss
            class="overflow-y-auto relative"
            style="max-height: 400px;"
          >
            <ScoreWeightMenu
              collapsible={false}
              showApplyButton
              weights={scoreWeights}
              {scoreNames}
              on:apply={(e) => {
                scoreWeights = e.detail;
                dismiss();
              }}
              on:cancel={dismiss}
            />
          </div>
        </ActionMenuButton>
      </div>
      {#if runningSampler}
        {#if samplerRunProgress == null}
          <div role="status" class="ml-3 w-8 h-8 grow-0 shrink-0 self-center">
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
        <div class="ml-3 flex-auto whitespace-nowrap self-center">
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
      {:else}
        <div class="p-3 border-r border-slate-200">
          <div class="h-full flex items-center gap-2">
            <!-- <ActionMenuButton
              buttonClass="btn px-1 py-0.5 hover:bg-slate-200 text-xs text-slate-600 font-bold"
              buttonTitle="Add a filter option"
              disabled={retrievingSlices}
            >
              <span slot="button-content"
                ><Fa icon={faSearch} class="inline mr-1" />
                Refine Search</span
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
            </ActionMenuButton> -->
            <button
              disabled={retrievingSlices}
              class="btn px-1 py-0.5 hover:bg-slate-200 text-xs text-slate-600 font-bold"
              on:click={() => (specEditorVisible = true)}
              ><Fa icon={faWrench} class="inline mr-1" /> Configure Slicing</button
            >
          </div>
        </div>
      {/if}
    </div>
  </div>
  <div class="flex-auto min-h-0 overflow-auto relative">
    {#if !!baseSlice}
      <div
        class="bg-white sticky top-0 z-10 {Object.values(
          enabledSliceControls
        ).some((v) => !!v) || Object.keys(savedSlices).length > 0
          ? ''
          : 'border-b-4 border-slate-200'}"
        bind:this={searchViewHeader}
      >
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
          allowFavorite={true}
          allowMultiselect={false}
          metricInfo={(n) => getMetric(metricInfo, n)}
          metricGetter={(s, name) => getMetric(s.metrics, name)}
          bind:metricNames
          bind:scoreNames
          bind:scoreWidthScalers
          allowShowScores={false}
          showCheckboxes={false}
          on:newsearch={(e) => {
            updateEditingControl(e.detail.type, e.detail.base_slice);
            toggleSliceControl(e.detail.type, true);
          }}
          on:saveslice
        />
      </div>
      {#if Object.keys(savedSlices).length > 0}
        <div
          class="{Object.values(enabledSliceControls).some((v) => !!v)
            ? ''
            : 'border-b-4 border-slate-200'} bg-white"
        >
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
            on:newsearch={(e) => {
              updateEditingControl(e.detail.type, e.detail.base_slice);
              toggleSliceControl(e.detail.type, true);
            }}
            on:saveslice
          />
        </div>
      {/if}
      {#if Object.values(enabledSliceControls).some((v) => !!v)}
        <div
          class="sampler-panel w-full mb-2 bg-white"
          bind:this={samplerPanel}
        >
          <div
            class="pt-3 rounded bg-slate-100 text-gray-700 border-slate-100 border-2 box-border"
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
      <div class="flex-auto relative w-full">
        {#if !!slices && slices.length > 0}
          <div class="w-full min-h-0" class:disable-div={runningSampler}>
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
              on:newsearch={(e) => {
                updateEditingControl(e.detail.type, e.detail.base_slice);
                toggleSliceControl(e.detail.type, true);
              }}
              on:saveslice
            />
          </div>
        {:else}
          <div
            class="w-full mt-6 flex-auto min-h-0 flex flex-col items-center justify-center text-slate-500"
          >
            <div>No slices yet!</div>
          </div>
        {/if}
      </div>
    {:else if !retrievingSlices}
      <div
        class="w-full flex-auto min-h-0 mt-6 flex flex-col items-center justify-center text-slate-500"
      >
        <div>No slices yet!</div>
      </div>
    {/if}
    {#if retrievingSlices}
      <div
        class="absolute top-0 left-0 bg-white/80 w-full h-full flex flex-col items-center justify-center z-20"
      >
        <div>Retrieving slices...</div>
        {#if showDetailLoadingMessage}
          <div class="mt-2 text-sm text-slate-500">
            It may take up to a minute to rank slices on your first visit.
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

{#if specEditorVisible}
  <div
    class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
    on:click={() => (specEditorVisible = false)}
    on:keydown={(e) => {}}
  />
  <div
    class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none"
  >
    <div
      class="w-2/3 h-2/3 z-20 rounded-md bg-white p-1 pointer-events-auto"
      style="min-width: 200px; max-width: 100%;"
    >
      <SliceSpecEditor
        {sliceSpec}
        {timestepDefinition}
        on:dismiss={(e) => {
          if (!!e.detail) sliceSpec = e.detail;
          specEditorVisible = false;
        }}
      />
    </div>
  </div>
{/if}

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
