<script lang="ts">
  import type { ModelMetrics, ModelSummary } from './model';
  import SidebarItem from './SidebarItem.svelte';
  import SliceFeature from './slices/slice_table/SliceFeature.svelte';
  import type {
    Slice,
    SliceFeatureBase,
    SliceMetric,
  } from './slices/utils/slice.type';
  import { SidebarTableWidths } from './utils/sidebarwidths';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';

  export let models: { [key: string]: ModelSummary } = {};
  export let activeModel: string | undefined;
  export let selectedSlice: SliceFeatureBase | null = null;

  export let metricToShow: string = 'AUROC';

  let sliceMetrics:
    | { [key: string]: SliceMetric | { [key: string]: SliceMetric } }
    | undefined;

  let metricScales: { [key: string]: (v: number) => number } = {};

  $: {
    let maxInstances = Object.values(models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) => Math.max(prev, curr.metrics?.n_val.instances ?? 0),
        0
      );
    let maxTrajectories = Object.values(models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) => Math.max(prev, curr.metrics?.n_val.trajectories ?? 0),
        0
      );
    let maxMetricValue = Object.values(models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) =>
          Math.max(
            prev,
            (curr.metrics!.performance[metricToShow] as number) ?? 0
          ),
        0
      );
    metricScales = {
      [metricToShow]:
        maxMetricValue > 1
          ? (v: number) => v / maxMetricValue
          : (v: number) => v,
      Timesteps: (v: number) => v / maxInstances,
      Trajectories: (v: number) => v / maxTrajectories,
      'Positive Rate': (v: number) => v,
    };
  }

  $: if (!!selectedSlice) loadSliceScores(selectedSlice);
  else sliceMetrics = undefined;

  async function loadSliceScores(sliceDef: SliceFeatureBase) {
    let sliceRequests: { [key: string]: SliceFeatureBase } = {
      toScore: sliceDef,
    };
    try {
      let results = await (
        await fetch(`/slices/score`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sliceRequests }),
        })
      ).json();
      let result = results.sliceRequestResults.toScore as Slice;
      if (!!result) sliceMetrics = result.metrics;
      else sliceMetrics = undefined;
      console.log(
        'slice metrics:',
        sliceMetrics,
        results,
        JSON.stringify({ sliceRequests })
      );
    } catch (e) {
      console.log('error calculating slice for sidebar:', e);
    }
  }
</script>

<div class="my-2 text-lg font-bold px-4">Models</div>
{#if !!selectedSlice}
  <div class="rounded bg-slate-100 p-3 mx-4">
    <div class="ml-2 flex text-xs font-bold text-slate-600">
      <div class="flex-auto">Within slice:</div>
      <button class="hover:opacity-50" on:click={() => (selectedSlice = null)}
        ><Fa icon={faXmark} /></button
      >
    </div>
    <div class="overflow-x-scroll whitespace-nowrap">
      <SliceFeature
        feature={selectedSlice}
        currentFeature={selectedSlice}
        canToggle={false}
      />
    </div>
  </div>
{/if}
<div class="overflow-x-scroll">
  <div class="flex items-start gap-2 px-4 pt-4 pb-2 text-xs text-slate-500">
    <div
      class="grow-0 shrink-0"
      style="width: {SidebarTableWidths.Checkbox}px;"
    ></div>
    <div
      class="grow-0 shrink-0"
      style="width: {SidebarTableWidths.ModelName}px;"
    >
      Model
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      Timesteps
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      Trajectories
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      {metricToShow}
    </div>
    <div class="grow-0 shrink-0" style="width: {SidebarTableWidths.Metric}px;">
      Positive Rate
    </div>
  </div>
  {#each Object.entries(models) as [modelName, model] (modelName)}
    <SidebarItem
      {model}
      {modelName}
      {metricToShow}
      {metricScales}
      customMetrics={sliceMetrics?.[modelName] ?? undefined}
      isActive={activeModel === modelName}
      on:click={() => (activeModel = modelName)}
    />
  {/each}
</div>
