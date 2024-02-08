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
  import {
    faSort,
    faSortDown,
    faSortUp,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { areSetsEqual } from './slices/utils/utils';

  export let models: { [key: string]: ModelSummary } = {};
  export let activeModel: string | undefined;
  export let selectedModels: string[] = [];
  export let selectedSlice: SliceFeatureBase | null = null;
  export let sliceSpec: string = 'default';

  export let metricToShow: string = 'AUROC';

  let sortField: string = 'name';
  let sortDescending: boolean = false;

  let modelOrder: string[] = [];

  let sliceMetrics:
    | { [key: string]: { [key: string]: SliceMetric } }
    | undefined;

  let metricScales: { [key: string]: (v: number) => number } = {};

  let metricOptions: string[] = [];
  $: if (Object.values(models).length > 0) {
    let modelsWithMetrics = Object.values(models).filter((m) => !!m.metrics);
    if (modelsWithMetrics.length > 0)
      metricOptions = Object.keys(
        modelsWithMetrics[0].metrics!.performance
      ).sort();
    else metricOptions = [];
  } else metricOptions = [];

  $: {
    let maxInstances = Object.values(models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) =>
          Math.max(prev, curr.metrics?.n_slice_eval.instances ?? 0),
        0
      );
    let maxTrajectories = Object.values(models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) =>
          Math.max(prev, curr.metrics?.n_slice_eval.trajectories ?? 0),
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

  $: if (!!selectedSlice) loadSliceScores(selectedSlice, sliceSpec);
  else sliceMetrics = undefined;

  async function loadSliceScores(sliceDef: SliceFeatureBase, spec: string) {
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
          body: JSON.stringify({
            sliceRequests,
            sliceSpec: spec,
            selectedModel: activeModel,
          }),
        })
      ).json();
      let result = results.sliceRequestResults.toScore as Slice;
      if (!!result)
        sliceMetrics = result.metrics as {
          [key: string]: { [key: string]: SliceMetric };
        };
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

  function hasSameTimestepDefinition(
    modelName: string,
    otherModelName: string
  ): boolean {
    if (!models[modelName] || !models[otherModelName]) return false;
    return (
      models[modelName].timestep_definition ==
      models[otherModelName].timestep_definition
    );
  }

  function setSort(field: string) {
    if (sortField != field) {
      sortField = field;
      sortDescending = false;
    } else {
      sortDescending = !sortDescending;
    }
  }

  $: {
    if (sortField == 'name') {
      modelOrder = Object.keys(models).sort();
      if (sortDescending) modelOrder = modelOrder.reverse();
    } else {
      let metricValues: { [key: string]: { [key: string]: number } };
      if (!!sliceMetrics)
        metricValues = Object.fromEntries(
          Object.keys(models).map((m) => [
            m,
            {
              Timesteps: sliceMetrics![m]['Timesteps']?.count ?? 0,
              Trajectories:
                (sliceMetrics![m]['Trajectories']?.count as number) ?? 0,
              [metricToShow]:
                (sliceMetrics![m][metricToShow]?.mean as number) ?? 0,
              'Positive Rate':
                (sliceMetrics![m]['Positive Rate']?.mean as number) ?? 0,
            },
          ])
        );
      else
        metricValues = Object.fromEntries(
          Object.entries(models).map(([modelName, model]) => [
            modelName,
            {
              Timesteps: model.metrics?.n_slice_eval.instances ?? 0,
              Trajectories: model.metrics?.n_slice_eval.trajectories ?? 0,
              [metricToShow]: model.metrics?.performance[metricToShow] ?? 0,
              'Positive Rate': model.metrics?.positive_rate ?? 0,
            },
          ])
        );
      modelOrder = Object.keys(models).sort(
        (a, b) =>
          (metricValues[a][sortField] - metricValues[b][sortField]) *
          (sortDescending ? -1 : 1)
      );
    }
  }
</script>

<div class="flex flex-col w-full h-full">
  <div class="my-2 px-4 flex justify-between grow-0 shrink-0">
    <div class="text-lg font-bold">Models</div>
    <select class="flat-select" bind:value={metricToShow}>
      {#each metricOptions as metricName}
        <option value={metricName}>{metricName}</option>
      {/each}
    </select>
  </div>
  {#if !!selectedSlice}
    <div class="rounded bg-slate-100 px-3 pt-3 mx-2 mb-2">
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
  <div class="px-2 overflow-auto flex-auto min-h-0">
    <div
      class="text-sm text-left inline-flex align-top slice-header whitespace-nowrap bg-slate-100 rounded-t border-b border-slate-600"
    >
      {#if !!selectedSlice}
        <div
          class="grow-0 shrink-0"
          style="width: {SidebarTableWidths.Checkbox}px;"
        ></div>
      {/if}
      <button
        class="p-2 text-left grow-0 shrink-0 hover:bg-slate-200 whitespace-nowrap"
        on:click={() => setSort('name')}
        class:font-bold={sortField == 'name'}
        style="width: {SidebarTableWidths.ModelName}px;"
      >
        Model {#if sortField == 'name'}
          <Fa
            icon={sortDescending ? faSortDown : faSortUp}
            class="inline text-xs"
          />
        {/if}
      </button>
      {#each ['Timesteps', 'Trajectories', metricToShow, 'Positive Rate'] as fieldName}
        <button
          class="p-2 rounded text-left grow-0 shrink-0 hover:bg-slate-200 whitespace-nowrap"
          class:font-bold={sortField == fieldName}
          on:click={() => setSort(fieldName)}
          style="width: {SidebarTableWidths.Metric}px;"
        >
          {fieldName}
          {#if sortField == fieldName}
            <Fa
              icon={sortDescending ? faSortDown : faSortUp}
              class="inline text-xs"
            />
          {/if}
        </button>
      {/each}
    </div>
    {#each modelOrder as modelName (modelName)}
      {@const model = models[modelName]}
      <SidebarItem
        {model}
        {modelName}
        {metricToShow}
        {metricScales}
        showCheckbox={!!selectedSlice}
        customMetrics={sliceMetrics?.[modelName] ?? undefined}
        isActive={activeModel === modelName}
        isChecked={selectedModels.includes(modelName) ||
          activeModel === modelName}
        allowCheck={!activeModel ||
          hasSameTimestepDefinition(modelName, activeModel)}
        on:click={() => (activeModel = modelName)}
        on:toggle={(e) => {
          let idx = selectedModels.indexOf(modelName);
          if (idx >= 0)
            selectedModels = [
              ...selectedModels.slice(0, idx),
              ...selectedModels.slice(idx + 1),
            ];
          else selectedModels = [...selectedModels, modelName];
        }}
      />
    {/each}
  </div>
</div>

<style>
  .slice-header {
    min-width: 100%;
  }
  .slice-header > * {
    flex: 0 0 auto;
  }
</style>
