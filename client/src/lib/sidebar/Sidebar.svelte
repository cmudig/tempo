<svelte:options accessors />

<script lang="ts">
  import type { ModelMetrics, ModelSummary } from '../model';
  import SidebarItem from './SidebarItem.svelte';
  import SliceFeature from '../slices/slice_table/SliceFeature.svelte';
  import type {
    Slice,
    SliceFeatureBase,
    SliceMetric,
  } from '../slices/utils/slice.type';
  import {
    faPlus,
    faSort,
    faSortDown,
    faSortUp,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';
  import { createEventDispatcher, getContext } from 'svelte';
  import type { Writable } from 'svelte/store';

  const dispatch = createEventDispatcher();

  let {
    models,
    currentModel,
  }: {
    models: Writable<{
      [key: string]: { spec: ModelSummary; metrics?: ModelMetrics };
    }>;
    currentModel: Writable<string | null>;
  } = getContext('models');
  export let selectedModels: string[] = [];

  export let metricToShow: string = 'AUROC';
  export let isLoadingModels: boolean = false;

  let sortField: string = 'name';
  let sortDescending: boolean = false;

  let modelOrder: string[] = [];

  let sliceMetrics:
    | { [key: string]: { [key: string]: SliceMetric } }
    | undefined;

  let metricScales: { [key: string]: (v: number) => number } = {};

  let metricOptions: string[] = [];
  $: if (Object.values($models).length > 0) {
    let modelsWithMetrics = Object.values($models).filter((m) => !!m.metrics);
    if (modelsWithMetrics.length > 0)
      metricOptions = Array.from(
        new Set(
          modelsWithMetrics
            .map((m) => Object.keys(m.metrics!.performance))
            .flat()
        )
      ).sort();
    else metricOptions = [];
  } else metricOptions = [];

  let availableMetrics: string[] = [];
  $: availableMetrics = [
    'Timesteps',
    'Trajectories',
    metricToShow,
    'Labels',
    'Predictions',
  ];
  $: if (sortField != 'name' && !availableMetrics.includes(sortField))
    sortField = 'name';

  const metricDescriptions: { [key: string]: string } = {
    Timesteps: 'Number of timesteps in the slice evaluation set',
    Trajectories: 'Number of unique trajectories in the slice evaluation set',
    Labels: 'True values for the target variable',
    Predictions: 'Predicted values for the target variable',
  };

  $: {
    let maxInstances = Object.values($models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) => Math.max(prev, curr.metrics?.n_test.instances ?? 0),
        0
      );
    let maxTrajectories = Object.values($models)
      .filter((m) => !!m.metrics)
      .reduce(
        (prev, curr) => Math.max(prev, curr.metrics?.n_test.trajectories ?? 0),
        0
      );
    let maxMetricValue = Object.values($models)
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
    };
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
      modelOrder = Object.keys($models).sort();
      if (sortDescending) modelOrder = modelOrder.reverse();
    } else {
      let metricValues: {
        [key: string]: { [key: string]: number };
      };
      if (!!sliceMetrics)
        metricValues = Object.fromEntries(
          Object.keys($models).map((m) => [
            m,
            {
              Timesteps: sliceMetrics![m]['Timesteps']?.count ?? 0,
              Trajectories:
                (sliceMetrics![m]['Trajectories']?.count as number) ?? 0,
              [metricToShow]:
                (sliceMetrics![m][metricToShow]?.mean as number) ?? 0,
              Labels: (sliceMetrics![m]['Labels']?.mean as number) ?? 0,
              Predictions:
                (sliceMetrics![m]['Predictions']?.mean as number) ?? 0,
            },
          ])
        );
      else
        metricValues = Object.fromEntries(
          Object.entries($models).map(([modelName, model]) => [
            modelName,
            {
              Timesteps: model.metrics?.n_test.instances ?? 0,
              Trajectories: model.metrics?.n_test.trajectories ?? 0,
              [metricToShow]: model.metrics?.performance[metricToShow] ?? 0,
              Labels:
                model.metrics?.labels?.value ??
                model.metrics?.labels?.mean ??
                0,
              Predictions:
                model.metrics?.predictions?.value ??
                model.metrics?.predictions?.mean ??
                0,
            },
          ])
        );
      modelOrder = Object.keys($models).sort(
        (a, b) =>
          (metricValues[a][sortField] - metricValues[b][sortField]) *
          (sortDescending ? -1 : 1)
      );
    }
  }

  let editingModelName: string | null = null;

  export function editModelName(modelName: string) {
    editingModelName = modelName;
  }
</script>

<div class="flex flex-col w-full h-full">
  <div class="w-full sticky top-0">
    <div class="py-2 px-4 flex items-center grow-0 shrink-0 gap-2">
      <div class="text-lg font-bold whitespace-nowrap shrink-1 overflow-hidden">
        Models
      </div>
      {#if isLoadingModels}
        <div role="status">
          <svg
            aria-hidden="true"
            class="w-4 h-4 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600"
            viewBox="0 0 100 101"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
              fill="currentColor"
            />
            <path
              d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
              fill="currentFill"
            />
          </svg>
        </div>
      {/if}
      <div class="flex-auto" />
      <ActionMenuButton
        buttonClass="bg-transparent px-1 hover:opacity-40"
        align="right"
        ><span slot="button-content"><Fa icon={faPlus} class="inline" /></span>
        <div slot="options">
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            on:click={() => dispatch('new', 'default')}
            >From Default Specification</a
          >
          {#if !!$currentModel}
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              on:click={() => dispatch('new', $currentModel)}
              >From <span class="font-mono">{$currentModel}</span></a
            >
          {/if}
        </div></ActionMenuButton
      >
      <ActionMenuButton
        buttonClass="bg-transparent px-1 hover:opacity-40"
        align="right"
      >
        <div slot="options">
          {#if selectedModels.length == 0}
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Create a copy of this model"
              on:click={() => dispatch('new', $currentModel)}>Duplicate</a
            >
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Rename this model"
              on:click={() => (editingModelName = $currentModel ?? null)}
              >Rename...</a
            >
          {/if}
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            title="Permanently delete these models"
            on:click={() =>
              dispatch('delete', [$currentModel, ...selectedModels])}>Delete</a
          >
        </div>
      </ActionMenuButton>
    </div>
    <div class="flex px-4 my-2 items-center w-full">
      <div class="shrink-0 text-xs mr-2 text-slate-600">Display metric:</div>
      {#if metricOptions.length > 0}
        <select
          class="flat-select-sm"
          style="min-width: 120px;"
          bind:value={metricToShow}
        >
          {#each metricOptions as metricName}
            <option value={metricName}>{metricName}</option>
          {/each}
        </select>
      {/if}
    </div>
  </div>
  {#if Object.keys($models).length == 0}
    <div
      class="w-full mt-6 flex-auto min-h-0 flex flex-col items-center justify-center text-slate-500"
    >
      <div>No models yet!</div>
    </div>
  {:else}
    <div class="overflow-y-auto flex-auto min-h-0">
      <!-- <div
        class="text-sm text-left inline-flex align-top slice-header whitespace-nowrap bg-slate-100 rounded-t border-b border-slate-600 sticky top-0 z-1"
      >
        <div
          class="grow-0 shrink-0"
          style="width: {SidebarTableWidths.Checkbox}px;"
        ></div>
        <button
          class="p-2 text-left grow-0 shrink-0 hover:bg-slate-200 whitespace-nowrap"
          on:click={() => setSort('name')}
          class:font-bold={sortField == 'name'}
          title={'Model name' +
            (sortField == 'name'
              ? ', sorted ' + (sortDescending ? 'descending' : 'ascending')
              : '')}
          style="width: {SidebarTableWidths.ModelName}px;"
        >
          Model {#if sortField == 'name'}
            <Fa
              icon={sortDescending ? faSortDown : faSortUp}
              class="inline text-xs"
            />
          {/if}
        </button>
        {#each availableMetrics as fieldName}
          <button
            class="p-2 rounded text-left grow-0 shrink-0 hover:bg-slate-200 whitespace-nowrap"
            class:font-bold={sortField == fieldName}
            title={(fieldName == metricToShow
              ? 'Selected performance metric'
              : metricDescriptions[fieldName] ?? '') +
              (sortField == fieldName
                ? ', sorted ' + (sortDescending ? 'descending' : 'ascending')
                : '')}
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
      </div> -->
      {#each modelOrder as modelName (modelName)}
        {@const model = $models[modelName]}
        <SidebarItem
          displayItem={{
            name: modelName,
            description:
              model.spec?.draft?.description ?? model.spec?.description,
            output_values: model.spec?.output_values,
          }}
          metrics={model.metrics ?? null}
          {metricToShow}
          {metricScales}
          isEditingName={editingModelName == modelName}
          isActive={$currentModel === modelName}
          isChecked={selectedModels.includes(modelName) ||
            $currentModel === modelName}
          allowCheck={$currentModel != modelName}
          on:click={() => {
            $currentModel = modelName;
            selectedModels = [];
          }}
          on:toggle={(e) => {
            let idx = selectedModels.indexOf(modelName);
            if (idx >= 0)
              selectedModels = [
                ...selectedModels.slice(0, idx),
                ...selectedModels.slice(idx + 1),
              ];
            else selectedModels = [...selectedModels, modelName];
          }}
          on:duplicate={(e) => dispatch('new', e.detail)}
          on:editname={(e) => (editingModelName = e.detail)}
          on:canceledit={(e) => (editingModelName = null)}
          on:rename={(e) => {
            editingModelName = null;
            dispatch('rename', e.detail);
          }}
          on:delete={(e) => dispatch('delete', [e.detail])}
        />
      {/each}
      {#if modelOrder.length > 1}
        <div class="p-4 text-center text-xs text-slate-500">
          Use checkboxes to select additional models for comparison.
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .slice-header {
    min-width: 100%;
  }
  .slice-header > * {
    flex: 0 0 auto;
  }
</style>
