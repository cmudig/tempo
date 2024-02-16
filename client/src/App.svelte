<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { ModelSummary } from './lib/model';
  import ModelEditor from './lib/ModelEditor.svelte';
  import ModelResultsView from './lib/ModelResultsView.svelte';
  import SlicesView from './lib/SlicesView.svelte';
  import Sidebar from './lib/Sidebar.svelte';
  import type { Slice, SliceFeatureBase } from './lib/slices/utils/slice.type';
  import { faHeart } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import SavedSlicesView from './lib/SavedSlicesView.svelte';
  import ResizablePanel from './lib/utils/ResizablePanel.svelte';

  let models: { [key: string]: ModelSummary } = {};

  enum View {
    results = 'Results',
    slices = 'Slices',
    editor = 'Edit',
  }
  let currentView: View = View.results;
  let showingSaved: boolean = false;

  let currentModel = 'vasopressor_8h';
  let selectedModels: string[] = [];

  let sliceSpec = 'default';
  let metricToShow: string = 'AUROC';

  let selectedSlice: SliceFeatureBase | null = null;
  $: if (currentView !== View.slices) selectedSlice = null;

  let savedSlices: { [key: string]: SliceFeatureBase[] } = {};

  onMount(async () => {
    await refreshModels();
  });

  async function refreshModels() {
    let result = await fetch('/models');
    models = (await result.json()).models;
    console.log('models:', models);
    if (!!refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refreshModels, 5000);
  }

  let refreshTimer: NodeJS.Timeout | null = null;

  onDestroy(() => {
    if (!!refreshTimer) clearTimeout(refreshTimer);
  });

  $: if (
    !!models[currentModel] &&
    !models[currentModel].metrics?.performance[metricToShow]
  ) {
    let availableMetrics = Object.keys(
      models[currentModel].metrics?.performance ?? {}
    ).sort();
    if (availableMetrics.length > 0) metricToShow = availableMetrics[0];
  }
</script>

<main class="w-screen h-screen flex flex-col">
  <div class="w-full h-12 grow-0 shrink-0 bg-slate-500 flex py-2 px-3">
    <div class="flex-auto" />
    <button
      class="btn {showingSaved ? 'btn-dark-blue' : 'btn-dark-slate'}"
      on:click={() => (showingSaved = !showingSaved)}
      ><Fa icon={faHeart} class="inline mr-2" /> Saved Slices</button
    >
  </div>
  <div class="flex-auto w-full flex h-0">
    {#if showingSaved}
      <SavedSlicesView bind:savedSlices bind:metricToShow />
    {:else}
      <ResizablePanel rightResizable width={540} maxWidth="40%" height="100%">
        <Sidebar
          {models}
          bind:metricToShow
          bind:activeModel={currentModel}
          bind:selectedModels
          bind:selectedSlice
          {sliceSpec}
        />
      </ResizablePanel>
      <div class="flex-auto h-full flex flex-col w-0">
        <div
          class="w-full px-4 py-2 flex gap-3 bg-slate-300 border-b border-slate-400"
        >
          {#each [View.results, View.slices, View.editor] as view}
            <button
              class="rounded my-2 py-1 px-6 text-center w-32 {currentView ==
              view
                ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
                : 'text-slate-700 hover:bg-slate-200'}"
              on:click={() => (currentView = view)}>{view}</button
            >
          {/each}
        </div>
        <div
          class="w-full flex-auto"
          class:overflow-y-auto={currentView != View.slices}
        >
          {#if currentView == View.results}
            <ModelResultsView
              modelName={currentModel}
              modelSummary={models[currentModel]}
            />
          {:else if currentView == View.slices}
            <SlicesView
              bind:selectedSlice
              bind:sliceSpec
              bind:savedSlices
              bind:metricToShow
              modelName={currentModel}
              timestepDefinition={models[currentModel]?.timestep_definition ??
                ''}
              modelsToShow={Array.from(
                new Set([...selectedModels, currentModel])
              )}
            />
          {:else if currentView == View.editor}
            <ModelEditor
              modelName={currentModel}
              on:viewmodel={(e) => {
                currentView = View.results;
                currentModel = e.detail;
              }}
              on:train={(e) => {
                currentModel = e.detail;
                refreshModels();
              }}
            />
          {/if}
        </div>
      </div>
    {/if}
  </div>
</main>
