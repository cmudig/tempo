<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { ModelSummary } from './lib/model';
  import ModelEditor from './lib/ModelEditor.svelte';
  import ModelResultsView from './lib/ModelResultsView.svelte';
  import SlicesView from './lib/SlicesView.svelte';
  import Sidebar from './lib/Sidebar.svelte';
  import type { Slice, SliceFeatureBase } from './lib/slices/utils/slice.type';

  let models: { [key: string]: ModelSummary } = {};

  enum View {
    results = 'Results',
    slices = 'Slices',
    editor = 'Edit',
  }
  let currentView = View.results;

  let currentModel = 'vasopressor_8h';
  let selectedModels: string[] = [];

  let selectedSlice: SliceFeatureBase | null = null;
  $: if (currentView !== View.slices) selectedSlice = null;

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
</script>

<main class="w-screen h-screen flex">
  <div
    class="border-r border-slate-400 h-full shrink-0 grow-0"
    style="width: 540px; max-width: 40%;"
  >
    <Sidebar
      {models}
      bind:activeModel={currentModel}
      bind:selectedModels
      bind:selectedSlice
    />
  </div>
  <div class="flex-auto h-full flex flex-col w-0">
    <div
      class="w-full px-4 py-2 flex gap-3 bg-slate-300 border-b border-slate-400"
    >
      {#each [View.results, View.slices, View.editor] as view}
        <button
          class="rounded my-2 py-1 px-6 text-center w-32 {currentView == view
            ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
            : 'text-slate-700 hover:bg-slate-200'}"
          on:click={() => (currentView = view)}>{view}</button
        >
      {/each}
    </div>
    <div
      class="w-full flex-auto"
      class:overflow-scroll={currentView != View.slices}
    >
      {#if currentView == View.results}
        <ModelResultsView modelName={currentModel} />
      {:else if currentView == View.slices}
        <SlicesView
          bind:selectedSlice
          modelName={currentModel}
          modelsToShow={Array.from(new Set([...selectedModels, currentModel]))}
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
</main>
