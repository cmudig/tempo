<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { ModelSummary } from './lib/model';
  import ModelEditor from './lib/ModelEditor.svelte';
  import ModelResultsView from './lib/ModelResultsView.svelte';

  let models: { [key: string]: ModelSummary } = {};

  enum View {
    results = 'Results',
    slices = 'Slices',
    editor = 'Edit',
  }
  let currentView = View.editor;

  let currentModel = 'vasopressor_8h';

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
    class="w-1/5 border-r border-slate-400 h-full shrink-0 grow-0"
    style="max-width: 500px;"
  >
    <div class="my-2 text-lg font-bold px-4">Models</div>
    {#each Object.entries(models) as [modelName, model]}
      <button
        class="flex items-center text-left py-2 px-4 font-mono w-full {currentModel ==
        modelName
          ? 'bg-blue-600 text-white hover:bg-blue-700'
          : 'hover:bg-slate-100'}"
        on:click={() => (currentModel = modelName)}
      >
        <div class="flex-auto">
          {modelName}
        </div>
        {#if model.training && !!model.status}
          <div
            class="text-xs font-sans {currentModel == modelName
              ? 'text-slate-50'
              : 'text-slate-500'}"
          >
            {model.status.state}
          </div>
        {/if}
      </button>
    {/each}
  </div>
  <div class="flex-auto h-full flex flex-col">
    <div class="w-full px-4 py-2 flex gap-6">
      {#each [View.results, View.slices, View.editor] as view}
        <button
          class="rounded-lg my-2 py-1 px-6 {currentView == view
            ? 'bg-blue-600 text-white hover:bg-blue-700'
            : 'text-slate-700 hover:bg-slate-200'}"
          on:click={() => (currentView = view)}>{view}</button
        >
      {/each}
    </div>
    <div class="w-full flex-auto overflow-scroll">
      {#if currentView == View.results}
        <ModelResultsView modelName={currentModel} />
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
