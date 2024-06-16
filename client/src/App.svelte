<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { type ModelSummary, metricsHaveWarnings } from './lib/model';
  import ModelEditor from './lib/model_editor/ModelEditor.svelte';
  import ModelResultsView from './lib/ModelResultsView.svelte';
  import SlicesView from './lib/slices/SlicesView.svelte';
  import Sidebar from './lib/sidebar/Sidebar.svelte';
  import type { Slice, SliceFeatureBase } from './lib/slices/utils/slice.type';
  import {
    faBook,
    faDatabase,
    faWarning,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import ResizablePanel from './lib/utils/ResizablePanel.svelte';
  import DatasetInfoView from './lib/DatasetInfoView.svelte';
  import logoUrl from './assets/logo_dark.svg';
  import QueryLanguageReferenceView from './lib/QueryLanguageReferenceView.svelte';

  let models: { [key: string]: ModelSummary } = {};

  enum View {
    editor = 'Specification',
    results = 'Metrics',
    slices = 'Slices',
  }
  let currentView: View = View.results;
  let showingSaved: boolean = false;
  let showingDatasetInfo: boolean = false;
  let showingQueryReference: boolean = false;

  let currentModel: string | null = null;
  let selectedModels: string[] = [];

  let sliceSpec = 'default';
  let metricToShow: string = 'AUROC';

  let selectedSlice: SliceFeatureBase | null = null;
  $: if (currentView !== View.slices) selectedSlice = null;

  // keys are slice specification names
  let savedSlices: { [key: string]: { [key: string]: SliceFeatureBase } } = {};

  onMount(async () => {
    await refreshModels();
  });

  async function refreshModels() {
    let result = await fetch('/models');
    models = (await result.json()).models;
    console.log('models:', models);
    if (Object.keys(models).length > 0) {
      if (!currentModel || !models[currentModel])
        currentModel = Object.keys(models).sort()[0];
    }
    if (!!refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refreshModels, 5000);
  }

  let oldCurrentModel: string | null = null;
  $: if (oldCurrentModel !== currentModel) {
    if (
      !!models &&
      !!currentModel &&
      !!models[currentModel] &&
      !models[currentModel].metrics
    )
      currentView = View.editor;
    if (
      !!currentModel &&
      !!models[currentModel] &&
      !models[currentModel].metrics?.performance[metricToShow]
    ) {
      let availableMetrics = Object.keys(
        models[currentModel].metrics?.performance ?? {}
      ).sort();
      if (availableMetrics.length > 0) metricToShow = availableMetrics[0];
    }
    oldCurrentModel = currentModel;
  }

  let refreshTimer: NodeJS.Timeout | null = null;

  onDestroy(() => {
    if (!!refreshTimer) clearTimeout(refreshTimer);
  });

  async function createModel(reference: string) {
    try {
      let newModel = await (
        await fetch(`/models/new/${reference}`, { method: 'POST' })
      ).json();
      currentModel = newModel.name;
      selectedModels = [];
      currentView = View.editor;
    } catch (e) {
      console.error('error creating new model:', e);
    }
    refreshModels();
  }
</script>

<main class="w-screen h-screen flex flex-col">
  <div
    class="w-full h-12 grow-0 shrink-0 bg-slate-700 flex py-2 px-4 items-center"
  >
    <div class="font-bold text-white h-full py-1">
      <img src={logoUrl} class="h-full" alt="Tempo" />
    </div>
    <div class="flex-auto" />
    <button
      class="mr-3 btn btn-dark-slate"
      on:click={() => (showingQueryReference = true)}
      ><Fa icon={faBook} class="inline mr-2" /> Query Reference</button
    >
    <button
      class="btn btn-dark-slate"
      on:click={() => (showingDatasetInfo = true)}
      ><Fa icon={faDatabase} class="inline mr-2" /> Dataset Info</button
    >
  </div>
  <div class="flex-auto w-full flex h-0">
    <ResizablePanel
      rightResizable
      width={540}
      minWidth={300}
      maxWidth={720}
      height="100%"
    >
      <Sidebar
        {models}
        bind:metricToShow
        bind:activeModel={currentModel}
        bind:selectedModels
        bind:selectedSlice
        {sliceSpec}
        on:new={(e) => createModel(e.detail)}
      />
    </ResizablePanel>
    <div class="flex-auto h-full flex flex-col w-0" style="z-index: 1;">
      <div class="w-full px-4 py-2 flex gap-3 bg-slate-200">
        {#each [View.editor, View.results, View.slices] as view}
          <button
            class="rounded my-2 py-1 text-center w-36 {currentView == view
              ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
              : 'text-slate-700 hover:bg-slate-300'}"
            on:click={() => (currentView = view)}
            >{#if view == View.results && !!currentModel && !!models && !!models[currentModel]?.metrics && metricsHaveWarnings(models[currentModel].metrics)}<Fa
                icon={faWarning}
                class="inline mr-1"
              />{/if}
            {view}</button
          >
        {/each}
      </div>
      <div
        class="w-full flex-auto"
        class:overflow-y-auto={currentView != View.slices}
      >
        {#if currentView == View.results}
          {#each !!currentModel ? Array.from(new Set( [currentModel, ...selectedModels] )) : [] as model}
            <ModelResultsView modelName={model} modelSummary={models[model]} />
          {/each}
        {:else if currentView == View.slices}
          <SlicesView
            bind:selectedSlice
            bind:sliceSpec
            bind:savedSlices
            bind:metricToShow
            modelName={currentModel}
            timestepDefinition={models[currentModel ?? '']
              ?.timestep_definition ?? ''}
            modelsToShow={!!currentModel
              ? Array.from(new Set([...selectedModels, currentModel]))
              : []}
          />
        {:else if currentView == View.editor}
          <ModelEditor
            modelName={currentModel}
            otherModels={selectedModels.filter((m) => m != currentModel)}
            on:viewmodel={(e) => {
              currentView = View.results;
              currentModel = e.detail;
            }}
            on:train={async (e) => {
              await refreshModels();
              currentModel = e.detail;
              selectedModels = [];
            }}
            on:delete={async () => {
              await refreshModels();
              currentView = View.results;
            }}
            on:finish={(e) => {
              refreshModels();
              if (e.detail.success) currentView = View.results;
            }}
          />
        {/if}
      </div>
    </div>
  </div>
  {#if showingDatasetInfo}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={() => (showingDatasetInfo = false)}
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none"
    >
      <div
        class="w-2/3 h-2/3 z-20 rounded-md bg-white p-1 pointer-events-auto"
        style="min-width: 200px; max-width: 100%;"
      >
        <DatasetInfoView on:close={() => (showingDatasetInfo = false)} />
      </div>
    </div>
  {:else if showingQueryReference}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={() => (showingQueryReference = false)}
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none"
    >
      <div
        class="w-2/3 h-2/3 z-20 rounded-md bg-white p-1 pointer-events-auto"
        style="min-width: 200px; max-width: 100%;"
      >
        <QueryLanguageReferenceView
          on:close={() => (showingQueryReference = false)}
        />
      </div>
    </div>
  {/if}
</main>
