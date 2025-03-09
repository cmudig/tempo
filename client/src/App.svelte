<script lang="ts">
  import { onDestroy, onMount, setContext } from 'svelte';
  import {
    type ModelSummary,
    metricsHaveWarnings,
    type ModelMetrics,
    type QueryResult,
    type QueryEvaluationResult,
  } from './lib/model';
  import ModelEditor from './lib/model_editor/ModelEditor.svelte';
  import ModelResultsView from './lib/model_metrics/ModelResultsView.svelte';
  import SlicesView from './lib/slices/SlicesView.svelte';
  import Sidebar from './lib/sidebar/Sidebar.svelte';
  import type { Slice, SliceFeatureBase } from './lib/slices/utils/slice.type';
  import {
    faBook,
    faDatabase,
    faBars,
    faWarning,
    faCaretLeft,
    faCaretRight,
    faSearch,
    faUserCircle,
    faWrench,
    faChartBar,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import ResizablePanel from './lib/utils/ResizablePanel.svelte';
  import DatasetInfoView from './lib/datasets/DatasetInfoView.svelte';
  import logoUrl from './assets/logo_dark.svg';
  import logoLightUrl from './assets/logo_light.svg';
  import QueryLanguageReferenceView from './lib/QueryLanguageReferenceView.svelte';
  import ModelTrainingView from './lib/ModelTrainingView.svelte';
  import { writable, type Writable } from 'svelte/store';
  import type { Dataset } from './lib/dataset';
  import DatasetView from './lib/datasets/DatasetView.svelte';
  import DatasetQueryScratchpad from './lib/datasets/DatasetQueryScratchpad.svelte';
  import ActionMenuButton from './lib/slices/utils/ActionMenuButton.svelte';

  export let csrf: Writable<string> = writable('');
  setContext('csrf', csrf);

  let currentDataset: Writable<string | null> = writable(null);
  let queryResultCache: Writable<{ [key: string]: QueryEvaluationResult }> =
    writable({});
  let dataFields: Writable<string[]> = writable([]);
  setContext('dataset', { currentDataset, queryResultCache, dataFields });

  let datasetOptions: { [key: string]: { spec: Dataset; models: string[] } } =
    {};

  let models: Writable<{
    [key: string]: { spec: ModelSummary; metrics?: ModelMetrics };
  }> = writable({});
  let currentModel: Writable<string | null> = writable(null);
  setContext('models', { models, currentModel });

  enum View {
    editor = 'Specification',
    results = 'Metrics',
    slices = 'Subgroups',
  }
  let currentView: View = View.results;
  let showingDatasetInfo: boolean = false;
  let showingDatasetManagement: boolean = false;
  let showingQueryBuilder: boolean = false;
  let queryHistory: string[] = [];
  let showingQueryReference: boolean = false;
  let showingLogin: boolean = false;
  let showingSignup: boolean = false;
  let loginErrorMessage: string | null = null;
  let currentUser: string | null = null;

  let selectedModels: string[] = [];

  let isLoadingDatasets: boolean = false;
  let isLoadingModels: boolean = false;
  let isLoadingUser: boolean = false;
  let showSidebar: boolean = true;

  let sliceSpec = 'default';
  let metricToShow: string = 'AUROC';

  let selectedSlice: SliceFeatureBase | null = null;
  $: if (currentView !== View.slices) selectedSlice = null;

  // keys are slice specification names
  let savedSlices: { [key: string]: { [key: string]: SliceFeatureBase } } = {};

  onMount(async () => {
    await checkCurrentUser();
    await refreshDatasets();
    await refreshModels();
  });

  async function checkCurrentUser() {
    isLoadingUser = true;
    try {
      let result = await fetch(import.meta.env.BASE_URL + 'user_info');
      console.log('user result:', result);
      if (result.status == 403) {
        showingSignup = true;
        isLoadingUser = false;
        return;
      }
      let response = await result.json();
      console.log('user response:', response);
      currentUser = response.user_id ?? null;
    } catch (e) {
      console.error('error getting current user:', e);
    }
    isLoadingUser = false;
  }

  async function refreshDatasets() {
    isLoadingDatasets = true;
    try {
      let result = await fetch(import.meta.env.BASE_URL + '/datasets');
      datasetOptions = await result.json();
    } catch (e) {
      console.error('error fetching datasets:', e);
    }
    if ($currentDataset == null)
      $currentDataset = window.localStorage.getItem('currentDataset');
    if (!$currentDataset || !datasetOptions[$currentDataset]) {
      if (Object.keys(datasetOptions).length > 0)
        $currentDataset = Object.keys(datasetOptions).sort()[0];
      else $currentDataset = null;
    }
    if (Object.keys(datasetOptions).length == 0)
      showingDatasetManagement = true;
    isLoadingDatasets = false;
  }

  async function refreshModels() {
    if (Object.keys(datasetOptions).length == 0 || $currentDataset == null)
      return;

    isLoadingModels = true;
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${$currentDataset}/models`
      );
      $models = (await result.json()).models;
      console.log('models:', $models);
      if (!$currentModel || !$models[$currentModel])
        setTimeout(
          () =>
            ($currentModel =
              Object.keys($models).length > 0
                ? Object.keys($models).sort()[0]
                : null),
          2000
        );
      isLoadingModels = false;
    } catch (e) {
      console.error('error fetching models:', e);
      isLoadingModels = false;
    }

    if (!!refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refreshModels, 10000);
  }

  let oldCurrentModel: string | null = null;
  $: if (oldCurrentModel !== $currentModel) {
    if (
      !!$models &&
      !!$currentModel &&
      !!$models[$currentModel] &&
      !$models[$currentModel].metrics
    )
      currentView = View.editor;
    if (
      !!$currentModel &&
      !!$models[$currentModel] &&
      !$models[$currentModel].metrics?.performance[metricToShow]
    ) {
      let availableMetrics = Object.keys(
        $models[$currentModel].metrics?.performance ?? {}
      ).sort();
      if (availableMetrics.length > 0) metricToShow = availableMetrics[0];
    }
    oldCurrentModel = $currentModel;
  }

  const manageDatasetsValue = '$!$!manage_datasets!$!$';

  let oldDataset: string | null = null;
  $: if (oldDataset !== $currentDataset) {
    if ($currentDataset == manageDatasetsValue) {
      $currentDataset = oldDataset;
      showingDatasetManagement = true;
    } else {
      refreshModels();
      window.localStorage.setItem('currentDataset', $currentDataset!);
      $queryResultCache = {};
      fetch(
        import.meta.env.BASE_URL + `/datasets/${$currentDataset}/data/fields`
      ).then((resp) =>
        resp.json().then((result) => ($dataFields = result.fields))
      );
      queryHistory = [];
      oldDataset = $currentDataset;
    }
  }

  let refreshTimer: NodeJS.Timeout | null = null;

  onDestroy(() => {
    if (!!refreshTimer) clearTimeout(refreshTimer);
  });

  async function createModel(reference: string) {
    try {
      let newModel = await (
        await fetch(
          import.meta.env.BASE_URL +
            `/datasets/${$currentDataset}/models/new/${reference}`,
          {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'X-CSRF-Token': $csrf },
          }
        )
      ).json();
      $currentModel = newModel.name;
      selectedModels = [];
      currentView = View.editor;
      setTimeout(() => {
        if (!!sidebar && !!$currentModel) sidebar?.editModelName($currentModel);
      }, 100);
    } catch (e) {
      console.error('error creating new model:', e);
    }
    refreshModels();
  }

  async function renameModel(modelName: string, newName: string) {
    if (modelName == newName || newName.length == 0) return;
    try {
      let result = await fetch(
        import.meta.env.BASE_URL +
          `/datasets/${$currentDataset}/models/${newName}`
      );
      if (result.status == 200) {
        alert('A model with that name already exists.');
        return;
      }
    } catch (e) {}

    try {
      await fetch(
        import.meta.env.BASE_URL +
          `/datasets/${$currentDataset}/models/${modelName}/rename`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': $csrf,
          },
          credentials: 'same-origin',
          body: JSON.stringify({
            name: newName,
          }),
        }
      );
      await refreshModels();
      $currentModel = newName;
    } catch (e) {
      console.error('error renaming model:', e);
    }
  }

  async function deleteModels(modelNames: string[]) {
    try {
      await Promise.all(
        modelNames.map((m) =>
          fetch(
            import.meta.env.BASE_URL +
              `/datasets/${$currentDataset}/models/${m}`,
            {
              method: 'DELETE',
              headers: {
                'X-CSRF-Token': $csrf,
              },
              credentials: 'same-origin',
            }
          )
        )
      );
    } catch (e) {
      console.error('error deleting model:', e);
    }
    refreshModels();
  }

  let sidebar: Sidebar;

  let trainingBar: ModelTrainingView;
  let refreshKey: any = {}; // set to a different object when need to refresh the main page

  let enteredUsername: string = '';
  let enteredPassword: string = '';
  let rememberMe: boolean = false;
  let loggingIn: boolean = false;
  async function login() {
    loggingIn = true;
    try {
      let result = await fetch(import.meta.env.BASE_URL + '/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': $csrf,
        },
        body: JSON.stringify({
          user_id: enteredUsername,
          password: enteredPassword,
          remember: rememberMe,
        }),
        credentials: 'same-origin',
      });
      if (result.status != 200) {
        loginErrorMessage = await result.text();
        return;
      }
      let response = await result.json();
      if (response.error) {
        loginErrorMessage = response.error;
        return;
      }
      loginErrorMessage = null;
      showingLogin = false;
      showingSignup = false;
      currentUser = response.user_id;
      refreshDatasets();
    } catch (e) {
      loginErrorMessage = 'Error occurred while logging in';
    }
    loggingIn = false;
  }

  async function signUp() {
    loggingIn = true;
    try {
      let result = await fetch(import.meta.env.BASE_URL + '/create_user', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': $csrf,
        },
        body: JSON.stringify({
          user_id: enteredUsername,
          password: enteredPassword,
          remember: rememberMe,
        }),
        credentials: 'same-origin',
      });
      if (result.status != 200) {
        loginErrorMessage = await result.text();
        return;
      }
      let response = await result.json();
      if (response.error) {
        loginErrorMessage = response.error;
        return;
      }
      loginErrorMessage = null;
      showingLogin = false;
      showingSignup = false;
      currentUser = response.user_id;
      showingDatasetManagement = false;
      refreshDatasets();
    } catch (e) {
      loginErrorMessage = 'Error occurred while creating account';
    }
    loggingIn = false;
  }
</script>

<svelte:document
  on:keydown={(e) => {
    if (
      e.key === 'Escape' &&
      (showingDatasetInfo ||
        showingDatasetManagement ||
        showingQueryBuilder ||
        showingQueryReference)
    ) {
      showingDatasetInfo = false;
      showingDatasetManagement = false;
      showingQueryBuilder = false;
      showingQueryReference = false;
      e.stopPropagation();
      e.preventDefault();
    } else if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
      showingQueryBuilder = !showingQueryBuilder;
      e.preventDefault();
    }
  }}
/>
<main class="w-screen h-screen flex flex-col">
  <div
    class="w-full h-12 grow-0 shrink-0 bg-slate-700 flex py-2 px-4 items-center"
  >
    <button
      class="mr-4 text-white hover:opacity-50"
      on:click={() => (showSidebar = !showSidebar)}
    >
      <Fa icon={faBars} class="text-lg inline" />
    </button>
    <div class="font-bold text-white h-full py-1">
      <img src={logoUrl} class="h-full" alt="Tempo" />
    </div>
    <div class="flex-auto" />
    <button
      class="mr-3 btn btn-dark-slate"
      on:click={() => (showingDatasetManagement = true)}
      ><Fa icon={faDatabase} class="inline mr-2" />
      {#if !!$currentDataset}Dataset <span
          class="font-normal font-mono text-slate-300 text-sm ml-1"
          >{$currentDataset}</span
        >{:else}Select Dataset{/if}
    </button>
    <ActionMenuButton buttonClass="btn btn-dark-slate" align="right">
      <span slot="button-content">
        <Fa icon={faWrench} class="inline mr-2" /> Tools
      </span>
      <div slot="options">
        <a
          href="#"
          tabindex="0"
          role="menuitem"
          title="Open an editor to test out Tempo queries"
          on:click={() => (showingQueryBuilder = true)}
          ><Fa icon={faSearch} class="inline mr-2" /> Query
          <span class="font-normal text-slate-500 text-sm ml-1"
            >{#if navigator.platform
              .toLowerCase()
              .startsWith('mac')}&#8984;K{:else}Cmd+K{/if}</span
          ></a
        >
        <a
          href="#"
          tabindex="0"
          role="menuitem"
          title="Show documentation for the query language"
          on:click={() => (showingQueryReference = true)}
          ><Fa icon={faBook} class="inline mr-2" /> Syntax Reference</a
        >
        <a
          href="#"
          tabindex="0"
          role="menuitem"
          title="Show summary of dataset characteristics"
          class={!$currentDataset ? 'pointer-events-none opacity-50' : ''}
          on:click={() => (showingDatasetInfo = true)}
          ><Fa icon={faChartBar} class="inline mr-2" /> Dataset Info</a
        >
        {#if currentUser !== null}
          <a
            href="/logout"
            tabindex="0"
            role="menuitem"
            title="Log out of this user account"
            ><div class="mb-1">
              <Fa icon={faUserCircle} class="inline mr-2" /> Log Out
            </div>
            <div class="text-xs text-slate-500">
              Logged in: <span class="font-mono">{currentUser}</span>
            </div></a
          >
        {/if}
      </div>
    </ActionMenuButton>
  </div>
  <div class="flex-auto w-full flex h-0">
    {#if showSidebar}
      <ResizablePanel
        rightResizable
        width={540}
        minWidth={360}
        maxWidth="50%"
        collapsible={false}
        height="100%"
      >
        <Sidebar
          bind:this={sidebar}
          bind:metricToShow
          bind:selectedModels
          {isLoadingModels}
          on:new={(e) => createModel(e.detail)}
          on:rename={(e) => renameModel(e.detail.old, e.detail.new)}
          on:delete={(e) => deleteModels(e.detail)}
        />
      </ResizablePanel>
    {/if}
    <div class="flex-auto h-full flex flex-col w-0" style="z-index: 1;">
      <div class="w-full px-4 py-2 flex gap-3 bg-slate-200">
        {#each [View.editor, View.results, View.slices] as view}
          <button
            class="rounded my-2 py-1 text-center w-36 {currentView == view
              ? 'bg-blue-600 text-white font-bold hover:bg-blue-700'
              : 'text-slate-700 hover:bg-slate-300'}"
            on:click={() => (currentView = view)}>{view}</button
          >
        {/each}
      </div>
      {#if !!$currentModel}
        <ModelTrainingView
          bind:this={trainingBar}
          datasetName={$currentDataset}
          modelNames={[$currentModel, ...selectedModels]}
          on:finish={(e) => {
            refreshModels();
            if (e.detail && currentView == View.editor)
              currentView = View.results;
            else refreshKey = {};
          }}
        />
      {/if}
      {#key refreshKey}
        <div
          class="w-full flex-auto h-0"
          class:overflow-y-auto={currentView != View.slices}
        >
          {#if !$currentModel}
            <div class="w-full h-full flex items-center justify-center">
              <div class="text-slate-500">No model selected</div>
            </div>
          {:else if currentView == View.results}
            {#each !!$currentModel ? Array.from(new Set( [$currentModel, ...selectedModels] )) : [] as model}
              <ModelResultsView
                modelName={model}
                modelSummary={$models[model]?.spec}
              />
            {/each}
          {:else if currentView == View.slices}
            <SlicesView
              bind:selectedSlice
              bind:sliceSpec
              bind:savedSlices
              bind:metricToShow
              modelName={$currentModel}
              timestepDefinition={$models[$currentModel ?? '']?.spec
                .timestep_definition ?? ''}
              modelsToShow={!!$currentModel
                ? Array.from(new Set([...selectedModels, $currentModel]))
                : []}
            />
          {:else if currentView == View.editor}
            <ModelEditor
              modelName={$currentModel}
              otherModels={selectedModels.filter((m) => m != $currentModel)}
              on:viewmodel={(e) => {
                currentView = View.results;
                $currentModel = e.detail;
              }}
              on:train={async (e) => {
                if (!!trainingBar) trainingBar.pollTrainingStatus();
                await refreshModels();
              }}
              on:delete={async () => {
                await refreshModels();
                currentView = View.results;
              }}
            />
          {/if}
        </div>
      {/key}
    </div>
  </div>
  {#if isLoadingDatasets || isLoadingUser}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-white/70 flex items-center justify-center flex-col"
    >
      <div class="text-center mb-4">Loading...</div>
      <div role="status">
        <svg
          aria-hidden="true"
          class="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600"
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
    </div>
  {/if}
  {#if showingLogin || showingSignup}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={async () => {
        showingLogin = false;
        showingSignup = false;
        await checkCurrentUser();
        await refreshDatasets();
      }}
      tabindex="0"
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none"
    >
      <div
        class="w-1/2 z-20 rounded-md bg-white p-6 pointer-events-auto overflow-hidden"
        style="min-width: 240px; max-width: 600px;"
      >
        <div class="flex justify-center mb-4">
          <img src={logoLightUrl} class="h-8" alt="Tempo" />
        </div>
        <div class="w-full leading-relaxed mb-4">
          Tempo is an interactive tool designed to help you prototype and
          evaluate predictive models on temporal event datasets, like health
          records. Log in or create an account below to try using Tempo on a
          sample dataset, or upload your own. All user data is deleted every 7
          days.
        </div>
        <div class="flex w-full justify-center">
          <form
            class="w-1/2 p-4 rounded bg-slate-100"
            style="min-width: 400px;"
            on:submit|preventDefault={() => {
              if (showingLogin) login();
              else signUp();
            }}
          >
            <fieldset id="login" class="bg-transparent">
              <legend class="font-bold"
                >{#if showingLogin}Sign In{:else}Create Account{/if}</legend
              >
              {#if !!loginErrorMessage}
                <div class="p-2 my-2 rounded bg-red-100 text-small">
                  {loginErrorMessage}
                </div>
              {/if}
              <div class="mt-2">
                <label class="text-sm" for="user_id">User ID</label>
                <input
                  class="flat-text-input w-full"
                  type="username"
                  name="user_id"
                  bind:value={enteredUsername}
                />
              </div>
              <div class="mt-2">
                <label class="text-sm" for="password">Password</label>
                <input
                  class="flat-text-input w-full"
                  type="password"
                  name="password"
                  bind:value={enteredPassword}
                />
              </div>
              <div class="mt-2">
                <label class="text-sm cursor-pointer"
                  ><input
                    type="checkbox"
                    name="remember"
                    bind:value={rememberMe}
                  /> Remember me</label
                >
              </div>
            </fieldset>
            <div class="">
              <input
                class="mt-2 btn btn-blue cursor-pointer"
                disabled={enteredUsername.length == 0 ||
                  enteredPassword.length == 0}
                type="submit"
                value={showingLogin ? 'Sign In' : 'Create Account'}
              />
            </div>
            <div class="mt-2 text-slate-600">
              {#if showingLogin}Don't have an account yet? <a
                  on:click={() => {
                    showingLogin = false;
                    showingSignup = true;
                    loginErrorMessage = null;
                  }}
                  href="#"
                  class="text-blue-500 hover:opacity-50">Create one</a
                >{:else}Already have an account? <a
                  on:click={() => {
                    showingLogin = true;
                    showingSignup = false;
                    loginErrorMessage = null;
                  }}
                  href="#"
                  class="text-blue-500 hover:opacity-50">Log in</a
                >
              {/if}
            </div>
          </form>
        </div>
      </div>
    </div>
  {:else if showingDatasetInfo}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={() => (showingDatasetInfo = false)}
      tabindex="0"
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none"
    >
      <div
        class="w-2/3 h-2/3 z-20 rounded-md bg-white pointer-events-auto"
        style="min-width: 300px; max-width: 90%;"
      >
        <DatasetInfoView
          on:close={() => (showingDatasetInfo = false)}
          spec={!!$currentDataset
            ? (datasetOptions[$currentDataset]?.spec ?? null)
            : null}
        />
      </div>
    </div>
  {:else if showingDatasetManagement}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={() => (showingDatasetManagement = false)}
      tabindex="0"
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex items-center justify-center pointer-events-none p-16"
    >
      <div
        class="w-full h-full z-20 rounded-md bg-white pointer-events-auto overflow-hidden"
      >
        <DatasetView
          bind:currentDataset={$currentDataset}
          {isLoadingDatasets}
          datasets={datasetOptions}
          on:close={() => (showingDatasetManagement = false)}
          on:refresh={refreshDatasets}
        />
      </div>
    </div>
  {:else if showingQueryBuilder}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-gray-800/30"
      on:click={() => (showingQueryBuilder = false)}
      tabindex="0"
      on:keydown={(e) => {}}
    />
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 flex mt-24 items-start justify-center pointer-events-none"
    >
      <div
        class="w-1/2 z-20 rounded-md bg-white pointer-events-auto shadow-lg"
        style="min-width: 300px; max-width: 70%;"
      >
        <DatasetQueryScratchpad bind:queryHistory />
      </div>
    </div>
  {:else if showingQueryReference}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full z-20 bg-black/70"
      on:click={() => (showingQueryReference = false)}
      tabindex="0"
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
