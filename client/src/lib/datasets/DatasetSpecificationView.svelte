<script lang="ts">
  import type { Writable } from 'svelte/store';
  import type { Dataset, DataSource } from '../dataset';
  import { createEventDispatcher, getContext } from 'svelte';
  import Fa from 'svelte-fa';
  import { faPlus, faXmark } from '@fortawesome/free-solid-svg-icons';
  import DataSourceEditor from './DataSourceEditor.svelte';
  import { areObjectsEqual, deepCopy } from '../slices/utils/utils';

  const dispatch = createEventDispatcher();

  let csrf: Writable<string> = getContext('csrf');

  export let datasetName: string | null = null;
  export let spec: Dataset | null = null;
  export let hasModels: boolean = false;
  let draftSpec: Dataset | null = null;
  let hasDraft: boolean = false;
  let draftSaved: boolean = true;
  let isLoadingSpec: boolean = false;

  $: if (!!datasetName && !!spec) {
    loadDraftSpec();
  }

  async function loadDraftSpec() {
    console.log('loading draft spec');
    isLoadingSpec = true;
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/draft`
      );
      let newDraft = (await result.json()).spec;
      console.log(
        'loaded draft spec:',
        newDraft,
        !!newDraft && Object.keys(newDraft).length > 0
      );
      hasDraft = !!newDraft && Object.keys(newDraft).length > 0;
      if (!hasDraft) newDraft = spec;
      setDraftSpec(newDraft);
      draftSaved = true;
    } catch (e) {
      console.error('Error loading draft spec:', e);
    }
    isLoadingSpec = false;
  }

  async function updateDraftSpec() {
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/draft`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': $csrf,
          },
          body: JSON.stringify({ spec: draftSpec }),
          credentials: 'same-origin',
        }
      );
      setDraftSpec((await result.json()).spec);
      console.log('updated draft');
      draftSaved = true;
      hasDraft = true;
    } catch (e) {
      console.error('Error saving draft spec:', e);
    }
  }

  async function buildDataset() {
    if (
      hasModels &&
      !confirm(
        'Are you sure you want to build the dataset? Model training and subgroup discovery results will be deleted. (Specifications will be saved.)'
      )
    )
      return;
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/spec`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': $csrf,
          },
          body: JSON.stringify({ spec: draftSpec }),
          credentials: 'same-origin',
        }
      );
      if (result.status != 200) {
        console.error('Error building dataset:', await result.text());
        return;
      }
      let taskID = (await result.json()).task_id;
      console.log('task ID:', taskID);
      dispatch('build', taskID);
      setTimeout(() => {
        loadDraftSpec();
      }, 100);
    } catch (e) {
      console.error('Error building dataset:', e);
    }
  }

  async function reset() {
    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/draft`,
        {
          method: 'DELETE',
          headers: {
            'X-CSRF-Token': $csrf,
          },
          credentials: 'same-origin',
        }
      );
      console.log(await result.text());
      if (!!spec) setDraftSpec(spec);
      dispatch('refresh');
      hasDraft = false;
      draftSaved = true;
    } catch (e) {
      console.error('Error saving draft spec:', e);
    }
  }

  function setDraftSpec(newDraft: Dataset) {
    console.log('setting draft spec to', newDraft, hasDraft);
    draftSpec = newDraft;
    dataSources = deepCopy(draftSpec?.data.sources ?? []);
  }

  let addingNewSource: boolean = false;
  let isUploading: boolean = false;
  let uploadError: string | null = null;

  let fileInput: HTMLInputElement;
  async function uploadFile() {
    if (!fileInput || !fileInput.files || fileInput.files.length == 0) return;
    isUploading = true;
    var data = new FormData();
    data.append('newfile', fileInput.files[0]);

    try {
      let result = await fetch(
        import.meta.env.BASE_URL + `/datasets/${datasetName}/draft/add_source`,
        {
          method: 'POST',
          body: data,
          headers: {
            'X-CSRF-Token': $csrf,
          },
          credentials: 'same-origin',
        }
      );
      if (result.status != 200) {
        uploadError = await result.text();
      } else {
        setDraftSpec((await result.json()).spec);
        uploadError = null;
        addingNewSource = false;
      }
    } catch (e) {
      uploadError = `${e}`;
    }
    isUploading = false;
  }

  let dataSources: DataSource[] = [];

  let saveDraftTimer: NodeJS.Timeout | null = null;
  $: if (
    !!draftSpec &&
    !areObjectsEqual(draftSpec?.data?.sources, dataSources)
  ) {
    console.log('saving draft');
    draftSaved = false;
    scheduleSaveDraft();
    draftSpec = {
      ...draftSpec,
      data: {
        ...draftSpec.data,
        sources: dataSources,
      },
      slices: draftSpec.slices,
    };
  }

  function scheduleSaveDraft() {
    if (!!saveDraftTimer) clearTimeout(saveDraftTimer);
    saveDraftTimer = setTimeout(updateDraftSpec, 5000);
  }
</script>

<div class="p-4 grow-0 shrink-0 w-full flex items-center gap-2">
  <div class="text-lg font-bold whitespace-nowrap truncate font-mono flex-auto">
    Specification for {datasetName}
  </div>
  <button class="btn btn-blue" on:click={buildDataset}> Build </button>
  {#if hasDraft || !draftSaved}
    <button class="my-1 btn btn-slate" on:click={reset}> Revert </button>
  {/if}
</div>
<div class="flex-auto min-h-0 overflow-auto w-full">
  {#if !!datasetName && !!draftSpec}
    {#if hasDraft || !draftSaved}
      <div
        class="mx-4 mb-2 flex items-center rounded {draftSaved
          ? 'bg-sky-100 text-sky-600'
          : 'bg-orange-100 text-orange-700'} transition-colors duration-300 px-3 text-sm"
      >
        <div class="flex-auto py-2">
          {#if draftSaved}<strong>Draft saved&nbsp;</strong> Changes have not yet
            been reflected in the version of the dataset used throughout the system.
          {:else}
            <strong>Unsaved changes&nbsp;</strong> A draft will be saved automatically...{/if}
        </div>
        {#if !draftSaved}
          <div class="shrink-0">
            <button
              class="btn-sm bg-orange-200 hover:bg-orange-300 text-orange-700"
              on:click={updateDraftSpec}>Save Draft</button
            >
          </div>
        {/if}
      </div>
    {/if}
    {#if !!draftSpec.error}
      <div class="mx-4 rounded my-2 p-3 text-red-500 bg-red-50 font-mono">
        {draftSpec.error}
      </div>
    {/if}
    <div class="px-4 font-bold mb-2">Data Sources</div>
    {#each dataSources as source, i}
      <DataSourceEditor
        bind:source
        on:delete={(e) => {
          dataSources = [
            ...dataSources.slice(0, i),
            ...dataSources.slice(i + 1),
          ];
        }}
      />
    {/each}
    {#if addingNewSource}
      <div class="mx-4 mb-2 rounded bg-slate-100 p-2 flex items-center gap-2">
        {#if isUploading}
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
        <input
          type="file"
          bind:this={fileInput}
          name="newfile"
          accept=".csv,.arrow"
          on:change={(e) => {
            if (!!e.target && e.target.files[0].size > 1024 * 1024 * 1024) {
              alert('Please upload files smaller than 1 GB.');
              e.target.value = '';
            }
          }}
        />
        <button class="btn btn-slate" on:click={uploadFile}>Upload</button>
        <div class="flex-auto" />
        <button
          class="hover:opacity-50"
          on:click={() => (addingNewSource = false)}
          ><Fa icon={faXmark} /></button
        >
      </div>
      {#if !!uploadError}
        <div class="mx-4 mb-2 text-red-500 text-sm">
          {uploadError}
        </div>
      {/if}
    {/if}
    <button
      class="mx-4 btn btn-slate"
      disabled={addingNewSource}
      on:click={(e) => (addingNewSource = true)}
    >
      <Fa icon={faPlus} class="inline mr-2" />Add New
    </button>
    <div class="px-4 font-bold mt-4 mb-2">Split Sizes</div>
    <div class="px-4 font-bold mt-4 mb-2">Subgroup Discovery Settings</div>
  {/if}
</div>
