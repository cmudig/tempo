<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Dataset } from '../dataset';
  import ResizablePanel from '../utils/ResizablePanel.svelte';
  import DatasetInfoView from './DatasetInfoView.svelte';
  import DatasetSidebar from './DatasetSidebar.svelte';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';

  const dispatch = createEventDispatcher();

  export let showSidebar: boolean = true;
  export let datasets: { [key: string]: { spec: Dataset; models: string[] } } =
    {};
  export let currentDataset: string | null = null;

  let clearingCache: boolean = false;
  let clearCacheMessage: string | null = null;
  async function clearCache(target: string) {
    try {
      clearCacheMessage = null;
      clearingCache = true;
      let result = await fetch(`/datasets/${currentDataset}/clear_cache`, {
        body: JSON.stringify({ target }),
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (result.status != 200) console.error('error clearing cache:', result);
      else clearCacheMessage = await result.text();
      clearingCache = false;
      setTimeout(() => (clearCacheMessage = null), 10000);
    } catch (e) {
      console.error('error clearing cache:', e);
      clearingCache = false;
    }
  }
</script>

<div class="w-full h-full flex">
  {#if showSidebar}
    <ResizablePanel
      rightResizable
      width={540}
      minWidth={360}
      maxWidth="50%"
      collapsible={false}
      height="100%"
    >
      <div class="mt-1">
        <DatasetSidebar {datasets} bind:currentDataset />
      </div>
      <!-- on:new={(e) => createModel(e.detail)}
    on:rename={(e) => renameModel(e.detail.old, e.detail.new)}
    on:delete={(e) => deleteModels(e.detail)} -->
    </ResizablePanel>
  {/if}
  <div class="flex-auto h-full flex flex-col">
    <div class="flex shrink-0 p-4">
      <div
        class="text-lg font-bold whitespace-nowrap flex-auto truncate font-mono"
      >
        {currentDataset}
      </div>
      <button
        class="text-slate-600 px-2 hover:opacity-50"
        on:click={() => dispatch('close')}
        ><Fa icon={faXmark} class="inline" /></button
      >
    </div>
    <div
      class="mb-4 px-4 grid gap-4 items-center"
      style="grid-template-columns: max-content auto;"
    >
      <button
        class="btn btn-red"
        disabled={clearingCache}
        on:click={() => clearCache('variables')}
      >
        Clear Variable Caches
      </button>
      <div class="flex-auto text-sm text-slate-500">
        Remove all computed variables for model training, slicing, and queries.
      </div>
      <button
        class="btn btn-red"
        disabled={clearingCache}
        on:click={() => clearCache('models')}
      >
        Clear Trained Models
      </button>
      <div class="flex-auto text-sm text-slate-500">
        Remove all model training results (without removing the specifications).
      </div>
      <button
        class="btn btn-red"
        disabled={clearingCache}
        on:click={() => clearCache('slices')}
      >
        Clear Subgroup Results
      </button>
      <div class="flex-auto text-sm text-slate-500">
        Remove discovered subgroups.
      </div>
      <button
        class="btn btn-red"
        disabled={clearingCache}
        on:click={() => clearCache('all')}
      >
        Clear All Caches
      </button>
      <div class="flex-auto text-sm text-slate-500">
        Remove all of the above cached data for this dataset, including the
        train/val/test split.
      </div>
    </div>
    {#if !!clearCacheMessage}
      <div class="px-4 mb-4 text-blue-500 text-sm">{clearCacheMessage}</div>
    {/if}
    <div class="min-h-0 flex-auto overflow-auto px-4 pb-4">
      <div class="p-4 border-2 border-slate-300 rounded-md">
        <DatasetInfoView showCloseButton={false} />
      </div>
    </div>
  </div>
</div>
