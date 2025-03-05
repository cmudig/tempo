<svelte:options accessors />

<script lang="ts">
  import { faPlus } from '@fortawesome/free-solid-svg-icons';
  import type { Dataset } from '../dataset';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher } from 'svelte';
  import SidebarItem from '../sidebar/SidebarItem.svelte';

  const dispatch = createEventDispatcher();

  export let datasets: { [key: string]: { spec: Dataset; models: string[] } } =
    {};
  export let currentDataset: string | null = null;
  export let selectedDatasets: string[] = [];
  export let editingDatasetName: string | null = null;
  export let isLoadingDatasets: boolean = false;

  export function editDatasetName(datasetName: string) {
    editingDatasetName = datasetName;
  }
</script>

<div class="flex flex-col w-full h-full">
  <div class="w-full sticky top-0">
    <div class="py-2 px-4 flex items-center grow-0 shrink-0 gap-2">
      <div class="text-lg font-bold whitespace-nowrap shrink-1 overflow-hidden">
        Datasets
      </div>
      {#if isLoadingDatasets}
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
      <button
        class="bg-transparent px-1 hover:opacity-40"
        on:click={() => dispatch('new')}
        ><Fa icon={faPlus} class="inline" /></button
      >
      <ActionMenuButton
        buttonClass="bg-transparent px-1 hover:opacity-40"
        align="right"
      >
        <div slot="options">
          {#if selectedDatasets.length == 0}
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Create a copy of this dataset"
              on:click={() => dispatch('new', currentDataset)}>Duplicate</a
            >
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              title="Rename this dataset"
              on:click={() => (editingDatasetName = currentDataset ?? null)}
              >Rename...</a
            >
          {/if}
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            title="Permanently delete these datasets"
            on:click={() =>
              dispatch('delete', [currentDataset, ...selectedDatasets])}
            >Delete</a
          >
        </div>
      </ActionMenuButton>
    </div>
    <div class="flex px-4 my-2 items-center w-full"></div>
  </div>
  {#if Object.keys(datasets).length == 0}
    <div
      class="w-full mt-6 flex-auto min-h-0 flex flex-col items-center justify-center text-slate-500"
    >
      <div>No datasets yet!</div>
    </div>
  {:else}
    <div class="overflow-y-auto flex-auto min-h-0">
      {#each Object.keys(datasets).sort() as dsName (dsName)}
        {@const dataset = datasets[dsName]}
        <SidebarItem
          displayItem={{
            name: dsName,
            description: `${dataset.models.length} model${dataset.models.length != 1 ? 's' : ''}`,
          }}
          displayItemType="dataset"
          isEditingName={editingDatasetName == dsName}
          isActive={currentDataset === dsName}
          showCheckbox={false}
          on:click={() => {
            currentDataset = dsName;
            selectedDatasets = [];
          }}
          on:toggle={(e) => {
            let idx = selectedDatasets.indexOf(dsName);
            if (idx >= 0)
              selectedDatasets = [
                ...selectedDatasets.slice(0, idx),
                ...selectedDatasets.slice(idx + 1),
              ];
            else selectedDatasets = [...selectedDatasets, dsName];
          }}
          on:duplicate={(e) => dispatch('new', e.detail)}
          on:editname={(e) => (editingDatasetName = e.detail)}
          on:canceledit={(e) => (editingDatasetName = null)}
          on:rename={(e) => {
            console.log('in sidebar', e.detail);
            editingDatasetName = null;
            dispatch('rename', e.detail);
          }}
          on:delete={(e) =>
            dispatch(
              'delete',
              Array.from(
                new Set([e.detail, currentDataset, ...selectedDatasets])
              )
            )}
        />
      {/each}
    </div>
  {/if}
</div>
