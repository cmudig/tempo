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
</script>

<div class="flex flex-col w-full h-full">
  <div class="w-full sticky top-0">
    <div class="py-2 px-4 flex items-center grow-0 shrink-0 gap-2">
      <div class="text-lg font-bold whitespace-nowrap shrink-1 overflow-hidden">
        Datasets
      </div>
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
              title="Rename this model"
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
          isChecked={selectedDatasets.includes(dsName) ||
            currentDataset === dsName}
          allowCheck={currentDataset != dsName}
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
            editingDatasetName = null;
            dispatch('rename', e.detail);
          }}
          on:delete={(e) => dispatch('delete', [e.detail])}
        />
      {/each}
    </div>
  {/if}
</div>
