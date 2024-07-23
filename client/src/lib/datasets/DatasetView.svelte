<script lang="ts">
  import type { Dataset } from '../dataset';
  import ResizablePanel from '../utils/ResizablePanel.svelte';
  import DatasetInfoView from './DatasetInfoView.svelte';
  import DatasetSidebar from './DatasetSidebar.svelte';

  export let showSidebar: boolean = true;
  export let datasets: { [key: string]: { spec: Dataset; models: string[] } } =
    {};
  export let currentDataset: string | null = null;
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
  <div class="flex-auto h-full">
    <DatasetInfoView />
  </div>
</div>
