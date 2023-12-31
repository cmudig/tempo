<script lang="ts">
  import type { ModelSummary } from './model';
  import SidebarItem from './SidebarItem.svelte';

  export let models: { [key: string]: ModelSummary } = {};
  export let activeModel: string | undefined;
</script>

<div class="my-2 text-lg font-bold px-4">Models</div>
{#each Object.entries(models) as [modelName, model]}
  <SidebarItem
    {model}
    {modelName}
    isActive={activeModel === modelName}
    on:click={() => (activeModel = modelName)}
  />
{/each}
