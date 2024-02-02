<script lang="ts">
  import { onMount } from "svelte";
  import type { ModelSummary } from "./lib/model";
  import SidebarItem from "./components/SidebarItem.svelte";
  import MainDashboard from "./components/MainDashboard.svelte";

  let models: ModelSummary[] = [];
  let activeModel: ModelSummary | undefined;

  onMount(async () => {
    let result = await fetch("/models");
    models = (await result.json()).models;
    console.log("models:", models);
    activeModel = models[0];
  });
</script>

<main class="m-4 font-sans">
  <div class="flex flex-row">
    <div class="mr-8">
      <h2>MODELS</h2>
      <div class="mt-2">
        {#each models as model}
          <SidebarItem
            {model}
            isActive={activeModel === model}
            on:click={() => (activeModel = model)}
          />
          <hr />
        {/each}
      </div>
    </div>

    <div>
      <h2>MODEL STATS</h2>
      {#if activeModel}
        <MainDashboard model={activeModel} />
      {/if}
    </div>
  </div>
</main>
