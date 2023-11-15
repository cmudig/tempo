<script lang="ts">
  import { onMount } from 'svelte';
  import type { ModelSummary } from './lib/model';

  let models: ModelSummary[] = [];

  onMount(async () => {
    let result = await fetch('/models');
    models = (await result.json()).models;
    console.log('models:', models);
  });
</script>

<main class="m-4">
  <h2 class="text-lg text-blue-600">Hello World</h2>
</main>
