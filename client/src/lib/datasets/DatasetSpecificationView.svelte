<script lang="ts">
  import type { Dataset } from '../dataset';

  export let datasetName: string | null = null;
  export let spec: Dataset | null = null;

  $: console.log('dataset spec:', spec);
</script>

{#if !!datasetName && !!spec}
  <div class="px-4 font-bold mb-2">Data Sources</div>
  <div class="px-4 font-bold mb-2">Split Sizes</div>
  <div class="px-4 font-bold mb-2">Subgroup Discovery Settings</div>
{/if}
