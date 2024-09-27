<script lang="ts">
  import { ModelArchitectureType, type ModelArchitectureInfo } from '../model';

  export let modelArchitecture: ModelArchitectureInfo | null = null;
  $: console.log('architecture type:', modelArchitecture);
</script>

{#if !!modelArchitecture}
  <div class="mb-1">
    <span class="text-sm">Model type:</span>
    <select bind:value={modelArchitecture.type} class="flat-select ml-2">
      {#each ['xgboost', 'dense', 'rnn', 'transformer'] as archType}
        <option value={archType}>{ModelArchitectureType[archType]}</option>
      {/each}
    </select>
  </div>
{/if}
