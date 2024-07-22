<script lang="ts">
  import ScoreFunctionEditor from './ScoreFunctionEditor.svelte';
  import type { ScoreFunction } from './scorefunctions';

  export let scoreFunctionSpec: ScoreFunction[] = [];
  export let changesPending: boolean = false;

  let anyChangesPending: boolean[] = [];
  $: changesPending = anyChangesPending.some((v) => v);
</script>

<div class="mt-2 mb-1 font-bold">Search Criteria</div>
<div class="text-slate-500 text-xs mb-2">
  Search for slices that contain more instances matching these characteristics:
</div>
{#each scoreFunctionSpec as scoreFunction, i}
  <ScoreFunctionEditor
    {scoreFunction}
    on:change={(e) => {
      scoreFunctionSpec = [
        ...scoreFunctionSpec.slice(0, i),
        e.detail,
        ...scoreFunctionSpec.slice(i + 1),
      ];
    }}
    topLevel
    allowDelete={false}
    bind:changesPending={anyChangesPending[i]}
  />
{/each}
