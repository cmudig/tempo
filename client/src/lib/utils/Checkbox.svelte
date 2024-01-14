<script lang="ts">
  import { faCheck } from '@fortawesome/free-solid-svg-icons';
  import { createEventDispatcher } from 'svelte';
  import Fa from 'svelte-fa/src/fa.svelte';

  export let checked = false;
  export let colorClass: string | null = null;
  export let disabled = false;

  const dispatch = createEventDispatcher();
</script>

<button
  class="mr-1 inline checkbox rounded flex items-center justify-center text-white {colorClass !=
  null
    ? colorClass
    : checked
      ? 'bg-blue-400'
      : 'bg-slate-300 hover:bg-slate-400'}"
  {disabled}
  class:opacity-50={disabled}
  on:click|stopPropagation={(e) => {
    checked = !checked;
    dispatch('change', checked);
  }}
>
  {#if checked}
    <Fa icon={faCheck} />
  {/if}
</button>

<style>
  .checkbox {
    width: 18px;
    height: 18px;
  }
</style>
