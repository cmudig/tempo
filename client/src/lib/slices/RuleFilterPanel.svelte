<script lang="ts">
  import Fa from 'svelte-fa';
  import { RuleFilterType, type RuleFilter } from './rulefilters';
  import { faFilter } from '@fortawesome/free-solid-svg-icons';
  import RuleFilterEditor from './RuleFilterEditor.svelte';

  export let ruleFilterSpec: RuleFilter | null = null;
  export let changesPending: boolean = false;
  export let valueNames: {
    [key: string]: [any, { [key: string]: any }];
  } | null = {};

  $: console.log('new rule filter:', ruleFilterSpec, valueNames);
</script>

<div class="mt-2 mb-1 font-bold">Rule Filters</div>
<div class="text-slate-500 text-xs mb-2">
  Choose which features are allowed in subgroup rules. More restrictions can
  improve search speed.
</div>
{#if !ruleFilterSpec}
  <button
    class="btn btn-slate disabled:opacity-50 shrink-0"
    on:click={() => {
      ruleFilterSpec = {
        type: 'constraint',
        logic: RuleFilterType.exclude,
        features: [],
        values: [],
      };
    }}><Fa icon={faFilter} class="inline mr-2" />Define Filter</button
  >
{:else}
  <RuleFilterEditor
    bind:ruleFilter={ruleFilterSpec}
    on:delete={(e) => {
      ruleFilterSpec = null;
    }}
    topLevel
    {valueNames}
    bind:changesPending
  />
{/if}
