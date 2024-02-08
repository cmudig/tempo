<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { faEllipsis } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import type {
    SliceChangeDescription,
    SliceVariableDescription,
    SliceVariableValueComparison,
  } from './slicedescription';
  import * as d3 from 'd3';
  import SliceMetricCategoryBar from '../slices/metric_charts/SliceMetricCategoryBar.svelte';

  const dispatch = createEventDispatcher();

  export let variable: SliceVariableDescription | null = null;
  export let valueComparison: SliceVariableValueComparison | null = null;
  export let change: SliceChangeDescription | null = null;

  export let expanded: boolean = false;

  const _ratioPercentFormat = d3.format('.0~%');
  const _ratioTimesFormat = d3.format('.2~');
  const ratioFormat = (v: number) =>
    v > 1 ? _ratioTimesFormat(v + 1) + ' times' : _ratioPercentFormat(v);
</script>

<div
  class="mb-2 p-2 rounded hover:bg-slate-100 relative"
  on:click={() => {
    expanded = !expanded;
    dispatch('toggle', (change ?? variable)?.variable);
  }}
  on:keypress={(e) => {
    if (e.key === 'Enter') {
      expanded = !expanded;
      dispatch('toggle', (change ?? variable)?.variable);
    }
  }}
  role="button"
  tabindex="-1"
>
  {#if !!variable || !!change}
    {@const varToDisplay = change ??
      variable ?? { variable: '', enrichments: [] }}
    <div class="text-sm font-mono mb-1">{varToDisplay.variable}</div>
    {#if !!valueComparison && expanded}
      <div class="flex items-start gap-2">
        <div
          class="w-16 shrink-0 grow-0 font-bold text-slate-500 text-xs text-right mt-2"
        >
          Overall
        </div>
        <div class="flex-auto h-12 overflow-visible">
          <SliceMetricCategoryBar
            width="100%"
            order={valueComparison.values}
            counts={Object.fromEntries(
              valueComparison.values.map((v, i) => [v, valueComparison.base[i]])
            )}
          />
        </div>
      </div>
      <div class="flex items-start gap-2">
        <div
          class="w-16 shrink-0 grow-0 font-bold text-slate-500 text-xs text-right mt-2"
        >
          In slice
        </div>
        <div class="flex-auto h-12 overflow-visible">
          <SliceMetricCategoryBar
            width="100%"
            order={valueComparison.values}
            counts={Object.fromEntries(
              valueComparison.values.map((v, i) => [
                v,
                valueComparison.slice[i],
              ])
            )}
          />
        </div>
      </div>
    {/if}
    {#each varToDisplay.enrichments as enrichment}
      {#if !!change}
        <div class="text-xs text-slate-600 mb-1">
          {ratioFormat(enrichment.ratio)} more likely to {#if enrichment.source_value != enrichment.destination_value}change
            from
            <strong>{enrichment.source_value}</strong> to
            <strong>{enrichment.destination_value}</strong>{:else}remain <strong
              >{enrichment.source_value}</strong
            >{/if}
          in slice than overall
        </div>
      {:else}
        <div class="text-xs text-slate-600 mb-1">
          {ratioFormat(enrichment.ratio)} more likely to be
          <strong>{enrichment.value}</strong>
          in slice than overall
        </div>
      {/if}
    {/each}
  {/if}
  {#if !!valueComparison && !expanded}
    <Fa
      class="absolute right-0 bottom-0 text-slate-400 mr-2 mb-2"
      icon={faEllipsis}
    />
  {/if}
</div>
