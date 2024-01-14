<script lang="ts">
  import {
    AllCategories,
    VariableCategory,
    type VariableDefinition,
    type VariableEvaluationSummary,
  } from './model';
  import Checkbox from './utils/Checkbox.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import * as d3 from 'd3';
  import { createEventDispatcher } from 'svelte';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import SliceMetricHistogram from './slices/metric_charts/SliceMetricHistogram.svelte';
  import SliceMetricCategoryBar from './slices/metric_charts/SliceMetricCategoryBar.svelte';

  const dispatch = createEventDispatcher();

  export let varName: string = '';
  export let varInfo: VariableDefinition | null = null;
  export let editing = false;

  export let timestepDefinition: string = '';

  let newVariableName: string | null = null;
  let newVariableQuery: string | null = null;

  let evaluationSummary: VariableEvaluationSummary | null = null;
  let evaluationError: string | null = null;
  let summaryIsStale: boolean = false;

  $: if (editing && !!varInfo) {
    if (newVariableName == null) {
      newVariableName = varName;
      newVariableQuery = varInfo!.query;
    }
  } else {
    newVariableName = null;
    newVariableQuery = null;
  }

  $: if (editing && !!newVariableQuery) {
    if (!evaluationSummary && !evaluationError) {
      evaluateQuery();
    }
  } else {
    evaluationSummary = null;
    evaluationError = null;
  }

  async function evaluateQuery() {
    evaluationTimer = null;
    let query = encodeURIComponent(
      `(${newVariableQuery}) ${timestepDefinition}`
    );
    try {
      let result = await (await fetch(`/data/query?q=${query}`)).json();
      console.log(result);
      if (result.error) {
        evaluationError = result.error;
        evaluationSummary = null;
      } else if (result.summary) {
        evaluationSummary = result.summary as VariableEvaluationSummary;
        evaluationError = null;
      }
      summaryIsStale = false;
    } catch (e) {
      evaluationError = `${e}`;
      evaluationSummary = null;
      summaryIsStale = false;
    }
  }

  let evaluationTimer: NodeJS.Timeout | null = null;
  const metricWidth = 176;
</script>

{#if !!varInfo && !!varName}
  <div class="ml-2 mb-1 flex items-center gap-1">
    <Checkbox
      checked={varInfo.enabled}
      on:change={(e) => {
        dispatch('toggle', e.detail);
      }}
    />
    <div class="w-2" />
    {#if editing}
      <div class="flex-auto flex flex-col gap-2 h-full">
        <input
          type="text"
          class="flat-text-input w-full font-mono text-sm"
          placeholder="Variable Name"
          bind:value={newVariableName}
        />
        <div class="flex flex-auto w-full">
          <div
            class="text-sm w-48 self-stretch bg-slate-200 rounded mr-2 p-2"
            class:opacity-50={summaryIsStale}
          >
            {#if !!evaluationError}
              <div class="text-red-600">
                {@html '<p>' +
                  evaluationError.replace('\n', '</p><p>') +
                  '</p>'}
              </div>
            {:else if !!evaluationSummary}
              <div class="mt-2">
                {#if evaluationSummary.type == 'binary' && !!evaluationSummary.rate}
                  <SliceMetricBar
                    value={evaluationSummary.rate}
                    width={metricWidth}
                  >
                    <span slot="caption">
                      <strong>{d3.format('.1%')(evaluationSummary.rate)}</strong
                      >
                      true,
                      <strong
                        >{d3.format('.1%')(1 - evaluationSummary.rate)}</strong
                      > false
                    </span>
                  </SliceMetricBar>
                {:else if evaluationSummary.type == 'continuous' && !!evaluationSummary.hist}
                  <SliceMetricHistogram
                    mean={evaluationSummary.mean ?? 0}
                    histValues={Object.fromEntries(
                      evaluationSummary.hist.bins
                        .slice(0, evaluationSummary.hist.bins.length - 1)
                        .map((b, i) => [b, evaluationSummary.hist.counts[i]])
                    )}
                    width={metricWidth}
                  />
                {:else if evaluationSummary.type == 'categorical' && !!evaluationSummary.counts}
                  <SliceMetricCategoryBar
                    order={Object.keys(evaluationSummary.counts).sort(
                      (a, b) =>
                        evaluationSummary.counts[b] -
                        evaluationSummary.counts[a]
                    )}
                    counts={evaluationSummary.counts}
                    width={metricWidth}
                  />
                {/if}

                <p class="mt-2 text-xs text-slate-500">
                  {evaluationSummary.n_trajectories} trajectories, {evaluationSummary.n_values}
                  values
                </p>
              </div>
            {/if}
          </div>
          <div class="flex-auto">
            <div class="mb-1 text-slate-500 text-xs w-32">Query</div>
            <input
              type="text"
              class="flat-text-input w-full font-mono"
              bind:value={newVariableQuery}
              on:input={() => {
                if (!!evaluationTimer) clearTimeout(evaluationTimer);
                evaluationTimer = setTimeout(evaluateQuery, 2000);
                summaryIsStale = true;
              }}
            />
            <div class="mt-2 flex gap-1">
              <button
                class="my-1 py-1 text-sm px-3 rounded text-slate-800 bg-slate-200 hover:bg-slate-300 font-bold"
                on:click={() => dispatch('cancel')}>Cancel</button
              >
              <button
                class="my-1 py-1 text-sm px-3 rounded text-slate-800 bg-slate-200 hover:bg-slate-300 font-bold"
                class:opacity-30={newVariableQuery == varInfo.query}
                disabled={newVariableQuery == varInfo.query}
                on:click={() =>
                  dispatch('save', {
                    name: newVariableName,
                    query: newVariableQuery,
                  })}>Save</button
              >
            </div>
          </div>
        </div>
      </div>
    {:else}
      <button
        class="font-mono hover:bg-slate-200 rounded flex-auto text-left px-2 py-1"
        on:click={() => dispatch('edit')}>{varName}</button
      >
    {/if}
  </div>
{/if}
