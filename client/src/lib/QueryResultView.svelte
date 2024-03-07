<script lang="ts">
  import Fa from 'svelte-fa';
  import type { QueryResult, VariableEvaluationSummary } from './model';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import SliceMetricCategoryBar from './slices/metric_charts/SliceMetricCategoryBar.svelte';
  import SliceMetricHistogram from './slices/metric_charts/SliceMetricHistogram.svelte';
  import * as d3 from 'd3';
  import { faDownload } from '@fortawesome/free-solid-svg-icons';

  export let query: string = '';
  export let evaluateQuery: boolean = true;
  export let delayEvaluation: boolean = false;
  export let showName: boolean = false;

  let evaluationTimer: number | null = null;

  $: if (evaluateQuery) {
    if (query.length == 0) {
      loadingSummary = false;
      summaryIsStale = false;
      evaluationError = null;
      evaluationSummary = null;
    }
    if (delayEvaluation) {
      summaryIsStale = true;
      if (!!evaluationTimer) clearTimeout(evaluationTimer);
      if (query.length > 0)
        evaluationTimer = setTimeout(liveEvaluateQuery, 2000);
    } else if (query.length > 0) {
      summaryIsStale = true;
      liveEvaluateQuery(query);
    }
  }
  export let metricWidth = 176;

  export let evaluationSummary: QueryResult | null = null;
  export let evaluationError: string | null = null;
  let evaluatedLength: number | null = null;
  let evaluatedType: string | null = null;
  let loadingSummary: boolean = false;
  let summaryIsStale: boolean = false;

  async function liveEvaluateQuery(q: string) {
    loadingSummary = true;
    let encodedQuery = encodeURIComponent(query);
    try {
      let result = await (await fetch(`/data/query?q=${encodedQuery}`)).json();
      if (result.error) {
        evaluationError = result.error;
        evaluatedLength = null;
        evaluatedType = null;
        evaluationSummary = null;
      } else if (result.query == query && !!result.result) {
        evaluationSummary = result.result as QueryResult;
        evaluatedType = result.result_type;
        evaluatedLength = result.n_values;
        evaluationError = null;
      }
      loadingSummary = false;
      summaryIsStale = false;
    } catch (e) {
      evaluationError = `${e}`;
      evaluationSummary = null;
      loadingSummary = false;
    }
  }

  let showHeaders = false;
  $: if (!!evaluationSummary)
    showHeaders =
      !!evaluationSummary.occurrences || !!evaluationSummary.durations;
</script>

<div class="text-sm w-full" class:opacity-50={summaryIsStale}>
  {#if loadingSummary}
    <div class="mb-1 text-slate-500 text-xs">Loading...</div>
  {:else if !!evaluationError}
    <div class="text-red-600">
      {@html '<p>' + evaluationError.replace('\n', '</p><p>') + '</p>'}
    </div>
  {:else if !!evaluationSummary}
    {#if showName && !!evaluationSummary.name}
      <div class="mb-2 font-mono">
        {evaluationSummary.name}
      </div>
    {:else}
      <div class="mb-1 text-slate-500 text-xs flex justify-between">
        <div class="font-bold">Query Result</div>
        <a
          class="hover:opacity-50 ml-2"
          title="Download query result for all data splits"
          href="/data/query?q={encodeURIComponent(query)}&dl=1"
          target="_blank"><Fa icon={faDownload} class="inline" /></a
        >
      </div>
    {/if}
    {#if !!evaluatedType}
      <div class="mb-2 text-slate-600 text-xs">
        {evaluatedType}
        {#if evaluatedLength != null}with {d3.format(',')(evaluatedLength)} values{/if}
      </div>
    {/if}
    {#if !!evaluationSummary.occurrences}
      {@const values = evaluationSummary.occurrences}
      {#if showHeaders}<div class="mb-1 text-xs text-slate-500">
          Occurrences per Trajectory
        </div>{/if}
      <div class="h-12">
        {#if values.type == 'binary' && !!values.mean}
          <SliceMetricBar
            value={values.mean}
            width={metricWidth}
            color="#d97706"
            showFullBar
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(values.mean)}</strong>
              true,
              <strong>{d3.format('.1%')(1 - values.mean)}</strong> false
            </span>
          </SliceMetricBar>
        {:else if values.type == 'continuous' && !!values.hist}
          <SliceMetricHistogram
            mean={values.mean ?? 0}
            histValues={values.hist}
            width={metricWidth}
          />
        {:else if values.type == 'categorical' && !!values.counts}
          <SliceMetricCategoryBar
            order={Object.keys(values.counts).sort(
              (a, b) => values.counts[b] - values.counts[a]
            )}
            counts={values.counts}
            width={metricWidth}
          />
        {/if}
      </div>
    {/if}
    {#if !!evaluationSummary.durations}
      {@const values = evaluationSummary.durations}
      {#if showHeaders}<div class="mb-1 text-xs text-slate-500">
          Durations
        </div>{/if}
      <div class="h-12">
        {#if values.type == 'binary' && !!values.mean}
          <SliceMetricBar
            value={values.mean}
            width={metricWidth}
            color="#d97706"
            showFullBar
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(values.mean)}</strong>
              true,
              <strong>{d3.format('.1%')(1 - values.mean)}</strong> false
            </span>
          </SliceMetricBar>
        {:else if values.type == 'continuous' && !!values.hist}
          <SliceMetricHistogram
            mean={values.mean ?? 0}
            histValues={values.hist}
            width={metricWidth}
          />
        {:else if values.type == 'categorical' && !!values.counts}
          <SliceMetricCategoryBar
            order={Object.keys(values.counts).sort(
              (a, b) => values.counts[b] - values.counts[a]
            )}
            counts={values.counts}
            width={metricWidth}
          />
        {/if}
      </div>
    {/if}
    {#if !!evaluationSummary.values}
      {@const values = evaluationSummary.values}
      {#if showHeaders}<div class="mb-1 text-xs text-slate-500">
          Values
        </div>{/if}
      <div class="h-12">
        {#if values.type == 'binary' && !!values.mean}
          <SliceMetricBar
            value={values.mean}
            width={metricWidth}
            color="#d97706"
            showFullBar
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(values.mean)}</strong>
              true,
              <strong>{d3.format('.1%')(1 - values.mean)}</strong> false
            </span>
          </SliceMetricBar>
        {:else if values.type == 'continuous' && !!values.hist}
          <SliceMetricHistogram
            mean={values.mean ?? 0}
            histValues={values.hist}
            width={metricWidth}
          />
        {:else if values.type == 'categorical' && !!values.counts}
          <SliceMetricCategoryBar
            order={Object.keys(values.counts).sort(
              (a, b) => values.counts[b] - values.counts[a]
            )}
            counts={values.counts}
            width={metricWidth}
          />
        {/if}
      </div>
      {#if !!values.missingness}
        <div class="mt-1 text-xs text-red-700">
          <strong
            >{(values.missingness < 0.1
              ? d3.format('.3~%')
              : d3.format('.1~%'))(values.missingness)}</strong
          >
          missing values
        </div>
      {/if}
    {/if}
  {/if}
</div>
