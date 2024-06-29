<script lang="ts">
  import Fa from 'svelte-fa';
  import type {
    QueryEvaluationResult,
    QueryResult,
    VariableEvaluationSummary,
  } from './model';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import SliceMetricCategoryBar from './slices/metric_charts/SliceMetricCategoryBar.svelte';
  import SliceMetricHistogram from './slices/metric_charts/SliceMetricHistogram.svelte';
  import * as d3 from 'd3';
  import { faDownload } from '@fortawesome/free-solid-svg-icons';
  import {
    createEventDispatcher,
    getContext,
    onDestroy,
    onMount,
  } from 'svelte';
  import type { Writable } from 'svelte/store';
  import { base64ToBlob } from './slices/utils/utils';

  const dispatch = createEventDispatcher();

  let {
    currentDataset,
    queryResultCache,
  }: {
    currentDataset: Writable<string | null>;
    queryResultCache: Writable<{ [key: string]: QueryEvaluationResult }>;
  } = getContext('dataset');

  let container: HTMLElement;
  export let query: string = '';
  export let evaluateQuery: boolean = true;
  export let delayEvaluation: boolean = false;
  export let showName: boolean = false;
  export let compact: boolean = false;

  let evaluationTimer: number | null = null;

  let mounted = false;
  let visible = false;
  let observer: IntersectionObserver;
  onMount(() => {
    mounted = true;
  });

  $: if (!observer && !!container) {
    observer = new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
    });
    observer.observe(container);
  }

  onDestroy(() => {
    if (!!observer) observer.unobserve(container);
  });

  let oldQuery: string = '';
  $: if (evaluateQuery && visible && oldQuery != query) {
    evaluateIfNeeded(query);
    oldQuery = query;
  }

  export let metricWidth = 176;

  export let evaluationSummary: QueryResult | null = null;
  export let evaluationError: string | null = null;
  let evaluatedLength: number | null = null;
  let evaluatedType: string | null = null;
  let loadingSummary: boolean = false;
  let summaryIsStale: boolean = false;

  function evaluateIfNeeded(q: string) {
    if (q.length == 0) {
      loadingSummary = false;
      summaryIsStale = false;
      evaluationError = null;
      evaluationSummary = null;
    }
    if (delayEvaluation && !$queryResultCache[q]) {
      summaryIsStale = true;
      if (!!evaluationTimer) clearTimeout(evaluationTimer);
      if (q.length > 0)
        evaluationTimer = setTimeout(liveEvaluateQuery, mounted ? 2000 : 500);
    } else if (q.length > 0) {
      summaryIsStale = true;
      liveEvaluateQuery(q);
    }
  }
  async function liveEvaluateQuery(q: string) {
    if (!visible) {
      oldQuery = '';
      return;
    }
    if ($currentDataset == null) {
      console.warn('cannot live evaluate query without a currentDataset prop');
      return;
    }
    let result: QueryEvaluationResult;
    if (!!$queryResultCache[query]) {
      result = $queryResultCache[query];
    } else {
      loadingSummary = true;
      let encodedQuery = encodeURIComponent(query);
      try {
        result = await (
          await fetch(
            `/datasets/${$currentDataset}/data/query?q=${encodedQuery}`
          )
        ).json();
        $queryResultCache = { ...$queryResultCache, [query]: result };
      } catch (e) {
        evaluationError = `${e}`;
        evaluationSummary = null;
        loadingSummary = false;
        return;
      }
    }
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
    dispatch('result', evaluationError == null);
    loadingSummary = false;
    summaryIsStale = false;
  }

  let showHeaders = false;
  $: if (!!evaluationSummary)
    showHeaders =
      !!evaluationSummary.occurrences || !!evaluationSummary.durations;

  let downloadProgress: string | null = null;
  let downloadTaskID: string | null = null;
  async function pollDownload() {
    if (!downloadTaskID) return;
    try {
      let result = await (await fetch(`/tasks/${downloadTaskID}`)).json();
      if (result.status == 'complete') {
        evaluationError = null;
        downloadProgress = null;
        downloadTaskID = null;
        downloadQueryResult();
      } else if (result.status == 'error') {
        downloadProgress = null;
        evaluationError = result.status_info;
        downloadTaskID = null;
      } else {
        evaluationError = null;
        downloadProgress =
          result.status_info?.message ?? result.status_info ?? result.status;
        setTimeout(pollDownload, 1000);
      }
    } catch (e) {
      console.error('error checking task status:', e);
      evaluationError = `${e}`;
      downloadProgress = null;
      downloadTaskID = null;
    }
  }
  async function downloadQueryResult() {
    downloadProgress = 'starting';
    try {
      let result = await (
        await fetch(
          `/datasets/${$currentDataset}/data/query?q=${encodeURIComponent(query)}&dl=1`
        )
      ).json();
      if (result.blob) {
        downloadProgress = null;
        let blob = base64ToBlob(result.blob, 'application/zip');
        let url = window.URL.createObjectURL(blob);
        let a = document.createElement('a');
        document.body.appendChild(a);
        a.style.display = 'none';
        a.href = url;
        a.download = result.filename;
        a.click();
        window.URL.revokeObjectURL(url);
      } else {
        downloadTaskID = result.id;
        downloadProgress =
          result.status_info?.message ?? result.status_info ?? result.status;
        setTimeout(pollDownload, 1000);
      }
    } catch (e) {
      console.error('error downloading query result:', e);
      evaluationError = `${e}`;
      downloadProgress = null;
    }
  }
</script>

<div
  class="text-sm w-full"
  class:opacity-50={summaryIsStale}
  bind:this={container}
>
  {#if !!downloadProgress}
    <div class="mb-1 text-slate-500 text-xs">
      Preparing download ({downloadProgress})
    </div>
  {:else if loadingSummary}
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
    {:else if !compact}
      <div class="mb-1 text-slate-500 text-xs flex justify-between">
        <div class="font-bold">Query Result</div>
        <button
          class="hover:opacity-50 ml-2"
          title="Download query result for all data splits"
          on:click={downloadQueryResult}
          ><Fa icon={faDownload} class="inline" /></button
        >
      </div>
    {/if}
    {#if !!evaluatedType && !compact}
      <div class="mb-2 text-slate-600 text-xs">
        {evaluatedType}
        {#if evaluatedLength != null}with {d3.format(',')(evaluatedLength)} values{/if}
      </div>
    {/if}
    {#if !!evaluationSummary.occurrences && !compact}
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
    {#if !!evaluationSummary.durations && !compact}
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
      {#if showHeaders && !compact}<div class="mb-1 text-xs text-slate-500">
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
