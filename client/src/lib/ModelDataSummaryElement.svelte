<script lang="ts">
  import type { ModelDataSummaryItem } from './model';
  import SliceMetricBar from './slices/metric_charts/SliceMetricBar.svelte';
  import SliceMetricCategoryBar from './slices/metric_charts/SliceMetricCategoryBar.svelte';
  import SliceMetricHistogram from './slices/metric_charts/SliceMetricHistogram.svelte';
  import * as d3 from 'd3';

  export let element: ModelDataSummaryItem;
  export let indentLevel: number = 0;
  export let metricWidth: number = 200;
</script>

{#if !!element}
  {#if element.type == 'group'}
    <div class="mb-4">
      <div
        class="mb-1 text-slate-600 flex-auto"
        class:font-bold={indentLevel < 1}
        class:text-sm={indentLevel <= 2}
        class:text-xs={indentLevel > 2}
      >
        {element.name}
      </div>
      {#each element.children ?? [] as child}
        <svelte:self element={child} indentLevel={indentLevel + 1} />
      {/each}
    </div>
  {:else if !!element.summary}
    {@const summary = element.summary}
    <div class="mb-4 flex items-center">
      <div
        class="mb-2 text-slate-600 flex-auto"
        class:font-bold={indentLevel < 1}
        class:text-sm={indentLevel <= 2}
        class:text-xs={indentLevel > 2}
      >
        {element.name}
      </div>
      <div style="width: {metricWidth + 16}px;">
        {#if summary.type == 'binary' && !!summary.rate}
          <SliceMetricBar
            value={summary.rate}
            width={metricWidth}
            color="#3b82f6"
            showFullBar
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(summary.rate)}</strong>
              true,
              <strong>{d3.format('.1%')(1 - summary.rate)}</strong> false
            </span>
          </SliceMetricBar>
        {:else if summary.type == 'continuous' && !!summary.hist}
          <SliceMetricHistogram
            mean={summary.mean ?? 0}
            histValues={Object.fromEntries(
              summary.hist.bins
                .slice(0, summary.hist.bins.length - 1)
                .map((b, i) => [b, summary.hist.counts[i]])
            )}
            width={metricWidth}
          />
        {:else if summary.type == 'categorical' && !!summary.counts}
          <SliceMetricCategoryBar
            order={Object.keys(summary.counts).sort(
              (a, b) => summary.counts[b] - summary.counts[a]
            )}
            counts={summary.counts}
            width={metricWidth}
          />
        {/if}
      </div>
    </div>
  {/if}
{/if}
