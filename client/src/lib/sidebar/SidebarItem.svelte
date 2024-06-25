<script lang="ts">
  import {
    type ModelSummary,
    decimalMetrics,
    metricsHaveWarnings,
    type ModelMetrics,
  } from '../model';
  import * as d3 from 'd3';
  import { SidebarTableWidths } from '../utils/sidebarwidths';
  import Checkbox from '../utils/Checkbox.svelte';
  import SliceMetricBar from '../slices/metric_charts/SliceMetricBar.svelte';
  import type { SliceMetric } from '../slices/utils/slice.type';
  import {
    faCheck,
    faEllipsisV,
    faWarning,
    faXmark,
  } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { createEventDispatcher } from 'svelte';
  import { MetricColors, makeCategoricalColorScale } from '../colors';
  import SliceMetricHistogram from '../slices/metric_charts/SliceMetricHistogram.svelte';
  import SliceMetricCategoryBar from '../slices/metric_charts/SliceMetricCategoryBar.svelte';
  import Tooltip from '../utils/Tooltip.svelte';
  import removeMd from 'remove-markdown';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';

  const dispatch = createEventDispatcher();

  export let model: ModelSummary;
  export let metrics: ModelMetrics;
  export let modelName: string;
  export let isActive: boolean;
  export let isChecked: boolean;
  export let metricToShow: string;
  export let customMetrics: { [key: string]: SliceMetric } | undefined =
    undefined;
  export let showCheckbox: boolean = true;
  export let allowCheck: boolean = true;
  export let checkDisabledReason: string | null = null;
  export let differences: string[] = [];

  export let metricScales: { [key: string]: (v: number) => number } = {};
  export let isEditingName: boolean = false;

  const accuracyFormat = d3.format('.1~%');
  const decimalFormat = d3.format(',.2~');
  const countFormat = d3.format(',');

  let hovering = false;

  let metricValues: { [key: string]: SliceMetric | null } | undefined;
  $: if (!!customMetrics)
    metricValues = {
      Timesteps: {
        type: 'binary',
        mean: customMetrics['Timesteps']?.count ?? 0,
      },
      Trajectories: {
        type: 'binary',
        mean: customMetrics['Trajectories']?.count ?? 0,
      },
      [metricToShow]: customMetrics[metricToShow] ?? null,
      Labels: customMetrics['Labels'] ?? null,
      Predictions: customMetrics['Predictions'] ?? null,
    };
  else if (!!model && !!metrics)
    metricValues = {
      Timesteps: {
        type: 'binary',
        mean: metrics.n_test.instances ?? 0,
      },
      Trajectories: {
        type: 'binary',
        mean: metrics.n_test.trajectories ?? 0,
      },
      [metricToShow]:
        metrics.performance[metricToShow] !== undefined
          ? {
              type: 'numeric',
              value: metrics.performance[metricToShow],
            }
          : null,
      Labels: metrics.labels ?? null,
      Predictions: metrics.predictions ?? null,
    };
  else metricValues = undefined;

  let newName: string | null = null;
  $: if (isEditingName && newName == null) newName = modelName;
  else if (!isEditingName) newName = null;
  let oldEditBox: HTMLInputElement;
  let editBox: HTMLInputElement;
  $: if (editBox !== oldEditBox) {
    if (!!editBox) {
      editBox.focus();
      editBox.select();
    }
    oldEditBox = editBox;
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  on:click
  class="w-full inline-flex slice-row items-center justify-center flex-wrap py-4 cursor-pointer {isActive
    ? 'bg-blue-100'
    : 'hover:bg-slate-100'} "
  on:mouseenter={() => (hovering = true)}
  on:mouseleave={() => (hovering = false)}
>
  <div
    class="font-mono p-2 text-sm flex grow shrink items-center gap-2"
    style="flex-basis: 0; min-width: 240px;"
  >
    {#if showCheckbox}
      <div
        class="grow-0 shrink-0"
        style="width: {SidebarTableWidths.Checkbox}px;"
      >
        {#if !allowCheck && !!checkDisabledReason}
          <Tooltip title={checkDisabledReason} position="right">
            <Checkbox disabled={isActive || !allowCheck} checked={isChecked} />
          </Tooltip>
        {:else}
          <Checkbox
            disabled={isActive || !allowCheck}
            checked={isChecked}
            on:change={(e) => dispatch('toggle')}
          />
        {/if}
      </div>
    {/if}
    <div class="flex-auto min-w-0">
      {#if isEditingName}
        <form
          class="w-full"
          on:submit|preventDefault={() =>
            dispatch('rename', { old: modelName, new: newName })}
        >
          <div class="flex w-full items-center gap-2">
            <input
              type="text"
              class="flat-text-input flex-auto"
              bind:value={newName}
              bind:this={editBox}
              on:blur={() => setTimeout(() => dispatch('canceledit'), 100)}
            />
            <button
              class="bg-transparent hover:opacity-60 text-slate-600 text-lg"
              type="button"
              on:mousedown|preventDefault|stopPropagation={() => {}}
              on:click|stopPropagation={() => {
                dispatch('canceledit');
              }}
              title="Cancel the rename"><Fa icon={faXmark} /></button
            >
            <button
              class="bg-transparent hover:opacity-60 text-slate-600 text-lg disabled:opacity-50"
              type="submit"
              disabled={!newName || newName.length == 0}
              title="Save the renamed model"><Fa icon={faCheck} /></button
            >
          </div>
        </form>
      {:else}
        <div class="whitespace-nowrap truncate">
          {#if !!metrics && metricsHaveWarnings(metrics)}
            <Fa class="text-orange-300 inline" icon={faWarning} />
          {/if}
          {modelName}
        </div>
      {/if}
      {#if !!model.description}
        <div class="text-slate-500 font-sans text-xs line-clamp-2">
          {removeMd(model.description)}
        </div>
      {/if}
      {#if differences.length > 0}
        <div class="text-xs text-slate-500 font-sans">
          <strong>&Delta;:</strong>
          {differences.join(', ')}
        </div>
      {/if}
    </div>
  </div>
  {#if hovering && !isEditingName}
    <div
      class="grow-0 shrink-0"
      style="width: {SidebarTableWidths.Checkbox}px;"
    >
      <ActionMenuButton
        buttonClass="bg-transparent px-1 hover:opacity-40"
        align="right"
      >
        <span slot="button-content"
          ><Fa icon={faEllipsisV} class="inline" /></span
        >
        <div slot="options">
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            title="Duplicate this model"
            on:click={() => dispatch('duplicate', modelName)}>Duplicate</a
          >
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            title="Rename this model"
            on:click={() => dispatch('editname', modelName)}>Rename...</a
          >
          <a
            href="#"
            tabindex="0"
            role="menuitem"
            title="Permanently delete this model"
            on:click={() => dispatch('delete', modelName)}>Delete</a
          >
        </div>
      </ActionMenuButton>
    </div>
  {/if}
  {#if !!metricValues}
    <div
      class="p-2 whitespace-nowrap grow-0 shrink-0 grid auto-rows-max text-xs gap-x-2 gap-y-0 items-center"
      style="width: 40%; min-width: 300px; max-width: {SidebarTableWidths.AllMetrics}px; grid-template-columns: max-content auto 96px;"
    >
      <div class="font-bold text-right">Timesteps</div>
      <SliceMetricBar
        value={metricValues['Timesteps']?.mean ?? 0}
        scale={metricScales['Timesteps'] ?? ((v) => v)}
        showFullBar
        horizontalLayout
        showTooltip={false}
        color={MetricColors.Timesteps}
        width={null}
      />
      <div>
        <strong>{countFormat(metricValues['Timesteps']?.mean ?? 0)}</strong>
      </div>

      <div class="font-bold text-right">Trajectories</div>
      <SliceMetricBar
        value={metricValues['Trajectories']?.mean ?? 0}
        scale={metricScales['Trajectories'] ?? ((v) => v)}
        horizontalLayout
        showTooltip={false}
        showFullBar
        color={MetricColors.Trajectories}
        width={null}
      />
      <div>
        <strong>{countFormat(metricValues['Trajectories']?.mean ?? 0)}</strong>
      </div>

      {#if !metricValues[metricToShow] && !metricValues.Predictions}
        <div class="col-span-full">
          <Tooltip title="Not enough data"><span>&mdash;</span></Tooltip>
        </div>
      {:else if !!metricValues[metricToShow]}
        {@const metric = metricValues[metricToShow]}
        <div class="font-bold text-right">{metricToShow}</div>
        <SliceMetricBar
          value={metric?.value ?? 0}
          scale={metricScales[metricToShow] ?? ((v) => v)}
          color={MetricColors.Accuracy}
          width={null}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong
            >{decimalMetrics.includes(metricToShow)
              ? decimalFormat(metric?.value ?? 0)
              : accuracyFormat(metric?.value ?? 0)}</strong
          >
        </div>
      {:else}
        <div class="col-span-full">&mdash;</div>
      {/if}

      {#if !!metricValues.Labels}
        {@const metric = metricValues.Labels}
        {#if metric.type == 'binary'}
          <div class="font-bold text-right">Labels</div>
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Labels}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
          </div>
        {:else if metric.type == 'numeric'}
          <div class="font-bold text-right">Labels</div>
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Labels}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
          </div>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            noParent
            title={'Labels'}
            width={null}
            horizontalLayout
            mean={metric.mean}
            color={MetricColors.Labels}
            histValues={metric.hist ?? {}}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            noParent
            width={null}
            title={'Labels'}
            horizontalLayout
            colorScale={makeCategoricalColorScale(MetricColors.Labels)}
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
          />
        {/if}
      {/if}

      {#if !!metricValues.Predictions}
        {@const metric = metricValues.Predictions}
        {#if metric.type == 'binary'}
          <div class="font-bold text-right">Predictions</div>
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Predictions}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
          </div>
        {:else if metric.type == 'numeric'}
          <div class="font-bold text-right">Predictions</div>
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Predictions}
            width={null}
            showFullBar
            horizontalLayout
            showTooltip={false}
          />
          <div>
            <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
          </div>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            noParent
            title={'Predictions'}
            width={null}
            horizontalLayout
            mean={metric.mean}
            color={MetricColors.Predictions}
            histValues={metric.hist ?? {}}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            noParent
            width={null}
            title={'Predictions'}
            horizontalLayout
            colorScale={makeCategoricalColorScale(MetricColors.Predictions)}
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
          />
        {/if}
      {/if}

      <!-- {#each metricNames as name, i}
    {@const metric = sliceForScores.metrics[name]}

    {#if !!metricInfo[name] && metricInfo[name].visible}
      {#if metric.type == 'binary'}
        <div class="font-bold text-right">{name}</div>
        <SliceMetricBar
          value={metric.mean}
          color={ColorWheel[i]}
          width={null}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong>{format('.1%')(metric.mean)}</strong>
        </div>
      {:else if metric.type == 'count'}
        <div class="font-bold text-right">{name}</div>
        <SliceMetricBar
          value={metric.share}
          width={null}
          color={ColorWheel[i]}
          showFullBar
          horizontalLayout
          showTooltip={false}
        />
        <div>
          <strong>{format(',')(metric.count)}</strong>
          <span style="font-size: 0.7rem;" class="italic text-gray-700"
            >({format('.1%')(metric.share)})</span
          >
        </div>
      {:else if metric.type == 'continuous'}
        <SliceMetricHistogram
          noParent
          title={name}
          width={null}
          horizontalLayout
          mean={metric.mean}
          color={ColorWheel[i]}
          histValues={metric.hist}
        />
      {:else if metric.type == 'categorical'}
        <SliceMetricCategoryBar
          noParent
          width={null}
          title={name}
          horizontalLayout
          colorScale={makeCategoricalColorScale(ColorWheel[i])}
          order={metricInfo[name].order}
          counts={metric.counts}
        />
      {/if}
    {/if}
  {/each}
</div>

    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      <SliceMetricBar
        value={metricValues['Timesteps']?.mean ?? 0}
        scale={metricScales['Timesteps'] ?? ((v) => v)}
        color={MetricColors.Timesteps}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Timesteps']?.mean ?? 0)}</strong>
        </span>
      </SliceMetricBar>
    </div>
    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      <SliceMetricBar
        value={metricValues['Trajectories']?.mean ?? 0}
        scale={metricScales['Trajectories'] ?? ((v) => v)}
        color={MetricColors.Trajectories}
        width={SidebarTableWidths.Metric - 20}
      >
        <span slot="caption">
          <strong>{countFormat(metricValues['Trajectories']?.mean ?? 0)}</strong
          >
        </span>
      </SliceMetricBar>
    </div>
    <div
      class="p-2 grow-0 shrink-0"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !metricValues[metricToShow] && !metricValues.Predictions}
        <Tooltip title="Not enough data"><span>&mdash;</span></Tooltip>
      {:else if !!metricValues[metricToShow]}
        {@const metric = metricValues[metricToShow]}
        <SliceMetricBar
          value={metric?.value ?? 0}
          scale={metricScales[metricToShow] ?? ((v) => v)}
          color={MetricColors.Accuracy}
          width={SidebarTableWidths.Metric - 20}
        >
          <span slot="caption">
            <strong
              >{decimalMetrics.includes(metricToShow)
                ? decimalFormat(metric?.value ?? 0)
                : accuracyFormat(metric?.value ?? 0)}</strong
            >
          </span>
        </SliceMetricBar>
      {:else}
        &mdash;
      {/if}
    </div>
    <div
      class="p-2 grow-0 shrink-0 whitespace-nowrap"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !!metricValues.Labels}
        {@const metric = metricValues.Labels}
        {#if metric.type == 'binary'}
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'numeric'}
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            mean={metric.mean}
            histValues={metric.hist}
            color={MetricColors.Labels}
            width={SidebarTableWidths.Metric - 20}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
            width={SidebarTableWidths.Metric - 20}
          />
        {/if}
      {/if}
    </div>
    <div
      class="p-2 grow-0 shrink-0 whitespace-nowrap"
      style="width: {SidebarTableWidths.Metric}px;"
    >
      {#if !!metricValues.Predictions}
        {@const metric = metricValues.Predictions}
        {#if metric.type == 'binary'}
          <SliceMetricBar
            value={metric.mean}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format('.1%')(metric?.mean ?? 0)}</strong> pos.
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'numeric'}
          <SliceMetricBar
            value={metric.value}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          >
            <span slot="caption">
              <strong>{d3.format(',.3~')(metric?.value ?? 0)}</strong>
            </span>
          </SliceMetricBar>
        {:else if metric.type == 'continuous'}
          <SliceMetricHistogram
            mean={metric.mean}
            histValues={metric.hist}
            color={MetricColors.Predictions}
            width={SidebarTableWidths.Metric - 20}
          />
        {:else if metric.type == 'categorical'}
          <SliceMetricCategoryBar
            order={model.output_values ??
              Object.keys(metric.counts ?? {}).sort()}
            counts={metric.counts}
            width={SidebarTableWidths.Metric - 20}
          />
        {/if}
      {/if}-->
    </div>
  {/if}
</div>

<style>
</style>
