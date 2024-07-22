<script lang="ts">
  import {
    SliceSearchControl,
    type Slice,
    type SliceMetricInfo,
    type SliceMetric,
  } from '../utils/slice.type';
  import SliceMetricBar from '../metric_charts/SliceMetricBar.svelte';
  import { format } from 'd3-format';
  import SliceMetricHistogram from '../metric_charts/SliceMetricHistogram.svelte';
  import SliceMetricCategoryBar from '../metric_charts/SliceMetricCategoryBar.svelte';
  import { createEventDispatcher, onMount } from 'svelte';
  import Select from 'svelte-select';
  import Fa from 'svelte-fa/src/fa.svelte';
  import {
    faPencil,
    faPlus,
    faRotateRight,
    faHeart,
  } from '@fortawesome/free-solid-svg-icons';
  import { faHeart as faHeartOutline } from '@fortawesome/free-regular-svg-icons';
  import ActionMenuButton from '../utils/ActionMenuButton.svelte';
  import { TableWidths } from './tablewidths';
  import SliceFeature from './SliceFeature.svelte';
  import { areObjectsEqual, featuresHaveSameTree } from '../utils/utils';
  import SliceFeatureEditor from './SliceFeatureEditor.svelte';
  import { featureToString, parseFeature } from '../utils/slice_parsing';
  import Checkbox from '../utils/Checkbox.svelte';
  import { interpolateViridis } from 'd3';
  import Tooltip from '../../utils/Tooltip.svelte';

  const dispatch = createEventDispatcher();

  const SearchCriteriaMetricGroup = 'Search Criteria';

  export let slice: Slice | null = null;
  export let scoreNames: Array<string> = [];
  export let showScores = false;
  export let metricGroups: string[] | null = null;
  export let metricNames: Array<any> = [];
  export let metricGetter: (slice: Slice, key: any) => SliceMetric = (
    slice,
    key
  ) => slice.metrics![key];
  export let positiveOnly = false;
  export let allowedValues: any = null;

  export let fixedFeatureOrder: Array<any> = [];

  export let customSlice: Slice | null = null; // if the slice is custom-created
  export let temporarySlice: Slice | null = null; // if a variable is adjusted dynamically

  export let scoreCellWidth = 72;
  export let scoreWidthScalers = {};
  export let metricInfo:
    | { [key: string]: SliceMetricInfo }
    | ((key: string) => SliceMetricInfo) = {};

  export let rowClass = '';
  export let maxIndent = 0;
  export let indent = 0;

  export let isSaved = false;
  export let isSelected = false;
  export let isEditing = false;

  export let showCheckbox: boolean = true;
  export let allowFavorite: boolean = true;
  export let allowEdit: boolean = true;
  export let allowSearch: boolean = true;
  export let allowSelect: boolean = false;

  let showButtons = false;

  const indentAmount = 24;

  export let showCreateSliceButton = false;

  /*let featureOrder = [];
  $: {
    let sliceForFeatures = slice || customSlice || temporarySlice;
    featureOrder = Object.keys(sliceForFeatures.featureValues);
    featureOrder.sort((a, b) => {
      let aIndex = fixedFeatureOrder.indexOf(a);
      let bIndex = fixedFeatureOrder.indexOf(b);
      if (aIndex < 0) aIndex = featureOrder.length;
      if (bIndex < 0) bIndex = featureOrder.length;
      if (aIndex == bIndex) return a.localeCompare(b);
      return b - a;
    });
  }*/

  let baseSlice: Slice | null = null;
  $: baseSlice = customSlice || slice;
  let sliceToShow: Slice | null = null;
  $: sliceToShow = temporarySlice || customSlice || slice;

  let sliceForScores: Slice | null = null;
  $: sliceForScores = revertedScores ? customSlice || slice : sliceToShow;

  let revertedScores: boolean = false;
  function temporaryRevertSlice(revert: boolean) {
    revertedScores = revert;
  }

  let groupedMetricNames: string[][] = [];
  $: if (metricNames.length > 0) {
    if (metricGroups != null)
      groupedMetricNames = metricGroups.map((g) =>
        metricNames.filter((n) => n[0] == g)
      );
    else groupedMetricNames = [metricNames];
  } else groupedMetricNames = [];
</script>

{#if !!sliceToShow && !!slice}
  <div
    class="slice-row py-1 px-2 {isSelected
      ? 'bg-blue-100 hover:bg-blue-50'
      : allowSelect
        ? 'bg-white hover:bg-slate-100'
        : 'bg-white'} {rowClass ? rowClass : ''} inline-flex items-center"
    style="min-width: 100%; margin-left: {indentAmount *
      (maxIndent - indent)}px;"
    on:mouseenter={() => (showButtons = true)}
    on:mouseleave={() => (showButtons = false)}
    on:click={(e) => {
      if (e.target.tagName.toLocaleLowerCase() !== 'div') return;
      if (allowSelect) dispatch('select', !isSelected);
    }}
    on:keypress={(e) => {
      if (allowSelect && e.key === 'Enter') dispatch('select', !isSelected);
    }}
    role={allowSelect ? 'button' : 'none'}
    tabindex="-1"
  >
    {#if showCheckbox}
      <div class="p-2" style="width: {TableWidths.Checkbox}px;">
        <Checkbox
          checked={isSelected}
          on:change={(e) => dispatch('select', !isSelected)}
        />
      </div>
    {/if}
    <div
      class="flex-auto w-0"
      style="min-width: {TableWidths.FeatureList}px; max-width: 800px;"
    >
      <div
        class="py-2 w-full text-xs min-w-0"
        class:opacity-50={revertedScores}
      >
        {#if isEditing}
          <SliceFeatureEditor
            featureText={featureToString(
              featuresHaveSameTree(slice.feature, sliceToShow.feature) &&
                slice.feature.type != 'base'
                ? slice.feature
                : sliceToShow.feature,
              false,
              positiveOnly
            )}
            {positiveOnly}
            {allowedValues}
            on:cancel={(e) => {
              isEditing = false;
              dispatch('endedit');
            }}
            on:save={(e) => {
              let newFeature = parseFeature(e.detail, allowedValues);
              console.log('new feature:', newFeature);
              isEditing = false;
              dispatch('endedit');
              dispatch('edit', newFeature);
            }}
          />
        {:else}
          <div class="flex pt-1 items-center whitespace-nowrap">
            <div style="flex: 0 1 auto;" class="overflow-x-auto">
              <SliceFeature
                feature={featuresHaveSameTree(
                  slice.feature,
                  sliceToShow.feature
                ) && slice.feature.type != 'base'
                  ? slice.feature
                  : sliceToShow.feature}
                currentFeature={sliceToShow.feature}
                canToggle={featuresHaveSameTree(
                  slice.feature,
                  sliceToShow.feature
                )}
                {positiveOnly}
                on:toggle
              />
            </div>
            {#if (showButtons || isSaved) && allowFavorite}
              <button
                class="bg-transparent hover:opacity-60 ml-1 px-1 text-slate-600 py-1"
                title="Add a new custom slice"
                on:click={() => dispatch('saveslice', sliceToShow)}
                ><Fa icon={isSaved ? faHeart : faHeartOutline} /></button
              >
            {/if}
            {#if showButtons}
              {#if showCreateSliceButton}
                <button
                  class="bg-transparent hover:opacity-60 ml-1 px-1 text-slate-600 py-1"
                  title="Add a new custom slice"
                  on:click={() => dispatch('create')}
                  ><Fa icon={faPlus} /></button
                >
              {/if}
              {#if allowEdit}
                <button
                  class="bg-transparent hover:opacity-60 ml-1 px-1 py-1 text-slate-600"
                  on:click={() => {
                    isEditing = true;
                    dispatch('beginedit');
                  }}
                  title="Temporarily modify the slice definition"
                  ><Fa icon={faPencil} /></button
                >
                {#if !!temporarySlice && !areObjectsEqual(temporarySlice, slice)}
                  <button
                    class="bg-transparent hover:opacity-60 ml-1 py-1 px-1 text-slate-600"
                    on:click={() => {
                      temporaryRevertSlice(false);
                      dispatch('reset');
                    }}
                    on:mouseenter={() => temporaryRevertSlice(true)}
                    on:mouseleave={() => temporaryRevertSlice(false)}
                    title="Reset the slice definition"
                    ><Fa icon={faRotateRight} /></button
                  >
                {/if}
              {/if}
            {/if}
          </div>
          {#if !!sliceForScores && !!metricGetter( sliceForScores, [SearchCriteriaMetricGroup, '0'] )}
            {@const metric = metricGetter(sliceForScores, [
              SearchCriteriaMetricGroup,
              '0',
            ])}
            <div
              class="ml-2 mt-3 flex items-center w-1/3 gap-2"
              style="max-width: 360px;"
            >
              <div class="text-slate-600">Search criteria</div>
              <div class="flex-auto">
                <SliceMetricBar
                  value={metric.mean}
                  scale={(v) => v}
                  color={'#be185d'}
                  width={null}
                  horizontalLayout
                  showFullBar
                  showTooltip={false}
                />
              </div>
              <div>
                <strong>{format('.1%')(metric.mean)}</strong>
              </div>
            </div>
          {/if}
        {/if}
      </div>
    </div>
    {#if !!sliceForScores && !!sliceForScores.metrics && !!metricInfo && groupedMetricNames.length > 0}
      {#each groupedMetricNames as metricGroup}
        <div
          class="metric-column p-2 whitespace-nowrap grow-0 shrink-0 grid auto-rows-max text-xs gap-x-2 gap-y-0 items-center"
          style="grid-template-columns: max-content auto 96px;"
        >
          {#each metricGroup as name}
            {@const metric = metricGetter(sliceForScores, name)}
            {@const displayName =
              metricGroup.length == 1
                ? ''
                : Array.isArray(name)
                  ? name[1]
                  : name}
            {@const mInfo =
              typeof metricInfo === 'function'
                ? metricInfo(name)
                : metricInfo[name]}
            {#if !metric}
              <div class="col-span-full">
                <Tooltip title="Not enough data"><span>&mdash;</span></Tooltip>
              </div>
            {:else if sliceForScores.isEmpty}
              <span class="text-slate-600">Empty</span>
            {:else if !!mInfo && mInfo.visible}
              {#if metric.type == 'binary'}
                <div class="font-bold text-right">{displayName}</div>
                <SliceMetricBar
                  value={metric.mean}
                  scale={mInfo.scale ?? ((v) => v)}
                  color={mInfo.color ?? null}
                  colorScale={mInfo.colorScale ?? interpolateViridis}
                  width={null}
                  horizontalLayout
                  showFullBar
                  showTooltip={false}
                />
                <div>
                  <strong>{format('.1%')(metric.mean)}</strong>
                </div>
              {:else if metric.type == 'numeric'}
                <div class="font-bold text-right">{displayName}</div>
                <SliceMetricBar
                  value={metric.value}
                  scale={mInfo.scale ?? ((v) => v)}
                  color={mInfo.color ?? null}
                  colorScale={mInfo.colorScale ?? interpolateViridis}
                  width={null}
                  horizontalLayout
                  showFullBar
                  showTooltip={false}
                ></SliceMetricBar>
                <div>
                  <strong>{format('.1%')(metric.value)}</strong>
                </div>
              {:else if metric.type == 'count'}
                <div class="font-bold text-right">{displayName}</div>
                <SliceMetricBar
                  value={metric.share}
                  width={null}
                  color={mInfo.color ?? null}
                  colorScale={mInfo.colorScale ?? interpolateViridis}
                  horizontalLayout
                  showFullBar
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
                  title={displayName}
                  width={null}
                  mean={metric.mean}
                  histValues={metric.hist}
                  color={mInfo.color ?? '#3b82f6'}
                />
              {:else if metric.type == 'categorical'}
                <SliceMetricCategoryBar
                  noParent
                  title={displayName}
                  width={null}
                  order={mInfo.order}
                  counts={metric.counts}
                />
              {/if}
            {/if}
          {/each}
        </div>
      {/each}
    {/if}
  </div>
{/if}

<style>
  .metric-column {
    width: 400px;
  }

  @media screen and (width < 1600px) {
    .metric-column {
      width: 320px;
    }
  }
</style>
