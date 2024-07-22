<script lang="ts">
  import type {
    Slice,
    SliceFeatureBase,
    SliceMetric,
    SliceMetricInfo,
  } from '../utils/slice.type';
  import SliceRow from './SliceRow.svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import Hoverable from '../utils/Hoverable.svelte';
  import {
    faAngleLeft,
    faAngleRight,
    faEye,
    faEyeSlash,
    faGripLinesVertical,
  } from '@fortawesome/free-solid-svg-icons';
  import {
    areObjectsEqual,
    areSetsEqual,
    withToggledFeature,
  } from '../utils/utils';
  import { TableWidths } from './tablewidths';
  import { createEventDispatcher } from 'svelte';
  import type { SliceFeature } from '../utils/slice.type';

  const dispatch = createEventDispatcher();

  export let showHeader = true;

  export let slices: Array<Slice> = [];
  export let selectedSlices: Array<SliceFeatureBase> = [];
  export let selectedSlice: SliceFeatureBase | null = null; // if not allowMultiselect
  export let savedSlices: { [key: string]: SliceFeatureBase } = {};

  export let baseSlice: Slice | null = null;
  export let sliceRequests: { [key: string]: SliceFeatureBase } = {};
  export let sliceRequestResults: { [key: string]: Slice } = {};

  export let fixedFeatureOrder: Array<any> = [];
  export let searchBaseSlice: any = null;

  export let showCheckboxes = true;
  export let allowShowScores = true;
  export let showScores = false;
  export let positiveOnly = false;

  export let valueNames: any = {};
  export let allowedValues: any = {};

  export let metricGroups: string[] | null = null;
  export let metricNames: any[] = [];
  export let metricDescriptions: { [key: string]: string } = {};
  export let metricGetter: (slice: Slice, key: any) => SliceMetric = (
    slice,
    key
  ) => slice.metrics![key] as SliceMetric;
  export let metricInfo:
    | { [key: string]: SliceMetricInfo }
    | ((key: string) => SliceMetricInfo) = {};
  export let scoreNames: string[] = [];
  export let scoreWidthScalers = {};

  export let allowFavorite: boolean = true;
  export let allowEdit: boolean = true;
  export let allowSearch: boolean = true;
  export let allowMultiselect: boolean = true;

  let editingSlice: string | null = null;
  let tempRevertedSlice: string | null = null;

  // Drag and drop metrics logic

  let clickingColumn: string | null = null; // prepare for drag
  let draggingColumn: string | null = null; // track drag action
  let dropRight = false;

  function metricDragStart(e: DragEvent, colName: string) {
    if (!!e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
    draggingColumn = colName;
  }

  function metricDragEnd(e: any, colName: string) {
    draggingColumn = null;
  }

  function metricDragEnter(e: any, colName: string) {
    if (!draggingColumn || colName == draggingColumn) {
      return false;
    }
    let names = metricGroups ?? metricNames;
    let startIdx = names.indexOf(draggingColumn);
    let endIdx = names.indexOf(colName);
    dropRight = startIdx < endIdx;
    e.target.classList.add('drop-zone');
    e.target.classList.add(dropRight ? 'drop-zone-r' : 'drop-zone-l');
  }

  function metricDragLeave(e: any, colName: string) {
    e.target.classList.remove('drop-zone');
    e.target.classList.remove('drop-zone-r');
    e.target.classList.remove('drop-zone-l');
  }

  function metricDrop(e: any, colName: string) {
    e.target.classList.remove('drop-zone');
    if (!!draggingColumn && draggingColumn != colName) {
      console.log(draggingColumn, colName);
      let names = metricGroups ?? metricNames;
      let startIdx = names.indexOf(draggingColumn);
      let endIdx = names.indexOf(colName);
      let newOrder = Array.from(names);
      newOrder.splice(startIdx, 1);
      let newNames = [
        ...newOrder.slice(0, endIdx),
        draggingColumn,
        ...newOrder.slice(endIdx),
      ];
      if (!!metricGroups) {
        console.log('new groups:', newNames);
        metricGroups = newNames;
        metricNames = metricGroups
          .map((g) => metricNames.map((n) => n[0] == g))
          .flat();
      } else metricNames = newNames;
    }
    return false;
  }

  function toggleSliceFeature(slice: Slice, feature: SliceFeature) {
    let allRequests = Object.assign({}, sliceRequests);
    let r;
    if (!!allRequests[slice.stringRep]) r = allRequests[slice.stringRep];
    else r = slice.feature;
    let selectionIdx = selectedSlices.findIndex((s) => areObjectsEqual(s, r!));
    r = withToggledFeature(r, slice.feature, feature);
    allRequests[slice.stringRep] = r;
    sliceRequests = allRequests;
    console.log('slice requests:', sliceRequests);
    if (selectionIdx >= 0)
      selectedSlices = [
        ...selectedSlices.slice(0, selectionIdx),
        r,
        ...selectedSlices.slice(selectionIdx + 1),
      ];
  }

  function resetSlice(slice: Slice) {
    let allRequests = Object.assign({}, sliceRequests);
    if (!!allRequests[slice.stringRep]) {
      let selectionIdx = selectedSlices.findIndex((s) =>
        areObjectsEqual(s, allRequests[slice.stringRep])
      );
      if (selectionIdx >= 0)
        selectedSlices = [
          ...selectedSlices.slice(0, selectionIdx),
          slice.feature,
          ...selectedSlices.slice(selectionIdx + 1),
        ];
    }
    delete allRequests[slice.stringRep];
    sliceRequests = allRequests;
  }

  function editSliceFeature(slice: Slice, newFeature: SliceFeatureBase) {
    let allRequests = Object.assign({}, sliceRequests);
    let selectionIdx = selectedSlices.findIndex((s) =>
      areObjectsEqual(s, allRequests[slice.stringRep] ?? slice.feature)
    );
    allRequests[slice.stringRep] = newFeature;
    sliceRequests = allRequests;
    console.log('slice requests:', sliceRequests);
    if (selectionIdx >= 0)
      selectedSlices = [
        ...selectedSlices.slice(0, selectionIdx),
        newFeature,
        ...selectedSlices.slice(selectionIdx + 1),
      ];
  }

  function selectSlice(slice: Slice, selected: boolean = true) {
    if (!allowMultiselect) {
      if (selected) {
        selectedSlice = slice.feature;
        selectedSlices = [slice.feature];
      } else {
        selectedSlice = null;
        selectedSlices = [];
      }
      console.log('selected:', selectedSlices);
      return;
    }

    if (selected) selectedSlices = [...selectedSlices, slice.feature];
    else {
      let idx = selectedSlices.findIndex((s) =>
        areObjectsEqual(s, slice.feature)
      );
      if (idx >= 0)
        selectedSlices = [
          ...selectedSlices.slice(0, idx),
          ...selectedSlices.slice(idx + 1),
        ];
    }
  }
</script>

<div class="relative">
  {#if showHeader}
    <div
      class="px-2 text-left inline-flex align-top font-bold slice-header whitespace-nowrap bg-slate-100 rounded-t border-b border-slate-600"
      style="min-width: 100%;"
    >
      {#if showCheckboxes}
        <div style="width: {TableWidths.Checkbox}px;">
          <div class="p-2 w-full h-full" />
        </div>
      {/if}
      <div
        class="flex-auto"
        style="min-width: {TableWidths.FeatureList}px; max-width: 800px;"
      >
        <div class="p-2">Slice</div>
      </div>
      {#if !!metricGroups}
        {#each metricGroups as groupName}
          <div
            class="grow-0 shrink-0 bg-slate-100 hover:bg-slate-200 metric-column"
            class:opacity-30={draggingColumn == groupName}
            draggable={clickingColumn == groupName}
            on:dragstart={(e) => metricDragStart(e, groupName)}
            on:dragend={(e) => metricDragEnd(e, groupName)}
            on:dragover|preventDefault={() => false}
            on:dragenter={(e) => metricDragEnter(e, groupName)}
            on:dragleave={(e) => metricDragLeave(e, groupName)}
            on:drop|preventDefault|stopPropagation={(e) =>
              metricDrop(e, groupName)}
          >
            <Hoverable class="potential-drop-zone p-2" let:hovering>
              <div class="flex items-center">
                <div>{groupName}</div>
                <div class="flex-1" />
                <button
                  class="ml-2 bg-transparent text-slate-400 cursor-move"
                  on:mousedown={() => (clickingColumn = groupName)}
                  on:mouseup={() => (clickingColumn = null)}
                  class:opacity-0={!hovering}
                  class:disabled={!hovering}
                  ><Fa icon={faGripLinesVertical} /></button
                >
              </div>
              {#if !!metricDescriptions && !!metricDescriptions[groupName]}
                <div
                  class="font-normal text-xs text-slate-600 whitespace-normal w-full"
                >
                  {metricDescriptions[groupName]}
                </div>
              {/if}
            </Hoverable>
          </div>
        {/each}
      {:else}
        <div
          class="flex flex-col grow shrink-0 bg-slate-100 hover:bg-slate-200"
          style="min-width: {TableWidths.AllMetrics}px; max-width: 500px;"
        >
          <div class="p-2">Metrics</div>
        </div>
      {/if}
      {#if showScores}
        {#each scoreNames as score, i}
          <div class="bg-slate-100" style="width: {TableWidths.Score}px;">
            <div class="p-2">
              {score}
            </div>
          </div>
        {/each}
      {/if}
      {#if allowShowScores}
        <div
          class="bg-slate-100 hover:bg-slate-200"
          on:click={() => (showScores = !showScores)}
        >
          <div class="w-full h-full px-4 flex justify-center items-center">
            {#if showScores}
              <Fa icon={faAngleLeft} />
            {:else}
              <Fa icon={faAngleRight} />
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}
  {#if !!baseSlice}
    <SliceRow
      slice={baseSlice}
      {scoreNames}
      {positiveOnly}
      scoreCellWidth={72}
      {scoreWidthScalers}
      {showScores}
      showCheckbox={showCheckboxes}
      {metricNames}
      {metricGroups}
      {metricInfo}
      {metricGetter}
      {allowedValues}
      {allowEdit}
      {allowFavorite}
      {allowSearch}
      allowSelect={false}
      isSaved={!!savedSlices[baseSlice.stringRep]}
      isSelected={!!selectedSlices.find((s) =>
        areObjectsEqual(
          s,
          (sliceRequestResults[baseSlice.stringRep] || baseSlice).feature
        )
      )}
      temporarySlice={tempRevertedSlice == baseSlice.stringRep
        ? baseSlice
        : sliceRequestResults[baseSlice.stringRep]}
      {fixedFeatureOrder}
      isEditing={baseSlice.stringRep == editingSlice}
      on:beginedit={(e) => (editingSlice = baseSlice.stringRep)}
      on:endedit={(e) => (editingSlice = null)}
      on:edit={(e) => editSliceFeature(baseSlice, e.detail)}
      on:toggle={(e) => toggleSliceFeature(baseSlice, e.detail)}
      on:reset={(e) => resetSlice(baseSlice)}
      on:temprevert={(e) =>
        (tempRevertedSlice = e.detail ? baseSlice.stringRep : null)}
      on:newsearch
      on:saveslice
      on:select={(e) =>
        selectSlice(
          sliceRequestResults[baseSlice.stringRep] || baseSlice,
          e.detail
        )}
    />
  {/if}
  {#each slices as slice, i (slice.stringRep || i)}
    {@const sliceToShow = sliceRequestResults[slice.stringRep] || slice}
    <SliceRow
      {slice}
      {scoreNames}
      {positiveOnly}
      scoreCellWidth={72}
      showCheckbox={showCheckboxes}
      {scoreWidthScalers}
      {showScores}
      {metricNames}
      {metricGroups}
      {metricInfo}
      {metricGetter}
      {allowedValues}
      {allowEdit}
      {allowFavorite}
      {allowSearch}
      allowSelect={!allowMultiselect}
      {fixedFeatureOrder}
      isSaved={!!Object.values(savedSlices).find((s) =>
        areObjectsEqual(s, sliceToShow.feature)
      )}
      isSelected={!!selectedSlices.find((s) =>
        areObjectsEqual(s, sliceToShow.feature)
      )}
      temporarySlice={tempRevertedSlice == slice.stringRep
        ? slice
        : sliceToShow}
      isEditing={slice.stringRep == editingSlice}
      on:beginedit={(e) => (editingSlice = slice.stringRep)}
      on:endedit={(e) => (editingSlice = null)}
      on:edit={(e) => editSliceFeature(slice, e.detail)}
      on:toggle={(e) => toggleSliceFeature(slice, e.detail)}
      on:reset={(e) => resetSlice(slice)}
      on:temprevert={(e) =>
        (tempRevertedSlice = e.detail ? slice.stringRep : null)}
      on:newsearch
      on:saveslice
      on:select={(e) => selectSlice(sliceToShow, e.detail)}
    />
  {/each}
</div>

<style>
  .metric-column {
    width: 360px;
  }

  @media screen and (width < 1600px) {
    .metric-column {
      width: 280px;
    }
  }
</style>
