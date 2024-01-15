<script lang="ts">
  import type { SliceFeatureBase } from '../slices/utils/slice.type';
  import { faCheck, faPlus, faXmark } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';
  import * as d3 from 'd3';
  import type {
    SliceChangeDescription,
    SliceDescription,
  } from './slicedescription';
  import SliceFeatureDetails from './SliceFeatureDetails.svelte';
  import SliceDetailsColumn from './SliceDetailsColumn.svelte';

  export let modelNames: string[] = [];
  export let slice: SliceFeatureBase | null = null;

  let selectedFeature: string | null = null;

  let loadingSliceDescription = false;
  let offset: number = 0;
  let loadedOffset: number = 0;
  let baseSliceDescription: SliceDescription | null = null;
  let offsetSliceDescription: SliceDescription | null = null;
  let sliceChanges: SliceChangeDescription[] | null = null;

  async function loadSliceDescription() {
    try {
      loadingSliceDescription = true;

      console.log('Loading slice description', slice);
      let result = await (
        await fetch(`/slices/${modelNames.join(',')}/compare`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            slice: slice,
            ...(offset != 0 ? { offset: offset } : {}),
          }),
        })
      ).json();
      loadingSliceDescription = false;
      if (offset != 0) {
        baseSliceDescription = result.source;
        offsetSliceDescription = result.destination;
        sliceChanges = result.top_changes;
      } else {
        baseSliceDescription = result;
      }
      loadedOffset = offset;
      console.log(baseSliceDescription, offsetSliceDescription);
    } catch (e) {
      console.error('error loading slice description:', e);
      loadingSliceDescription = false;
    }
  }

  $: if (slice != null) {
    loadSliceDescription();
  } else {
    baseSliceDescription = null;
    offsetSliceDescription = null;
    sliceChanges = null;
  }

  $: if (slice == null) loadedOffset = offset;

  let searchText: string = '';

  let filteredFeatures: string[] | null = null;

  $: if (searchText.length > 0 && !!baseSliceDescription) {
    filteredFeatures = Object.keys(baseSliceDescription.all_variables)
      .filter((v) =>
        v.toLocaleLowerCase().includes(searchText.toLocaleLowerCase())
      )
      .sort((a, b) => a.length - b.length);
  } else filteredFeatures = null;
</script>

{#if !!slice}
  {#if loadingSliceDescription}
    <div class="w-full h-full flex flex-col items-center justify-center">
      <div>Loading slice details...</div>
      <div role="status" class="w-8 h-8 grow-0 shrink-0 mt-2">
        <svg
          aria-hidden="true"
          class="text-gray-200 animate-spin stroke-gray-600 w-8 h-8 align-middle"
          viewBox="-0.5 -0.5 99.5 99.5"
          xmlns="http://www.w3.org/2000/svg"
        >
          <ellipse
            cx="50"
            cy="50"
            rx="45"
            ry="45"
            fill="none"
            stroke="currentColor"
            stroke-width="10"
          />
          <path
            d="M 50 5 A 45 45 0 0 1 95 50"
            stroke-width="10"
            stroke-linecap="round"
            fill="none"
          />
        </svg>
      </div>
    </div>
  {:else}
    <div class="w-full h-full flex flex-col gap-2 p-2">
      <div class="w-full h-8 flex">
        <div class="w-1/3 font-bold text-sm text-slate-600 py-1.5 px-2">
          Correlated features
        </div>
        <div class="flex-auto font-bold text-sm text-slate-600 px-2">
          {#if offset != 0}
            Correlated features at <select
              class="ml-2 flat-select font-normal"
              bind:value={offset}
            >
              {#each [24, 18, 16, 12, 8, 6, 5, 4, 3, 2, 1] as t}
                <option value={t}>{t} timestep{t != 1 ? 's' : ''} back</option>
              {/each}
              {#each [-1, -2, -3, -4, -5, -6, -8, -12, -16, -18, -24] as t}
                <option value={t}
                  >{-t} timestep{-t != 1 ? 's' : ''} forward</option
                >
              {/each}
            </select>
            <button class="ml-2 hover:opacity-50" on:click={() => (offset = 0)}
              ><Fa icon={faXmark} class="inline" /></button
            >
            {#if loadedOffset != offset}
              <button
                class="ml-2 hover:opacity-50"
                on:click={loadSliceDescription}
              >
                <Fa icon={faCheck} class="inline" />
              </button>
            {/if}
          {/if}
        </div>
        <input
          type="search"
          bind:value={searchText}
          class="ml-2 w-48 shrink-1 flat-text-input"
          placeholder="Find variable..."
        />
        {#if selectedFeature != null}
          <button
            class="btn btn-slate ml-2"
            on:click={() => (selectedFeature = null)}>Show All</button
          >
        {/if}
      </div>
      <div class="flex-auto min-h-0 w-full flex">
        <div class="w-1/3 overflow-auto pt-2 p-1">
          {#if !!baseSliceDescription}
            <SliceDetailsColumn
              sliceDescription={baseSliceDescription}
              bind:selectedFeature
              {filteredFeatures}
            />
          {/if}
        </div>
        {#if offset != 0}
          <div class="w-1/3 overflow-auto pt-2 p-1">
            {#if !!sliceChanges}
              <SliceDetailsColumn
                changeDescription={sliceChanges}
                bind:selectedFeature
                {filteredFeatures}
              />
            {/if}
          </div>
          <div class="w-1/3 overflow-auto pt-2 p-1">
            {#if !!offsetSliceDescription}
              <SliceDetailsColumn
                sliceDescription={offsetSliceDescription}
                bind:selectedFeature
                {filteredFeatures}
              />
            {/if}
          </div>
        {:else}
          <div class="w-2/3 h-full p-2 flex items-center justify-center">
            <button class="btn btn-slate" on:click={() => (offset = -1)}
              ><Fa class="inline mr-2" icon={faPlus} /> Add timestep offset for comparison</button
            >
          </div>
        {/if}
      </div>
    </div>
  {/if}
{/if}
