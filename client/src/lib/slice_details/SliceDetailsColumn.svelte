<script lang="ts">
  import SliceFeatureDetails from './SliceFeatureDetails.svelte';
  import type {
    SliceChangeDescription,
    SliceDescription,
  } from './slicedescription';

  export let selectedFeature: string | null = null;
  export let sliceDescription: SliceDescription | null = null;
  export let changeDescription: SliceChangeDescription[] | null = null;

  export let filteredFeatures: string[] | null = null;

  const minFeaturesToShow = 5;
  let numFeaturesShown: number = 5;
</script>

{#if !!sliceDescription}
  {#if selectedFeature != null}
    <SliceFeatureDetails
      variable={sliceDescription.top_variables.find(
        (v) => v.variable == selectedFeature
      ) ?? { variable: selectedFeature, enrichments: [] }}
      valueComparison={sliceDescription.all_variables[selectedFeature]}
      expanded
      on:toggle={() => (selectedFeature = null)}
    />
  {:else}
    {@const featuresToShow =
      filteredFeatures != null
        ? filteredFeatures.map(
            (v) =>
              sliceDescription?.top_variables.find(
                (tv) => tv.variable == v
              ) ?? { variable: v, enrichments: [] }
          )
        : sliceDescription.top_variables}
    {#each featuresToShow.slice(0, numFeaturesShown) as variable}
      <SliceFeatureDetails
        {variable}
        valueComparison={sliceDescription.all_variables[variable.variable]}
        expanded={selectedFeature == variable.variable}
        on:toggle={() => (selectedFeature = variable.variable)}
      />
    {/each}
    <div class="flex items-center justify-center gap-2">
      {#if numFeaturesShown > minFeaturesToShow}
        <button
          class="btn btn-slate"
          on:click={() => (numFeaturesShown = minFeaturesToShow)}
          >Show Less</button
        >
      {/if}
      {#if numFeaturesShown < featuresToShow.length}
        <button
          class="btn btn-slate"
          on:click={() => (numFeaturesShown += minFeaturesToShow)}
          >Show More</button
        >
      {/if}
    </div>
  {/if}
{:else if !!changeDescription}
  {#if selectedFeature != null}
    {@const featureChanges = changeDescription.find(
      (v) => v.variable == selectedFeature
    )}
    {#if !!featureChanges}
      <SliceFeatureDetails
        change={featureChanges}
        expanded
        on:toggle={() => (selectedFeature = null)}
      />
    {:else}
      <div class="text-slate-400 text-center">No top changes</div>
    {/if}
  {:else}
    {@const featuresToShow =
      filteredFeatures != null
        ? filteredFeatures.map(
            (v) =>
              changeDescription?.find((tv) => tv.variable == v) ?? {
                variable: v,
                enrichments: [],
              }
          )
        : changeDescription}
    {#each featuresToShow.slice(0, numFeaturesShown) as variable}
      <SliceFeatureDetails
        change={variable}
        expanded={selectedFeature == variable.variable}
        on:toggle={() => (selectedFeature = variable.variable)}
      />
    {/each}
    <div class="flex items-center justify-center gap-2">
      {#if numFeaturesShown > minFeaturesToShow}
        <button
          class="btn btn-slate"
          on:click={() => (numFeaturesShown = minFeaturesToShow)}
          >Show Less</button
        >
      {/if}
      {#if numFeaturesShown < featuresToShow.length}
        <button
          class="btn btn-slate"
          on:click={() => (numFeaturesShown += minFeaturesToShow)}
          >Show More</button
        >
      {/if}
    </div>
  {/if}
{/if}
