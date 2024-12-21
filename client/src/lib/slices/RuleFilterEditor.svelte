<script lang="ts">
  import { createEventDispatcher, getContext } from 'svelte';
  import {
    RuleFilterCombination,
    RuleFilterType,
    type RuleFilter,
  } from './rulefilters';
  import Fa from 'svelte-fa';
  import {
    faChevronDown,
    faDiagramPredecessor,
    faTrash,
  } from '@fortawesome/free-solid-svg-icons';
  import type { Writable } from 'svelte/store';
  import type { ModelMetrics, ModelSummary, QueryResult } from '../model';
  import ActionMenuButton from './utils/ActionMenuButton.svelte';
  import { areObjectsEqual, deepCopy } from './utils/utils';
  import QueryResultView from '../QueryResultView.svelte';
  import Checkbox from '../slices/utils/Checkbox.svelte';
  import SearchableMultiselect from './utils/SearchableMultiselect.svelte';

  const dispatch = createEventDispatcher();

  let { currentDataset }: { currentDataset: Writable<string | null> } =
    getContext('dataset');

  let {
    models,
    currentModel,
  }: {
    models: Writable<{
      [key: string]: { spec: ModelSummary; metrics?: ModelMetrics };
    }>;
    currentModel: Writable<string | null>;
  } = getContext('models');

  export let valueNames: {
    [key: string]: [any, { [key: string]: any }];
  } | null = {};

  let inverseValueNames: { [key: string]: string } = {};
  $: if (!!valueNames)
    inverseValueNames = Object.fromEntries(
      Object.entries(valueNames).map((entry) => [entry[1][0], entry[0]])
    );
  else inverseValueNames = {};

  export let ruleFilter: RuleFilter | null = null;
  export let topLevel: boolean = false;
  export let allowDelete: boolean = true;
  export let changesPending: boolean = false;

  const Combinations = [
    { value: 'and', name: 'and' },
    { value: 'or', name: 'or' },
  ];
  const FilterLogics = [
    { value: RuleFilterType.exclude, name: 'Exclude' },
    { value: RuleFilterType.include, name: 'Require' },
  ];

  function valueSetForFeatures(features: any[]): string[] {
    if (!valueNames) return [];
    if (features.length == 0) return [];
    return Array.from(
      new Set(
        features
          .map((f) => Object.values(valueNames[inverseValueNames[f]][1]))
          .flat()
      )
    );
  }
</script>

{#if !!ruleFilter}
  {#if topLevel}
    <div class="rounded bg-slate-100 w-full p-2 mb-2">
      {#if !!valueNames}
        <svelte:self
          {ruleFilter}
          {valueNames}
          on:delete
          on:change={(e) => (ruleFilter = e.detail)}
          {allowDelete}
        />
      {:else}
        <div class="w-full m-12 text-slate-600 text-center">
          The filter can't edited right now because the subgroup discovery
          algorithm failed to run.
        </div>
      {/if}
    </div>
  {:else}
    <div>
      <div class="flex items-center gap-2 mt-2 mb-1">
        {#if allowDelete}
          <button
            class="bg-transparent hover:opacity-60 mx-1 text-slate-600"
            on:click={() => dispatch('delete')}><Fa icon={faTrash} /></button
          >
        {/if}
        <button
          class="bg-transparent hover:opacity-60 mx-1 text-slate-600"
          title="Add more conditions, joined by AND or OR"
          on:click={() =>
            dispatch('change', {
              type: 'combination',
              combination: RuleFilterCombination.and,
              lhs: ruleFilter,
              rhs: {
                type: 'constraint',
                logic: RuleFilterType.exclude,
                features: [],
                values: [],
              },
            })}
          ><Fa icon={faDiagramPredecessor} class="text-slate-600" /></button
        >
      </div>
      {#if ruleFilter.type == 'constraint'}
        <div class="mt-2">
          <select
            class="flat-select text-sm"
            value={ruleFilter.logic}
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(ruleFilter ?? {}), {
                  logic: e.target?.value,
                })
              )}
          >
            {#each FilterLogics as logic}
              <option value={logic.value}>{logic.name}</option>
            {/each}
          </select>
        </div>
        {#if !!valueNames}
          {@const selectedFeatures = ruleFilter.features ?? []}
          <div class="flex items-center gap-2 mt-2">
            <div class="text-slate-600 text-sm">Subgroups with features:</div>
            <!-- <select
              class="flat-select text-sm"
              value={!!ruleFilter.features && ruleFilter.features.length > 0
                ? ruleFilter.features[0]
                : ''}
              on:change={(e) =>
                dispatch(
                  'change',
                  Object.assign(deepCopy(ruleFilter ?? {}), {
                    features:
                      e.target?.value.length > 0 ? [e.target?.value] : [],
                  })
                )}
            >
              <option value=""></option>
              {#each Object.entries(valueNames) as featureEntry}
                <option value={featureEntry[0]}>{featureEntry[1][0]}</option>
              {/each}
            </select> -->
            <ActionMenuButton
              buttonClass="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg p-1.5"
              buttonTitle="Select features to {ruleFilter.logic ==
              RuleFilterType.exclude
                ? 'exclude'
                : 'require'}"
              buttonActiveClass="text-slate-800 outline outline-blue-400"
              singleClick={false}
              ><span slot="button-content"
                >{#if selectedFeatures.length == 0}Select features{:else}{selectedFeatures[0]}
                  {#if selectedFeatures.length > 1}and {selectedFeatures.length -
                      1} others{/if}{/if}
                <Fa
                  icon={faChevronDown}
                  style="transform: translateY(-2px); font-size: 0.6em;"
                  class="inline"
                /></span
              >
              <div slot="options">
                <SearchableMultiselect
                  choices={Object.entries(valueNames)
                    .sort((a, b) => a[1][0] - b[1][0])
                    .map((item) => ({ value: item[1][0], name: item[1][0] }))}
                  selected={ruleFilter.features ?? []}
                  on:change={(e) => {
                    console.log('change', e.detail);
                    dispatch(
                      'change',
                      Object.assign(deepCopy(ruleFilter ?? {}), {
                        features: e.detail,
                      })
                    );
                  }}
                />
              </div></ActionMenuButton
            >
          </div>
          {#if !!ruleFilter.features && ruleFilter.features.length > 0}
            {@const selectedValues = ruleFilter.values ?? []}
            <div class="flex items-center gap-2 mt-2">
              <div class="text-slate-600 text-sm">Taking values:</div>
              <ActionMenuButton
                buttonClass="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg p-1.5"
                buttonTitle="Select features to {ruleFilter.logic ==
                RuleFilterType.exclude
                  ? 'exclude'
                  : 'require'}"
                buttonActiveClass="text-slate-800 outline outline-blue-400"
                singleClick={false}
                ><span slot="button-content"
                  >{#if selectedValues.length == 0}(any value){:else}{selectedValues[0]}
                    {#if selectedValues.length > 1}and {selectedValues.length -
                        1} others{/if}{/if}
                  <Fa
                    icon={faChevronDown}
                    style="transform: translateY(-2px); font-size: 0.6em;"
                    class="inline"
                  /></span
                >
                <div slot="options">
                  <SearchableMultiselect
                    choices={valueSetForFeatures(ruleFilter.features ?? [])
                      .sort()
                      .map((item) => ({ value: item, name: item }))}
                    selected={ruleFilter.values ?? []}
                    on:change={(e) => {
                      console.log('change', e.detail);
                      dispatch(
                        'change',
                        Object.assign(deepCopy(ruleFilter ?? {}), {
                          values: e.detail,
                        })
                      );
                    }}
                  />
                </div></ActionMenuButton
              >
            </div>
          {/if}
        {/if}
      {:else if ruleFilter.type == 'combination'}
        <div class="mt-2 text-slate-600 text-sm">Exclude if...</div>
        <div class="function-container">
          <svelte:self
            ruleFilter={ruleFilter.lhs}
            {valueNames}
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(ruleFilter ?? {}), { lhs: e.detail })
              )}
            on:delete={(e) => dispatch('change', ruleFilter.rhs)}
          />
        </div>
        <select
          class="flat-select text-sm"
          value={ruleFilter.combination}
          on:change={(e) =>
            dispatch(
              'change',
              Object.assign(deepCopy(ruleFilter ?? {}), {
                combination: e.target.value,
              })
            )}
        >
          {#each Combinations as combination}
            <option value={combination.value}>{combination.name}</option>
          {/each}
        </select>
        <div class="function-container">
          <svelte:self
            ruleFilter={ruleFilter.rhs}
            {valueNames}
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(ruleFilter ?? {}), { rhs: e.detail })
              )}
            on:delete={(e) => dispatch('change', ruleFilter.lhs)}
          />
        </div>
      {/if}
    </div>
  {/if}
{/if}

<style>
  .function-container {
    @apply ml-2 pl-2 border-l-2 border-slate-300 my-2 flex items-center;
  }
</style>
