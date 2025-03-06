<script lang="ts">
  import { createEventDispatcher, getContext } from 'svelte';
  import type { ScoreFunction } from './scorefunctions';
  import Fa from 'svelte-fa';
  import {
    faDiagramPredecessor,
    faTrash,
  } from '@fortawesome/free-solid-svg-icons';
  import type { Writable } from 'svelte/store';
  import type { ModelMetrics, ModelSummary, QueryResult } from '../model';
  import ActionMenuButton from './utils/ActionMenuButton.svelte';
  import { areObjectsEqual, deepCopy } from './utils/utils';
  import QueryResultView from '../QueryResultView.svelte';

  const dispatch = createEventDispatcher();

  let csrf: Writable<string> = getContext('csrf');
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

  export let scoreFunction: ScoreFunction | null = null;
  export let topLevel: boolean = false;
  export let allowDelete: boolean = true;
  export let showTypeControl: boolean = true;
  export let changesPending: boolean = false;

  const FunctionTypes = [
    { value: 'constant', name: 'Constant' },
    { value: 'relation', name: 'Comparative Relation' },
    { value: 'logical', name: 'Logical Predicate' },
    { value: 'model_property', name: 'Model Property' },
  ];
  const Relations = [
    { value: '=', name: 'equals' },
    { value: '!=', name: 'does not equal' },
    { value: '<', name: 'less than' },
    { value: '>', name: 'greater than' },
    { value: '<=', name: 'less than or equal to' },
    { value: '>=', name: 'greater than or equal to' },
    { value: 'in', name: 'is one of' },
    { value: 'not-in', name: 'is not one of' },
  ];
  const LogicalOperations = [
    { value: 'and', name: 'and' },
    { value: 'or', name: 'or' },
  ];
  const ModelProperties = [
    { value: 'label', name: 'True Label' },
    { value: 'prediction', name: 'Prediction' },
    { value: 'prediction_probability', name: 'Prediction Probability' },
    { value: 'correctness', name: 'Correctness' },
    { value: 'deviation', name: 'Deviation' },
    { value: 'abs_deviation', name: 'Absolute Deviation' },
  ];

  function convertConstantValue(val: string): string | number {
    if (val.toLocaleLowerCase() == 'false') return 0;
    if (val.toLocaleLowerCase() == 'true') return 1;
    let rawNumber = parseFloat(val);
    let intNumber = parseInt(val);
    if (!Number.isNaN(rawNumber)) {
      if (rawNumber == intNumber) return intNumber;
      else return rawNumber;
    }
    return val;
  }

  function changeType(newType: string): ScoreFunction {
    if (!scoreFunction) return { type: 'constant', value: 0 };
    if (newType == scoreFunction.type) return scoreFunction;
    else if (newType == 'relation') {
      return {
        type: 'relation',
        relation: '=',
        lhs: scoreFunction,
        rhs: { type: 'constant', value: 0 },
      };
    } else if (newType == 'logical') {
      return {
        type: 'logical',
        relation: 'and',
        lhs: scoreFunction,
        rhs: scoreFunction,
      };
    } else if (newType == 'model_property') {
      return {
        type: 'model_property',
        property: 'label',
        model_name: $currentModel ?? '',
      };
    } else return { type: 'constant', value: 0 };
  }

  let oldScoreFunction: ScoreFunction | null = null;
  $: if (topLevel && !areObjectsEqual(oldScoreFunction, scoreFunction)) {
    evaluateIfNeeded(scoreFunction, oldScoreFunction == null);
    oldScoreFunction = scoreFunction;
  }

  let evaluationSummary: QueryResult | null = null;
  let evaluationError: string | null = null;
  let evaluationTimer: NodeJS.Timeout | null = null;
  let loadingEvaluation: boolean = false;

  function evaluateIfNeeded(
    q: ScoreFunction | null,
    immediate: boolean = false
  ) {
    evaluationError = null;
    evaluationSummary = null;
    changesPending = true;
    if (!!evaluationTimer) clearTimeout(evaluationTimer);
    if (!!q) {
      if (immediate) liveEvaluateQuery();
      else evaluationTimer = setTimeout(liveEvaluateQuery, 2000);
    }
  }
  async function liveEvaluateQuery() {
    let result: { error?: string; result?: QueryResult };
    try {
      result = await (
        await fetch(
          import.meta.env.BASE_URL +
            `/datasets/${$currentDataset}/slices/${$currentModel}/validate_score_function`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-CSRF-Token': $csrf,
            },
            credentials: 'same-origin',
            body: JSON.stringify({ score_function: scoreFunction }),
          }
        )
      ).json();
    } catch (e) {
      evaluationError = `${e}`;
      evaluationSummary = null;
      changesPending = false;
      return;
    }
    console.log('evaluated query:', result);
    changesPending = false;

    if (result.error) {
      evaluationError = result.error;
      evaluationSummary = null;
    } else if (!!result.result) {
      evaluationSummary = result.result;
      evaluationError = null;
      dispatch('change', scoreFunction);
    }
  }
</script>

{#if !!scoreFunction}
  {#if topLevel}
    <div class="rounded bg-slate-100 w-full p-2 mb-2 flex items-center">
      <div class="flex-auto">
        <svelte:self
          {scoreFunction}
          on:delete
          on:change={(e) => (scoreFunction = e.detail)}
          {allowDelete}
        />
      </div>
      <div class="w-48 mr-2">
        {#if !!evaluationSummary || !!evaluationError}
          <QueryResultView
            evaluateQuery={false}
            {evaluationSummary}
            {evaluationError}
          />
        {:else}
          <div class="text-slate-500 text-xs">
            Validating search criteria...
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <div
      class={scoreFunction.type == 'relation' || scoreFunction.type == 'logical'
        ? ''
        : 'flex items-center gap-2'}
    >
      {#if (scoreFunction.type == 'constant' || scoreFunction.type == 'model_property') && showTypeControl}
        <select
          class="flat-select text-sm"
          value={scoreFunction.type}
          on:change={(e) => dispatch('change', changeType(e.target.value))}
        >
          <option value="model_property">Model Property</option>
          <option value="constant">Constant</option>
        </select>
      {:else if scoreFunction.type == 'relation'}
        <div class="inline-flex items-center order-last gap-2">
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
                type: 'logical',
                relation: 'and',
                lhs: scoreFunction,
                rhs: {
                  type: 'relation',
                  relation: '=',
                  lhs: {
                    type: 'model_property',
                    property: 'label',
                    model_name: $currentModel ?? '',
                  },
                  rhs: { type: 'constant', value: 0 },
                },
              })}
            ><Fa icon={faDiagramPredecessor} class="text-slate-600" /></button
          >
          <!-- <ActionMenuButton buttonClass="bg-transparent px-1 hover:opacity-40"
          ><span slot="button-content"
            ><Fa icon={faDiagramPredecessor} class="inline" /></span
          >
          <div slot="options">
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              on:click={() =>
                dispatch('change', {
                  type: 'relation',
                  relation: '=',
                  lhs: scoreFunction,
                  rhs: { type: 'constant', value: 0 },
                })}>Comparative Relation (=, &gt;, etc.)</a
            >
            <a
              href="#"
              tabindex="0"
              role="menuitem"
              on:click={() =>
                dispatch('change', {
                  type: 'logical',
                  relation: 'and',
                  lhs: scoreFunction,
                  rhs: scoreFunction,
                })}>Logical Predicate (And/Or)</a
            >
          </div></ActionMenuButton
        > -->
        </div>
      {/if}
      {#if scoreFunction.type == 'constant' && scoreFunction.value !== undefined}
        {#if Array.isArray(scoreFunction.value)}
          {#each scoreFunction.value as element, i}
            <input
              class="flat-text-input"
              type="text"
              value={scoreFunction.value}
              placeholder="Value"
              on:change={(e) =>
                dispatch(
                  'change',
                  Object.assign(deepCopy(scoreFunction), {
                    value: [
                      ...scoreFunction.value?.slice(0, i),
                      convertConstantValue(e.target.value ?? ''),
                      ...scoreFunction?.value.slice(i + 1),
                    ],
                  })
                )}
            />
          {/each}
        {:else}
          <input
            class="flat-text-input"
            type="text"
            value={scoreFunction.value}
            placeholder="Value"
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(scoreFunction), {
                  value: convertConstantValue(e.target.value ?? ''),
                })
              )}
          />
        {/if}
      {:else if scoreFunction.type == 'relation' || scoreFunction.type == 'logical'}
        <div class="function-container">
          <svelte:self
            scoreFunction={scoreFunction.lhs}
            showTypeControl={false}
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(scoreFunction), { lhs: e.detail })
              )}
            on:delete={(e) => dispatch('change', scoreFunction.rhs)}
          />
        </div>
        <select
          class="flat-select text-sm"
          value={scoreFunction.relation}
          on:change={(e) =>
            dispatch(
              'change',
              Object.assign(deepCopy(scoreFunction), {
                relation: e.target.value,
              })
            )}
        >
          {#each scoreFunction.type == 'relation' ? Relations : LogicalOperations as relation}
            <option value={relation.value}>{relation.name}</option>
          {/each}
        </select>
        <div class="function-container">
          <svelte:self
            scoreFunction={scoreFunction.rhs}
            on:change={(e) =>
              dispatch(
                'change',
                Object.assign(deepCopy(scoreFunction), { rhs: e.detail })
              )}
            on:delete={(e) => dispatch('change', scoreFunction.lhs)}
          />
        </div>
      {:else if scoreFunction.type == 'model_property'}
        <select
          class="flat-select text-sm"
          value={scoreFunction.model_name}
          on:change={(e) =>
            dispatch(
              'change',
              Object.assign(deepCopy(scoreFunction), {
                model_name: e.target.value,
              })
            )}
        >
          {#each Object.keys($models).sort() as modelName}
            <option value={modelName}>{modelName}</option>
          {/each}
        </select>
        <select
          class="flat-select text-sm"
          value={scoreFunction.property}
          on:change={(e) =>
            dispatch(
              'change',
              Object.assign(deepCopy(scoreFunction), {
                property: e.target.value,
              })
            )}
        >
          {#each ModelProperties as property}
            <option value={property.value}>{property.name}</option>
          {/each}
        </select>
      {/if}
    </div>
  {/if}
{/if}

<style>
  .function-container {
    @apply ml-2 pl-2 border-l-2 border-slate-300 my-2 flex items-center;
  }
</style>
