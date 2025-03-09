<script lang="ts">
  import {
    createEventDispatcher,
    getContext,
    onDestroy,
    onMount,
  } from 'svelte';
  import QueryResultView from '../QueryResultView.svelte';
  import TextareaAutocomplete from '../slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from '../utils/query_autocomplete';
  import type { Writable } from 'svelte/store';
  import { faCheck, faChevronDown } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';
  import DatasetInfoView from './DatasetInfoView.svelte';
  import QueryLanguageReferenceView from '../QueryLanguageReferenceView.svelte';
  import QueryEditorTextarea from '../model_editor/QueryEditorTextarea.svelte';
  import {
    QueryTemplatesNoTimestepDefs,
    QueryTemplatesTimestepDefsOnly,
  } from '../model_editor/querytemplates';

  let {
    dataFields,
  }: {
    dataFields: Writable<string[]>;
  } = getContext('dataset');

  const dispatch = createEventDispatcher();

  export let queryHistory: string[] = [];
  let query: string = '';
  let finalQuery: string = '';
  let queryInput: QueryEditorTextarea;

  let autocompleteVisible: boolean = false;

  let showingDatasetInfo: boolean = false;
  let showingQueryReference: boolean = false;

  let hints: string[] = [
    '{field}',
    '2 * {field} - 30',
    '{field 1, field 2, field 3}',
    'case when {field} < 50 then "Low" else "High" end',
    'case when {field} < 2 then "0-2" when {field} < 4 then "2-4" else "> 4" end',
    'mean {field} from #now - 1 hour to #now every 4 hours from #mintime to #maxtime',
    'first {field} from #now to #now + 10 mins every 1 hour from #mintime to #maxtime',
    'exists {field 1, field 2} from #mintime to #now every 1 hour from #mintime to #maxtime',
    '{field} impute mean',
  ];
  let hintIndex: number = Math.floor(Math.random() * hints.length);
  let hintTimer: NodeJS.Timeout | null = null;

  onMount(() => {
    hintTimer = setTimeout(advanceHint, 5000);
    queryInput?.focus();
  });

  function advanceHint() {
    hintIndex = (hintIndex + 1) % hints.length;
    hintTimer = setTimeout(advanceHint, 5000);
  }

  onDestroy(() => {
    if (!!hintTimer) clearTimeout(hintTimer);
    hintTimer = null;
  });

  function addToHistory() {
    if (queryHistory.includes(finalQuery)) {
      let idx = queryHistory.indexOf(finalQuery);
      queryHistory = [
        ...queryHistory.slice(0, idx),
        ...queryHistory.slice(idx + 1),
      ];
    }
    queryHistory = [finalQuery, ...queryHistory];
  }
</script>

<div class="w-full relative overflow-y-auto" style="max-height: 80vh;">
  <div
    class="flex items-stretch p-4 gap-4 sticky top-0 bg-white z-30 {showingDatasetInfo ||
    showingQueryReference ||
    queryHistory.length > 0
      ? 'rounded-t-md border-b border-slate-400'
      : 'rounded-md'}"
  >
    <div class="flex-auto flex flex-col">
      <div class="relative w-full flex-auto min-h-32 mb-2">
        <QueryEditorTextarea
          bind:this={queryInput}
          class="resize-none appearance-none p-2 caret-blue-600 leading-tight w-full focus:outline-none focus:border-blue-600"
          textClass="font-mono text-base"
          templates={[
            ...QueryTemplatesNoTimestepDefs,
            ...QueryTemplatesTimestepDefsOnly,
          ]}
          bind:value={query}
          bind:autocompleteVisible
          placeholder={'Enter a query, such as: ' + hints[hintIndex]}
          on:keydown={(e) => {
            if (
              e.key === 'Enter' &&
              !(e.shiftKey || e.altKey) &&
              !autocompleteVisible
            ) {
              console.log('enter key');
              finalQuery = query;
              e.preventDefault();
            }
          }}
        />
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <button class="btn btn-blue" on:click={() => (finalQuery = query)}
          >Evaluate</button
        >
        <button
          class="btn {showingQueryReference ? 'btn-dark-slate' : 'btn-slate'}"
          on:click={() => {
            showingQueryReference = !showingQueryReference;
            showingDatasetInfo = false;
          }}>Syntax <Fa icon={faChevronDown} class="inline" /></button
        >
        <button
          class="btn {showingDatasetInfo ? 'btn-dark-slate' : 'btn-slate'}"
          on:click={() => {
            showingDatasetInfo = !showingDatasetInfo;
            showingQueryReference = false;
          }}>Dataset Info <Fa icon={faChevronDown} class="inline" /></button
        >
      </div>
    </div>
    <div
      class="text-sm w-48 shrink-0 grow-0 self-stretch ml-2 p-2"
      class:hidden={finalQuery.length == 0}
    >
      <QueryResultView
        query={finalQuery}
        evaluateQuery={true}
        delayEvaluation={false}
        on:result={(e) => {
          if (e.detail) addToHistory();
        }}
      />
    </div>
  </div>
  {#if showingDatasetInfo}
    <div class="w-full pb-4" style="max-height: 50vh;">
      <DatasetInfoView showHeader={false} />
    </div>
  {:else if showingQueryReference}
    <div class="w-full" style="height: 50vh;">
      <QueryLanguageReferenceView showHeader={false} />
    </div>
  {:else if queryHistory.length > 0}
    <div class="w-full">
      {#each queryHistory as historyItem (historyItem)}
        <button
          class="p-4 w-full flex items-stretch gap-4 appearance-none hover:bg-slate-100 text-left"
          on:click={() => {
            query = historyItem;
            queryInput.focus();
            queryInput.select();
          }}
        >
          <div
            class="text-slate-700 font-mono leading-relaxed flex-auto text-sm pl-2"
          >
            {historyItem}
          </div>
          <div class="text-sm w-48 shrink-0 grow-0 self-stretch ml-2 p-2">
            <QueryResultView
              compact
              query={historyItem}
              evaluateQuery={true}
              delayEvaluation={false}
            />
          </div>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  textarea::placeholder {
    @apply text-slate-500;
  }
</style>
