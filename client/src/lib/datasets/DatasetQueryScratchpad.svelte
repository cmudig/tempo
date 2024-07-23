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
  import { faCheck } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa';

  let {
    dataFields,
  }: {
    dataFields: Writable<string[]>;
  } = getContext('dataset');

  const dispatch = createEventDispatcher();

  export let queryHistory: string[] = [];
  let query: string = '';
  let finalQuery: string = '';
  let queryInput: HTMLElement;

  let autocompleteVisible: boolean = false;

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

  let copiedText: boolean = false;
  function copyQuery() {
    navigator.clipboard.writeText(queryInput.value);
    copiedText = true;
    setTimeout(() => (copiedText = false), 5000);
  }

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

  $: if (query !== finalQuery) finalQuery = '';
</script>

<div class="w-full">
  <div class="flex items-stretch p-4 gap-4">
    <div class="flex-auto flex flex-col">
      <div class="relative w-full flex-auto h-24 mb-2">
        <textarea
          class="resize-none appearance-none text-slate-700 font-mono p-2 caret-blue-600 leading-relaxed w-full h-full font-mono text-base focus:outline-none focus:border-blue-600"
          spellcheck={false}
          bind:value={query}
          bind:this={queryInput}
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
        {#if $dataFields.length > 0}
          <TextareaAutocomplete
            ref={queryInput}
            resolveFn={(query, prefix) =>
              getAutocompleteOptions($dataFields, query, prefix)}
            replaceFn={performAutocomplete}
            triggers={['{', '#', ',']}
            delimiterPattern={/[\s\(\[\]\)](?=[\{#])/}
            menuItemTextFn={(v) => v.value}
            maxItems={3}
            menuItemClass="p-2"
            on:replace={(e) => (query = e.detail)}
            bind:visible={autocompleteVisible}
          />
        {/if}
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <button class="btn btn-blue" on:click={() => (finalQuery = query)}
          >Evaluate</button
        >
        <button
          on:click={() => {
            query = '';
            queryInput.focus();
          }}
          class="btn btn-slate">Clear</button
        >
        <button on:click={copyQuery} class="btn btn-slate">Copy Query</button>
        <div
          class="transition-opacity duration-300 text-slate-500 text-sm"
          class:opacity-0={!copiedText}
        >
          <Fa icon={faCheck} class="inline mr-1" /> Copied to clipboard.
        </div>
      </div>
    </div>
    <div
      class="text-sm w-48 shrink-0 grow-0 self-stretch ml-2 p-2"
      class:hidden={finalQuery.length == 0 || finalQuery != query}
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
  {#if queryHistory.length > 0}
    <div class="overflow-y-auto w-full border-t border-slate-400">
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
