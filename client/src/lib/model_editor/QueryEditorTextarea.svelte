<svelte:options accessors />

<script lang="ts">
  import highlight from 'custom-syntax-highlighter';
  import { highlightPatterns } from './syntaxhighlight';
  import QueryTemplateMenu from './QueryTemplateMenu.svelte';
  import TextareaAutocomplete from '../slices/utils/TextareaAutocomplete.svelte';
  import {
    getAutocompleteOptions,
    performAutocomplete,
  } from '../utils/query_autocomplete';
  import { getContext } from 'svelte';
  import type { Writable } from 'svelte/store';

  let { dataFields }: { dataFields: Writable<string[]> } =
    getContext('dataset');

  export let textClass: string | undefined = undefined;
  export let style: string | undefined = undefined;
  export let value: string | null = null;

  export let templates: {
    title: string;
    children: { name: string; query: string }[];
  }[] = [];
  export let autocompleteVisible: boolean = false;

  let highlightedView: HTMLElement;
  let highlightedViewID: string =
    'highlightView-' +
    new Array(20)
      .fill(0)
      .map(() => Math.floor(Math.random() * 10))
      .join('');

  $: if (!!highlightedView) {
    highlightedView.innerText = formatQueryForHighlights(value ?? '');
    highlight({
      selector: `#${highlightedViewID}`,
      patterns: highlightPatterns,
    });
  }

  let queryInput: HTMLTextAreaElement;

  function formatQueryForHighlights(query: string) {
    // replace <word> with a token
    query = query.replaceAll(/<([A-z]+?)>/g, '####TOKEN####$1####ENDTOKEN####');
    return query.replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  }

  $: if (!!queryInput) {
    document.addEventListener('selectionchange', onSelectionChange);
  } else {
    document.removeEventListener('selectionchange', onSelectionChange);
  }

  function onSelectionChange(e) {
    const activeElement = document.activeElement;
    if (activeElement !== queryInput || !value) return;

    const selectionStart = queryInput.selectionStart;
    const selectionEnd = queryInput.selectionEnd;
    // look for overlaps with any tokens
    let precedingTokenMatch = value.slice(0, selectionStart).search(/<[^>]*$/);
    let followingTokenMatch = value.slice(selectionEnd).match(/^[^<]*>/);
    let newSelectionStart = selectionStart;
    let newSelectionEnd = selectionEnd;
    if (
      precedingTokenMatch >= 0 &&
      value.slice(precedingTokenMatch).search(/^<([A-z]+)>/) >= 0
    )
      newSelectionStart = precedingTokenMatch;
    if (
      !!followingTokenMatch &&
      value
        .slice(0, selectionEnd + followingTokenMatch[0].length)
        .search(/<([A-z]+)>$/) >= 0
    )
      newSelectionEnd = selectionEnd + followingTokenMatch[0].length;
    if (newSelectionStart != selectionStart || newSelectionEnd != selectionEnd)
      queryInput.setSelectionRange(newSelectionStart, newSelectionEnd);
  }
  let editorFocused: boolean = false;

  export function focus() {
    if (!!queryInput) {
      queryInput.focus();
      editorFocused = true;
    }
  }

  $: if (!!queryInput && !!value) {
    queryInput.style.height = '1px';
    queryInput.style.height = 12 + queryInput.scrollHeight + 'px';
  }
</script>

<textarea
  spellcheck={false}
  class="{textClass ?? 'font-mono text-xs'} {$$props.class ??
    'flat-text-input w-full h-full leading-tight'}"
  style="color: transparent; {style ?? ''}"
  bind:this={queryInput}
  bind:value
  on:input
  on:focus={() => (editorFocused = true)}
  on:blur={() => (editorFocused = false)}
  on:keydown
  on:keypress
  on:keyup
/>
<div
  class="border-2 border-transparent {textClass ??
    'text-slate-700 font-mono text-xs'} p-2 leading-tight pointer-events-none bg-transparent w-full h-full absolute top-0 left-0 text-wrap whitespace-pre-wrap break-words"
  id={highlightedViewID}
  bind:this={highlightedView}
></div>
<TextareaAutocomplete
  ref={queryInput}
  resolveFn={(query, prefix) =>
    getAutocompleteOptions($dataFields, query, prefix)}
  replaceFn={performAutocomplete}
  triggers={['{', '#']}
  delimiterPattern={/[\s\(\[\]\)](?=[\{#])/}
  menuItemTextFn={(v) => v.value}
  maxItems={3}
  menuItemClass="p-2"
  bind:visible={autocompleteVisible}
  on:replace={(e) => (value = e.detail)}
/>
<div class="flex gap-2 items-center">
  <QueryTemplateMenu
    disabled={!editorFocused}
    {templates}
    on:insert={(e) => {
      if (!editorFocused) return;
      document.execCommand('insertText', false, e.detail.query);
    }}
  />
  <slot name="buttons" />
</div>
