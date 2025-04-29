<script lang="ts">
  import Fa from 'svelte-fa';
  import type { DataSource } from '../dataset';
  import { faChevronRight, faTrash } from '@fortawesome/free-solid-svg-icons';
  import { createEventDispatcher } from 'svelte';
  import { areObjectsEqual } from '../slices/utils/utils';

  const dispatch = createEventDispatcher();

  export let source: DataSource;

  let expanded: boolean = false;

  let customFields: string[] = [];
  $: if (!!source) {
    if (source.type == 'attributes') customFields = ['id_field'];
    else if (source.type == 'events')
      customFields = ['id_field', 'time_field', 'type_field', 'value_field'];
    else if (source.type == 'intervals')
      customFields = [
        'id_field',
        'start_time_field',
        'end_time_field',
        'type_field',
        'value_field',
      ];
  } else customFields = [];

  const fieldNames: { [key: string]: string } = {
    id_field: 'ID',
    time_field: 'Time',
    start_time_field: 'Start Time',
    end_time_field: 'End Time',
    type_field: 'Type',
    value_field: 'Value',
  };

  function getDefaultFieldName(field: string): string {
    if (field == 'type_field')
      return source.type == 'events' ? 'eventtype' : 'intervaltype';
    return {
      id_field: 'id',
      time_field: 'time',
      start_time_field: 'starttime',
      end_time_field: 'endtime',
      value_field: 'value',
    }[field];
  }

  let oldSource: DataSource | undefined = undefined;
  $: if (!areObjectsEqual(oldSource, source)) {
    filterFields();
    oldSource = source;
  }

  function filterFields() {
    source = Object.fromEntries(
      Object.entries(source).filter(
        ([k, v]) =>
          !Object.keys(fieldNames).includes(k) ||
          (customFields.includes(k) && v.length > 0)
      )
    ) as DataSource;
  }
</script>

<div class="mx-4 mb-2 rounded bg-slate-100">
  <div class="px-4 py-2 flex items-center gap-2">
    <button
      on:click={() => (expanded = !expanded)}
      class="hover:opacity-50 p-1 shrink-0"
    >
      <Fa
        class="inline mr-2 {expanded ? 'rotate-90' : ''}"
        icon={faChevronRight}
      />
    </button>
    <select class="flat-select" bind:value={source.type}>
      <option value="">Select data type</option>
      <option value="attributes">Attributes</option>
      <option value="events">Events</option>
      <option value="intervals">Intervals</option>
    </select>
    <div class="font-mono text-sm truncate flex-auto">
      {source.path}
    </div>
    <button
      class="hover:opacity-50 shrink-0 text-slate-500 px-2"
      title="Delete data source"
      on:click={(e) => dispatch('delete')}><Fa icon={faTrash} /></button
    >
  </div>
  {#if expanded}
    <div class="px-4 pb-2">
      <div class="font-bold mb-1 text-sm">Field Names</div>
      <div class="text-slate-600 text-xs mb-2">
        All of the fields below are required in the uploaded file. Enter custom
        names to use those columns instead of the default names.
      </div>
      <div
        class="grid gap-2 text-sm items-center"
        style="grid-template-columns: max-content auto;"
      >
        {#each customFields as field (field)}
          <label class="text-slate-600" for={field}>{fieldNames[field]} </label>
          <input
            class="flat-text-input"
            bind:value={source[field]}
            id={field}
            placeholder={getDefaultFieldName(field)}
            autocomplete="false"
            autocapitalize="false"
          />
        {/each}
      </div>
    </div>
  {/if}
</div>
