<script lang="ts">
  import { faChevronRight, faPlus } from '@fortawesome/free-solid-svg-icons';
  import ActionMenuButton from '../slices/utils/ActionMenuButton.svelte';
  import Fa from 'svelte-fa';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  export let templates: {
    title: string;
    children: { name: string; query: string }[];
  }[] = [];
  export let disabled: boolean = false;

  let hoveringCategory: string | null = null;

  let actionButton: ActionMenuButton;
</script>

<ActionMenuButton
  {disabled}
  buttonClass="btn-slate btn-sm"
  allowFocus={false}
  singleClick
  bind:this={actionButton}
>
  <span slot="button-content"
    ><Fa icon={faPlus} class="inline mr-1" />Template</span
  >
  <div slot="options" class="overflow-visible">
    <ul>
      {#each templates as category (category.title)}
        <li
          class="relative text-gray-700 block px-4 py-2 text-sm overflow-visible hover:bg-slate-100"
          on:mouseenter={() => (hoveringCategory = category.title)}
          on:mouseleave={() => (hoveringCategory = null)}
        >
          <div class="w-full h-full flex justify-between items-center">
            <div>{category.title}</div>
            <Fa icon={faChevronRight} class="text-xs" />
          </div>
          {#if hoveringCategory == category.title}
            <div
              class="z-50 absolute top-0"
              style="left: calc(100% - 12px); margin-left: 12px;"
            >
              <div
                class="bg-white ring-1 ring-black ring-opacity-5 focus:outline-none rounded-md shadow-lg w-64 overflow-y-auto"
                style="max-height: 240px;"
              >
                <ul class="py-2 text-sm text-gray-700">
                  {#each category.children as template}
                    <li>
                      <a
                        href="#"
                        on:mousedown|preventDefault={() => {}}
                        on:mouseup|preventDefault={() => {
                          dispatch('insert', template);
                          hoveringCategory = null;
                          actionButton.hideOptionsMenu();
                        }}
                        ><div class="font-semibold">{template.name}</div>
                        <div class="text-xs text-slate-500 truncate">
                          {template.query.replaceAll(/<\w+>/g, '...')}
                        </div></a
                      >
                    </li>
                  {/each}
                </ul>
              </div>
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  </div>
</ActionMenuButton>
