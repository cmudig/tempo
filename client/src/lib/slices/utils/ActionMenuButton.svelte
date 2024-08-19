<svelte:options accessors />

<script lang="ts">
  import { faEllipsisVertical } from '@fortawesome/free-solid-svg-icons';
  import { createEventDispatcher, onMount } from 'svelte';
  import Fa from 'svelte-fa/src/fa.svelte';

  const dispatch = createEventDispatcher();

  export let visible = false;

  export let buttonClass =
    'bg-transparent hover:opacity-60 text-slate-600 py-2 px-1 mr-2';
  export let buttonTitle = 'Show more actions';
  export let buttonStyle = '';
  export let align: 'left' | 'center' | 'right' = 'left';

  export let menuWidth = 240;

  export let disabled: boolean = false;

  export let singleClick: boolean = true;

  export let allowFocus: boolean = true;

  let optionsMenuOpacity = 0.0;
  let optionsMenu: Element;

  $: if (visible) window.addEventListener('keydown', escapeOptionsMenu, true);
  else window.removeEventListener('keydown', escapeOptionsMenu, true);

  function escapeOptionsMenu(e) {
    if (e.key === 'Escape') {
      hideOptionsMenu();
      e.stopPropagation();
      e.preventDefault();
    }
  }

  export function showOptionsMenu() {
    optionsMenuOpacity = 0;
    visible = true;
    setTimeout(() => (optionsMenuOpacity = 1.0), 10);
    if (!!optionsMenu && allowFocus) optionsMenu.focus();
    dispatch('show');
  }

  export function hideOptionsMenu(e: Event = undefined) {
    if (!allowFocus && !!e) e.preventDefault();
    optionsMenuOpacity = 0;
    setTimeout(() => (visible = false), 200);
    dispatch('hide');
  }

  function dismiss() {
    visible = false;
  }
</script>

<div class="relative">
  <button
    class={buttonClass}
    style={buttonStyle}
    id="menu-button"
    title={buttonTitle}
    {disabled}
    on:click|stopPropagation={allowFocus ? showOptionsMenu : undefined}
    on:mousedown={allowFocus
      ? undefined
      : (e) => {
          showOptionsMenu();
          e.preventDefault();
          e.stopPropagation();
        }}
    aria-expanded={visible}
    aria-label="Options menu"
    aria-haspopup="true"
  >
    <slot name="button-content">
      <Fa icon={faEllipsisVertical} class="inline text-center" />
    </slot>
  </button>
  {#if visible}
    <div
      class="fixed top-0 left-0 right-0 bottom-0 w-full h-full"
      style="z-index: 999;"
      on:mousedown|stopPropagation={hideOptionsMenu}
      on:keydown={(e) => {}}
    />
    <div
      class="absolute left-0 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none transition-opacity duration-200 overflow-visible"
      style="opacity: {optionsMenuOpacity}; width: {menuWidth}px; transform: translateX({align ==
      'right'
        ? '-100%'
        : align == 'center'
          ? '-50%'
          : '0'}); z-index: 1000;"
      role="menu"
      aria-orientation="vertical"
      aria-labelledby="menu-button"
      bind:this={optionsMenu}
      on:click|stopPropagation={singleClick ? hideOptionsMenu : () => {}}
      on:keydown={(e) => {}}
    >
      <div class="menu-options py-1" role="none">
        <slot name="options" {dismiss} />
      </div>
    </div>
  {/if}
</div>

<style>
  .menu-options :global(a) {
    @apply text-gray-700 block px-4 py-2 text-sm;
  }

  .menu-options :global(a:hover) {
    @apply bg-slate-100;
  }

  .menu-options :global(a:active) {
    @apply bg-slate-200;
  }
</style>
