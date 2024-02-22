<script lang="ts">
  import { faInfoCircle } from '@fortawesome/free-solid-svg-icons';
  import Fa from 'svelte-fa/src/fa.svelte';

  export let title: string = '';
  export let position: 'top' | 'bottom' | 'left' | 'right' = 'top';
  export let hoverTargetClass: string = 'text-slate-300 hover:text-slate-500';
</script>

<div class="tooltip relative inline-block" aria-haspopup="true">
  <slot>
    <Fa icon={faInfoCircle} class={hoverTargetClass} />
  </slot>
  <div
    class="tooltiptext bg-slate-800/90 border border-slate-200 text-white font-normal font-sans text-sm rounded shadow p-2"
    class:top={position == 'top'}
    class:bottom={position == 'bottom'}
    class:left={position == 'left'}
    class:right={position == 'right'}
    role="tooltip"
  >
    {title}
  </div>
</div>

<style>
  .top {
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
  }

  .bottom {
    top: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
  }

  .right {
    top: 50%;
    transform: translateY(-50%);
    left: calc(100% + 8px);
  }

  .left {
    top: 50%;
    transform: translateY(-50%);
    right: calc(100% + 8px);
  }

  /* Tooltip text */
  .tooltip .tooltiptext {
    visibility: hidden;
    text-align: center;

    /* Position the tooltip text - see examples below! */
    position: absolute;
    z-index: 999;

    width: max-content;
    max-width: 400px;
  }

  /* Show the tooltip text when you mouse over the tooltip container */
  .tooltip:hover .tooltiptext {
    visibility: visible;
  }
</style>
