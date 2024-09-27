<!--
  @component
  Generates an SVG bar chart.
 -->
<script>
  import { createEventDispatcher, getContext } from 'svelte';

  const dispatch = createEventDispatcher();

  const { data, x, xGet, yGet, xScale, yScale, xDomain, custom, width } =
    getContext('LayerCake');

  /** @type {String} [fill='#00bbff'] - The shape's fill color. This is technically optional because it comes with a default value but you'll likely want to replace it with your own color. */
  export let fill = '#00bbff';

  export let fillFn = null;

  export let clickable = false;

  const hoverStroke = '#333';
  const hoverStrokeWidth = 1;
  const selectStrokeWidth = 3;

  let hoveredIndex = null;

  function getSelected(customVals, d) {
    return (
      !!customVals && !!customVals.selectedGet && customVals.selectedGet(d)
    );
  }
</script>

<g class="bar-group">
  {#each $data as d, i}
    {#if $x(d) < 0 && $x(d) == $x(d)}
      <rect
        class="group-rect"
        data-id={i}
        x={Math.max(0, $xGet(d))}
        y={$yGet(d)}
        height={$yScale.bandwidth()}
        width={$xScale(0) - Math.max(0, $xGet(d))}
        fill={!!fillFn ? fillFn(d) : fill}
        stroke={hoveredIndex == i || getSelected($custom, d)
          ? hoverStroke
          : 'none'}
        stroke-width={getSelected($custom, d)
          ? selectStrokeWidth
          : hoveredIndex == i
          ? hoverStrokeWidth
          : 0}
        class:pointer={clickable}
      />
    {:else if $x(d) == $x(d)}
      <rect
        class="group-rect"
        data-id={i}
        x={$xScale(0)}
        y={$yGet(d)}
        height={$yScale.bandwidth()}
        width={Math.min($xGet(d), $width) - $xScale(0)}
        fill={!!fillFn ? fillFn(d) : fill}
        stroke={hoveredIndex == i || getSelected($custom, d)
          ? hoverStroke
          : 'none'}
        stroke-width={getSelected($custom, d)
          ? selectStrokeWidth
          : hoveredIndex == i
          ? hoverStrokeWidth
          : 0}
        class:pointer={clickable}
      />
    {/if}
    <rect
      class="hover-zone"
      class:pointer={clickable}
      x={0}
      y={$yGet(d)}
      height={$yScale.bandwidth()}
      width={$width}
      fill="none"
      stroke="none"
      on:mouseenter={() => {
        hoveredIndex = i;
        dispatch('hover', d);
      }}
      on:mouseleave={() => {
        hoveredIndex = null;
        dispatch('hover', null);
      }}
      on:click={() => dispatch('click', d)}
      on:keydown={(event) => {
        if (event.code === 13) dispatch('click', d);
      }}
    />
  {/each}
</g>

<style>
  .hover-zone {
    pointer-events: all;
  }
</style>
