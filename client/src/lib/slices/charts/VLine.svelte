<!--
  @component
  Generates a vertical line that optionally shows the values in each series.
 -->
<script>
  import { getContext } from 'svelte';
  import { getTextWidth } from '../utils/utils';
  import * as d3 from 'd3';

  const { data, x, xGet, y, xScale, yGet, r, width, height } =
    getContext('LayerCake');

  /** @type {String} [stroke='#ab00d6'] - The stroke color of the line. */
  export let color = '#bbb';

  export let xValue = null;

  export let title = null;

  export let strokeWidth = 2;

  export let showMatchingPoints = false;

  export let formatValue = '.3';

  export let pointSize = 6;

  export let rColorMap = null;

  let hovering = false;

  let alignTextLeft = true;
  $: alignTextLeft = $xScale(xValue) <= $width - 100;

  let dataToLabel = null;
  $: if (showMatchingPoints) {
    let filteredData = $data.filter((d) => $x(d) == xValue);
    let nodes = filteredData.map((d) => ({
      fx: $xGet(d),
      targetY: $yGet(d),
      y: $yGet(d),
    }));
    if (filteredData.length > 1) {
      // Use a force simulation to push apart labels
      let force = d3
        .forceSimulation()
        .nodes(nodes)
        .force('collide', d3.forceCollide(18))
        .force('y', d3.forceY((d) => d.targetY).strength(1))
        .stop();

      for (let i = 0; i < 50; i++) force.tick();
    }
    dataToLabel = filteredData.map((d, i) => ({ d, labelY: nodes[i].y }));
  } else {
    dataToLabel = null;
  }
</script>

{#if xValue != null}
  <g
    class="vline"
    transform="translate({$xScale(xValue)}, 0)"
    on:mouseenter={() => (hovering = true)}
    on:mouseleave={() => (hovering = false)}
  >
    <line
      class="vline line"
      class:emphasized={hovering}
      x1={0}
      x2={0}
      y1={0}
      y2={$height}
      stroke-width={strokeWidth}
      stroke={color}
    />
    {#if !!title}
      <g
        class="vline-label"
        transform="translate({alignTextLeft ? 8 : -8}, 14)"
      >
        <rect
          x={alignTextLeft ? -4 : -getTextWidth(title, '11pt sans-serif') - 4}
          y={-14}
          rx="3"
          ry="3"
          width={getTextWidth(title, '11pt sans-serif') + 8}
          height={20}
          class="label-background"
        />
        <text
          class="title-label"
          class:emphasized={hovering}
          x={0}
          y={0}
          text-anchor={alignTextLeft ? 'start' : 'end'}
          style="fill: {color};">{@html title}</text
        >
      </g>
    {/if}
    {#if showMatchingPoints}
      {#each dataToLabel as datum}
        {@const valueText = d3.format(formatValue)($y(datum.d))}
        <ellipse
          class:emphasized={hovering}
          cx={0}
          cy={$yGet(datum.d)}
          rx={pointSize / 2}
          ry={pointSize / 2}
          fill={!!rColorMap ? rColorMap(datum.d) : color}
        />
        <g
          class="vline-point-label"
          transform="translate({alignTextLeft ? 8 : -8}, {datum.labelY})"
        >
          <rect
            x={alignTextLeft
              ? -4
              : -getTextWidth(valueText, '11pt sans-serif') - 4}
            y={-14}
            rx="3"
            ry="3"
            width={getTextWidth(valueText, '11pt sans-serif') + 8}
            height={20}
            class="label-background"
          />
          <text
            class="label"
            class:emphasized={hovering}
            x={0}
            y={0}
            text-anchor={alignTextLeft ? 'start' : 'end'}
            style="fill: {!!rColorMap ? rColorMap(datum.d) : color};"
            >{valueText}</text
          >
        </g>
      {/each}
    {/if}
  </g>
{/if}

<style>
  .label-background {
    fill: #f1f5f9;
    fill-opacity: 0.8;
  }

  .label {
    font-size: 11pt;
    font-weight: 400;
  }

  .title-label.emphasized {
    fill: #333 !important;
  }

  .line.emphasized {
    stroke: #333 !important;
  }
</style>
