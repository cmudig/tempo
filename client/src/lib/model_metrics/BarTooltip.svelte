<script>
  import { getContext } from 'svelte';

  const { data, x, y, xScale, yScale, custom, width } = getContext('LayerCake');

  export let formatText = null;
  export let dx = null; // in plot coordinates
  export let dy = null; // in plot coordinates

  export let tooltipWidth = 240;

  let padding = 8;

  function formatTooltip(d) {
    if (!!formatText) return formatText(d);
    return Object.keys(d).map((k) => `<p><strong>${k}:</strong> ${d[k]}</p>`);
  }

  function getHovered(customVals, d) {
    let fn = (customVals || {}).hoveredGet || (() => false);
    return fn(d);
  }
</script>

{#each $data as d, i}
  {#if getHovered($custom, d)}
    <div
      class="tooltip"
      style="left: {Math.min(
        $width - tooltipWidth,
        Math.max(0, $xScale(dx != null ? $x(d) + dx : $x(d)))
      ) + padding}px; top: {$yScale(
        dy != null ? $y(d) - dy : $y(d)
      )}px; padding: {padding}px; {tooltipWidth != null
        ? 'width: ' + tooltipWidth + 'px;'
        : ''}"
    >
      {@html formatTooltip(d)}
    </div>
  {/if}
{/each}

<style>
  .tooltip {
    background-color: rgba(235, 235, 235, 0.9);
    border-radius: 4px;
    z-index: 100;
    color: #333;
    font-weight: 300;
    font-size: 10pt;
    position: absolute;
  }
</style>
