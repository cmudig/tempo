<!--
  @component
  Generates an SVG area shape using the `area` function from [d3-shape](https://github.com/d3/d3-shape).
 -->
<script>
  import { createEventDispatcher, getContext } from 'svelte';

  const dispatch = createEventDispatcher();

  const { data, xGet, yGet, r, width, height } = getContext('LayerCake');

  /** @type {String} [stroke='#ab00d6'] - The shape's fill color. This is technically optional because it comes with a default value but you'll likely want to replace it with your own color. */
  export let stroke = '#ab00d6';

  export let filterR = null;

  export let allowHover = false;
  export let allowSelect = false;

  $: path =
    'M' +
    $data
      .filter((d) => filterR == null || filterR == $r(d))
      .map((d) => {
        return $xGet(d) + ',' + $yGet(d);
      })
      .join('L');

  function closestPointTo(x, y) {
    let closestPoint = null;
    let closestDistance = 1e9;
    for (let d of $data) {
      let distance = Math.sqrt(
        Math.pow(x - $xGet(d), 2.0) + Math.pow(y - $yGet(d), 2.0)
      );
      if (distance < closestDistance) {
        closestPoint = d;
        closestDistance = distance;
      }
    }
    return closestPoint;
  }
</script>

<path class="path-line" d={path} {stroke} />
<rect
  x={0}
  y={0}
  width={$width}
  height={$height}
  fill="transparent"
  style="pointer-events: all;"
  on:pointermove={(e) => {
    if (!e.target || !allowHover) return;
    let rect = e.target.getBoundingClientRect();
    let mouseX = e.clientX - rect.left; //x position within the element.
    let mouseY = e.clientY - rect.top; //y position within the element.
    dispatch('hover', closestPointTo(mouseX, mouseY));
  }}
  on:mouseleave={() => {
    if (allowHover) dispatch('hover', null);
  }}
  on:click={(e) => {
    if (!e.target || !allowSelect) return;
    let rect = e.target.getBoundingClientRect();
    let mouseX = e.clientX - rect.left; //x position within the element.
    let mouseY = e.clientY - rect.top; //y position within the element.
    dispatch('click', closestPointTo(mouseX, mouseY));
  }}
/>

<style>
  .path-line {
    fill: none;
    stroke-linejoin: round;
    stroke-linecap: round;
    stroke-width: 2;
  }
</style>
