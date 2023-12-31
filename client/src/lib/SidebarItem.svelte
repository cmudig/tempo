<script lang="ts">
  import type { ModelSummary } from './model';

  export let model: ModelSummary;
  export let modelName: string;
  export let isActive: boolean;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  on:click
  class="grid grid-cols-5 gap-8 pt-6 pb-6 cursor-pointer {isActive
    ? 'bg-blue-600 text-white hover:bg-blue-700'
    : 'hover:bg-slate-100'} "
>
  <div class="col-span-2">
    <p class="text-slate-600 mb-2 text-[10px]">Predictive Target</p>
    <p class="text-[12px]">{modelName}</p>
  </div>
  <div>
    <p class="text-slate-600 mb-2 text-[10px]">Accuracy</p>
    <p class="text-[12px]">
      {Math.floor((model?.['metrics']?.['roc_auc'] ?? 0) * 100).toFixed(1) +
        '%'}
    </p>
  </div>
  <div>
    <p class="text-slate-600 mb-2 text-[10px]">Count</p>
    <p class="text-[12px]">{(model.metrics || {}).n_val}</p>
  </div>
  {#if model.training && !!model.status}
    <div
      class="text-xs font-sans {isActive ? 'text-slate-50' : 'text-slate-500'}"
    >
      {model.status.state}
    </div>
  {/if}
</div>
