<script lang="ts">
    import type { ModelSummary } from "../lib/model";

    export let model: ModelSummary;
    export let isActive: boolean;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
    on:click
    class="grid grid-cols-4 gap-8 pt-6 pb-6 hover:bg-slate-100 drop-shadow-xl {isActive
        ? 'bg-sky-100'
        : ''} "
>
    <div class="col-span-2">
        <p class="text-slate-600 mb-2 text-[10px]">Predictive Target</p>
        <p class="text-[12px]">{model?.["target_name"]}</p>
    </div>
    <div>
        <p class="text-slate-600 mb-2 text-[10px]">Accuracy</p>
        <p class="text-[12px]">
            {Math.floor((model?.["metrics"]?.["roc_auc"] ?? 0) * 100).toFixed(
                1,
            ) + "%"}
        </p>
    </div>
    <div>
        <p class="text-slate-600 mb-2 text-[10px]">Count</p>
        <p class="text-[12px]">{model?.["n_patients"]}</p>
    </div>
</div>
