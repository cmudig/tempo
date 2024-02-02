<script lang="ts">
  import type { ModelSummary } from "../lib/model";

  export let model: ModelSummary;

  function getMetricItem(key: keyof ModelSummary["metrics"]) {
    return model?.["metrics"]?.[key] ?? 0;
  }
</script>

<div class="mt-2">
  <div class="grid grid-cols-5 gap-8 pt-6 pb-6">
    <div>
      <p>Accuracy</p>
      <p>
        {#key model}
          {Math.floor(getMetricItem("roc_auc") * 100).toFixed(1) + "%"}
        {/key}
      </p>
    </div>
    <div>
      <p>Sensitivity</p>
      <p>
        {#key model}
          {Math.floor(getMetricItem("sensitivity") * 100).toFixed(1) + "%"}
        {/key}
      </p>
    </div>
    <div>
      <p>Specificity</p>
      <p>
        {#key model}
          {Math.floor(getMetricItem("specificity") * 100).toFixed(1) + "%"}
        {/key}
      </p>
    </div>
    <div>
      <p>ROC</p>
    </div>
    <div>
      <p>Count</p>
      <p>{model?.["n_patients"]}</p>
    </div>
  </div>
</div>
