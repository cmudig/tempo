<script lang="ts">
  import {
    ModelArchitectureType,
    type HyperparameterSpec,
    type ModelArchitectureInfo,
  } from '../model';
  import HyperparameterInput from './HyperparameterInput.svelte';

  export let modelArchitecture: ModelArchitectureInfo | null = null;
  $: if (!!modelArchitecture && !modelArchitecture.num_samples)
    modelArchitecture = { ...modelArchitecture, num_samples: 10 };

  const DefaultHyperparameters: { [key: string]: HyperparameterSpec } = {
    num_epochs: { type: 'grid search', value: [5, 10, 20] },
    batch_size: { type: 'fix', value: 32 },
    lr: { type: 'log uniform', value: [1e-4, 1e-2] },
    num_layers: { type: 'grid search', value: [1, 2, 3] },
    hidden_dim: { type: 'grid search', value: [32, 64, 128] },
    num_heads: { type: 'grid search', value: [2, 4, 8] },
    dropout: { type: 'grid search', value: [0, 0.1, 0.2] },
    weight_decay: { type: 'log uniform', value: [1e-4, 0.1] },
  };

  const HyperparameterNames: { [key: string]: string } = {
    num_epochs: '# Training Epochs',
    batch_size: 'Batch Size',
    lr: 'Learning Rate',
    num_layers: 'Number of Layers',
    hidden_dim: 'Hidden Size',
    num_heads: '# Attention Heads',
    dropout: 'Dropout',
    weight_decay: 'Weight Decay',
  };

  function addHyperParameter(
    parameter: string,
    { type, value }: { type: string; value: any }
  ) {
    if (modelArchitecture) {
      modelArchitecture.hyperparameters = {
        ...modelArchitecture.hyperparameters,
        [parameter]: { type: type, value: value },
      };
    }
  }

  let oldArchitectureType: string | null = null;
  $: if (modelArchitecture && modelArchitecture.type !== oldArchitectureType) {
    oldArchitectureType = modelArchitecture.type;
    let newHyperparameters: { [key: string]: HyperparameterSpec } = {};
    let hyperparamsToUse: string[] = [];
    if (modelArchitecture.type == 'xgboost') hyperparamsToUse = [];
    else if (modelArchitecture.type == 'dense')
      hyperparamsToUse = [
        'num_epochs',
        'batch_size',
        'lr',
        'num_layers',
        'hidden_dim',
        'dropout',
        'weight_decay',
      ];
    else if (modelArchitecture.type == 'rnn')
      hyperparamsToUse = [
        'num_epochs',
        'batch_size',
        'lr',
        'num_layers',
        'hidden_dim',
        'dropout',
        'weight_decay',
      ];
    else if (modelArchitecture.type == 'transformer')
      hyperparamsToUse = [
        'num_epochs',
        'batch_size',
        'lr',
        'num_layers',
        'hidden_dim',
        'num_heads',
        'dropout',
        'weight_decay',
      ];
    hyperparamsToUse.forEach((paramName) => {
      newHyperparameters[paramName] =
        modelArchitecture!.hyperparameters[paramName] ??
        DefaultHyperparameters[paramName];
    });
    modelArchitecture = {
      ...modelArchitecture,
      hyperparameters: newHyperparameters,
    };
  }
</script>

{#if !!modelArchitecture}
  <div class="mb-1 flex flex-row gap-4 items-center">
    <span class="text-sm">Model type</span>
    <select bind:value={modelArchitecture.type} class="flat-select ml-2">
      {#each ['xgboost', 'dense', 'rnn', 'transformer'] as archType}
        <option value={archType}>{ModelArchitectureType[archType]}</option>
      {/each}
    </select>
  </div>

  {#if modelArchitecture.type != 'xgboost'}
    <h3 class="font-bold mt-3">Hyperparameters</h3>
    <div class="text-slate-500 text-xs mb-2">
      Specify which hyperparameters values should be searched during training.
    </div>

    <div class="mb-4 flex flex-row gap-4 items-center">
      <span class="text-sm">Number of Training Runs</span>
      <input
        type="number"
        class="flat-text-input w-24"
        min="1"
        max="50"
        bind:value={modelArchitecture.num_samples}
      />
      <span class="text-xs text-slate-500"
        >Sample the hyperparameter space and pick the model with the best
        validation loss.</span
      >
    </div>
    <div
      class="grid items-stretch w-full grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
    >
      {#each Object.entries(modelArchitecture.hyperparameters) as [hyperparamName, hyperparamValue] (hyperparamName)}
        <div class="bg-gray-100 rounded-lg p-4">
          <h4 class="font-bold">{HyperparameterNames[hyperparamName]}</h4>
          <HyperparameterInput
            history={hyperparamValue}
            on:change={(e) => {
              addHyperParameter(hyperparamName, e.detail);
            }}
          />
        </div>
      {/each}
    </div>
  {/if}
{/if}
