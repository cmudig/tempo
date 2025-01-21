<script lang="ts">
  import { ModelArchitectureType, type ModelArchitectureInfo } from '../model';
  import HyperparameterInput from './HyperparameterInput.svelte'

  export let modelArchitecture: ModelArchitectureInfo | null = null;
  $: console.log('architecture type1:', modelArchitecture);

  function addHyperParameter(parameter: string, {type, value}: {type: string, value: any}) {
    if (modelArchitecture) {
      modelArchitecture.hyperparameters = {
        ...modelArchitecture.hyperparameters,
        [parameter]: {type: type, value: value}
      };
    }
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

  <div class="flex flex-row flex-wrap gap-4">
      {#if modelArchitecture.type != 'xgboost'}
        <h3 class="font-bold mt-3">Hyperparameters</h3>
        <div class="text-slate-500 text-xs mb-2">
          Specify which hyperparameters values should be searched during training.
        </div>

        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[100px] h-[230px]">
          <h4 class="font-bold">Use Tuner?</h4>
          <label class="relative inline-flex items-center cursor-pointer mt-4">
            <input type="checkbox" class="sr-only peer" bind:checked={modelArchitecture.tuner}>
            <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
          </label>
        </div>

        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Number of Epochs</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.num_epochs}
            on:change={e => {
              addHyperParameter('num_epochs',e.detail)
            }} 
          />
        </div>
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Batch size</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.batch_size}
            on:change={e => {
              addHyperParameter('batch_size',e.detail)
            }} 
          />
        </div>
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Learning rate</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.lr}
            on:change={e => {
              addHyperParameter('lr',e.detail)
            }} 
          />
        </div>
    {/if}
    
    {#if modelArchitecture.type === 'rnn'}
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Number of layers</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.num_layers}
            on:change={e => {
              addHyperParameter('num_layers',e.detail)
            }} 
          />
        </div>
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Hidden size</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.hidden_size}
            on:change={e => {
              addHyperParameter('hidden_size',e.detail)
            }} 
          />
        </div>
    {/if}

    {#if modelArchitecture.type === 'transformer'}
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Number of heads</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.num_heads}
            on:change={e => {
              addHyperParameter('num_heads',e.detail)
            }} 
          />
        </div>
        <div class="bg-gray-100 rounded-lg p-4 shadow-sm mb-4 w-[300px] h-[230px]">
          <h4 class="font-bold">Hidden dimension</h4>
          <HyperparameterInput  
            history={modelArchitecture.hyperparameters.hidden_dim}
            on:change={e => {
              addHyperParameter('hidden_dim',e.detail)
            }} 
          />
        </div>
    {/if}

  </div>

{/if}
