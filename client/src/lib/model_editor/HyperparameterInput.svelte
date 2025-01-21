<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    export let type: string = 'fix';
    export let value: any = 0;
    export let history: any;

    const dispatch = createEventDispatcher();
    let options = ['fix', 'uniform', 'grid search', 'log uniform'];
    
    type = history?.type ?? 'uniform';
    // Variables for different input types
    let fixedValue = history?.value ?? "";
    let uniformLower = history?.value[0] ?? "";
    let uniformUpper = history?.value[1] ?? "";
    let logUniformLower = history?.value[0] ?? "";
    let logUniformUpper = history?.value[1] ?? "";
    let gridValues = history?.value ?? "";
    let gridError = '';
    let uniformError = '';
    let logUniformError = '';
    let fixedError = '';

    function handleTypeChange(e: Event) {
        const target = e.target as HTMLSelectElement;
        type = target.value;
        
        // Reset values when type changes
        if (type === 'fix') {
            value = fixedValue;
        } else if (type === 'uniform') {
            value = [uniformLower, uniformUpper];
        } else if (type === 'log uniform') {
            value = [logUniformLower, logUniformUpper];
        } else if (type === 'grid search') {
            value = gridValues;
        }
        
        dispatch('change', {type, value});
    }

    function handleValueChange() {
        try {
            if (type === 'fix') {
                value = handleFixedValue(fixedValue.toString());
            } else if (type === 'uniform') {
                value = handleUniformValue(uniformLower.toString(), uniformUpper.toString());
            } else if (type === 'log uniform') {
                value = handleLogUniformValue(logUniformLower.toString(), logUniformUpper.toString());
            } else if (type === 'grid search') {
                value = handleGridSearchValue(gridValues);
            }
            dispatch('change', {type, value});
        } catch (error) {
            console.error(error);
            // Could add error handling UI here
        }
    }

    // For fixed values, validate it's a non-zero number
    function handleFixedValue(val: string): number {
        try {
            const num = Number(val);
            if (isNaN(num)) {
                throw new Error('Fixed value must be a number');
            }
            if (num === 0) {
                throw new Error('Value cannot be zero');
            }
            fixedError = '';
            return num;
        } catch (error) {
            if (error instanceof Error) {
                fixedError = error.message;
            }
            throw error;
        }
    }

    // For uniform range, validate both non-zero bounds and return as array
    function handleUniformValue(lower: string, upper: string): [number, number] {
        try {
            const lowerNum = Number(lower);
            const upperNum = Number(upper);
            if (isNaN(lowerNum) || isNaN(upperNum)) {
                throw new Error('Bounds must be numbers');
            }
            if (lowerNum >= upperNum) {
                throw new Error('Lower bound must be less than upper bound');
            }
            uniformError = '';
            return [lowerNum, upperNum];
        } catch (error) {
            if (error instanceof Error) {
                uniformError = error.message;
            }
            throw error;
        }
    }

    // For log uniform, validate positive non-zero bounds and return as array
    function handleLogUniformValue(lower: string, upper: string): [number, number] {
        try {
            const [lowerNum, upperNum] = handleUniformValue(lower, upper);
            if (lowerNum <= 0 || upperNum <= 0) {
                throw new Error('Bounds must be positive for log uniform');
            }
            logUniformError = '';
            return [lowerNum, upperNum];
        } catch (error) {
            if (error instanceof Error) {
                logUniformError = error.message;
            }
            throw error;
        }
    }

    // For grid search, parse comma-separated non-zero values into array of numbers
    function handleGridSearchValue(val: string): number[] {
        try {
            const numbers = val.split(',')
                .map(v => v.trim())
                .map(v => {
                    const num = Number(v);
                    if (isNaN(num)) {
                        throw new Error('All values must be numbers');
                    }
                    return num;
                });
            gridError = '';
            return numbers;
        } catch (error) {
            if (error instanceof Error) {
                gridError = error.message;
            }
            gridValues = '';
            return [];
        }
    }

</script>

<div class="flex flex-row gap-2 items-center">
    <div class="flex flex-col gap-2">
      <div class="flex flex-row gap-2 items-center">
        <span class="text-sm">Value type: </span>
        <select bind:value={type} on:change={handleTypeChange} class="flat-select ml-auto">
            {#each options as option}
                <option value={option}>{option}</option>
            {/each}
        </select>
      </div>

      {#if type === 'fix'}
        <div class="text-slate-500 text-sm mb-2">
          Use this value:
        </div>
        <div class="flex flex-col gap-2">
          <input type="number" 
            bind:value={fixedValue}
            placeholder=" Enter fixed value" 
            class="border rounded-lg p-1 w-full {fixedError ? 'border-red-500' : ''}"
            on:change={handleValueChange}/>
          {#if fixedError}
            <span class="text-red-500 text-sm">{fixedError}</span>
          {/if}
        </div>
      {:else if type === 'uniform'}
        <div class="text-slate-500 text-sm mb-2 whitespace-pre-line">
          Find the best value between the lower 
          and upper bounds, sampling uniformly.
        </div>
        <div class="flex flex-row gap-2 items-center">
          <div class="flex flex-col gap-2">
            <div class="flex flex-row gap-2 items-center">
              <input type="number" 
                     bind:value={uniformLower} 
                     placeholder=" Lower bound" 
                     class="border rounded-lg p-1 w-1/2 {uniformError ? 'border-red-500' : ''}" 
                     on:change={handleValueChange}/>
              <input type="number" 
                     bind:value={uniformUpper} 
                     placeholder=" Upper bound" 
                     class="border rounded-lg p-1 w-1/2 {uniformError ? 'border-red-500' : ''}" 
                     on:change={handleValueChange}/>
            </div>
            {#if uniformError}
              <span class="text-red-500 text-sm">{uniformError}</span>
            {/if}
          </div>
        </div>
      {:else if type === 'log uniform'}
        <div class="text-slate-500 text-sm mb-2">
          Find the best value between the lower and upper bounds, sampling logarithmically.
        </div>
        <div class="flex flex-col gap-2">
          <div class="flex flex-row gap-2 items-center">
            <input type="number" 
                   bind:value={logUniformLower} 
                   placeholder=" Lower bound" 
                   class="border rounded-lg p-1 w-1/2 {logUniformError ? 'border-red-500' : ''}" 
                   on:change={handleValueChange}/>
            <input type="number" 
                   bind:value={logUniformUpper} 
                   placeholder=" Upper bound" 
                   class="border rounded-lg p-1 w-1/2 {logUniformError ? 'border-red-500' : ''}" 
                   on:change={handleValueChange}/>
          </div>
          {#if logUniformError}
            <span class="text-red-500 text-sm">{logUniformError}</span>
          {/if}
        </div>
      {:else if type === 'grid search'}
        <div class="text-slate-500 text-sm mb-2">
          Find the best value among the following values (enter numbers separated by commas).
        </div>
        <div class="flex flex-col gap-2">
          <input type="text" 
                 bind:value={gridValues} 
                 placeholder="Comma-separated values" 
                 class="border rounded-lg p-1 w-full {gridError ? 'border-red-500' : ''}" 
                 on:change={handleValueChange}/>
          {#if gridError}
            <span class="text-red-500 text-sm">{gridError}</span>
          {/if}
        </div>
      {/if}
    </div>
</div>