<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Fa from 'svelte-fa/src/fa.svelte';
  import { faXmark } from '@fortawesome/free-solid-svg-icons';

  const dispatch = createEventDispatcher();

  let scrollParent: HTMLElement;

  function scrollTo(id: string) {
    let offset = scrollParent.querySelector(`#${id}`)?.offsetTop ?? 0;
    scrollParent.scrollTo({ top: offset, behavior: 'smooth' });
  }
</script>

<div class="flex flex-col w-full h-full">
  <div class="w-full py-4 px-4 flex justify-between">
    <div class="font-bold">Tempo Query Language Reference</div>
    <button
      class="text-slate-600 px-2 hover:opacity-50"
      on:click={() => dispatch('close')}
      ><Fa icon={faXmark} class="inline" /></button
    >
  </div>
  <div class="w-full flex flex-auto min-h-0">
    <div
      class="w-1/5 px-3 h-full shrink-0"
      style="min-width: 280px; max-width: 360px;"
    >
      <div class="flex-auto min-h-0 overflow-auto mt-4">
        <button class="section-link" on:click={() => scrollTo('literals')}
          >Literals</button
        >
        <button class="section-link" on:click={() => scrollTo('data-fields')}
          >Data Fields</button
        >
        <button class="section-link" on:click={() => scrollTo('element-wise')}
          >Element-wise Operations</button
        >
        <button class="section-link" on:click={() => scrollTo('logical')}
          >Logical Operations</button
        >
        <button class="section-link" on:click={() => scrollTo('timestep-defs')}
          >Timestep Definitions</button
        >
        <button class="section-link" on:click={() => scrollTo('aggregations')}
          >Aggregations</button
        >
        <button
          class="section-link"
          on:click={() => scrollTo('transformations')}>Transformations</button
        >
        <button
          class="section-link"
          on:click={() => scrollTo('discretizations')}>Discretizations</button
        >
        <button class="section-link" on:click={() => scrollTo('variables')}
          >Inline Variables</button
        >
        <button class="section-link" on:click={() => scrollTo('constants')}
          >Constants</button
        >
        <button class="section-link" on:click={() => scrollTo('functions')}
          >Functions</button
        >
      </div>
    </div>
    <div
      class="flex-auto pb-4 overflow-y-auto h-full relative"
      bind:this={scrollParent}
    >
      <div class="w-full px-4 pb-4">
        <span id="literals" />
        <div class="reference-header">Literals</div>
        <div class="reference-element">
          <div class="sample">123, 45.67, inf, -inf</div>
          <div class="explanation">Numerical literals</div>
        </div>
        <div class="reference-element">
          <div class="sample">30 sec, 2 minutes, 1 h, 3 days</div>
          <div class="explanation">
            Duration literals. Standard abbreviations, full names, and plural
            versions of time units are accepted. (Time data is assumed to be
            input in units of seconds.)
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">"alpha", 'bravo'</div>
          <div class="explanation">String literals</div>
        </div>
        <div class="reference-element">
          <div class="sample">/AaBb/, /aabb/i</div>
          <div class="explanation">
            Regular expression pattern, optionally case-insensitive
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">[x, y], (x, y)</div>
          <div class="explanation">Value lists</div>
        </div>

        <span id="data-fields" />
        <div class="reference-header">Data Fields</div>
        <div class="reference-element">
          <div class="sample">&lbrace;field&rbrace;</div>
          <div class="explanation">
            Select a field from attributes, events, or intervals
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">&lbrace;field1, field2&rbrace;</div>
          <div class="explanation">
            Select multiple events or intervals of the same type as a single
            series
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">&lbrace;attr:field&rbrace;</div>
          <div class="explanation">Find a field from attributes only</div>
        </div>
        <div class="reference-element">
          <div class="sample">&lbrace;event:field&rbrace;</div>
          <div class="explanation">Find a field from events only</div>
        </div>
        <div class="reference-element">
          <div class="sample">&lbrace;interval:field&rbrace;</div>
          <div class="explanation">Find a field from intervals only</div>
        </div>

        <span id="element-wise" />
        <div class="reference-header">Element-Wise Operations</div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">+</span> expr2, expr1
            <span class="keyword">-</span>
            expr2, expr1 <span class="keyword">*</span> expr2, expr1
            <span class="keyword">/</span> expr2
          </div>
          <div class="explanation">
            Arithmetic element-wise addition, subtraction, multiplication,
            division
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">&gt;</span> expr2, expr1
            <span class="keyword">&gt;=</span>
            expr2, expr1 <span class="keyword">&lt;</span> expr2, expr1
            <span class="keyword">&lt;=</span>
            expr2
          </div>
          <div class="explanation">Numerical element-wise comparison</div>
        </div>
        <div class="reference-element">
          <div class="sample">expr1 <span class="keyword">=</span> expr2</div>
          <div class="explanation">Element-wise equality</div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">!=</span> expr2, expr1
            <span class="keyword">&lt;&gt;</span> expr2
          </div>
          <div class="explanation">Element-wise non-equality</div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">between</span> lower
            <span class="keyword">and</span> upper
          </div>
          <div class="explanation">
            Element-wise test if greater than or equal to lower, and less than
            upper
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">in</span> [x, y, ...]
          </div>
          <div class="explanation">
            Element-wise test if value is contained in the given list
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">not in</span> [x, y, ...]
          </div>
          <div class="explanation">
            Element-wise test if value is not contained in the given list
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">contains</span> "string", expr
            <span class="keyword">contains</span> /pattern/
          </div>
          <div class="explanation">
            Element-wise test if value contains the given string or regular
            expression pattern
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">startswith</span> "string", expr
            <span class="keyword">startswith</span> /pattern/
          </div>
          <div class="explanation">
            Element-wise test if value starts with the given string or regular
            expression pattern
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">endswith</span> "string", expr
            <span class="keyword">endswith</span> /pattern/
          </div>
          <div class="explanation">
            Element-wise test if value ends with the given string or regular
            expression pattern
          </div>
        </div>

        <span id="logical" />
        <div class="reference-header">Logical Operations</div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">and</span> expr2
          </div>
          <div class="explanation">Element-wise logical AND</div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr1 <span class="keyword">or</span> expr2
          </div>
          <div class="explanation">Element-wise logical OR</div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">not</span> expr
          </div>
          <div class="explanation">Element-wise negation</div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">case when</span> expr1
            <span class="keyword">then</span>
            value1 <span class="keyword">when</span> expr2
            <span class="keyword">then</span>
            value2 ... <span class="keyword">else</span> fallback
            <span class="keyword">end</span>
          </div>
          <div class="explanation">
            Element-wise case expression, evaluates each 'when' expression
            sequentially and stores the result. Values must be broadcastable to
            a consistent size.
          </div>
        </div>

        <span id="timestep-defs" />
        <div class="reference-header">Timestep Definitions</div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">every</span> duration
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end
          </div>
          <div class="explanation">
            Creates a time index that contains a row for each time from the
            given start time to the end time in each trajectory, separated by
            the duration. Use <span class="constant">#mintime</span> or
            <span class="constant">#maxtime</span> to denote the earliest and
            latest times in each trajectory, respectively. (If the
            <span class="font-mono"
              ><span class="keyword">from</span>
              start <span class="keyword">to</span> end</span
            >
            clause is omitted, the bounds are assumed to be
            <span class="constant">#mintime</span>
            and
            <span class="constant">#maxtime</span>.)
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">at every</span> &lbrace;field&rbrace;
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end
          </div>
          <div class="explanation">
            Creates a time index that contains a row for the time of each event
            from the given start time to the end time in each trajectory. Use <span
              class="constant">#mintime</span
            >
            or <span class="constant">#maxtime</span> to denote the earliest and
            latest times in each trajectory, respectively. (If the
            <span class="font-mono"
              ><span class="keyword">from</span>
              start <span class="keyword">to</span> end</span
            >
            clause is omitted, the bounds are assumed to be
            <span class="constant">#mintime</span>
            and
            <span class="constant">#maxtime</span>.)
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">at every</span>
            <span class="function">start</span>(&lbrace;field&rbrace;)
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end
          </div>
          <div class="explanation">
            Creates a time index that contains a row for the start time of each
            <strong>interval</strong> from the given start time to the end time
            in each trajectory. Use <span class="constant">#mintime</span>
            or <span class="constant">#maxtime</span> to denote the earliest and
            latest times in each trajectory, respectively.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">at every</span>
            <span class="function">end</span>(&lbrace;field&rbrace;)
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end
          </div>
          <div class="explanation">
            Creates a time index that contains a row for the end time of each
            <strong>interval</strong> from the given start time to the end time
            in each trajectory. Use <span class="constant">#mintime</span>
            or <span class="constant">#maxtime</span> to denote the earliest and
            latest times in each trajectory, respectively.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="keyword">at</span> [x, y, ...]
          </div>
          <div class="explanation">
            Creates a time index that contains a row for each trajectory for
            each element in the given value list. Use <span class="constant"
              >#mintime</span
            >
            or <span class="constant">#maxtime</span> to denote the earliest and
            latest times in each trajectory, respectively.
          </div>
        </div>

        <span id="aggregations" />
        <div class="reference-header">Aggregations</div>
        <div class="reference-element">
          <div class="sample">
            <span class="parameter">mean</span> expr
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end [timestep definition]
          </div>
          <div class="explanation">
            Aggregates the values of the given <strong>event expression</strong>
            between time bounds evaluated at every element of the time index. Use
            <span class="constant">#now</span> to denote the current time of the
            time index when expressing time bounds.
          </div>
          <div class="explanation">
            Supported aggregation functions:
            {@html [
              'sum',
              'mean',
              'median',
              'min',
              'max',
              'any',
              'all',
              'all nonnull',
              'first',
              'last',
              'exists',
              'exists nonnull',
              'count',
              'count distinct',
              'count nonnull',
            ]
              .map((fn) => `<span class="font-mono">${fn}</span>`)
              .join(', ')}.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="parameter">mean</span> expr
            <span class="keyword">before</span> end [timestep definition]
          </div>
          <div class="explanation">
            Performs an aggregation as above, using <span class="constant"
              >#mintime</span
            > as the lower bound.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="parameter">mean</span> expr
            <span class="keyword">after</span> start [timestep definition]
          </div>
          <div class="explanation">
            Performs an aggregation as above, using <span class="constant"
              >#maxtime</span
            > as the upper bound.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="parameter">mean value</span> expr
            <span class="keyword">from</span>
            start <span class="keyword">to</span> end [timestep definition]
          </div>
          <div class="explanation">
            Aggregates the values of the given <strong
              >interval expression</strong
            >
            between time bounds evaluated at every element of the time index. Use
            <span class="constant">#now</span> to denote the current time of the
            time index when expressing time bounds.
          </div>
          <div class="explanation">
            Supported aggregation functions:
            {@html [
              'sum',
              'mean',
              'median',
              'min',
              'max',
              'any',
              'all',
              'all nonnull',
              'first',
              'last',
              'exists',
              'exists nonnull',
              'count',
              'count distinct',
              'count nonnull',
              'integral',
            ]
              .map((fn) => `<span class="font-mono">${fn}</span>`)
              .join(', ')}.
          </div>
          <div class="explanation">
            The second keyword can be <span class="parameter">value</span>,
            <span class="parameter">amount</span>,
            <span class="parameter">rate</span>, or
            <span class="parameter">duration</span>. This specifies if the
            interval's value should be transformed based on how much it overlaps
            with the time bounds. For example, if the stored interval represents
            a total quantity of drug administered over the interval,
            <span class="parameter">sum amount</span> will calculate the total amount
            of that drug administered within the time bounds, assuming it was delivered
            at a constant rate.
          </div>
        </div>

        <span id="transformations" />
        <div class="reference-header">Transformations</div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">where</span> predicate
          </div>
          <div class="explanation">
            Replaces values in expr in which the predicate's value is false with
            a missing value.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">carry</span> 1 hour, expr
            <span class="keyword">carry</span> 4 steps
          </div>
          <div class="explanation">
            Carries forward non-missing values by the given duration or the
            given number of timesteps. Only modifies rows of expr that have
            missing values.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">impute</span>
            <span class="parameter">mean</span>
          </div>
          <div class="explanation">
            Replaces missing values with the given function of the non-missing
            values. Supported functions: <span class="parameter">mean</span>,
            <span class="parameter">median</span>
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">impute</span> value
          </div>
          <div class="explanation">
            Replaces missing values with the given constant value.
          </div>
        </div>

        <span id="discretizations" />
        <div class="reference-header">Discretizations</div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">cut</span> 4
            <span class="parameter">quantiles</span>, expr
            <span class="keyword">cut</span>
            3 <span class="parameter">quantiles</span>
            <span class="keyword">named</span> ["Low", "Medium", "High"]
          </div>
          <div class="explanation">
            Discretizes values into the specified number of equally-spaced
            quantiles. If names are provided, the number of names should be
            equal to the number of quantiles. If names are not provided, default
            interval names will be used.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">cut</span> 3
            <span class="parameter">bins</span>, expr
            <span class="keyword">cut</span>
            5 <span class="parameter">bins</span>
            <span class="keyword">named</span> ["a", "b", ...]
          </div>
          <div class="explanation">
            Discretizes values into the specified number of equally-spaced
            numerical bins. If names are provided, the number of names should be
            equal to the number of quantiles. If names are not provided, default
            interval names will be used.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">cut</span>
            <span class="parameter">bins</span> [-inf, 0, 1, ...]
          </div>
          <div class="explanation">
            Discretizes values into the bins with the specified cutoffs. All
            bins are inclusive on the lower bound and exclusive on the upper
            bound, such that if N bin cutoffs are provided, N - 1 bins will be
            created. Use <span class="font-mono">-inf</span>
            and <span class="font-mono">inf</span> to create open-ended bins on either
            side. Names can be provided as shown above, using a value list that contains
            N - 1 names.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">cut</span>
            <span class="parameter">quantiles</span> [0, 0.2, 0.8, 1]
          </div>
          <div class="explanation">
            Discretizes values into the bins with the specified quantile
            cutoffs. All bins are inclusive on the lower bound and exclusive on
            the upper bound, such that if N bin cutoffs are provided, N - 1 bins
            will be created. Names can be provided as shown above, using a value
            list that contains N - 1 names.
          </div>
        </div>

        <span id="variables" />
        <div class="reference-header">Inline Variables</div>
        <div class="reference-element">
          <div class="sample">
            expr <span class="keyword">with</span> varname
            <span class="keyword">as</span> expr2
          </div>
          <div class="explanation">
            Creates a variable named 'varname' that can be used inside expr.
          </div>
        </div>

        <span id="constants" />
        <div class="reference-header">Constants</div>
        <div class="reference-element">
          <div class="sample">
            <span class="constant">#mintime</span>
          </div>
          <div class="explanation">
            Stores the minimum time value for any event or interval for each
            trajectory.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="constant">#maxtime</span>
          </div>
          <div class="explanation">
            Stores the maximum time value for any event or interval for each
            trajectory.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="constant">#value</span>
          </div>
          <div class="explanation">
            Stores the value of an expression being operated on by a 'where'
            clause. Only valid within the condition of a 'where' clause.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="constant">#now</span>
          </div>
          <div class="explanation">
            Stores the current time being evaluated in a timestep definition.
            Only valid when a timestep definition is present.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="constant">#indexvalue</span>
          </div>
          <div class="explanation">
            Stores the current value of an event or interval selected within a
            timestep definition. Only valid for timesteps defined by an "at
            every" expression.
          </div>
        </div>

        <span id="functions" />
        <div class="reference-header">Functions</div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">time</span>(expr)
          </div>
          <div class="explanation">
            Takes an event series as input and returns a series containing the
            times of each event.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">start</span>(expr)
          </div>
          <div class="explanation">
            Takes an interval series as input and returns an event series
            representing the starts of each interval (interval values are
            preserved).
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">starttime</span>(expr)
          </div>
          <div class="explanation">
            Takes an interval series as input and returns an event series where
            the values are the start times of each interval.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">end</span>(expr)
          </div>
          <div class="explanation">
            Takes an interval series as input and returns an event series
            representing the ends of each interval (interval values are
            preserved).
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">endtime</span>(expr)
          </div>
          <div class="explanation">
            Takes an interval series as input and returns an event series where
            the values are the end times of each interval.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">abs</span>(expr)
          </div>
          <div class="explanation">
            Returns the element-wise absolute values of the given expression.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">max</span>(expr1, expr2)
          </div>
          <div class="explanation">
            Returns the element-wise maximum of the two given expressions.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">min</span>(expr1, expr2)
          </div>
          <div class="explanation">
            Returns the element-wise minimum of the two given expressions.
          </div>
        </div>
        <div class="reference-element">
          <div class="sample">
            <span class="function">extract</span>(expr, pattern),
            <span class="function">extract</span>(expr, pattern, index)
          </div>
          <div class="explanation">
            Returns a capture group from the given regular expression evaluated
            on each element in expr. If index is provided, returns the capture
            group with that index (starting from 0).
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .section-link {
    @apply w-full text-left mb-1 py-1 text-sm px-4 rounded text-slate-800;
  }
  .section-link:hover {
    @apply bg-slate-200;
  }
  .reference-header {
    @apply sticky top-0 bg-white z-10 font-bold py-4 mt-2 text-slate-600;
  }
  .reference-element {
    @apply mb-3;
  }
  .reference-element .sample {
    @apply font-mono mr-2;
  }
  .reference-element .explanation {
    @apply text-slate-500 text-sm mt-1;
  }
  .reference-element .detail {
    @apply text-slate-500 text-xs mt-1;
  }
  .keyword {
    @apply text-blue-700;
  }
  .parameter {
    @apply text-green-700 font-mono;
  }
  .function {
    @apply text-violet-700 font-mono;
  }
  .constant {
    @apply text-pink-700 font-mono;
  }
</style>
