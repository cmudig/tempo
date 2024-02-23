import lark
import re
import csv
import datetime
from query_language.data_types import *
import json
import os
import random
import tqdm
import slice_finding as sf

GRAMMAR = """
start: variable_expr | variable_list

time_index: "EVERY"i atom time_bounds            -> periodic_time_index // periodic time literal
    | "AT EVERY"i atom time_bounds  -> event_time_index
    | "AT"i "(" expr ("," expr)* ")"             -> array_time_index

time_bounds: "FROM"i expr "TO"i expr

variable_list: variable_expr
    | "(" variable_expr ("," variable_expr)* ")"
variable_expr: [named_variable] expr
named_variable: (/[A-Za-z][^:]*/i | VAR_NAME) ":"

// Expression Parsing
 
case_when: "WHEN"i expr "THEN"i expr

agg_method: VAR_NAME AGG_TYPE?
AGG_TYPE: "rate"i|"amount"i|"value"i|"duration"i

?expr: variable_list time_index             -> time_series
    | expr "WHERE"i expr                    -> where_clause
    | expr "CARRY"i (time_quantity | step_quantity)  -> carry_clause
    | expr "IMPUTE"i (VAR_NAME | LITERAL)            -> impute_clause
    | expr "WITH"i VAR_NAME "AS"i logical   -> with_clause
    | logical
    
?logical: logical "AND"i negation                -> logical_and
    | logical "OR"i negation                  -> logical_or
    | negation

?negation: "NOT"i negation                        -> negate
    | comparison

?comparison: comparison ">=" agg_expr                   -> geq
    | comparison "<=" agg_expr                   -> leq
    | comparison ">" agg_expr                    -> gt
    | comparison "<" agg_expr                    -> lt
    | comparison "=" agg_expr                    -> eq
    | comparison ("!="|"<>") agg_expr            -> ne
    | comparison "IN"i value_list            -> isin
    | agg_expr

?agg_expr: agg_method expr time_bounds
    | sum

?sum: sum "+" product                       -> expr_add
    | sum "-" product                       -> expr_sub
    | product

?product: product "*" atom                      -> expr_mul
    | product "/" atom                      -> expr_div
    | atom

value_list: ("("|"[") LITERAL ("," LITERAL)* (")"|"]")

atom: VAR_NAME "(" expr ("," expr)* ")"                 -> function_call
    | DATA_NAME                             -> data_element
    | time_quantity
    | LITERAL                               -> literal
    | "#NOW"i                                -> now 
    | "#VALUE"i                              -> where_value
    | "#MINTIME"i                              -> min_time
    | "#MAXTIME"i                              -> max_time
    | "#INDEXVALUE"i                            -> index_value
    | "CASE"i (case_when)+ "ELSE"i expr "END"i -> case_expr     // if/else
    | "(" expr ")"
    | VAR_NAME                               -> var_name

time_quantity: LITERAL UNIT
step_quantity: LITERAL /steps?/i
UNIT: /hours?|minutes?|seconds?|hrs?|mins?|secs?|[hms]/i

DATA_NAME: /\{[^}]*\}/
VAR_NAME: /(?!(and|or|not|case|when|else|in|then|every|at|from|to|with|as)\b)[A-Za-z][A-Za-z0-9_]*/ 

LITERAL: SIGNED_NUMBER | QUOTED_STRING
QUOTED_STRING: /["'`][^"'`]*["'`]/


%import common (WORD, WS, SIGNED_NUMBER, LETTER)

%ignore WS
"""

def get_all_trajectory_ids(attributes, events, intervals):
    all_ids = []
    if len(attributes.get_ids()):
        all_ids.append(attributes.get_ids().values)
    if len(events.get_ids()):
        all_ids.append(events.get_ids().unique())
    elif len(intervals.get_ids()):
        all_ids.append(intervals.get_ids().unique())
    return np.unique(np.concatenate(all_ids))

class EvaluateExpression(lark.visitors.Transformer):
    def __init__(self, attributes, events, intervals, eventtype_macros=None):
        super().__init__()
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.time_index = None
        self.eventtype_macros = eventtype_macros if eventtype_macros is not None else {}
        self.value_placeholder = None
        self.index_value_placeholder = None
        self.variables = {}
        self._all_ids = None
        
    def get_all_ids(self):
        if self._all_ids is not None: return self._all_ids
        self._all_ids = get_all_trajectory_ids(self.attributes, self.events, self.intervals)
        return self._all_ids
        
    def _get_data_element(self, query):
        comps = query.split(":")
        el_name = comps[-1]
        # substitute with macro if available
        if el_name in self.eventtype_macros:
            el_name = self.eventtype_macros[el_name].strip()
        if "," in el_name:
            el_name = list(csv.reader([el_name], skipinitialspace=True))[0]
            # Substitute macros again
            el_name = [x.strip() for el in el_name for x in self.eventtype_macros.get(el, el).split(",")]
        if len(comps) > 1:
            scope = comps[0].lower()
            # Only search within the given scope
            if scope == "attr":
                if isinstance(el_name, list): raise ValueError(f"Cannot jointly retrieve multiple data elements from Attributes")
                return self.attributes.get(el_name)
            else:   
                if scope == "event":
                    return self.events.get(el_name)
                elif scope == "interval":
                    return self.intervals.get(el_name)
                else:
                    raise ValueError(f"Unknown data element scope {scope}")
                
        # Search for the element in attributes, events, intervals
        candidates = (
            [self.attributes.get(el_name)] if not isinstance(el_name, list) and self.attributes.has(el_name) else [] + 
            [self.events.get(el_name), self.intervals.get(el_name)]
        )
        candidates = [c for c in candidates if len(c) > 0]
        if len(candidates) > 1:
            raise ValueError(f"Multiple data elements found with name {el_name}. Try specifying a scope such as \{{attr:{el_name}\}} (or event: or interval:).")
        elif len(candidates) == 0:
            raise KeyError(f"No data element found with name {el_name}")
        return candidates[0]
        
    def data_element(self, args):
        match = re.match(r"\{([^\}]+)\}", args[0], flags=re.I)
        query = match.group(1)
        return self._get_data_element(query)
        
    def var_name(self, args):
        if args[0] in self.variables:
            return self.variables[args[0]]
        raise KeyError(f"No variable named {args[0]}")
        
    def time_quantity(self, args):
        return Duration(self._parse_literal(args[0]), args[1])
        
    def time_bounds(self, args):
        # Broadcast and convert to TimeIndex if needed
        start, end = args
        if isinstance(start, TimeIndex) and not isinstance(end, TimeIndex):
            end = TimeIndex(pd.DataFrame({
                start.id_field: start.get_ids(), 
                start.time_field: make_aligned_value_series(start, end)
            }), id_field=start.id_field, time_field=start.time_field)
        elif isinstance(end, TimeIndex) and not isinstance(start, TimeIndex):
            start = TimeIndex(pd.DataFrame({
                end.id_field: end.get_ids(), 
                end.time_field: make_aligned_value_series(end, start)
            }), id_field=end.id_field, time_field=end.time_field)
        elif isinstance(start, Attributes) and isinstance(end, Attributes):
            start = TimeIndex.from_attributes(start)
            end = TimeIndex.from_attributes(end)
        elif isinstance(start, Attributes) and not isinstance(end, Attributes):
            start = TimeIndex.from_attributes(start)
            end = start.with_times(end)
        elif isinstance(end, Attributes) and not isinstance(start, Attributes):
            end = TimeIndex.from_attributes(end)
            start = end.with_times(start)
        
        return (start, end)
    
    def _parse_literal(self, literal):
        if re.search(r"[\"'`]", literal) is not None:
            return re.sub(r"[\"'`]", "", literal)
        try:
            amt = float(literal)
            if round(amt) == amt:
                amt = int(amt)
            return amt
        except ValueError:
            raise ValueError("Literal must be either a number or quote-wrapped string")
            
    def literal(self, args): return self._parse_literal(args[0])

    def now(self, args): 
        if self.time_index is None:
            raise ValueError(f"'now' keyword can only be used within a time-series expression, ending with an 'at'/'every'/'at every' clause.")
        return self.time_index
    def where_value(self, args):
        if self.value_placeholder is None:
            raise ValueError(f"'value' keyword can only be used within a where clause to refer to the data being filtered.")
        return self.value_placeholder
    def index_value(self, args):
        if self.index_value_placeholder is None:
            raise ValueError(f"'indexvalue' keyword can only be used within a time series defined with 'at every' event or interval.")
        return self.index_value_placeholder
    def atom(self, args): return args[0]
    
    def min_time(self, args):
        min_t = min(self.events.get_times().min(), self.intervals.get_start_times().min())
        ids = self.get_all_ids()
        return Attributes(pd.Series(np.ones(len(ids)) * min_t, index=ids, name='mintime'))
    def max_time(self, args): 
        max_t = max(self.events.get_times().max(), self.intervals.get_end_times().max())
        # Offset the time by 1 so that it includes all events and intervals
        if isinstance(max_t, (datetime.datetime, np.datetime64)):
            max_t += datetime.timedelta(seconds=1)
        else:
            max_t += 1
        ids = self.get_all_ids()
        return Attributes(pd.Series(np.ones(len(ids)) * max_t, index=ids, name='maxtime'))

    def isin(self, args):
        return args[0].isin(args[1])
    
    def value_list(self, args): return [self._parse_literal(v) for v in args]
        
    def expr_add(self, args): return args[0] + args[1]
    def expr_sub(self, args): 
        print("SUB", args, args[0] - args[1])
        return args[0] - args[1]
    def expr_mul(self, args): return args[0] * args[1]
    def expr_div(self, args): return args[0] / args[1]
    def gt(self, args): return args[0] > args[1]
    def lt(self, args): return args[0] < args[1]
    def geq(self, args): return args[0] >= args[1]
    def leq(self, args): return args[0] <= args[1]
    def eq(self, args): return args[0] == args[1]
    def ne(self, args): return args[0] != args[1]
    
    def negate(self, args): return ~args[0]
    
    def logical_and(self, args): return args[0] & args[1]
    def logical_or(self, args): return args[0] | args[1]

    def agg_expr(self, args):
        agg_method = args[0]
        expr = args[1]
        time_bounds = args[-1]
        
        if self.time_index is not None:
            assert len(time_bounds[0]) == len(self.time_index), f"Start time bounds for aggregation (length {len(time_bounds[0])}) must be equal length to overall time index (length {len(self.time_index)})"
            assert len(time_bounds[1]) == len(self.time_index), f"End time bounds for aggregation (length {len(time_bounds[1])}) must be equal length to overall time index (length {len(self.time_index)})"
            if isinstance(expr, TimeSeries):
                # Convert the expression to an Events
                expr = expr.to_events()
            if isinstance(expr, Events):
                return expr.bin_aggregate(self.time_index, *time_bounds, agg_method[0])
            elif isinstance(expr, (Intervals, Compilable)):
                result = expr.bin_aggregate(self.time_index, *time_bounds, agg_method[1], agg_method[0])
                return result
            else:
                raise ValueError(f"Only Events and Intervals can be bin-aggregated")
        else:
            if isinstance(expr, (Events, TimeSeries)):
                return expr.aggregate(*time_bounds, agg_method[0])
            elif isinstance(expr, Intervals):
                result = expr.aggregate(*time_bounds, agg_method[1], agg_method[0])
                return result
            else:
                raise ValueError(f"Only Events and Intervals can be aggregated")            
        
    def agg_method(self, args):
        if len(args) > 1:
            return (args[0].value, args[1].value)
        return (args[0].value, "value")
        
    def case_expr(self, args):
        whens = args[:-1]
        else_clause = args[-1]
        
        if (any(isinstance(clause, Compilable) for when in args[:-1] for clause in when.children) or 
            isinstance(else_clause, Compilable)):
            # The entire case expression needs to be a Compilable
            whens = [tuple(Compilable(c) if not isinstance(c, Compilable) else c
                           for c in when.children) for when in whens]
            if not isinstance(else_clause, Compilable):
                else_clause = Compilable(else_clause)
            result = else_clause
            for condition, value in reversed(whens):
                result = value.where(condition, result)
            return result
        
        result = else_clause
        if isinstance(result, Duration): result = result.value()
        
        for when in reversed(whens):
            condition, value = when.children
            if isinstance(value, Duration): value = value.value()
            if isinstance(value, (Events, Attributes, Intervals, TimeSeries)):
                # Need to broadcast if one element is an Attributes
                if isinstance(value, Attributes) and isinstance(condition, (Events, Intervals, TimeSeries)):
                    value = condition.with_values(make_aligned_value_series(condition, value))
                elif isinstance(condition, Attributes) and isinstance(value, (Events, Intervals, TimeSeries)):
                    condition = make_aligned_value_series(value, condition)
                    
                if len(value.get_values()) != len(condition.get_values()):
                    raise ValueError(f"Case expression operands must be same length")
                result = value.where(condition.fillna(False).astype(bool), result)
                result = result.where(~condition.isna(), pd.NA)
            elif isinstance(result, (Events, Attributes, Intervals, TimeSeries)):
                if len(result.get_values()) != len(condition.get_values()):
                    raise ValueError(f"Case expression operands must be same length")
                result = result.where(~condition.fillna(False).astype(bool), value)
            elif isinstance(condition, (Attributes, Events, Intervals, TimeSeries)):
                # We need to broadcast both value and result to condition's type
                result = condition.apply(lambda x: pd.NA if pd.isna(x) else (value if x else result))
                
        return result
        
    def carry_clause(self, args):
        # Defines how far the values in the time series should be
        # carried forward within a given ID
        var_exp = args[0]
        if isinstance(args[0], Compilable): raise NotImplementedError("Carry forward not yet implemented for nested aggregations")
        if isinstance(args[1], lark.Tree) and args[1].data == "step_quantity":
            steps = int(args[1].children[0].value)
            return var_exp.carry_forward_steps(steps)
        else:
            return var_exp.carry_forward_duration(args[1])
            
    @lark.v_args(tree=True)
    def step_quantity(self, tree):
        return tree
    
    def impute_clause(self, args):
        # Defines how NaN values should be substituted
        var_exp = args[0]
        if isinstance(var_exp, Compilable):
            method = "constant"
            if args[1].value in ("mean", "median"):
                method = args[1].value
                constant_value = None
            else:
                constant_value = self._parse_literal(args[1].value)
            return var_exp.impute(method=method, constant_value=constant_value)
        
        nan_mask = ~pd.isna(var_exp.get_values())
        if args[1].value in ("mean", "median"):
            impute_method = args[1].value.lower()
            numpy_func = {"mean": np.nanmean, "median": np.nanmedian}[impute_method]
            return var_exp.where(nan_mask, numpy_func(var_exp.get_values().replace(pd.NA, np.nan).astype(float)))
        else:
            impute_method = self._parse_literal(args[1].value)
            return var_exp.where(nan_mask, impute_method)
            
    def function_call(self, args):
        function_name = args[0].value.lower()
        operands = args[1:]
        if function_name in ("time", "start", "end"):
            if len(operands) != 1: raise ValueError(f"time function requires exactly one operand")
            if function_name == "time":
                if not hasattr(operands[0], "get_times"):
                    raise ValueError("time function requires an object with a single time value per datum")
                return operands[0].with_values(operands[0].get_times())
            elif function_name == "start":
                if not hasattr(operands[0], "get_start_times"):
                    raise ValueError("start function requires interval objects")
                return operands[0].start_events().with_values(operands[0].get_start_times())
            elif function_name == "end":
                if not hasattr(operands[0], "get_end_times"):
                    raise ValueError("end function requires interval objects")
                return operands[0].end_events().with_values(operands[0].get_end_times())
        elif function_name in ("abs", ):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            return getattr(operands[0], function_name)()
        elif function_name in ("max", "min"):
            if len(operands) != 2: raise ValueError(f"{function_name} function requires exactly two operands")
            numpy_func = np.maximum if function_name == "max" else np.minimum
            if isinstance(operands[0], TimeIndex):
                return operands[0].with_times(numpy_func(operands[0].get_times(), make_aligned_value_series(operands[0], operands[1])))
            elif isinstance(operands[1], TimeIndex):
                return operands[1].with_times(numpy_func(operands[1].get_times(), make_aligned_value_series(operands[1], operands[0])))
            if isinstance(operands[0], (Attributes, Events, Intervals, TimeSeries)):
                return operands[0].with_values(numpy_func(operands[0].get_values(), make_aligned_value_series(operands[0], operands[1])))
            elif isinstance(operands[1], (Attributes, Events, Intervals, TimeSeries)):
                return operands[1].with_values(numpy_func(operands[1].get_values(), make_aligned_value_series(operands[1], operands[0])))
            else:
                raise ValueError(f"{function_name} function requires at least one parameter to be Attributes, Events, Intervals, TimeIndex, or TimeSeries")
        else:
            raise ValueError(f"Unknown function '{function_name}'")

    def variable_list(self, args):
        if len(args) == 1: return args[0]
        if all(isinstance(a, Attributes) for a in args):
            return AttributeSet(pd.concat([a.series for a in args], axis=1))
        elif all(isinstance(a, TimeSeries) for a in args):
            return TimeSeriesSet.from_series(args)
        raise ValueError("Variable list must contain either all Attributes or all TimeSeries objects")

class EvaluateQuery(lark.visitors.Interpreter):
    def __init__(self, attributes, events, intervals, variable_transform=None, eventtype_macros=None, cache=None, verbose=False):
        super().__init__()
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.cache = cache
        self.eventtype_macros = eventtype_macros if eventtype_macros is not None else {}
        # If provided, this should be a tuple of (description, transform). The
        # description should be a string uniquely identifying this transform,
        # and transform should be a function that will be called on any variable 
        # expressions before returning or saving to cache. The function should
        # take as input a TimeSeriesQueryable, and it should return a transformed 
        # instance of the input.
        if variable_transform is not None:
            self.variable_transform_desc, self.variable_transform = variable_transform
        else:
            self.variable_transform_desc = None
            self.variable_transform = None
        self.verbose = verbose
        self.evaluator = EvaluateExpression(self.attributes, self.events, self.intervals, self.eventtype_macros)
        
    def get_all_ids(self):
        return self.evaluator.get_all_ids()
    
    def _make_time_index(self, idx):
        if isinstance(idx, Attributes):
            return TimeIndex.from_attributes(idx)
        elif isinstance(idx, TimeIndex):
            return idx
        else:
            raise ValueError(f"Cannot convert {type(idx)} object to TimeIndex")

    def periodic_time_index(self, tree):
        duration = self.evaluator.transform(tree.children[0])
                
        start_time = self._make_time_index(self.evaluator.transform(tree.children[1].children[0]))
        end_time = self._make_time_index(self.evaluator.transform(tree.children[1].children[1]))
        
        return TimeIndex.range(start_time, end_time, duration)
        
    def event_time_index(self, tree):
        events = self.evaluator.transform(tree.children[0])

        start_time = self.evaluator.transform(tree.children[-1].children[0])
        end_time = self.evaluator.transform(tree.children[-1].children[1])
        if isinstance(start_time, Attributes) and isinstance(end_time, Attributes):
            pass
        elif isinstance(start_time, Attributes) and not isinstance(end_time, Attributes):
            end_time = start_time.with_values(end_time)
        elif not isinstance(start_time, Attributes) and isinstance(end_time, Attributes):
            start_time = end_time.with_values(start_time)
        elif isinstance(start_time, (float, int)) and not isinstance(end_time, (float, int)):
            ids = np.unique(events.get_ids())
            start_time = Attributes(pd.Series(np.ones(len(ids)) * start_time, index=ids))
            end_time = Attributes(pd.Series(np.ones(len(ids)) * end_time, index=ids))
        else:
            raise ValueError(f"Unsupported time types for event index: '{type(start_time)}' and '{type(end_time)}'")
        
        if len(tree.children) > 2:
            assert isinstance(events, Intervals), "Interval position may only be used in event index when the data element is an interval"
            if tree.children[1].value.lower() == "start":
                events = events.start_events()
            elif tree.children[1].value.lower() == "end":
                events = events.end_events()
            else:
                raise ValueError(f"Unrecognized interval position '{tree.children[1].value}'")
        
        if not isinstance(events, Events):
            raise ValueError(f"Expected 'at every' data element to evaluate to an Events object, but instead got '{type(events)}'")
        index, filtered_events = TimeIndex.from_events(events, starts=start_time, ends=end_time, return_filtered_events=True)
        self.evaluator.index_value_placeholder = TimeSeries(index, filtered_events.get_values())
        return index
        
    def array_time_index(self, tree):
        times = [self.evaluator.transform(c) for c in tree.children]
        return TimeIndex.from_times(times)
        
    def _parse_variable_expr(self, tree, evaluator, time_index_tree=None, cache_only=False):            
        # Parse where clauses first (these require top-down processing in case of a value placeholder)
        tree_desc = str(tree.children[1])
        options_desc = str(tree.children[2]) if len(tree.children) > 2 else ''
        
        var_name = tree.children[0].children[0].value if tree.children[0] and tree.children[0].children[0].value else None
        if isinstance(tree.children[1], (TimeSeries, TimeSeriesSet)):
            var_exp = tree.children[1]
        elif self.cache is not None:
            var_exp = self.cache.lookup((tree_desc, options_desc), time_index_tree=time_index_tree, transform_info=self.variable_transform_desc)
        else: var_exp = None
        if cache_only and var_exp is None: return tree
        elif var_exp is not None:
            if var_name is not None:
                var_exp = var_exp.rename(var_name)
            return var_exp.compress()

        self._preprocess_nested_aggregations(tree, evaluator)
            
        set_variables = set()
        for node in tree.iter_subtrees():
            if node is None: continue
            new_children = []
            for n in node.children:
                if isinstance(n, lark.Tree) and n.data == "with_clause":
                    # Defining a temporary variable
                    base_expr, with_var_name = self._parse_with_clause(n, evaluator)
                    set_variables.add(with_var_name)
                    new_children.append(base_expr)
                else:
                    new_children.append(n)
            node.children = new_children
        
        for node in tree.iter_subtrees():
            if node is None: continue
            node.children = [lark.Tree('atom', [self._parse_where_clause(n, evaluator)]) if isinstance(n, lark.Tree) and n.data == "where_clause" else n for n in node.children]
            
        try:
            # We only cache the main expression, so variable names and options can be adjusted later without recomputing
            # expensive aggregations
            if var_exp is None:
                var_exp = evaluator.transform(tree.children[1])
                print("After parsng:", var_exp)
                if isinstance(var_exp, Compilable): var_exp = var_exp.execute()
                if evaluator.time_index is not None:
                    if isinstance(var_exp, Attributes):
                        # Cast the attributes over the time index
                        var_exp = TimeSeries(evaluator.time_index, make_aligned_value_series(evaluator.time_index, var_exp))
                    elif isinstance(var_exp, TimeIndex):
                        # Use the times as the time series values
                        print("Converting index to series", var_exp, var_exp.get_times())
                        var_exp = TimeSeries(var_exp, var_exp.get_times())
                        
            if var_name is not None:
                var_exp = var_exp.rename(var_name)
            
        except Exception as e:
            raise ValueError(f"Exception occurred when processing variable '{var_name}': {e}")
        else:
            if self.variable_transform is not None:
                var_exp = self.variable_transform(var_exp)
                # var_exp = var_exp.with_values(pd.Series(sf.discretization.discretize_column(var_exp.name, var_exp.get_values(), self.discretization_spec)[0], 
                #                                         name=var_exp.name))
            var_exp = var_exp.compress()
            
            if self.cache is not None:
                self.cache.save((tree_desc, options_desc), var_exp, transform_info=self.variable_transform_desc, time_index_tree=time_index_tree)
            return var_exp
        
    def _parse_time_series(self, tree, evaluator):
        time_index_tree = tree.children[-1] if len(tree.children) > 1 else None
        time_index = self.visit(tree.children[-1]) if len(tree.children) > 1 else None
        evaluator.time_index = time_index
        pbar = tqdm.tqdm(tree.children[0].children) if self.verbose else tree.children[0].children
        variable_definitions = [self._parse_variable_expr(child, evaluator, time_index_tree=time_index_tree) for child in pbar]
        evaluator.time_index = None
        evaluator.index_value_placeholder = None

        if time_index is not None and not all(isinstance(v, TimeSeries) for v in variable_definitions):
            raise ValueError(f"All variables must evaluate to a TimeSeries when a time index is provided")
        if len(variable_definitions) == 1:
            return variable_definitions[0]
        else:
            return TimeSeriesSet.from_series(variable_definitions)
        
    def _preprocess_nested_aggregations(self, tree, evaluator):
        for node in tree.iter_subtrees_topdown():
            if node is None: continue
            if not (isinstance(node, lark.Tree) and node.data == "agg_expr"): continue
            # For aggregation expressions, convert all inner aggregation expressions
            # to Compilables! This will enable us to calculate dynamic values
            # during the aggregation.
            
            for desc in node.iter_subtrees():
                if desc is None or desc == node: continue
                desc.children = [Compilable(evaluator.transform(n)) if isinstance(n, lark.Tree) and n.data == "agg_expr" else n for n in desc.children]
                
        
    def _parse_where_clause(self, tree, evaluator):
        base = evaluator.transform(tree.children[0])
        evaluator.value_placeholder = base
        where = evaluator.transform(tree.children[1])
        evaluator.value_placeholder = None
        if isinstance(where, Compilable):
            return Compilable(base).filter(where)
        elif isinstance(base, (Events, Intervals, EventSet, IntervalSet, Compilable)):
            return base.filter(where)
        else:
            return base.where(where, pd.NA)
        
    def _parse_with_clause(self, tree, evaluator):
        var_name = tree.children[1].value
        var_value = evaluator.transform(tree.children[-1])
        if isinstance(var_value, Compilable): var_value = var_value.execute()
        print("variable", var_name, var_value)
        evaluator.variables[var_name] = var_value
        return tree.children[0], var_name
        
    def start(self, tree):
        # First replace all time series
        if isinstance(tree.children[0], lark.Tree) and tree.children[0].data == "time_series":
            return self._parse_time_series(tree.children[0], self.evaluator)

        if self.use_cache:
            # First parse cached expressions
            for node in tree.iter_subtrees():
                if node is None: continue
                node.children = [self._parse_variable_expr(n, self.evaluator, cache_only=True) if isinstance(n, lark.Tree) and n.data == "variable_expr" else n for n in node.children]
        
        # Parse time series first
        for node in tree.iter_subtrees():
            if node is None: continue
            node.children = [self._parse_time_series(n, self.evaluator) if isinstance(n, lark.Tree) and n.data == "time_series" else n for n in node.children]

        # Then parse detached variable expressions
        for node in tree.iter_subtrees():
            if node is None: continue
            node.children = [self._parse_variable_expr(n, self.evaluator) if isinstance(n, lark.Tree) and n.data == "variable_expr" else n for n in node.children]
            
        if isinstance(tree.children[0], lark.Tree): 
            return self.evaluator.transform(tree.children[0])
        return tree.children[0]
    
class QueryResultCache:
    """
    Saves query results (optionally pre-discretized) to a cache directory. Query
    results can also be saved to an in-memory cache if desired.
    """
    def __init__(self, cache_dir):
        super().__init__()
        assert cache_dir is not None, "Cache directory must be provided"
        self.cache_dir = cache_dir
        self._query_cache = {}
        self._in_memory_cache = {} # for items to be stored in memory instead of loaded from disk every time
    
    def load_cache(self):
        if not self.cache_dir: return
        if not os.path.exists(self.cache_dir): os.mkdir(self.cache_dir)
        
        # Load cache information
        if os.path.exists(os.path.join(self.cache_dir, "query_cache.json")):
            with open(os.path.join(self.cache_dir, "query_cache.json"), "r") as file:
                self._query_cache = json.load(file)
        else:
            self._query_cache = {}
            
    def _make_cache_key(self, tree, time_index_tree, transform_info):
        return (str(tree) + 
                ("_" + str(time_index_tree) if time_index_tree is not None else "") + 
                ("_" + str(transform_info) if transform_info is not None else ""))
    
    def lookup(self, tree, time_index_tree=None, transform_info=None, save_in_memory=False):
        """Returns the result of a variable parse if it exists in the cache."""
        self.load_cache()
        query_cache_key = self._make_cache_key(tree, time_index_tree, transform_info)
        
        if query_cache_key in self._in_memory_cache:
            return self._in_memory_cache[query_cache_key]
        elif query_cache_key in self._query_cache:
            result_info = self._query_cache[query_cache_key]
            if "time_index_tree" in result_info:
                index = self.lookup("time_index_" + result_info["time_index_tree"], time_index_tree=None, save_in_memory=True)
            else:
                index = None
            fpath = os.path.join(self.cache_dir, result_info["fname"])
            if not os.path.exists(fpath): return
            df = pd.read_feather(fpath)
            result = TimeSeriesQueryable.deserialize(result_info["meta"], df, **({"index": index} if index is not None else {}))
            if save_in_memory:
                self._in_memory_cache[query_cache_key] = result
            return result
        return None
        
    def save(self, tree, result, time_index_tree=None, transform_info=None):
        """Saves the given result object to the cache for the given tree description."""
        query_cache_name = self._make_cache_key(tree, time_index_tree, transform_info)
        if query_cache_name in self._query_cache and (time_index_tree is None or "time_index_" + str(time_index_tree) in self._query_cache):
            return
        
        if time_index_tree is not None and isinstance(result, (TimeSeries, TimeSeriesSet)):
            time_index_key = "time_index_" + str(time_index_tree)
            if time_index_key not in self._query_cache:
                self.save(time_index_key, result.index)
            meta, df = result.serialize(include_index=False)
        else:
            meta, df = result.serialize()
            
        fname = ('%015x' % random.randrange(16**15)) + ".arrow" # 15-character long random hex string
        self._query_cache[query_cache_name] = {
            "meta": meta,
            "fname": fname,
            **({"time_index_tree": str(time_index_tree)} if time_index_tree is not None else {})
        }
        df.to_feather(os.path.join(self.cache_dir, fname))
        with open(os.path.join(self.cache_dir, "query_cache.json"), "w") as file:
            json.dump(self._query_cache, file)

    
class TrajectoryDataset:
    def __init__(self, attributes, events, intervals, eventtype_macros=None, cache_dir=None):
        self.attributes = attributes if attributes is not None else AttributeSet(pd.DataFrame([]))
        self.events = events if events is not None else EventSet(pd.DataFrame({
            "id": [],
            "eventtype": [],
            "time": [],
            "value": [],
        }))
        self.intervals = intervals if intervals is not None else IntervalSet(pd.DataFrame({
            "id": [],
            "starttime": [],
            "endtime": [],
            "intervaltype": [],
            "value": []
        }))
        self.parser = lark.Lark(GRAMMAR, parser="earley")
        self.eventtype_macros = eventtype_macros
        if cache_dir is not None: self.cache = QueryResultCache(cache_dir)
        else: self.cache = None
        
    def get_ids(self):
        return get_all_trajectory_ids(self.attributes, self.events, self.intervals)
    
    def query(self, query_string, variable_transform=None, use_cache=True):
        query_evaluator = EvaluateQuery(self.attributes, 
                                        self.events, 
                                        self.intervals, 
                                        eventtype_macros=self.eventtype_macros, 
                                        variable_transform=variable_transform,
                                        cache=self.cache if use_cache else None, 
                                        verbose=True)
        tree = self.parser.parse(query_string)
        result = query_evaluator.visit(tree)
        return result
    
    def set_macros(self, macros):
        self.eventtype_macros = macros
        
if __name__ == '__main__':
    ids = [100, 101, 102]
    attributes = AttributeSet(pd.DataFrame({
        'start': [20, 31, 112],
        'end': [91, 87, 168],
        'a1': [3, 5, 1],
        'a2': [10, pd.NA, 42],
        'a3': [61, 21, pd.NA]
    }, index=ids))

    events = EventSet(pd.DataFrame([{
        'id': np.random.choice(ids),
        'time': np.random.randint(0, 100),
        'eventtype': np.random.choice(['e1', 'e2', 'e3']),
        'value': np.random.uniform(0, 100)
    } for _ in range(100)]))

    intervals = IntervalSet(pd.DataFrame([{
        'id': np.random.choice(ids),
        'starttime': np.random.randint(0, 50),
        'endtime': np.random.randint(50, 100),
        'intervaltype': np.random.choice(['i1', 'i2']),
        'value': np.random.uniform(0, 100)
    } for _ in range(10)]))

    dataset = TrajectoryDataset(attributes, events, intervals)
    # print(dataset.query("(min e2: min {'e1', e2} from now - 30 seconds to now, max e2: max {e2} from now - 30 seconds to now) at every {e1} from {start} to {end}"))
    # print(dataset.query("min {'e1', e2} from now - 30 seconds to now at every {e1} from {start} to {end}"))
    # print(dataset.query("myagg: mean ((now - (last time({e1}) from -1000 to now)) at every {e1} from 0 to {end}) from {start} to {end}"))
    # print(dataset.query("(my_age: (last {e1} from #now - 10 sec to #now) impute 'Missing') every 3 sec from #mintime to #maxtime"))
    # print(dataset.query("mean {e1} * 3 from now - 30 s to now"))
    # print(dataset.query("max(mean {e2} from now - 30 seconds to now, mean {e1} from now - 30 seconds to now) at every {e2} from {start} to {end}"))
    print(events.get('e1'))
    print(dataset.query("mean {e1} where {e1} > (last {e1} from #now - 30 sec to #now) from #now to #now + 30 sec every 30 sec from {start} to {end}", use_cache=False))
    print(dataset.query("mean (case when {e1} > (last {e2} from #now - 30 sec to #now) then {e1} else 0 end) from #now to #now + 30 sec every 30 sec from {start} to {end}", use_cache=False))