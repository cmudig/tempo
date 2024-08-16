import lark
import re
import csv
import datetime
from .data_types import *
import json
import os
import random
import tqdm
from divisi.utils import convert_to_native_types
from ..compute.filesystem import LocalFilesystem

GRAMMAR = """
start: variable_expr | variable_list

time_index: "EVERY"i atom [time_bounds]            -> periodic_time_index // periodic time literal
    | "AT EVERY"i atom [time_bounds]  -> event_time_index
    | "AT"i "(" expr ("," expr)* ")"             -> array_time_index

time_bounds: "FROM"i expr "TO"i expr              -> time_bounds_both_ends
    | "BEFORE"i expr                              -> time_bounds_upper
    | "AFTER"i expr                               -> time_bounds_lower
    | "AT"i expr                               -> time_bounds_instant

variable_list: variable_expr
    | "(" variable_expr ("," variable_expr)* ")"
variable_expr: [named_variable] expr
named_variable: (/[A-Za-z][^:]*/i | VAR_NAME) ":"

// Expression Parsing
 
case_when: "WHEN"i expr "THEN"i expr

agg_method: VAR_NAME AGG_OPTIONS* AGG_TYPE?
AGG_TYPE: "rate"i|"amount"i|"value"i|"duration"i
AGG_OPTIONS: "distinct"i|"nonnull"i

?cut_names: "NAMED"i value_list
?cut_expr: LITERAL CUT_TYPE [cut_names]        -> auto_cut
    | CUT_TYPE value_list [cut_names]          -> manual_cut
CUT_TYPE: /bins?/i|/quantiles?/i

?expr: variable_list time_index             -> time_series
    | expr "WHERE"i expr                    -> where_clause
    | expr "CARRY"i (time_quantity | step_quantity)  -> carry_clause
    | expr "IMPUTE"i (VAR_NAME | LITERAL)            -> impute_clause
    | expr "WITH"i VAR_NAME "AS"i logical   -> with_clause
    | expr "CUT"i cut_expr                  -> cut_clause
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
    | comparison "BETWEEN"i agg_expr "AND"i agg_expr  -> between
    | comparison "CONTAINS"i agg_expr  -> contains
    | comparison "STARTSWITH"i agg_expr  -> startswith
    | comparison "ENDSWITH"i agg_expr  -> endswith
    | comparison ("!="|"<>") agg_expr            -> ne
    | comparison "IN"i value_list            -> isin
    | comparison "NOT"i "IN"i value_list            -> isnotin
    | agg_expr

?agg_expr: agg_method expr time_bounds
    | sum

?sum: sum "+" product                       -> expr_add
    | sum "-" product                       -> expr_sub
    | product

?product: product "*" exponent                      -> expr_mul
    | product "/" exponent                      -> expr_div
    | exponent

?exponent: exponent "^" atom                      -> expr_pow
    | atom

value_list: ("("|"[") LITERAL ("," LITERAL)* (")"|"]")

atom: VAR_NAME "(" expr ("," expr)* ")"                 -> function_call
    | DATA_NAME ["AS"i UNIT]                            -> data_element
    | time_quantity
    | LITERAL                               -> literal
    | "#NOW"i                                -> now 
    | "#VALUE"i                              -> where_value
    | "#MINTIME"i                              -> min_time
    | "#MAXTIME"i                              -> max_time
    | "#INDEXVALUE"i                            -> index_value
    | "CASE"i (case_when)+ "ELSE"i expr "END"i -> case_expr     // if/else
    | "(" expr ")" "AS"i UNIT                      -> unit_expr
    | "(" expr ")"
    | VAR_NAME                               -> var_name

time_quantity: LITERAL UNIT
step_quantity: LITERAL /steps?/i
UNIT: /years?|days?|hours?|minutes?|seconds?|yrs?|hrs?|mins?|secs?|[hmsdy]/i

DATA_NAME: /\{[^}]*\}/
VAR_NAME: /(?!(and|or|not|case|when|else|in|then|every|at|from|to|with|as)\b)[A-Za-z][A-Za-z0-9_]*/ 

LITERAL: SIGNED_NUMBER | QUOTED_STRING | /-?inf(inity)?/i | /\\/(?!\\/)(\\\\\/|\\\\\\\|[^\\/])*?\\/i?/
QUOTED_STRING: /["'`][^"'`]*["'`]/


%import common (WORD, WS, SIGNED_NUMBER, LETTER)

%ignore WS
"""

def get_all_trajectory_ids(attributes, events, intervals):
    all_ids = []
    if attributes is not None:
        for attr_set in attributes:
            if len(attr_set.get_ids()):
                all_ids.append(attr_set.get_ids().values)
    if events is not None:
        for event_set in events:
            if len(event_set.get_ids()):
                all_ids.append(event_set.get_ids().values)
    if intervals is not None:
        for interval_set in intervals:
            if len(interval_set.get_ids()):
                all_ids.append(interval_set.get_ids().values)
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
        self._mintimes = None
        self._maxtimes = None
        
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
        candidates = []
        if len(comps) > 1:
            # Only search within the given scope
            scope = comps[0].lower()
            if scope == "attr" and isinstance(el_name, list): raise ValueError(f"Cannot jointly retrieve multiple data elements from Attributes")
            if scope not in ("attr", "event", "interval"): raise ValueError(f"Unknown data element scope {scope}")
        else:
            scope = None
            
        if (scope is None or scope == "attr") and not isinstance(el_name, list):
            candidates += [attr_set.get(el_name) for attr_set in self.attributes if attr_set.has(el_name)]
        if scope is None or scope == "event":
            candidates += [event_set.get(el_name) for event_set in self.events]
        if scope is None or scope == "interval":
            candidates += [interval_set.get(el_name) for interval_set in self.intervals]

        candidates = [c for c in candidates if len(c) > 0]
        if len(candidates) > 1:
            raise ValueError(f"Multiple data elements found with name {comps[-1]}. Try specifying a scope such as {{attr:{comps[-1]}}} (or event: or interval:).")
        elif len(candidates) == 0:
            raise KeyError(f"No data element found with name {query}")
        return candidates[0]
        
    def data_element(self, args):
        match = re.match(r"\{([^\}]+)\}", args[0], flags=re.I)
        query = match.group(1)
        value = self._get_data_element(query)
        if len(args) > 1 and args[1]:
            value /= Duration(1, args[1])
        return value
        
    def var_name(self, args):
        if args[0] in self.variables:
            return self.variables[args[0]]
        raise KeyError(f"No variable named {args[0]}")
        
    def time_quantity(self, args):
        return Duration(self._parse_literal(args[0]), args[1])
        
    def time_bounds(self, args):
        start, end = args
        if isinstance(start, Compilable):
            start = start.execute()
        if isinstance(end, Compilable):
            end = end.execute()
            
        # If either the start or end are Events, this creates an implicit
        # time index (overriding the initial one).
        if isinstance(start, Events) or isinstance(end, Events):
            if isinstance(start, Events) and isinstance(end, Events):
                if len(start.get_values()) != len(end.get_values()):
                    raise ValueError(f"Event sets used for aggregation bounds must be same length")
                new_index = TimeIndex.from_events(start)
                start = new_index
                end = TimeIndex.from_events(end)
            elif isinstance(start, Events):
                new_index = TimeIndex.from_events(start)
                start = new_index
                end = start.with_times(make_aligned_value_series(start, end))
            elif isinstance(end, Events):
                new_index = TimeIndex.from_events(end)
                end = new_index
                start = end.with_times(make_aligned_value_series(end, start))
        else:
            # Broadcast and convert to TimeIndex if needed
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
                if self.time_index is not None:
                    start = self.time_index.with_times(make_aligned_value_series(self.time_index, start))
                    end = self.time_index.with_times(make_aligned_value_series(self.time_index, end))
                else:
                    start = TimeIndex.from_attributes(start)
                    end = TimeIndex.from_attributes(end)
            elif isinstance(start, Attributes) and not isinstance(end, Attributes):
                if self.time_index is not None:
                    start = self.time_index.with_times(make_aligned_value_series(self.time_index, start))
                else:
                    start = TimeIndex.from_attributes(start)
                end = start.with_times(make_aligned_value_series(start, end))
            elif isinstance(end, Attributes) and not isinstance(start, Attributes):
                if self.time_index is not None:
                    end = self.time_index.with_times(make_aligned_value_series(self.time_index, end))
                else:
                    end = TimeIndex.from_attributes(end)
                start = end.with_times(make_aligned_value_series(end, start))
            new_index = None
            
        return (start, end, new_index)
    
    def time_bounds_both_ends(self, args):
        return self.time_bounds(args)
    
    def time_bounds_upper(self, args):
        start, end, new_index = self.time_bounds([self.min_time([]), args[0]])
        return start, self._perform_binary_numpy_function([start, end], "max", np.maximum), new_index
    
    def time_bounds_lower(self, args):
        start, end, new_index = self.time_bounds([args[0], self.max_time([])])
        return self._perform_binary_numpy_function([start, end], "min", np.minimum), end, new_index

    def time_bounds_instant(self, args):
        return self.time_bounds([args[0], args[0]])
    
    def _parse_literal(self, literal):
        if literal.startswith('/'):
            # Parse as a regex
            pattern = re.match(r'^/(.*)/(i?)', literal)
            if pattern is None:
                raise ValueError(f"Cannot parse regular expression from literal {literal}")
            return re.compile(pattern.group(1), flags=re.I if "i" in pattern.group(2) else 0)
        
        if re.search(r"[\"'`]", literal) is not None:
            return re.sub(r"[\"'`]", "", literal)
        try:
            amt = float(literal)
            if not np.isinf(amt) and round(amt) == amt:
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
        return Compilable(self.value_placeholder) if self.time_index is not None else self.value_placeholder
    def index_value(self, args):
        if self.index_value_placeholder is None:
            raise ValueError(f"'indexvalue' keyword can only be used within a time series defined with 'at every' event or interval.")
        return Compilable(self.index_value_placeholder)
    def atom(self, args): return args[0]
    
    def unit_expr(self, args): 
        return args[0] / Duration(1, args[1])
    
    def min_time(self, args):
        if self._mintimes is not None: return self._mintimes
        
        event_times = np.concatenate([event_set.get_times() for event_set in self.events] if self.events else [np.array([])])
        event_ids = np.concatenate([event_set.get_ids() for event_set in self.events] if self.events else [np.array([])])
        event_mins = pd.Series(event_times, name='times').groupby(event_ids).agg("min")
        
        interval_times = np.concatenate([interval_set.get_start_times() for interval_set in self.intervals] if self.intervals else [np.array([])])
        interval_ids = np.concatenate([interval_set.get_ids() for interval_set in self.intervals] if self.intervals else [np.array([])])
        interval_mins = pd.Series(interval_times, name='times').groupby(interval_ids).agg("min")

        ids = self.get_all_ids()
        all_mins = pd.merge(pd.Series(ids, name="id"), pd.merge(event_mins, interval_mins, how='outer', left_index=True, right_index=True), left_on="id", right_index=True).set_index("id").min(axis=1)
        self._mintimes = Attributes(all_mins.rename("mintime"))
        return self._mintimes
    
    def max_time(self, args): 
        if self._maxtimes is not None: return self._maxtimes
        
        event_times = np.concatenate([event_set.get_times() for event_set in self.events] if self.events else [np.array([])])
        event_ids = np.concatenate([event_set.get_ids() for event_set in self.events] if self.events else [np.array([])])
        event_maxes = pd.Series(event_times, name='times').groupby(event_ids).agg("max")
        
        interval_times = np.concatenate([interval_set.get_end_times() for interval_set in self.intervals] if self.intervals else [np.array([])])
        interval_ids = np.concatenate([interval_set.get_ids() for interval_set in self.intervals] if self.intervals else [np.array([])])
        interval_maxes = pd.Series(interval_times, name='times').groupby(interval_ids).agg("max")

        ids = self.get_all_ids()
        all_maxes = pd.merge(pd.Series(ids, name="id"), pd.merge(event_maxes, interval_maxes, how='outer', left_index=True, right_index=True), left_on="id", right_index=True).set_index("id").max(axis=1)
        # Offset the time by 1 so that it includes all events and intervals
        if pd.api.types.is_datetime64_any_dtype(all_maxes.dtype):
            all_maxes += datetime.timedelta(seconds=1)
        else:
            all_maxes += 1
        self._maxtimes = Attributes(all_maxes.rename("maxtime"))
        return self._maxtimes

    def isin(self, args):
        return args[0].isin(args[1])
    def isnotin(self, args):
        return ~args[0].isin(args[1])
    
    def value_list(self, args): return [self._parse_literal(v) for v in args]
        
    def expr_add(self, args): return args[0] + args[1]
    def expr_sub(self, args): return args[0] - args[1]
    def expr_mul(self, args): return args[0] * args[1]
    def expr_div(self, args): return args[0] / args[1]
    def expr_pow(self, args): return args[0] ** args[1]
    def gt(self, args): return args[0] > args[1]
    def lt(self, args): return args[0] < args[1]
    def geq(self, args): return args[0] >= args[1]
    def leq(self, args): return args[0] <= args[1]
    def eq(self, args): return args[0] == args[1]
    def ne(self, args): return args[0] != args[1]
    def between(self, args): return (args[0] >= args[1]) & (args[0] < args[2])
    
    def contains(self, args):
        strings = args[0].get_values().astype(str)
        return args[0].with_values(strings.str.contains(args[1]))
    def startswith(self, args):
        strings = args[0].get_values().astype(str)
        return args[0].with_values(strings.str.startswith(args[1]))
    def endswith(self, args):
        strings = args[0].get_values().astype(str)
        return args[0].with_values(strings.str.startswith(args[1]))
    
    def negate(self, args): return ~args[0]
    
    def logical_and(self, args): return args[0] & args[1]
    def logical_or(self, args): return args[0] | args[1]

    def agg_expr(self, args):
        agg_method = args[0]
        expr = args[1]
        *time_bounds, time_index = args[-1]
        has_inner_time_index = time_index is not None # if this is true, the return value will be an Events!
        if time_index is None: time_index = self.time_index
        
        if time_index is not None:
            assert len(time_bounds[0]) == len(time_index), f"Start time bounds for aggregation (length {len(time_bounds[0])}) must be equal length to overall time index (length {len(time_index)})"
            assert len(time_bounds[1]) == len(time_index), f"End time bounds for aggregation (length {len(time_bounds[1])}) must be equal length to overall time index (length {len(time_index)})"
            agg_result = None
            if isinstance(expr, TimeSeries):
                # Convert the expression to an Events
                expr = expr.to_events()
            if isinstance(expr, Events):
                agg_result = expr.bin_aggregate(time_index, *time_bounds, agg_method[0])
            elif isinstance(expr, (Intervals, Compilable)):
                agg_result = expr.bin_aggregate(time_index, *time_bounds, agg_method[1], agg_method[0])
            else:
                raise ValueError(f"Only Events and Intervals can be bin-aggregated")
            if has_inner_time_index:
                return agg_result.to_events()
            return agg_result
        else:
            if isinstance(expr, (Events, TimeSeries)):
                return expr.aggregate(*time_bounds, agg_method[0])
            elif isinstance(expr, Intervals):
                result = expr.aggregate(*time_bounds, agg_method[1], agg_method[0])
                return result
            else:
                raise ValueError(f"Only Events and Intervals can be aggregated")            
        
    def agg_method(self, args):
        results = {}
        for arg in args[1:]:
            results.setdefault(arg.type, []).append(arg.value)
        agg_func = args[0].value
        if "AGG_OPTIONS" in results:
            agg_func += " " + " ".join(sorted(x.lower() for x in results["AGG_OPTIONS"]))
        return (agg_func, results.get("AGG_TYPE", ["value"])[0])
        
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
                result = result.where(~condition.fillna(0).astype(bool), value)
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
            return var_exp.astype(np.float64).where(nan_mask, numpy_func(var_exp.get_values().replace(pd.NA, np.nan).astype(float)))
        else:
            impute_method = self._parse_literal(args[1].value)
            dtype = var_exp.get_values().dtype
            if isinstance(dtype, pd.CategoricalDtype):
                var_exp = var_exp.with_values(var_exp.get_values().astype(dtype.categories.dtype))
                dtype = var_exp.get_values().dtype
            elif isinstance(impute_method, str) and pd.api.types.is_numeric_dtype(dtype):
                # convert all values to strings
                var_exp = var_exp.with_values(var_exp.get_values().astype(pd.StringDtype()))
                dtype = var_exp.get_values().dtype
            scalar = dtype.type(impute_method)
            return var_exp.with_values(var_exp.get_values().where(nan_mask, scalar))
            
    def _perform_binary_numpy_function(self, operands, function_name, numpy_func):
        if isinstance(operands[0], Compilable):
            return getattr(operands[0], function_name)(operands[1])
        if isinstance(operands[1], Compilable):
            return getattr(operands[1], function_name)(operands[0])
        elif isinstance(operands[0], TimeIndex):
            return operands[0].with_times(numpy_func(operands[0].get_times(), make_aligned_value_series(operands[0], operands[1])))
        elif isinstance(operands[1], TimeIndex):
            return operands[1].with_times(numpy_func(operands[1].get_times(), make_aligned_value_series(operands[1], operands[0])))
        if isinstance(operands[0], (Attributes, Events, Intervals, TimeSeries)):
            return operands[0].with_values(numpy_func(operands[0].get_values(), make_aligned_value_series(operands[0], operands[1])))
        elif isinstance(operands[1], (Attributes, Events, Intervals, TimeSeries)):
            return operands[1].with_values(numpy_func(operands[1].get_values(), make_aligned_value_series(operands[1], operands[0])))
        else:
            raise ValueError(f"{function_name} function requires at least one parameter to be Attributes, Events, Intervals, TimeIndex, or TimeSeries")

    def function_call(self, args):
        function_name = args[0].value.lower()
        operands = args[1:]
        if function_name in ("time", "starttime", "endtime"):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            if function_name == "time":
                if isinstance(operands[0], Compilable):
                    return operands[0].time()
                if not hasattr(operands[0], "get_times"):
                    raise ValueError("time function requires an object with time values")
                return operands[0].with_values(operands[0].get_times())
            elif function_name == "starttime":
                if isinstance(operands[0], Compilable):
                    return operands[0].start().time()
                if not hasattr(operands[0], "get_start_times"):
                    raise ValueError("starttime function requires interval objects")
                return operands[0].start_events().with_values(operands[0].get_start_times())
            elif function_name == "endtime":
                if isinstance(operands[0], Compilable):
                    return operands[0].end().time()
                if not hasattr(operands[0], "get_end_times"):
                    raise ValueError("endtime function requires interval objects")
                return operands[0].end_events().with_values(operands[0].get_end_times())
        elif function_name in ("duration",):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            return (operands[0].end_events().with_values(operands[0].get_end_times())
                    - operands[0].start_events().with_values(operands[0].get_start_times()))
        elif function_name in ("start", "end"):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            if function_name == "start":
                if isinstance(operands[0], Compilable):
                    return operands[0].start()
                if not hasattr(operands[0], "get_start_times"):
                    raise ValueError("start function requires interval objects")
                return operands[0].start_events()
            elif function_name == "end":
                if isinstance(operands[0], Compilable):
                    return operands[0].end()
                if not hasattr(operands[0], "get_end_times"):
                    raise ValueError("end function requires interval objects")
                return operands[0].end_events()
        elif function_name in ("abs", ):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            return getattr(operands[0], function_name)()
        elif function_name in ("max", "min"):
            if len(operands) != 2: raise ValueError(f"{function_name} function requires exactly two operands")
            numpy_func = np.maximum if function_name == "max" else np.minimum
            return self._perform_binary_numpy_function(operands, function_name, numpy_func)
        elif function_name in ("extract", ):
            if len(operands) not in (2, 3): raise ValueError(f"{function_name} function takes as input a series, a pattern, and optionally an index of a capturing group")
            pattern = operands[1]
            if isinstance(pattern, re.Pattern) and not pattern.groups:
                pattern = re.compile("(" + pattern.pattern + ")", flags=pattern.flags)
            return operands[0].with_values(operands[0].get_values().str.extract(pattern)[operands[2] if len(operands) > 2 else 0])
        elif function_name == "shift":
            if len(operands) != 2: raise ValueError(f"{function_name} function requires exactly two operands")
            return operands[0].shift(operands[1])
        elif function_name in ("previous", "next"):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly two operands")
            return operands[0].shift(1 if function_name == "next" else -1)
        else:
            raise ValueError(f"Unknown function '{function_name}'")

    def variable_list(self, args):
        if len(args) == 1: return args[0]
        if all(isinstance(a, Attributes) for a in args):
            return AttributeSet(pd.concat([a.series for a in args], axis=1))
        elif all(isinstance(a, TimeSeries) for a in args):
            return TimeSeriesSet.from_series(args)
        raise ValueError("Variable list must contain either all Attributes or all TimeSeries objects")
    
    def auto_cut(self, args):
        num_bins = args[0]
        if not isinstance(num_bins, (float, int)) and int(num_bins) == num_bins:
            raise ValueError("Cut must either be followed by an integer bin count or a list of bin cutoffs")
        cut_type = args[1].value
        return CutOperator(int(num_bins), cut_type, names=args[2] if len(args) > 2 else None)
    
    def manual_cut(self, args):
        cut_type = args[0].value
        bins = args[1]
        return CutOperator(np.array(bins), cut_type, names=args[2] if len(args) > 2 else None)
    
    def cut_clause(self, args):
        base_values, cut_op = args
        return cut_op.apply(base_values)
        

class EvaluateQuery(lark.visitors.Interpreter):
    def __init__(self, attributes, events, intervals, variable_transform=None, eventtype_macros=None, cache=None, verbose=False, update_fn=None):
        super().__init__()
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.cache = cache
        self.eventtype_macros = eventtype_macros if eventtype_macros is not None else {}
        self.update_fn = update_fn
        # If provided, this should be a tuple of (description, transform_fn, restore_fn). The
        # description should be a string uniquely identifying this transform,
        # and transform should be a function that will be called on any variable 
        # expressions before saving to cache. The function should
        # take as input a TimeSeriesQueryable, and it should return either a 
        # tuple (transformed, info). The info will be stored in the cache. If the
        # query result is retrieved from the cache, the restore_fn will be called
        # with two arguments, the stored TimeSeriesQueryable and the stored info,
        # and it should return a restored version of the time series object.
        if variable_transform is not None:
            self.variable_transform_desc, self.variable_transform, self.variable_restore = variable_transform
        else:
            self.variable_transform_desc = None
            self.variable_transform = None
            self.variable_restore = None
        self.verbose = verbose
        self.evaluator = EvaluateExpression(self.attributes, self.events, self.intervals, self.eventtype_macros)
        
    def get_all_ids(self):
        return self.evaluator.get_all_ids()
    
    def _make_time_index(self, idx):
        if isinstance(idx, Attributes):
            return TimeIndex.from_attributes(idx)
        elif isinstance(idx, TimeIndex):
            return idx
        elif isinstance(idx, (TimeSeries, TimeSeriesSet)):
            return idx.index
        else:
            raise ValueError(f"Cannot convert {type(idx)} object to TimeIndex")

    def periodic_time_index(self, tree):
        duration = self.evaluator.transform(tree.children[0])
                
        if tree.children[1] is not None:
            start_time = self._make_time_index(self.evaluator.transform(tree.children[1].children[0]))
            end_time = self._make_time_index(self.evaluator.transform(tree.children[1].children[1]))
        else:
            start_time = self._make_time_index(self.evaluator.min_time([]))
            end_time = self._make_time_index(self.evaluator.max_time([]))
            
        return TimeIndex.range(start_time, end_time, duration)
        
    def event_time_index(self, tree):
        events = self.evaluator.transform(tree.children[0])

        if tree.children[-1] is not None:
            start_time = self.evaluator.transform(tree.children[-1].children[0])
            end_time = self.evaluator.transform(tree.children[-1].children[1])
        else:
            start_time = self.evaluator.min_time([])
            end_time = self.evaluator.max_time([])
            
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
            raise ValueError(f"Expected 'at every' data element to evaluate to an Events object, but instead got '{type(events).__name__}'")
        index, filtered_events = TimeIndex.from_events(events, starts=start_time, ends=end_time, return_filtered_events=True)
        self.evaluator.index_value_placeholder = TimeSeries(index, filtered_events.get_values())
        return index
        
    def array_time_index(self, tree):
        times = [self.evaluator.transform(c) for c in tree.children]
        return TimeIndex.from_times(times)
        
    def _parse_variable_expr(self, tree, evaluator, time_index_tree=None, cache_only=False):            
        # Parse where clauses first (these require top-down processing in case of a value placeholder)
        if not isinstance(tree, lark.Tree):
            return tree

        tree_desc = str(tree.children[1])
        options_desc = str(tree.children[2]) if len(tree.children) > 2 else ''
        
        var_name = tree.children[0].children[0].value if tree.children[0] and tree.children[0].children[0].value else None
        if isinstance(tree.children[1], (TimeSeries, TimeSeriesSet)):
            var_exp = tree.children[1]
        elif self.cache is not None:
            var_exp = self.cache.lookup((tree_desc, options_desc), time_index_tree=time_index_tree, transform_info=self.variable_transform_desc)
            if self.variable_transform_desc is not None and var_exp is not None:
                var_exp, transform_data = var_exp
                if self.variable_restore is not None:
                    if transform_data is None: var_exp = None
                    else: var_exp = self.variable_restore(var_exp, transform_data)
        else: var_exp = None
        if cache_only and var_exp is None: return tree
        elif var_exp is not None:
            if var_name is not None:
                var_exp = var_exp.rename(var_name)
            return var_exp.compress()

        try:
            # We only cache the main expression, so variable names and options can be adjusted later without recomputing
            # expensive aggregations
            if var_exp is None:
                var_exp = self._parse_variable_value(tree.children[1], evaluator)            
                
            if var_name is not None:
                var_exp = var_exp.rename(var_name)
        except Exception as e:
            raise ValueError(f"Exception occurred when processing variable '{var_name}': {e}")
        else:
            if self.variable_transform is not None:
                var_exp, transform_data = self.variable_transform(var_exp)
            else:
                transform_data = None
            var_exp = var_exp.compress()
            
            if self.cache is not None:
                self.cache.save((tree_desc, options_desc), 
                                var_exp, 
                                transform_info=self.variable_transform_desc, 
                                transform_data=transform_data,
                                time_index_tree=time_index_tree)
            return var_exp
        
    def _parse_variable_value(self, tree, evaluator):
        self._preprocess_nested_aggregations(tree, evaluator)
            
        set_variables = set()
        tree_parent = lark.Tree(None, [tree]) # in case one of the searched clauses is the root
        for node in tree_parent.iter_subtrees():
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
        
        for node in tree_parent.iter_subtrees():
            if node is None: continue
            node.children = [lark.Tree('atom', [self._parse_where_clause(n, evaluator)]) if isinstance(n, lark.Tree) and n.data == "where_clause" else n for n in node.children]
            
        tree = tree_parent.children[0]
        var_exp = evaluator.transform(tree)
        if isinstance(var_exp, Compilable): var_exp = var_exp.execute()
        if evaluator.time_index is not None:
            if isinstance(var_exp, Attributes):
                # Cast the attributes over the time index
                var_exp = TimeSeries(evaluator.time_index, make_aligned_value_series(evaluator.time_index, var_exp))
            elif isinstance(var_exp, TimeIndex):
                # Use the times as the time series values
                var_exp = TimeSeries(var_exp, var_exp.get_times())
            elif (isinstance(var_exp, Events) and len(var_exp) == len(evaluator.time_index) and 
                    (var_exp.get_ids().values == evaluator.time_index.get_ids().values).all()):
                # This is an Events but is perfectly aligned to the time index
                var_exp = TimeSeries(TimeIndex.from_events(var_exp), var_exp.get_values())
        return var_exp

    def _parse_time_series(self, tree, evaluator):
        time_index_tree = tree.children[-1] if len(tree.children) > 1 else None
        time_index = self.visit(tree.children[-1]) if len(tree.children) > 1 else None
        evaluator.time_index = time_index
        if self.update_fn is None:
            pbar = tqdm.tqdm(tree.children[0].children) if self.verbose else tree.children[0].children
        else:
            def progress_iterable():
                for i, c in enumerate(tree.children[0].children):
                    yield c
                    self.update_fn(i + 1, len(tree.children[0].children))
            pbar = progress_iterable()
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
            
            for desc in node.children[1].iter_subtrees():
                if desc is None or desc == node: continue
                desc.children = [Compilable(evaluator.transform(n)) 
                                 if (isinstance(n, lark.Tree) and n.data in ("agg_expr", "now")) else n 
                                 for n in desc.children]
                
        
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
        var_value = self._parse_variable_value(tree.children[-1], evaluator)
        if isinstance(var_value, Compilable): var_value = var_value.execute()
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
        self.cache_dir = LocalFilesystem(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self._query_cache = {}
        self._in_memory_cache = {} # for items to be stored in memory instead of loaded from disk every time
    
    def load_cache(self):
        if not self.cache_dir: return
        
        # Load cache information
        if self.cache_dir.exists("query_cache.json"):
            self._query_cache = self.cache_dir.read_file("query_cache.json")
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
            if not self.cache_dir.exists(result_info["fname"]): return
            df = self.cache_dir.read_file(result_info["fname"], format="feather")
            result = TimeSeriesQueryable.deserialize(result_info["meta"], df, **({"index": index} if index is not None else {}))
            if transform_info is not None:
                result = (result, result_info.get("transform_data", None))
            if save_in_memory:
                self._in_memory_cache[query_cache_key] = result
            return result
        return None
        
    def save(self, tree, result, time_index_tree=None, transform_info=None, transform_data=None):
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
        if transform_data is not None:
            self._query_cache[query_cache_name]["transform_data"] = convert_to_native_types(transform_data)
        self.cache_dir.write_file(df, fname, format='feather')
        self.cache_dir.write_file(self._query_cache, "query_cache.json")

    
class QueryEngine:
    def __init__(self, attributes, events, intervals, eventtype_macros=None, cache_fs=None):
        self.attributes = attributes if attributes is not None else [AttributeSet(pd.DataFrame([]))]
        self.events = events if events is not None else [EventSet(pd.DataFrame({
            "id": [],
            "eventtype": [],
            "time": [],
            "value": [],
        }))]
        self.intervals = intervals if intervals is not None else [IntervalSet(pd.DataFrame({
            "id": [],
            "starttime": [],
            "endtime": [],
            "intervaltype": [],
            "value": []
        }))]
        self.parser = lark.Lark(GRAMMAR, parser="earley")
        self.eventtype_macros = eventtype_macros
        if cache_fs is not None: self.cache = QueryResultCache(cache_fs)
        else: self.cache = None
        
    def get_ids(self):
        return get_all_trajectory_ids(self.attributes, self.events, self.intervals)
    
    def query(self, query_string, variable_transform=None, use_cache=True, update_fn=None):
        query_evaluator = EvaluateQuery(self.attributes, 
                                        self.events, 
                                        self.intervals, 
                                        eventtype_macros=self.eventtype_macros, 
                                        variable_transform=variable_transform,
                                        cache=self.cache if use_cache else None, 
                                        update_fn=update_fn,
                                        verbose=True)
        tree = self.parse(query_string)
        result = query_evaluator.visit(tree)
        return result
    
    def parse(self, query, keep_all_tokens=False):
        if keep_all_tokens:
            parser = lark.Lark(GRAMMAR, parser="earley", keep_all_tokens=True)
        else:
            parser = self.parser
        return parser.parse(query)
    
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

    dataset = QueryEngine(attributes, events, intervals)
    # print(dataset.query("(min e2: min {'e1', e2} from now - 30 seconds to now, max e2: max {e2} from now - 30 seconds to now) at every {e1} from {start} to {end}"))
    # print(dataset.query("min {e1} from #now - 30 seconds to #now cut 3 quantiles impute 'Missing' at every {e1} from #mintime to #maxtime"))
    # print(dataset.query("myagg: mean ((now - (last time({e1}) from -1000 to now)) at every {e1} from 0 to {end}) from {start} to {end}"))
    # print(dataset.query("(my_age: (last {e1} from #now - 10 sec to #now) impute 'Missing') every 3 sec from #mintime to #maxtime"))
    # print(dataset.query("mean {e1} * 3 from now - 30 s to now"))
    # print(dataset.query("max(mean {e2} from now - 30 seconds to now, mean {e1} from now - 30 seconds to now) at every {e2} from {start} to {end}"))
    print(events.get('e1'))
    print(dataset.query("{e1} - (last {e1} before {e1})"))
    # print(dataset.query("mean {e1} where {e1} > (last {e1} from #now - 30 sec to #now) from #now to #now + 30 sec every 30 sec from {start} to {end}", use_cache=False))
    # print(dataset.query("mean (case when {e1} > (last {e2} from #now - 30 sec to #now) then {e1} else 0 end) from #now to #now + 30 sec every 30 sec from {start} to {end}", use_cache=False))