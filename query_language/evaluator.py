import lark
import re
import csv
from query_language.data_types import *
import json
import os
import random
import tqdm

class EvaluateExpression(lark.visitors.Transformer):
    def __init__(self, attributes, events, intervals, time_index=None, eventtype_macros=None):
        super().__init__()
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.time_index = time_index
        self.eventtype_macros = eventtype_macros if eventtype_macros is not None else {}
        
    def _get_data_element(self, query):
        comps = query.split(":")
        el_name = comps[-1]
        # substitute with macro if available
        if el_name in self.eventtype_macros:
            el_name = self.eventtype_macros[el_name]
        if "," in el_name:
            el_name = list(csv.reader([el_name], skipinitialspace=True))[0]
            # Substitute macros again
            el_name = [x for el in el_name for x in self.eventtype_macros.get(el, el).split(",")]
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
            raise ValueError(f"Multiple data elements found with name {el_name}. Try specifying a scope such as \{attr:{el_name}\} (or event: or interval:).")
        elif len(candidates) == 0:
            raise KeyError(f"No data element found with name {el_name}")
        return candidates[0]
        
    def data_element(self, args):
        match = re.match(r"\{([^\}]+)\}", args[0], flags=re.I)
        query = match.group(1)
        return self._get_data_element(query)
        
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

    def now(self, args): return self.time_index
    def atom(self, args): return args[0]

    def isin(self, args):
        return args[0].isin(args[1])
    
    def value_list(self, args): return [self._parse_literal(v) for v in args]
        
    def expr_add(self, args): return args[0] + args[1]
    def expr_sub(self, args): return args[0] - args[1]
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
        
        assert self.time_index is not None, "Cannot perform an aggregation without an overall time index"
        assert len(time_bounds[0]) == len(self.time_index), f"Start time bounds for aggregation (length {len(time_bounds[0])}) must be equal length to overall time index (length {len(self.time_index)})"
        assert len(time_bounds[1]) == len(self.time_index), f"End time bounds for aggregation (length {len(time_bounds[1])}) must be equal length to overall time index (length {len(self.time_index)})"
        if isinstance(expr, Events):
            return expr.aggregate(self.time_index, *time_bounds, agg_method[0])
        elif isinstance(expr, Intervals):
            result = expr.aggregate(self.time_index, *time_bounds, agg_method[1], agg_method[0])
            return result
        else:
            raise ValueError(f"Only Events and Intervals can be aggregated")
        
    def agg_method(self, args):
        if len(args) > 1:
            return (args[0].value, args[1].value)
        return (args[0].value, None)
        
    def case_expr(self, args):
        whens = args[:-1]
        else_clause = args[-1]
        result = else_clause
        if isinstance(result, Duration): result = result.value()
        
        for when in reversed(whens):
            condition, value = when.children
            if isinstance(value, Duration): value = value.value()
            if isinstance(value, (Events, Attributes, Intervals, TimeSeries)):
                if len(value.get_values()) != len(condition.get_values()):
                    raise ValueError(f"Case expression operands must be same length")
                result = value.where(condition, result)
            elif isinstance(result, (Events, Attributes, Intervals, TimeSeries)):
                if len(result.get_values()) != len(condition.get_values()):
                    raise ValueError(f"Case expression operands must be same length")
                result = result.where(~condition, value)
            elif isinstance(condition, (Events, Intervals)):
                # We need to broadcast both value and result to condition's type
                result = condition.apply(lambda x: value if x else result)
                
        return result
        
    def where_clause(self, args): return args[0]
    
    def function_call(self, args):
        function_name = args[0].value.lower()
        operands = args[1:]
        if function_name == "timeof":
            if len(operands) != 1: raise ValueError(f"timeof function requires exactly one operand")
            return operands[0].with_values(operands[0].get_times())
        elif function_name in ("abs", ):
            if len(operands) != 1: raise ValueError(f"{function_name} function requires exactly one operand")
            return getattr(operands[0], function_name)()
        elif function_name in ("max", "min"):
            if len(operands) != 2: raise ValueError(f"{function_name} function requires exactly two operands")
            numpy_func = np.maximum if function_name == "max" else np.minimum
            if isinstance(operands[0], (Attributes, Events, Intervals, TimeSeries)):
                return operands[0].with_values(numpy_func(operands[0].get_values(), make_aligned_value_series(operands[0], operands[1])))
            elif isinstance(operands[1], (Attributes, Events, Intervals, TimeSeries)):
                return operands[1].with_values(numpy_func(operands[1].get_values(), make_aligned_value_series(operands[1], operands[0])))
            else:
                raise ValueError(f"{function_name} function requires at least one parameter to be Attributes, Events, Intervals, or TimeSeries")
        else:
            raise ValueError(f"Unknown function '{function_name}'")

class EvaluateQuery(lark.visitors.Interpreter):
    def __init__(self, attributes, events, intervals, eventtype_macros=None, cache_dir=None, verbose=False):
        super().__init__()
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.time_index_tree = None
        self.cache_dir = cache_dir
        self.eventtype_macros = eventtype_macros if eventtype_macros is not None else {}
        self.verbose = verbose
        self._query_cache = {}
        self.use_cache = True
        self.load_cache()
        
    def load_cache(self):
        if not self.cache_dir: return
        if not os.path.exists(self.cache_dir): os.mkdir(self.cache_dir)
        
        # Load cache information
        if os.path.exists(os.path.join(self.cache_dir, "query_cache.json")):
            with open(os.path.join(self.cache_dir, "query_cache.json"), "r") as file:
                self._query_cache = json.load(file)
        else:
            self._query_cache = {}
    
    def cache_lookup(self, tree):
        """Returns the result of a variable parse if it exists in the cache."""
        if not self.cache_dir or not self.use_cache: return
        if str(tree) in self._query_cache:
            result_info = self._query_cache[str(tree)]
            fpath = os.path.join(self.cache_dir, result_info["fname"])
            if not os.path.exists(fpath): return
            df = pd.read_feather(fpath)
            return TimeSeriesQueryable.deserialize(result_info["meta"], df)
        
    def save_to_cache(self, tree, result):
        """Saves the given result object to the cache for the given tree description."""
        if not self.cache_dir or not self.use_cache: return
        meta, df = result.serialize()
        fname = ('%015x' % random.randrange(16**15)) + ".arrow" # 15-character long random hex string
        self._query_cache[str(tree)] = {
            "meta": meta,
            "fname": fname
        }
        df.to_feather(os.path.join(self.cache_dir, fname))
        with open(os.path.join(self.cache_dir, "query_cache.json"), "w") as file:
            json.dump(self._query_cache, file)
        
    def _make_time_index(self, idx):
        if isinstance(idx, Attributes):
            return TimeIndex.from_attributes(idx)
        elif isinstance(idx, TimeIndex):
            return idx
        else:
            raise ValueError(f"Cannot convert {type(idx)} object to TimeIndex")

    def periodic_time_index(self, tree):
        evaluator = EvaluateExpression(self.attributes, self.events, self.intervals, None, self.eventtype_macros)
        duration = evaluator.transform(tree.children[0])
                
        start_time = self._make_time_index(evaluator.transform(tree.children[1].children[0]))
        end_time = self._make_time_index(evaluator.transform(tree.children[1].children[1]))
        
        return TimeIndex.range(start_time, end_time, duration)
        
    def event_time_index(self, tree):
        evaluator = EvaluateExpression(self.attributes, self.events, self.intervals, None, self.eventtype_macros)
        events = evaluator.transform(tree.children[0])

        start_time = evaluator.transform(tree.children[-1].children[0])
        end_time = evaluator.transform(tree.children[-1].children[1])
        
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
        return TimeIndex.from_events(events, starts=start_time, ends=end_time)
        
    def _parse_variable_expr(self, tree, evaluator):            
        var_name = tree.children[0].children[0].value if tree.children[0] and tree.children[0].children[0].value else None
        
        try:
            # We only cache the main expression, so variable names and options can be adjusted later without recomputing
            # expensive aggregations
            if (var_exp := self.cache_lookup((tree.children[1], self.time_index_tree))) is None:
                var_exp = evaluator.transform(tree.children[1])
                if evaluator.time_index is not None and isinstance(var_exp, Attributes):
                    # Cast the attributes over the time index
                    var_exp = TimeSeries(evaluator.time_index, make_aligned_value_series(evaluator.time_index, var_exp))
                self.save_to_cache((tree.children[1], self.time_index_tree), var_exp)
            
            if var_name is not None:
                var_exp = var_exp.rename(var_name)
            
            if len(tree.children) > 2:
                # Options clauses are executed IN ORDER of appearance
                for child in tree.children[2:]:
                    if child.data == "where_clause":
                        # Defines when the time series values should be converted to NaN
                        variable_filter = evaluator.transform(child)
                        var_exp = var_exp.where(variable_filter, pd.NA)
                    elif child.data == "carry_clause":
                        # Defines how far the values in the time series should be
                        # carried forward within a given ID
                        if child.children[0].data == "step_quantity":
                            steps = int(child.children[0].children[0].value)
                            var_exp = var_exp.carry_forward_steps(steps)
                        else:
                            duration = evaluator.transform(child.children[0])
                            var_exp = var_exp.carry_forward_duration(duration)
                    elif child.data == "impute_clause":
                        # Defines how NaN values should be substituted
                        nan_mask = ~pd.isna(var_exp.get_values())
                        impute_method = child.children[0].value.lower()
                        if impute_method in ("mean", "median"):
                            numpy_func = {"mean": np.nanmean, "median": np.nanmedian}[impute_method]
                            var_exp = var_exp.where(nan_mask, numpy_func(var_exp.get_values()))
                        else:
                            try:
                                constant_val = float(impute_method)
                                if round(constant_val) == constant_val:
                                    constant_val = int(constant_val)
                            except ValueError:
                                raise ValueError(f"Impute method must either be a number or 'mean', 'median', not '{impute_method}'")
                            else:
                                var_exp = var_exp.where(nan_mask, constant_val)
        except Exception as e:
            raise ValueError(f"Exception occurred when processing variable '{var_name}': {e}")
        else:
            return var_exp
        
    def _parse_variable_list(self, tree, time_index):
        evaluator = EvaluateExpression(self.attributes, self.events, self.intervals, time_index, self.eventtype_macros)
        pbar = tqdm.tqdm(tree.children) if self.verbose else tree.children
        return [self._parse_variable_expr(child, evaluator) for child in pbar]
        
    def start(self, tree):
        self.time_index_tree = tree.children[-1] if len(tree.children) > 1 else None
        time_index = self.visit(tree.children[-1]) if len(tree.children) > 1 else None
        variable_definitions = self._parse_variable_list(tree.children[0], time_index)
        if time_index is not None and not all(isinstance(v, TimeSeries) for v in variable_definitions):
            raise ValueError(f"All variables must evaluate to a TimeSeries when a time index is provided")
        if len(variable_definitions) == 1:
            return variable_definitions[0]
        else:
            return TimeSeriesSet.from_series(variable_definitions)
    
GRAMMAR = """
start: variable_list time_index?

time_index: "EVERY"i atom time_bounds            -> periodic_time_index // periodic time literal
    | "AT EVERY"i atom INTERVAL_POSITION? time_bounds  -> event_time_index

INTERVAL_POSITION: "START"i|"END"i

time_bounds: "FROM"i expr "TO"i expr

variable_list: variable_expr
    | "(" variable_expr ("," variable_expr)* ")"
variable_expr: [named_variable] expr ("[" option_clause ("," option_clause)* "]")?
named_variable: (/[A-Za-z][^:]*/i | VAR_NAME) ":"

option_clause: "WHERE"i expr                    -> where_clause
    | "CARRY"i (time_quantity | step_quantity)  -> carry_clause
    | "IMPUTE"i (VAR_NAME | LITERAL)            -> impute_clause

// Expression Parsing
 
case_when: "WHEN"i expr "THEN"i expr

agg_method: VAR_NAME AGG_TYPE?
AGG_TYPE: "rate"i|"amount"i|"value"i

?expr: agg_method expr time_bounds              -> agg_expr
    | "CASE"i (case_when)+ "ELSE"i expr "END"i -> case_expr     // if/else
    | logical
    
?logical: logical "AND"i negation                -> logical_and
    | logical "OR"i negation                  -> logical_or
    | negation

?negation: "NOT"i negation                        -> negate
    | comparison

?comparison: comparison ">=" sum                   -> geq
    | comparison "<=" sum                   -> leq
    | comparison ">" sum                    -> gt
    | comparison "<" sum                    -> lt
    | comparison "=" sum                    -> eq
    | comparison ("!="|"<>") sum            -> ne
    | comparison "IN"i value_list            -> isin
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
    | "NOW"i                                -> now 
    | "(" expr ")"

time_quantity: LITERAL UNIT
step_quantity: LITERAL /steps?/i
UNIT: /hours?|minutes?|seconds?|hrs?|mins?|secs?|[hms]/i

DATA_NAME: /\{[^}]*\}/
VAR_NAME: /(?!(and|or|not|case|when|else|in|then|every|at|from|to)\b)[A-Za-z][A-Za-z_]*/ 

LITERAL: SIGNED_NUMBER | QUOTED_STRING
QUOTED_STRING: /["'`][^"'`]*["'`]/


%import common (WORD, WS, SIGNED_NUMBER, LETTER)

%ignore WS
"""

class TrajectoryDataset:
    def __init__(self, attributes, events, intervals, eventtype_macros=None, cache_dir=None):
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.parser = lark.Lark(GRAMMAR, parser="earley")
        self.query_evaluator = EvaluateQuery(attributes, events, intervals, eventtype_macros=eventtype_macros, cache_dir=cache_dir)
        
    def query(self, query_string, use_cache=True):
        tree = self.parser.parse(query_string)
        self.query_evaluator.use_cache = use_cache
        result = self.query_evaluator.visit(tree)
        self.query_evaluator.use_cache = True
        return result
    
    def set_macros(self, macros):
        self.query_evaluator.eventtype_macros = macros
        
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
        'charttime': np.random.randint(0, 100),
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

    dataset = TrajectoryDataset(attributes, events, intervals, cache_dir="_tmp_cache")
    # print(dataset.query("(min e2: min {'e1', e2} from now - 30 seconds to now, max e2: max {e2} from now - 30 seconds to now) at every {e1} from {start} to {end}"))
    # print(dataset.query("min {'e1', e2} from now - 30 seconds to now at every {e1} from {start} to {end}"))
    print(dataset.query("exists {i1} from now - 30 seconds to now every 30 seconds from {start} + 3 sec to {end}"))
    # print(dataset.query("mean {e1} * 3 from now - 30 s to now"))
    # print(dataset.query("max(mean {e2} from now - 30 seconds to now, mean {e1} from now - 30 seconds to now) at every {e2} from {start} to {end}"))