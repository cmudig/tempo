import pandas as pd
import numpy as np
import random
from query_language.numba_functions import *
from numba.typed import List
from numba import njit

def make_aligned_value_series(value_set, other):
    """value_set must have get_ids() and get_values()"""
    if isinstance(other, Attributes):
        # Merge on the id field
        broadcast_attrs = pd.merge(pd.DataFrame({"id": value_set.get_ids()}), other.series,
                                    left_on="id", right_index=True,
                                    how='left')
        return broadcast_attrs[other.name]
    elif hasattr(other, "get_values"):
        if len(other.get_values()) != len(value_set.get_values()):
            raise ValueError(f"Event sets must be same length")
        return other.get_values()
    elif isinstance(other, Duration):
        return other.value()
    elif isinstance(other, pd.DataFrame):
        raise ValueError("Can't perform binary operations on Events with a DataFrame")
    return other

def compress_series(v):
    if pd.api.types.is_object_dtype(v.dtype) or isinstance(v.dtype, pd.CategoricalDtype):
        # Convert category types if needed
        if len(v.unique()) < len(v) * 0.5:
            v = v.astype("category")
            
        if isinstance(v.dtype, pd.CategoricalDtype) and pd.api.types.is_object_dtype(v.dtype.categories.dtype):
            try:
                v = v.cat.rename_categories(v.dtype.categories.astype(int))
            except ValueError:
                pass
            
        # Only return the categorical version if the categories are non-numeric
        if not (isinstance(v.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(v.dtype.categories.dtype)):
            return v
    isnan = pd.isna(v)
    try:
        if np.array_equal(v[~isnan], v[~isnan].astype(int)):
            has_nans = isnan.sum() > 0
            if v.min() >= 0 and v.max() < 2**8:
                return v.astype(pd.UInt8Dtype() if has_nans else np.uint8)
            elif v.abs().max() < 2**7:
                return v.astype(pd.Int8Dtype() if has_nans else np.int8)
            if v.min() >= 0 and v.max() < 2**16:
                return v.astype(pd.UInt16Dtype() if has_nans else np.uint16)
            elif v.abs().max() < 2**15:
                return v.astype(pd.Int16Dtype() if has_nans else np.int16)
            return v.astype(pd.Int64Dtype() if has_nans else np.int64)
    except:
        pass
    if pd.api.types.is_numeric_dtype(v):
        return v.astype(np.float32)
    return v

    
EXCLUDE_SERIES_METHODS = ("_repr_latex_",)

class TimeSeriesQueryable:
    """Base class for time-series data structures"""
    @staticmethod
    def deserialize(metadata, df, **kwargs):
        assert "type" in metadata, "Serialized time series information must have a 'type' key"
        if metadata["type"] == "Attributes":
            return Attributes.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "AttributeSet":
            return AttributeSet.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "Events":
            return Events.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "EventSet":
            return EventSet.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "Intervals":
            return Intervals.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "IntervalSet":
            return IntervalSet.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "TimeIndex":
            return TimeIndex.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "TimeSeries":
            return TimeSeries.deserialize(metadata, df, **kwargs)
        elif metadata["type"] == "TimeSeriesSet":
            return TimeSeriesSet.deserialize(metadata, df, **kwargs)
        else:
            raise ValueError(f"Unknown serialization type '{metadata['type']}'")

class Compilable:
    """
    A wrapper around a data series (Attributes, Events, or Intervals) that saves
    a compute graph when operations are called on it.
    """
    def __init__(self, data_or_fn, name=None, leaves=None):
        if isinstance(data_or_fn, str):
            self.fn = data_or_fn
            self.data = None
            self.leaves = leaves if leaves is not None else {}
        elif isinstance(data_or_fn, (float, int, np.number)):
            self.data = data_or_fn
            self.name = None
            self.leaves = {}
        else:
            self.data = data_or_fn
            if name is None: name = 'var_' + ('%015x' % random.randrange(16**15))
            self.name = name
            self.leaves = {name: self}
            
    @staticmethod
    def wrap(data):
        if isinstance(data, str):
            # Make sure this gets inserted as a string LITERAL, not a variable
            return Compilable(repr(data))
        return Compilable(data)
        
    def function_string(self):
        if self.data is not None: return self.name if self.name is not None else self.data
        else: return self.fn
        
    def mono_parent(self, string):
        return Compilable(string, leaves=self.leaves)
    
    def execute(self):
        fn = self.get_executable_function()
        inputs = {v: self.leaves[v].data for v in self.leaves}
        return fn(**inputs)
    
    def get_executable_function(self):
        """
        Returns a tuple (fn, args), where fn is a function that can be called
        with the given list of arguments to return the computed value of the
        expression.
        """
        args = [f"{k}=None" for k in self.leaves.keys()]
        results = {}
        exec(f"def compiled_fn({', '.join(args)}): return {self.function_string()}", 
             globals(), results)
        return results["compiled_fn"]
        
    def bin_aggregate(self, index, start_times, end_times, agg_type, agg_func):
        """
        Performs an aggregation within given time bins. Since the element being
        aggregated is not yet computed (within a Compilable instance), each
        leaf element of the compiled expression must either be a pre-aggregated
        series with the SAME time index as the current one, or an Events/Intervals
        instance. All non-preaggregated series must be of the same type and have
        the same time values.
        
        index: TimeIndex to use as the master time index
        start_times, end_times: TimeIndex objects
        agg_func: string name or function to use to aggregate values
        """
        agg_func = agg_func.lower()
        ids = start_times.get_ids()
        assert (ids == end_times.get_ids()).all(), "Start times and end times must have equal sets of IDs"
        starts = np.array(start_times.get_times(), dtype=np.int64)
        ends = np.array(end_times.get_times(), dtype=np.int64)
        assert (starts <= ends).all(), "Start times must be <= end times"
        
        # TODO: This method of combining inputs in matrices only works when all
        # inputs are numerical. If a preaggregated input is a string, the line
        # marked below will crash; if a different input is a string, it will be
        # silently converted to a number and carried through to numba as a number,
        # meaning that if different columns are converted to numbers differently,
        # equality operations between them may not be correct.
        preaggregated_input_names = []
        preaggregated_inputs = []
        series_input_names = []
        series_inputs = None
        result_name = None
        series_type = None # None, "events" or "intervals"
        for name, value in self.leaves.items():
            value = value.data
            if isinstance(value, TimeSeries):
                assert value.index == index, "TimeSeries inside aggregation expression does not match current aggregation index"
                if pd.api.types.is_object_dtype(value.get_values().values.dtype) or pd.api.types.is_string_dtype(value.get_values().values.dtype):
                    raise NotImplementedError("Nested aggregations currently only work on numerical values")
                preaggregated_input_names.append(name)
                # This line crashes when the preaggregated series values are strings
                preaggregated_inputs.append(value.get_values().values.astype(np.float64).reshape(-1, 1))
            else:
                series_input_names.append(name)
                if isinstance(value, Events):
                    if series_type != None and series_type != "events":
                        raise ValueError("Cannot have both un-aggregated Events and Intervals inside an aggregation expression")
                    series_type = "events"
                elif isinstance(value, Intervals):
                    if series_type != None and series_type != "intervals":
                        raise ValueError("Cannot have both un-aggregated Events and Intervals inside an aggregation expression")
                    series_type = "intervals"
                else:
                    raise ValueError(f"Unsupported aggregation expression type {str(type(value))}")
                
                if pd.api.types.is_object_dtype(value.get_values().values.dtype) or pd.api.types.is_string_dtype(value.get_values().values.dtype):
                    raise NotImplementedError("Nested aggregations currently only work on numerical values")
                
                if series_inputs is None:
                    series_inputs = value.prepare_aggregation_inputs(agg_func)
                    series_inputs = (*series_inputs[:-2], series_inputs[-2].reshape(-1, 1))
                else:
                    new_series_inputs = value.prepare_aggregation_inputs(agg_func)
                    assert new_series_inputs[0].equals(series_inputs[0]), "IDs do not match among unaggregated expressions"
                    assert new_series_inputs[1].equals(series_inputs[1]), "Times do not match among unaggregated expressions"
                    if isinstance(value, Intervals):
                        assert new_series_inputs[2].equals(series_inputs[2]), "Times do not match among unaggregated expressions"
                    series_inputs = (*series_inputs[:-1],
                                    np.hstack([series_inputs[-1], new_series_inputs[-2].reshape(-1, 1)]))
                    
                if result_name is None: result_name = value.name
                
        preaggregated_inputs = np.hstack(preaggregated_inputs)
        compiled_fn = njit()(self.get_executable_function())
        lcls = {}
        arg_assignments = ([f"{n}=preagg[{i}]" for i, n in enumerate(preaggregated_input_names)] + 
                           [f"{n}=series_vals[:,{i}]" for i, n in enumerate(series_input_names)])
        exec(f"def wrapped_fn(fn): return lambda series_vals, preagg: fn({', '.join(arg_assignments)})", globals(), lcls)
        wrapped_fn = lcls['wrapped_fn']
        compiled_fn = njit()(wrapped_fn(compiled_fn))
        
        if series_type == "events":
            grouped_values = numba_join_events_dynamic(List(ids.values.tolist()),
                                                starts, 
                                                ends, 
                                                compiled_fn,
                                                series_inputs[0].values, 
                                                series_inputs[1].values,
                                                series_inputs[2],
                                                preaggregated_inputs,
                                                AGG_FUNCTIONS[agg_func])
            grouped_values = convert_numba_result_dtype(grouped_values, agg_func)
        elif series_type  == "intervals":
            grouped_values = numba_join_intervals_dynamic(List(ids.values.tolist()),
                                                starts, 
                                                ends, 
                                                compiled_fn,
                                                series_inputs[0].values, 
                                                series_inputs[1].values,
                                                series_inputs[2].values,
                                                series_inputs[3],
                                                preaggregated_inputs,
                                                agg_type,
                                                AGG_FUNCTIONS[agg_func])
            grouped_values = convert_numba_result_dtype(grouped_values, agg_func)
            
        # TODO do we need to convert back categorical types?
        
        assert len(grouped_values) == len(index)
        
        return TimeSeries(index, pd.Series(grouped_values, name=result_name or "aggregated_series").convert_dtypes())
        
    def where(self, condition, other):
        if not isinstance(other, Compilable):
            other = Compilable.wrap(other)
        if not isinstance(condition, Compilable):
            condition = Compilable.wrap(condition)
            
        return Compilable(f"np.where({condition.function_string()}, {self.function_string()}, {other.function_string()})",
                          leaves={**self.leaves, **condition.leaves, **other.leaves})
    
    def filter(self, condition):
        if not isinstance(condition, Compilable):
            condition = Compilable.wrap(condition)
            
        return Compilable(f"({self.function_string()})[{condition.function_string()}]",
                          leaves={**self.leaves, **condition.leaves})
    
    def impute(self, method='mean', constant_value=None):
        if method == 'constant':
            return self.mono_parent(f"np.where(np.isnan({self.function_string()}), {constant_value}, {self.function_string()})")
        return self.mono_parent(f"np.where(np.isnan({self.function_string()}), np.nan{method}({self.function_string()}), {self.function_string()})")
    
    def __abs__(self): return self.mono_parent(f"abs({self.function_string()})")
    def __neg__(self): return self.mono_parent(f"-({self.function_string()})")
    def __pos__(self): return self.mono_parent(f"+({self.function_string()})")
    def __invert__(self): return self.mono_parent(f"~({self.function_string()})")
    
    def _handle_binary_op(self, opname, other, reverse=False):
        if not isinstance(other, Compilable):
            other = Compilable.wrap(other)
            
        fn_strings = (self.function_string(), other.function_string())
        if reverse: fn_strings = fn_strings[1], fn_strings[0]
        return Compilable(f"({fn_strings[0]}) {opname} ({fn_strings[1]})",
                          leaves={**self.leaves, **other.leaves})
        
    def __eq__(self, other): return self._handle_binary_op("==", other)
    def __ge__(self, other): return self._handle_binary_op(">=", other)
    def __gt__(self, other): return self._handle_binary_op(">", other)
    def __le__(self, other): return self._handle_binary_op("<=", other)
    def __ne__(self, other): return self._handle_binary_op("!=", other)
    def __lt__(self, other): return self._handle_binary_op("<", other)
    
    def __add__(self, other): return self._handle_binary_op("+", other)
    def __and__(self, other): return self._handle_binary_op("and", other)
    def __floordiv__(self, other): return self._handle_binary_op("//", other)
    def __mod__(self, other): return self._handle_binary_op("%", other)
    def __mul__(self, other): return self._handle_binary_op("*", other)
    def __or__(self, other): return self._handle_binary_op("or", other)
    def __pow__(self, other): return self._handle_binary_op("**", other)
    def __sub__(self, other): return self._handle_binary_op("-", other)
    def __truediv__(self, other): return self._handle_binary_op("/", other)
    def __xor__(self, other): return self._handle_binary_op("^", other)

    def __radd__(self, other): return self._handle_binary_op("+", other, reverse=True)
    def __rand__(self, other): return self._handle_binary_op("and", other, reverse=True)
    def __rdiv__(self, other): return self._handle_binary_op("/", other, reverse=True)
    def __rfloordiv__(self, other): return self._handle_binary_op("//", other, reverse=True)
    def __rmatmul__(self, other): return self._handle_binary_op("@", other, reverse=True)
    def __rmod__(self, other): return self._handle_binary_op("%", other, reverse=True)
    def __rmul__(self, other): return self._handle_binary_op("*", other, reverse=True)
    def __ror__(self, other): return self._handle_binary_op("or", other, reverse=True)
    def __rpow__(self, other): return self._handle_binary_op("**", other, reverse=True)
    def __rsub__(self, other): return self._handle_binary_op("-", other, reverse=True)
    def __rtruediv__(self, other): return self._handle_binary_op("/", other, reverse=True)
    def __rxor__(self, other): return self._handle_binary_op("^", other, reverse=True)
        

class Attributes(TimeSeriesQueryable):
    def __init__(self, series):
        """The series' index should be the set of instance IDs"""
        self.series = series
        self.name = self.series.name
        
    def __repr__(self):
        return f"<Attributes '{self.name}': {len(self.series)} values>\n{repr(self.series)}"
    
    def __len__(self): return len(self.series)
    
    def get_ids(self):
        return self.series.index
    
    def get_values(self):
        return self.series
    
    def compress(self):
        """Returns a new TimeSeries with values compressed to the minimum size
        needed to represent them."""
        return self.with_values(compress_series(self.get_values()))
        
    def with_values(self, new_values):
        return Attributes(pd.Series(new_values, index=self.series.index, name=self.series.name))
    
    def serialize(self):
        return {"type": "Attributes", "name": self.name}, pd.DataFrame(self.series)
    
    @staticmethod
    def deserialize(metadata, df):
        return Attributes(df[df.columns[0]])
    
    def filter(self, mask):
        """Returns a new Attributes with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return Attributes(self.series[mask])
        
    def __getattr__(self, name):
        if hasattr(self.series, name) and name not in EXCLUDE_SERIES_METHODS:
            pd_method = getattr(self.series, name)
            if callable(pd_method):
                def wrap_pandas_method(*args, **kwargs):
                    args = [a.get_values() if hasattr(a, "get_values") else a for a in args]
                    kwargs = {k: v.get_values() if hasattr(v, "get_values") else v for k, v in kwargs.items()}
                    result = pd_method(*args, **kwargs)
                    if isinstance(result, pd.Series) and (self.series.index == result.index).all():
                        return Attributes(result)
                    if isinstance(result, pd.Series):
                        raise ValueError(f"Cannot complete pandas method call '{name}' on {type(self)} because it returned a Series that isn't aligned with the original Series.")
                    return result
                return wrap_pandas_method
        raise AttributeError(name)
    
    def preserve_nans(self, new_values):
        return new_values.where(~pd.isna(self.series), pd.NA)
        
    def __abs__(self): return Attributes(self.preserve_nans(self.series.__abs__()))
    def __neg__(self): return Attributes(self.preserve_nans(self.series.__neg__()))
    def __pos__(self): return Attributes(self.preserve_nans(self.series.__pos__()))
    def __invert__(self): return Attributes(self.preserve_nans(self.series.__invert__()))
    
    def _handle_binary_op(self, opname, other):
        if isinstance(other, (Events, Intervals, TimeIndex, TimeSeries, Compilable)):
            return NotImplemented
        if isinstance(other, Attributes):
            return Attributes(self.preserve_nans(getattr(self.series, opname)(other.series).rename(self.name)))
        if isinstance(other, Duration):
            return Attributes(self.preserve_nans(getattr(self.series, opname)(other.value()).rename(self.name)))
        return Attributes(self.preserve_nans(getattr(self.series, opname)(other)))
        
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)
    
    def __add__(self, other): return self._handle_binary_op("__add__", other)
    def __and__(self, other): return self._handle_binary_op("__and__", other)
    def __floordiv__(self, other): return self._handle_binary_op("__floordiv__", other)
    def __mod__(self, other): return self._handle_binary_op("__mod__", other)
    def __mul__(self, other): return self._handle_binary_op("__mul__", other)
    def __or__(self, other): return self._handle_binary_op("__or__", other)
    def __pow__(self, other): return self._handle_binary_op("__pow__", other)
    def __sub__(self, other): return self._handle_binary_op("__sub__", other)
    def __truediv__(self, other): return self._handle_binary_op("__truediv__", other)
    def __xor__(self, other): return self._handle_binary_op("__xor__", other)

    def __radd__(self, other): return self._handle_binary_op("__radd__", other)
    def __rand__(self, other): return self._handle_binary_op("__rand__", other)
    def __rdiv__(self, other): return self._handle_binary_op("__rdiv__", other)
    def __rfloordiv__(self, other): return self._handle_binary_op("__rfloordiv__", other)
    def __rmatmul__(self, other): return self._handle_binary_op("__rmatmul__", other)
    def __rmod__(self, other): return self._handle_binary_op("__rmod__", other)
    def __rmul__(self, other): return self._handle_binary_op("__rmul__", other)
    def __ror__(self, other): return self._handle_binary_op("__ror__", other)
    def __rpow__(self, other): return self._handle_binary_op("__rpow__", other)
    def __rsub__(self, other): return self._handle_binary_op("__rsub__", other)
    def __rtruediv__(self, other): return self._handle_binary_op("__rtruediv__", other)
    def __rxor__(self, other): return self._handle_binary_op("__rxor__", other)

class AttributeSet(TimeSeriesQueryable):
    def __init__(self, df):
        """The df's index should be the set of instance IDs"""
        self.df = df.sort_index(kind='stable')
        
    def serialize(self):
        return {"type": "AttributeSet"}, self.df
    
    @staticmethod
    def deserialize(metadata, df):
        return AttributeSet(df)
        
    def get_ids(self):
        return self.df.index
    
    def filter(self, mask):
        """Returns a new AttributeSet with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return AttributeSet(self.df[mask])
        
    def has(self, attribute_name): return attribute_name in self.df.columns
    
    def get(self, attribute_name):
        return Attributes(self.df[attribute_name])
    
    def __repr__(self):
        return f"<AttributeSet: {len(self.df)} rows, {self.df.shape[1]} attributes>"
    
class Events(TimeSeriesQueryable):
    def __init__(self, df, type_field="eventtype", time_field="time", value_field="value", id_field="id", name=None):
        self.df = df
        self.type_field = type_field
        self.time_field = time_field
        self.id_field = id_field
        self.value_field = value_field
        self.event_types = self.df[type_field].unique()
        # Convert types if needed
        if pd.api.types.is_string_dtype(self.df[self.value_field].dtype):
            new_values = pd.to_numeric(self.df[self.value_field], errors='coerce')
            if (pd.isna(new_values) == pd.isna(self.df[self.value_field])).all():
                self.df = self.df.assign(**{self.value_field: new_values})
            
        if name is None:
            self.name = ', '.join(self.event_types)
        else:
            self.name = name
        
    def serialize(self):
        return {
            "type": "Events", 
            "type_field": self.type_field,
            "time_field": self.time_field,
            "value_field": self.value_field,
            "id_field": self.id_field,
            "name": self.name
        }, self.df
    
    @staticmethod
    def deserialize(metadata, df):
        return Events(df,
                      type_field=metadata["type_field"],
                      time_field=metadata["time_field"],
                      value_field=metadata["value_field"],
                      id_field=metadata["id_field"],
                      name=metadata["name"])
        
    def filter(self, mask):
        """Returns a new Events with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return Events(self.df[mask],
                      type_field=self.type_field,
                      time_field=self.time_field,
                      id_field=self.id_field,
                      value_field=self.value_field,
                      name=self.name)
        
    def __repr__(self):
        return f"<Events '{self.name}': {len(self.df)} values>\n{repr(self.df)}"
    
    def __len__(self):
        return len(self.df)
    
    def get_ids(self): return self.df[self.id_field]
    def get_types(self): return self.df[self.type_field]
    def get_times(self): return self.df[self.time_field]
    def get_values(self): return self.df[self.value_field]
    
    def preserve_nans(self, new_values):
        return new_values.where(~pd.isna(self.get_values()), pd.NA)
        
    def __getattr__(self, name):
        value_series = self.df[self.value_field]
        if hasattr(value_series, name) and name not in EXCLUDE_SERIES_METHODS:
            pd_method = getattr(value_series, name)
            if callable(pd_method):
                def wrap_pandas_method(*args, **kwargs):
                    args = [make_aligned_value_series(self, a) if isinstance(a, TimeSeriesQueryable) else a for a in args]
                    kwargs = {k: make_aligned_value_series(self, v) if isinstance(v, TimeSeriesQueryable) else v for k, v in kwargs.items()}
                    result = pd_method(*args, **kwargs)
                    if isinstance(result, pd.Series) and (value_series.index == result.index).all():
                        return self.with_values(result)
                    if isinstance(result, pd.Series):
                        raise ValueError(f"Cannot complete pandas method call '{name}' on {type(self)} because it returned a Series that isn't aligned with the original Series.")
                    return result
                return wrap_pandas_method
        raise AttributeError(name)

    def aggregate(self, start_times, end_times, agg_func):
        """
        Performs an aggregation that returns a single value per ID. Returns an
        Attributes object.
        """
        result = self.bin_aggregate(start_times, start_times, end_times, agg_func)
        return Attributes(result.series.set_axis(result.index.get_ids()))
        
    def prepare_aggregation_inputs(self, agg_func):
        event_ids = self.df[self.id_field]
        event_times = self.df[self.time_field].astype(np.float64)
        event_values = self.df[self.value_field]
        if isinstance(event_values.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(event_values.dtype):
            # Convert to numbers before using numba
            if agg_func not in CATEGORICAL_SUPPORT_AGG_FUNCTIONS:
                raise ValueError(f"Cannot use agg_func {agg_func} on categorical data")
            event_values, uniques = pd.factorize(event_values)
            event_values = np.where(pd.isna(event_values), np.nan, event_values).astype(np.float64)
        else:
            event_values = event_values.values.astype(np.float64)
            uniques = None
        
        return event_ids, event_times, event_values, uniques
        
    def bin_aggregate(self, index, start_times, end_times, agg_func):
        """
        Performs an aggregation within given time bins.
        
        index: TimeIndex to use as the master time index
        start_times, end_times: TimeIndex objects
        agg_func: string name or function to use to aggregate values
        """
        agg_func = agg_func.lower()
        ids = start_times.get_ids()
        assert (ids == end_times.get_ids()).all(), "Start times and end times must have equal sets of IDs"
        starts = np.array(start_times.get_times(), dtype=np.int64)
        ends = np.array(end_times.get_times(), dtype=np.int64)
        assert (starts <= ends).all(), "Start times must be <= end times"
        
        event_ids, event_times, event_values, uniques = self.prepare_aggregation_inputs(agg_func)
        
        grouped_values = numba_join_events(List(ids.values.tolist()),
                                             starts, 
                                             ends, 
                                             event_ids.values, 
                                             event_times.values,
                                             event_values,
                                             AGG_FUNCTIONS[agg_func])
        grouped_values = convert_numba_result_dtype(grouped_values, agg_func)
        
        if uniques is not None and agg_func in TYPE_PRESERVING_AGG_FUNCTIONS:
            grouped_values = np.where(np.isnan(grouped_values), 
                                    np.nan, 
                                    uniques[np.where(np.isnan(grouped_values), -1, grouped_values).astype(int)])
        
        assert len(grouped_values) == len(index)
        return TimeSeries(index, pd.Series(grouped_values, name=self.name).convert_dtypes())
        
    def compress(self):
        """Returns a new TimeSeries with values compressed to the minimum size
        needed to represent them."""
        return self.with_values(compress_series(self.get_values()))
        
    def with_values(self, new_values, preserve_nans=False):
        return Events(self.df.assign(**{self.value_field: self.preserve_nans(new_values) if preserve_nans else new_values}),
                      type_field=self.type_field,
                      time_field=self.time_field,
                      id_field=self.id_field,
                      value_field=self.value_field,
                      name=self.name)
        
    def __abs__(self): return self.with_values(self.df[self.value_field].__abs__(), preserve_nans=True)
    def __neg__(self): return self.with_values(self.df[self.value_field].__neg__(), preserve_nans=True)
    def __pos__(self): return self.with_values(self.df[self.value_field].__pos__(), preserve_nans=True)
    def __invert__(self): return self.with_values(self.df[self.value_field].__invert__(), preserve_nans=True)

    def _handle_binary_op(self, opname, other):
        if isinstance(other, Compilable): return NotImplemented
        return self.with_values(getattr(self.df[self.value_field], opname)(make_aligned_value_series(self, other)), preserve_nans=True)
    
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)
    
    def __add__(self, other): return self._handle_binary_op("__add__", other)
    def __and__(self, other): return self._handle_binary_op("__and__", other)
    def __floordiv__(self, other): return self._handle_binary_op("__floordiv__", other)
    def __mod__(self, other): return self._handle_binary_op("__mod__", other)
    def __mul__(self, other): return self._handle_binary_op("__mul__", other)
    def __or__(self, other): return self._handle_binary_op("__or__", other)
    def __pow__(self, other): return self._handle_binary_op("__pow__", other)
    def __sub__(self, other): return self._handle_binary_op("__sub__", other)
    def __truediv__(self, other): return self._handle_binary_op("__truediv__", other)
    def __xor__(self, other): return self._handle_binary_op("__xor__", other)

    def __radd__(self, other): return self._handle_binary_op("__radd__", other)
    def __rand__(self, other): return self._handle_binary_op("__rand__", other)
    def __rdiv__(self, other): return self._handle_binary_op("__rdiv__", other)
    def __rfloordiv__(self, other): return self._handle_binary_op("__rfloordiv__", other)
    def __rmatmul__(self, other): return self._handle_binary_op("__rmatmul__", other)
    def __rmod__(self, other): return self._handle_binary_op("__rmod__", other)
    def __rmul__(self, other): return self._handle_binary_op("__rmul__", other)
    def __ror__(self, other): return self._handle_binary_op("__ror__", other)
    def __rpow__(self, other): return self._handle_binary_op("__rpow__", other)
    def __rsub__(self, other): return self._handle_binary_op("__rsub__", other)
    def __rtruediv__(self, other): return self._handle_binary_op("__rtruediv__", other)
    def __rxor__(self, other): return self._handle_binary_op("__rxor__", other)

    
class EventSet(TimeSeriesQueryable):
    def __init__(self, df, type_field="eventtype", time_field="time", id_field="id", value_field="value"):
        self.df = df.sort_values([id_field, time_field], kind='stable')
        self.type_field = type_field
        self.time_field = time_field
        self.id_field = id_field
        self.value_field = value_field
        
    def get_ids(self): return self.df[self.id_field]
    def get_types(self): return self.df[self.type_field]
    def get_times(self): return self.df[self.time_field]
    def get_values(self): return self.df[self.value_field]
    
    def serialize(self):
        return {
            "type": "EventSet", 
            "type_field": self.type_field,
            "time_field": self.time_field,
            "value_field": self.value_field,
            "id_field": self.id_field,
        }, self.df
    
    @staticmethod
    def deserialize(metadata, df):
        return EventSet(df,
                      type_field=metadata["type_field"],
                      time_field=metadata["time_field"],
                      value_field=metadata["value_field"],
                      id_field=metadata["id_field"])
        
    def filter(self, mask):
        """Returns a new EventSet with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return EventSet(self.df[mask],
                      type_field=self.type_field,
                      time_field=self.time_field,
                      id_field=self.id_field,
                      value_field=self.value_field)
        
    def has(self, eventtype): return (self.df[self.type_field] == eventtype).sum() > 0
    
    def get(self, eventtype, name=None):
        new_df = self.df[(self.df[self.type_field] == eventtype) if isinstance(eventtype, str) else (self.df[self.type_field].isin(eventtype))].copy()
        try: new_df = new_df.assign(**{self.value_field: new_df[self.value_field].astype(np.float64)})
        except: pass
        return Events(new_df,
                      type_field=self.type_field, 
                      time_field=self.time_field,
                      value_field=self.value_field,
                      id_field=self.id_field,
                      name=name or (eventtype if isinstance(eventtype, str) else ", ".join(eventtype)))
        
    def __repr__(self):
        return f"<EventSet: {len(self.df)} rows, {len(self.df[self.type_field].unique())} event types>"
        
class Intervals(TimeSeriesQueryable):
    def __init__(self, df, type_field="intervaltype", start_time_field="starttime", end_time_field="endtime", value_field="value", id_field="id", name=None):
        self.df = df
        self.type_field = type_field
        self.start_time_field = start_time_field
        self.end_time_field = end_time_field
        self.id_field = id_field
        self.value_field = value_field
        self.event_types = self.df[type_field].unique()
        # Convert types if needed
        if pd.api.types.is_string_dtype(self.df[self.value_field].dtype):
            new_values = pd.to_numeric(self.df[self.value_field], errors='coerce')
            if (pd.isna(new_values) == pd.isna(self.df[self.value_field])).all():
                self.df = self.df.assign(**{self.value_field: new_values})
        if name is None:
            self.name = ', '.join(self.event_types)
        else:
            self.name = name

    def serialize(self):
        return {
            "type": "Intervals", 
            "type_field": self.type_field,
            "start_time_field": self.start_time_field,
            "end_time_field": self.end_time_field,
            "value_field": self.value_field,
            "id_field": self.id_field,
            "name": self.name
        }, self.df
    
    @staticmethod
    def deserialize(metadata, df):
        return Intervals(df,
                      type_field=metadata["type_field"],
                      start_time_field=metadata["start_time_field"],
                      end_time_field=metadata["end_time_field"],
                      value_field=metadata["value_field"],
                      id_field=metadata["id_field"],
                      name=metadata["name"])
        
    def filter(self, mask):
        """Returns a new Intervals with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return Intervals(self.df[mask],
                      type_field=self.type_field,
                      start_time_field=self.start_time_field,
                      end_time_field=self.end_time_field,
                      id_field=self.id_field,
                      value_field=self.value_field,
                      name=self.name)
        
    def __len__(self): return len(self.df)
    
    def get_ids(self):
        return self.df[self.id_field]

    def get_values(self):
        return self.df[self.value_field]
    
    def get_start_times(self):
        return self.df[self.start_time_field]
    
    def get_end_times(self):
        return self.df[self.end_time_field]
    
    def start_events(self):
        """returns an Events where the time is the start time of each interval"""
        return Events(self.df.drop(columns=[self.end_time_field]),
                      type_field=self.type_field,
                      time_field=self.start_time_field,
                      value_field=self.value_field,
                      id_field=self.id_field,
                      name=self.name)
        
    def end_events(self):
        """returns an Events where the time is the end time of each interval"""
        return Events(self.df.drop(columns=[self.start_time_field]),
                      type_field=self.type_field,
                      time_field=self.end_time_field,
                      value_field=self.value_field,
                      id_field=self.id_field,
                      name=self.name)
        
    def compress(self):
        """Returns a new TimeSeries with values compressed to the minimum size
        needed to represent them."""
        return self.with_values(compress_series(self.get_values()))
        
    def __repr__(self):
        return f"<Intervals '{self.name}': {len(self.df)} values>\n{repr(self.df)}"
    
    def preserve_nans(self, new_values):
        return new_values.where(~pd.isna(self.get_values()), pd.NA)
        
    def __getattr__(self, name):
        value_series = self.df[self.value_field]
        if hasattr(value_series, name) and name not in EXCLUDE_SERIES_METHODS:
            pd_method = getattr(value_series, name)
            if callable(pd_method):
                def wrap_pandas_method(*args, **kwargs):
                    args = [make_aligned_value_series(self, a) if isinstance(a, TimeSeriesQueryable) else a for a in args]
                    kwargs = {k: make_aligned_value_series(self, v) if isinstance(v, TimeSeriesQueryable) else v for k, v in kwargs.items()}
                    result = pd_method(*args, **kwargs)
                    if isinstance(result, pd.Series) and (value_series.index == result.index).all():
                        return self.with_values(result)
                    if isinstance(result, pd.Series):
                        raise ValueError(f"Cannot complete pandas method call '{name}' on {type(self)} because it returned a Series that isn't aligned with the original Series.")
                    return result
                return wrap_pandas_method
        raise AttributeError(name)
    
    def aggregate(self, start_times, end_times, agg_type, agg_func):
        """
        Performs an aggregation that returns a single value per ID. Returns an
        Attributes object.
        """
        result = self.bin_aggregate(start_times, start_times, end_times, agg_type, agg_func)
        return Attributes(result.series.set_axis(result.index.get_ids()))
        
    def prepare_aggregation_inputs(self, agg_func):
        event_ids = self.df[self.id_field]
        interval_starts = self.df[self.start_time_field].astype(np.float64)
        interval_ends = self.df[self.end_time_field].astype(np.float64)
        interval_values = self.df[self.value_field]
        
        if isinstance(interval_values.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(interval_values.dtype):
            # Convert to numbers before using numba
            if agg_func not in CATEGORICAL_SUPPORT_AGG_FUNCTIONS:
                raise ValueError(f"Cannot use agg_func {agg_func} on categorical data")
            interval_values, uniques = pd.factorize(interval_values)
            interval_values = np.where(pd.isna(interval_values), np.nan, interval_values).astype(np.float64)
        else:
            interval_values = interval_values.values
            uniques = None
        
        return event_ids, interval_starts, interval_ends, interval_values, uniques
        
    def bin_aggregate(self, index, start_times, end_times, agg_type, agg_func):
        """
        index: TimeIndex to use as the master time index
        start_times, end_times: TimeIndex objects
        agg_type: either "value", "amount", "rate", or "duration" - determines how value
            will be used
        agg_func: string name or function to use to aggregate values. "integral"
            on a "rate" agg_type specifies that the values should be multiplied
            by the time interval length
        """
        agg_func = agg_func.lower()
        ids = start_times.get_ids()
        assert (ids == end_times.get_ids()).all(), "Start times and end times must have equal sets of IDs"
        starts = np.array(start_times.get_times(), dtype=np.int64)
        ends = np.array(end_times.get_times(), dtype=np.int64)
        assert (starts <= ends).all(), "Start times must be <= end times"
        
        event_ids, interval_starts, interval_ends, interval_values, uniques = self.prepare_aggregation_inputs(agg_func)
        
        grouped_values = numba_join_intervals(List(ids.values.tolist()),
                                             starts, 
                                             ends, 
                                             event_ids.values, 
                                             interval_starts.values,
                                             interval_ends.values,
                                             interval_values,
                                             agg_type.lower(),
                                             AGG_FUNCTIONS[agg_func])
        grouped_values = convert_numba_result_dtype(grouped_values, agg_func)
        
        if uniques is not None and agg_func in TYPE_PRESERVING_AGG_FUNCTIONS:
            grouped_values = np.where(np.isnan(grouped_values), 
                                    np.nan, 
                                    uniques[np.where(np.isnan(grouped_values), -1, grouped_values).astype(int)])
        
        assert len(grouped_values) == len(index)
        return TimeSeries(index, pd.Series(grouped_values, name=self.name).replace(np.nan, pd.NA))
    
    def with_values(self, new_values, preserve_nans=False):
        return Intervals(self.df.assign(**{self.value_field: self.preserve_nans(new_values) if preserve_nans else new_values}),
                      type_field=self.type_field,
                      start_time_field=self.start_time_field,
                      end_time_field=self.end_time_field,
                      id_field=self.id_field,
                      value_field=self.value_field,
                      name=self.name)
        
    def __abs__(self): return self.with_values(self.df[self.value_field].__abs__(), preserve_nans=True)
    def __neg__(self): return self.with_values(self.df[self.value_field].__neg__(), preserve_nans=True)
    def __pos__(self): return self.with_values(self.df[self.value_field].__pos__(), preserve_nans=True)
    def __invert__(self): return self.with_values(self.df[self.value_field].__invert__(), preserve_nans=True)

    def _handle_binary_op(self, opname, other):
        if isinstance(other, Compilable): return NotImplemented
        return self.with_values(getattr(self.df[self.value_field], opname)(make_aligned_value_series(self, other)), preserve_nans=True)
    
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)
    
    def __add__(self, other): return self._handle_binary_op("__add__", other)
    def __and__(self, other): return self._handle_binary_op("__and__", other)
    def __floordiv__(self, other): return self._handle_binary_op("__floordiv__", other)
    def __mod__(self, other): return self._handle_binary_op("__mod__", other)
    def __mul__(self, other): return self._handle_binary_op("__mul__", other)
    def __or__(self, other): return self._handle_binary_op("__or__", other)
    def __pow__(self, other): return self._handle_binary_op("__pow__", other)
    def __sub__(self, other): return self._handle_binary_op("__sub__", other)
    def __truediv__(self, other): return self._handle_binary_op("__truediv__", other)
    def __xor__(self, other): return self._handle_binary_op("__xor__", other)

    def __radd__(self, other): return self._handle_binary_op("__radd__", other)
    def __rand__(self, other): return self._handle_binary_op("__rand__", other)
    def __rdiv__(self, other): return self._handle_binary_op("__rdiv__", other)
    def __rfloordiv__(self, other): return self._handle_binary_op("__rfloordiv__", other)
    def __rmatmul__(self, other): return self._handle_binary_op("__rmatmul__", other)
    def __rmod__(self, other): return self._handle_binary_op("__rmod__", other)
    def __rmul__(self, other): return self._handle_binary_op("__rmul__", other)
    def __ror__(self, other): return self._handle_binary_op("__ror__", other)
    def __rpow__(self, other): return self._handle_binary_op("__rpow__", other)
    def __rsub__(self, other): return self._handle_binary_op("__rsub__", other)
    def __rtruediv__(self, other): return self._handle_binary_op("__rtruediv__", other)
    def __rxor__(self, other): return self._handle_binary_op("__rxor__", other)
    

class IntervalSet(TimeSeriesQueryable):
    def __init__(self, df, type_field="intervaltype", start_time_field="starttime", end_time_field="endtime", value_field="value", id_field="id"):
        self.df = df.sort_values([id_field, start_time_field], kind='stable')
        self.type_field = type_field
        self.start_time_field = start_time_field
        self.end_time_field = end_time_field
        self.value_field = value_field
        self.id_field = id_field
        
    def serialize(self):
        return {
            "type": "IntervalSet", 
            "type_field": self.type_field,
            "start_time_field": self.start_time_field,
            "end_time_field": self.end_time_field,
            "value_field": self.value_field,
            "id_field": self.id_field,
        }, self.df
    
    @staticmethod
    def deserialize(metadata, df):
        return IntervalSet(df,
                      type_field=metadata["type_field"],
                      start_time_field=metadata["start_time_field"],
                      end_time_field=metadata["end_time_field"],
                      value_field=metadata["value_field"],
                      id_field=metadata["id_field"])
        
    def get_ids(self):
        return self.df[self.id_field]
    
    def get_types(self):
        return self.df[self.type_field]
    
    def get_start_times(self):
        return self.df[self.start_time_field]
    
    def get_end_times(self):
        return self.df[self.end_time_field]
    
    def filter(self, mask):
        """Returns a new Intervals with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return IntervalSet(self.df[mask],
                      type_field=self.type_field,
                      start_time_field=self.start_time_field,
                      end_time_field=self.end_time_field,
                      id_field=self.id_field,
                      value_field=self.value_field)
    
    def has(self, eventtype): return (self.df[self.type_field] == eventtype).sum() > 0
    
    def get(self, eventtype):
        new_df = self.df[(self.df[self.type_field] == eventtype) if isinstance(eventtype, str) else (self.df[self.type_field].isin(eventtype))]
        return Intervals(new_df,
                      type_field=self.type_field, 
                      start_time_field=self.start_time_field,
                      end_time_field=self.end_time_field,
                      value_field=self.value_field,
                      id_field=self.id_field)

    def __repr__(self):
        return f"<IntervalSet: {len(self.df)} rows, {len(self.df[self.type_field].unique())} interval types>"
    
class Duration(TimeSeriesQueryable):
    def __init__(self, amount, unit="s"):
        if unit.lower() in ("day", "d", "days"):
            self._value = amount * 3600 * 24
        elif unit.lower() in ("hour", "h", "hr", "hours", "hrs"):
            self._value = amount * 3600
        elif unit.lower() in ("minute", "min", "mins", "minutes", "m"):
            self._value = amount * 60
        elif unit.lower() in ("second", "sec", "seconds", "secs", "s"):
            self._value = amount
        else:
            raise ValueError(f"Unrecognized unit '{unit}'")
        
    def value(self):
        return self._value
    
    def __repr__(self):
        return f"<Duration {self.value()}s>"
    
    def __abs__(self): return Duration(self.value().__abs__())
    def __neg__(self): return Duration(self.value().__neg__())
    def __pos__(self): return Duration(self.value().__pos__())
    def __invert__(self): return Duration(self.value().__invert__())

    def _handle_binary_op(self, opname, other):
        if isinstance(other, (Events, Attributes, Intervals, TimeIndex, TimeSeries)):
            return NotImplemented
        if isinstance(other, Duration):
            return Duration(getattr(self.value(), opname)(other.value()))
        return Duration(getattr(self.value(), opname)(other))
    
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)
    
    def __add__(self, other): return self._handle_binary_op("__add__", other)
    def __and__(self, other): return self._handle_binary_op("__and__", other)
    def __floordiv__(self, other): return self._handle_binary_op("__floordiv__", other)
    def __mod__(self, other): return self._handle_binary_op("__mod__", other)
    def __mul__(self, other): return self._handle_binary_op("__mul__", other)
    def __or__(self, other): return self._handle_binary_op("__or__", other)
    def __pow__(self, other): return self._handle_binary_op("__pow__", other)
    def __sub__(self, other): return self._handle_binary_op("__sub__", other)
    def __truediv__(self, other): return self._handle_binary_op("__truediv__", other)
    def __xor__(self, other): return self._handle_binary_op("__xor__", other)

    def __radd__(self, other): return self._handle_binary_op("__radd__", other)
    def __rand__(self, other): return self._handle_binary_op("__rand__", other)
    def __rdiv__(self, other): return self._handle_binary_op("__rdiv__", other)
    def __rfloordiv__(self, other): return self._handle_binary_op("__rfloordiv__", other)
    def __rmatmul__(self, other): return self._handle_binary_op("__rmatmul__", other)
    def __rmod__(self, other): return self._handle_binary_op("__rmod__", other)
    def __rmul__(self, other): return self._handle_binary_op("__rmul__", other)
    def __ror__(self, other): return self._handle_binary_op("__ror__", other)
    def __rpow__(self, other): return self._handle_binary_op("__rpow__", other)
    def __rsub__(self, other): return self._handle_binary_op("__rsub__", other)
    def __rtruediv__(self, other): return self._handle_binary_op("__rtruediv__", other)
    def __rxor__(self, other): return self._handle_binary_op("__rxor__", other)
    

    
class TimeIndex(TimeSeriesQueryable):
    def __init__(self, timesteps, id_field="id", time_field="time"):
        """timesteps: a dataframe with instance ID and whose values indicate
            times in the instance's trajectory."""
        self.timesteps = timesteps.sort_values(id_field, kind='stable')
        self.id_field = id_field
        self.time_field = time_field
    
    def serialize(self):
        return {
            "type": "TimeIndex", 
            "time_field": self.time_field,
            "id_field": self.id_field,
        }, self.timesteps
    
    @staticmethod
    def deserialize(metadata, df):
        return TimeIndex(df,
                      time_field=metadata["time_field"],
                      id_field=metadata["id_field"])
            
    def __len__(self): return len(self.timesteps)
    
    def __eq__(self, other): return (
        isinstance(other, TimeIndex) and 
        (self.get_ids() == other.get_ids()).all() and
        (self.get_times() == other.get_times()).all()
    )
    
    def __ne__(self, other): return not (self == other)
    
    def get_ids(self):
        return self.timesteps[self.id_field]
    
    def get_times(self):
        return self.timesteps[self.time_field]
    
    def get_values(self):
        return self.timesteps[self.time_field]
    
    def filter(self, mask):
        """Returns a new time index with only steps for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return TimeIndex(self.timesteps[mask], id_field=self.id_field, time_field=self.time_field)
        
    @staticmethod
    def from_constant(ids, constant_time):
        """Constructs a time index with the same time for each ID."""
        return TimeIndex(pd.DataFrame({
            "id": ids,
            "time": [constant_time for _ in ids]
        }))
        
    @staticmethod
    def from_events(events, starts=None, ends=None, return_filtered_events=False):
        """Creates a time index from the timesteps and IDs represented in the given
        Events object"""
        event_times = events.df[[events.id_field, events.time_field]]
        mask = np.ones(len(event_times), dtype=bool)
        if starts is not None:
            mask &= event_times[events.time_field] >= make_aligned_value_series(events, starts)
        if ends is not None:
            mask &= event_times[events.time_field] < make_aligned_value_series(events, ends)
        mask &= ~event_times.duplicated([events.id_field, events.time_field])
        result = TimeIndex(event_times[mask].reset_index(drop=True), 
                         id_field=events.id_field, 
                         time_field=events.time_field)
        if return_filtered_events:
            return result, events.filter(mask)
        return result
        
    @staticmethod
    def from_attributes(attributes, id_field="id"):
        """Creates a time index from the timesteps and IDs represented in the given
        Attributes object (one per ID)"""
        attribute_df = pd.DataFrame({id_field: attributes.series.index, attributes.name: attributes.series.reset_index(drop=True)})
        return TimeIndex(attribute_df, 
                         id_field=id_field, 
                         time_field=attributes.name)
        
    @staticmethod
    def from_times(times):
        """Constructs a time index from a series of other time indexes, Attributes, or Events."""
        # Concatenate all the time indexes together, then re-sort
        indexes = []
        for time_element in times:
            if isinstance(time_element, (Events, EventSet)):
                indexes.append(TimeIndex.from_events(time_element))
            elif isinstance(time_element, Attributes):
                indexes.append(TimeIndex.from_attributes(time_element))
            elif isinstance(time_element, TimeIndex):
                indexes.append(time_element)
            elif isinstance(time_element, (TimeSeries, TimeSeriesSet)):
                indexes.append(time_element.index)
            else:
                raise ValueError(f"Unsupported argument of type '{type(time_element)}' for from_times")
        return TimeIndex(pd.DataFrame({
            "id": np.concatenate([i.get_ids().values for i in indexes]),
            "time": np.concatenate([i.get_times().values for i in indexes]),
        }).sort_values(["id", "time"]))
        
    @staticmethod
    def range(starts, ends, interval=Duration(1, 'hr')):
        """Creates a time index where each timestep is interval apart starting
        from each start to each end"""
        if not all(starts.get_ids() == ends.get_ids()):
            raise ValueError(f"Starts and ends must match IDs exactly")
        
        combined = pd.DataFrame({starts.id_field: starts.get_ids(), "start": starts.get_times(), "end": ends.get_times()})
        # remove nan times
        combined = combined.dropna(axis=0)
        start_df = (combined
            .apply(lambda row: pd.Series({starts.id_field: row["id"], starts.time_field: np.arange(row["start"], row["end"], interval.value())}), axis=1)
            .explode(starts.time_field)
            .reset_index(drop=True))
        # Remove timesteps where no value is present
        start_df = start_df[~pd.isna(start_df[starts.time_field])]
        start_df[starts.time_field] = start_df[starts.time_field].astype(np.int64)
        return TimeIndex(start_df, id_field=starts.id_field, time_field=starts.time_field)
        
    def add(self, duration, invert_self=False):
        """duration: either a Duration or an Attributes containing durations in
            seconds"""
        if isinstance(duration, Duration):
            return TimeIndex(self.timesteps.assign(**{self.time_field: self.timesteps[self.time_field] + duration.value()}),
                             id_field=self.id_field,
                             time_field=self.time_field)
        elif isinstance(duration, Attributes):
            increments = pd.merge(duration.series, self.timesteps, how='right', left_index=True, right_on=self.id_field)
            return TimeIndex(self.timesteps.assign(**{self.time_field: self.timesteps[self.time_field] + increments[duration.series.name]}),
                             id_field=self.id_field,
                             time_field=self.time_field)
        elif hasattr(duration, "get_values"):
            # Create a TimeSeries containing the result of subtracting the given value from the times
            return TimeSeries(self, (-1 if invert_self else 1) * self.timesteps[self.time_field] + duration.get_values())
        else:
            return NotImplemented
    
    def subtract(self, duration):
        """duration: either a Duration or an Attributes containing durations in
            seconds"""
        if isinstance(duration, Duration):
            return TimeIndex(self.timesteps.assign(**{self.time_field: self.timesteps[self.time_field] - duration.value()}),
                             id_field=self.id_field,
                             time_field=self.time_field)
        elif isinstance(duration, Attributes):
            increments = pd.merge(duration.series.rename("__merged_duration"), self.timesteps, how='right', left_index=True, right_on=self.id_field)
            return TimeIndex(self.timesteps.assign(**{self.time_field: self.timesteps[self.time_field] - increments["__merged_duration"]}),
                             id_field=self.id_field,
                             time_field=self.time_field)
        elif hasattr(duration, "get_values"):
            # Create a TimeSeries containing the result of subtracting the given value from the times
            return TimeSeries(self, self.timesteps[self.time_field].reset_index(drop=True) - duration.get_values().reset_index(drop=True))
        else:
            return NotImplemented

    def __len__(self):
        return len(self.timesteps)
    
    def __repr__(self):
        return f"<TimeIndex: {len(self.timesteps[self.id_field].unique())} IDs, {len(self.timesteps)} steps>\n{repr(self.timesteps)}"

    def with_times(self, new_times):
        return TimeIndex(self.timesteps.assign(**{self.time_field: new_times}),
                         id_field=self.id_field,
                         time_field=self.time_field)
        
    def __getattr__(self, name):
        value_series = self.timesteps[self.time_field]
        if hasattr(value_series, name) and name not in EXCLUDE_SERIES_METHODS:
            pd_method = getattr(value_series, name)
            if callable(pd_method):
                def wrap_pandas_method(*args, **kwargs):
                    args = [make_aligned_value_series(self, a) if isinstance(a, TimeSeriesQueryable) else a for a in args]
                    kwargs = {k: make_aligned_value_series(self, v) if isinstance(v, TimeSeriesQueryable) else v for k, v in kwargs.items()}
                    result = pd_method(*args, **kwargs)
                    if isinstance(result, pd.Series) and (value_series.index == result.index).all():
                        return self.with_times(result)
                    if isinstance(result, pd.Series):
                        raise ValueError(f"Cannot complete pandas method call '{name}' on {type(self)} because it returned a Series that isn't aligned with the original Series.")
                    return result
            return wrap_pandas_method
        raise AttributeError(name)

    def __abs__(self): return self.with_times(self.get_times().__abs__())
    def __neg__(self): return self.with_times(self.get_times().__neg__())
    def __pos__(self): return self.with_times(self.get_times().__pos__())

    def __add__(self, other): return self.add(other)
    def __sub__(self, other): return self.subtract(other)

    def __radd__(self, other): return self.add(other)
    def __rsub__(self, other): return self.add(other, invert_self=True)

    def _handle_binary_op(self, opname, other):
        if isinstance(other, Compilable): return NotImplemented
        return TimeSeries(self, getattr(self.timesteps[self.time_field], opname)(make_aligned_value_series(self, other)))
    
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)    
    

class TimeSeries(TimeSeriesQueryable):
    def __init__(self, index, series):
        """
        index: a TimeIndex
        series: a pandas Series containing values of the same length as the TimeIndex.
            The series name will be used as the time series' name; the series index
            will not be used.
        """
        self.index = index
        self.series = series
        self.name = self.series.name
        assert len(self.index) == len(self.series)
        
    def serialize(self, include_index=True):
        if include_index:
            index_meta, index_df = self.index.serialize()
            return {
                "type": "TimeSeries", 
                "name": self.name,
                "index_meta": index_meta
            }, pd.concat([index_df.reset_index(drop=True),
                        pd.DataFrame(self.series).reset_index(drop=True)], axis=1)
        else:
            return {
                "type": "TimeSeries", 
                "name": self.name,
            }, pd.DataFrame(self.series.reset_index(drop=True))
    
    @staticmethod
    def deserialize(metadata, df, index=None):
        if index is not None:
            return TimeSeries(index, df[df.columns[0]].rename(metadata["name"]))
        
        index = TimeIndex.deserialize(metadata["index_meta"], df[df.columns[:2]])
        return TimeSeries(index, df[df.columns[2]])
    
    def filter(self, mask):
        """Returns a new time series with an updated index and values with only
        values for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return TimeSeries(self.index.filter(mask), self.series[mask].reset_index(drop=True))
        
    def __len__(self):
        return len(self.series)
    
    def __repr__(self):
        return f"<TimeSeries {self.name}: {len(self.series)} rows>\n{repr(self.index.timesteps.assign(**{self.series.name or 'Series': self.series.values}))}"
    
    def compress(self):
        """Returns a new TimeSeries with values compressed to the minimum size
        needed to represent them."""
        return self.with_values(compress_series(self.get_values()))
        
    def to_events(self):
        return Events(self.index.timesteps.reset_index(drop=True).assign(
            eventtype=self.name or "timeseries_event", 
            value=self.series.reset_index(drop=True)),
                      time_field=self.index.time_field,
                      id_field=self.index.id_field)
        
    def get_ids(self):
        return self.index.get_ids()

    def get_times(self):
        return self.index.get_times()
    
    def get_values(self):
        return self.series
    
    def preserve_nans(self, new_values):
        return new_values.where(~pd.isna(self.get_values()), pd.NA)
        
    def with_values(self, new_values, preserve_nans=False):
        return TimeSeries(self.index, self.preserve_nans(new_values) if preserve_nans else new_values)
    
    def carry_forward_steps(self, steps):
        """Carries forward by the given number of timesteps."""
        return self.with_values(self.series.reset_index(drop=True).groupby(self.index.get_ids().reset_index(drop=True)).ffill(limit=steps))
    
    def carry_forward_duration(self, duration):
        """Carries forward by the given amount of time (if the start time of the
        time series element falls within duration of the start time of the last
        non-nan element)."""
        if isinstance(duration, Duration): duration = duration.value()
        try:
            float(duration)
        except ValueError:
            raise ValueError(f"carry_forward_duration requires a scalar value or Duration")
        if not pd.api.types.is_numeric_dtype(self.series.dtype):
            # Convert to numbers before using numba
            codes, uniques = pd.factorize(self.series)
            codes = np.where(pd.isna(self.series), np.nan, codes).astype(np.float64)
        else:
            codes = self.series.values.astype(np.float64)
            uniques = None
        result = numba_carry_forward(List(self.index.get_ids().values.tolist()), 
                                     np.array(self.index.get_times().values, dtype=np.int64), 
                                     codes, 
                                     duration)
        if uniques is not None:
            result = np.where(np.isnan(result), np.nan, uniques[np.where(np.isnan(result), -1, result).astype(int)])
        return self.with_values(pd.Series(result, index=self.series.index, name=self.series.name).replace(np.nan, pd.NA))

    def aggregate(self, start_times, end_times, agg_func):
        """
        Performs an aggregation that returns a single value per ID. Returns an
        Attributes object.
        """
        # Construct an events object using the time series index time as the time
        events = Events(pd.DataFrame({
            "id": self.index.get_ids(),
            "time": self.index.get_times(),
            "eventtype": self.name or "event",
            "value": self.series.values
        }))
        result = events.bin_aggregate(start_times, start_times, end_times, agg_func)
        return Attributes(result.series.set_axis(result.index.get_ids()))
    
    def __getattr__(self, name):
        if hasattr(self.series, name) and name not in EXCLUDE_SERIES_METHODS:
            pd_method = getattr(self.series, name)
            if callable(pd_method):
                def wrap_pandas_method(*args, **kwargs):
                    args = [make_aligned_value_series(self, a) if isinstance(a, TimeSeriesQueryable) else a for a in args]
                    kwargs = {k: make_aligned_value_series(self, v) if isinstance(v, TimeSeriesQueryable) else v for k, v in kwargs.items()}
                    result = pd_method(*args, **kwargs)
                    if isinstance(result, pd.Series) and len(self.series) == len(result):
                        return TimeSeries(self.index, result)
                    if isinstance(result, pd.Series):
                        raise ValueError(f"Cannot complete pandas method call '{name}' on {type(self)} because it returned a Series that isn't aligned with the original Series.")
                    return result
            return wrap_pandas_method
        raise AttributeError(name)
    
    def __abs__(self): return self.with_values(self.series.__abs__(), preserve_nans=True)
    def __neg__(self): return self.with_values(self.series.__neg__(), preserve_nans=True)
    def __pos__(self): return self.with_values(self.series.__pos__(), preserve_nans=True)
    def __invert__(self): return self.with_values(self.series.__invert__(), preserve_nans=True)
    
    def _handle_binary_op(self, opname, other):
        if isinstance(other, (Events, Intervals, TimeIndex, Compilable)):
            return NotImplemented
        if isinstance(other, Attributes):
            return self.with_values(getattr(self.series, opname)(make_aligned_value_series(self, other)), preserve_nans=True)
        if isinstance(other, TimeSeries):
            return self.with_values(getattr(self.series, opname)(other.series), preserve_nans=True)
        if isinstance(other, Duration):
            return self.with_values(getattr(self.series, opname)(other.value()), preserve_nans=True)
        return self.with_values(getattr(self.series, opname)(other), preserve_nans=True)
        
    def __eq__(self, other): return self._handle_binary_op("__eq__", other)
    def __ge__(self, other): return self._handle_binary_op("__ge__", other)
    def __gt__(self, other): return self._handle_binary_op("__gt__", other)
    def __le__(self, other): return self._handle_binary_op("__le__", other)
    def __ne__(self, other): return self._handle_binary_op("__ne__", other)
    def __lt__(self, other): return self._handle_binary_op("__lt__", other)
    
    def __add__(self, other): return self._handle_binary_op("__add__", other)
    def __and__(self, other): return self._handle_binary_op("__and__", other)
    def __floordiv__(self, other): return self._handle_binary_op("__floordiv__", other)
    def __mod__(self, other): return self._handle_binary_op("__mod__", other)
    def __mul__(self, other): return self._handle_binary_op("__mul__", other)
    def __or__(self, other): return self._handle_binary_op("__or__", other)
    def __pow__(self, other): return self._handle_binary_op("__pow__", other)
    def __sub__(self, other): return self._handle_binary_op("__sub__", other)
    def __truediv__(self, other): return self._handle_binary_op("__truediv__", other)
    def __xor__(self, other): return self._handle_binary_op("__xor__", other)

    def __radd__(self, other): return self._handle_binary_op("__radd__", other)
    def __rand__(self, other): return self._handle_binary_op("__rand__", other)
    def __rdiv__(self, other): return self._handle_binary_op("__rdiv__", other)
    def __rfloordiv__(self, other): return self._handle_binary_op("__rfloordiv__", other)
    def __rmatmul__(self, other): return self._handle_binary_op("__rmatmul__", other)
    def __rmod__(self, other): return self._handle_binary_op("__rmod__", other)
    def __rmul__(self, other): return self._handle_binary_op("__rmul__", other)
    def __ror__(self, other): return self._handle_binary_op("__ror__", other)
    def __rpow__(self, other): return self._handle_binary_op("__rpow__", other)
    def __rsub__(self, other): return self._handle_binary_op("__rsub__", other)
    def __rtruediv__(self, other): return self._handle_binary_op("__rtruediv__", other)
    def __rxor__(self, other): return self._handle_binary_op("__rxor__", other)


class TimeSeriesSet:
    def __init__(self, index, values):
        """
        index: a TimeIndex
        values: a DataFrame containing the same number of rows as 
        """
        self.index = index
        self.values = values
        assert len(self.index) == len(self.values)

    def serialize(self, include_index=True):
        if include_index:
            index_meta, index_df = self.index.serialize()
            return {
                "type": "TimeSeriesSet", 
                "index_meta": index_meta
            }, pd.concat([index_df.reset_index(drop=True),
                        self.values.reset_index(drop=True)], axis=1)
        else:
            return {
                "type": "TimeSeriesSet", 
            }, self.values.reset_index(drop=True)

    @staticmethod
    def deserialize(metadata, df, index=None):
        if index is not None:
            return TimeSeriesSet(index, df)
        index = TimeIndex.deserialize(metadata["index_meta"], df[df.columns[:2]])
        return TimeSeriesSet(index, df[df.columns[2:]])
        
    def filter(self, mask):
        """Returns a new time series set with an updated index and values with only
        values for which the mask is True."""
        if hasattr(mask, "get_values"): mask = mask.get_values()
        return TimeSeriesSet(self.index.filter(mask), self.values[mask].reset_index(drop=True))
        
    @staticmethod
    def from_series(time_series):
        """Creates a time series set from a list of TimeSeries objects with
        the same index"""
        if len(time_series) == 0:
            raise ValueError("Need at least 1 time series")
        for series in time_series:
            assert (isinstance(series, TimeSeries) and 
                    (series.index.get_ids().values == time_series[0].index.get_ids().values).all() and
                    (series.index.get_times().values == time_series[0].index.get_times().values).all()), "TimeSeries must be identically indexed"
        return TimeSeriesSet(time_series[0].index, 
                             pd.DataFrame({series.name or i: series.series for i, series in enumerate(time_series)}))
        
    def compress(self):
        """Returns a new TimeSeries with values compressed to the minimum size
        needed to represent them."""
        return TimeSeriesSet(self.index, pd.DataFrame({col: compress_series(self.values[col]) for col in self.values.columns}))
        
    def has(self, col): return col in self.values.columns
    
    def get(self, col):
        if col not in self.values:
            raise ValueError(f"TimeSeriesSet has no column named '{col}'")
        return TimeSeries(self.index, self.values[col])
    
    def __len__(self):
        return len(self.values)
    
    def __repr__(self):
        return (f"<TimeSeriesSet: {len(self.values)} rows, {len(self.values.columns)} columns>" + 
                f"\n{repr(pd.concat([self.index.timesteps.reset_index(drop=True), self.values.reset_index(drop=True)], axis=1))}")
    
if __name__ == "__main__":
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
    } for _ in range(50)]))
    
    intervals = IntervalSet(pd.DataFrame([{
        'id': np.random.choice(ids),
        'starttime': np.random.randint(0, 50),
        'endtime': np.random.randint(50, 100),
        'intervaltype': np.random.choice(['i1', 'i2']),
        'value': np.random.uniform(0, 100)
    } for _ in range(10)]))
    
    # print(events.get('e1').df, attributes.get('a2').fillna(0).series)
    # print((attributes.get('a2').fillna(0) < events.get('e1')).df)
    
    # print(intervals.get('i1').df)
    
    start_times = TimeIndex.from_attributes(attributes.get('start'))
    end_times = TimeIndex.from_attributes(attributes.get('end'))
    times = TimeIndex.range(start_times, end_times, Duration(30))
    
    print(events.get('e1'))
    compiled_expression = intervals.get('i1') - Compilable(events.get('e1').bin_aggregate(
        times,
        times - Duration(30),
        times,
        "last"
    ))

    # compiled_expression = Compilable(events.get('e1')).filter(events.get('e1') < Compilable(events.get('e1').bin_aggregate(
    #     times,
    #     times - Duration(30),
    #     times,
    #     "mean"
    # )))

    print(compiled_expression.bin_aggregate(
        times,
        times, times + Duration(30),
        "amount",
        "sum"
    ))
    
QUERY_RESULT_TYPENAMES = {
    Attributes: "Attributes",
    Events: "Events",
    Intervals: "Intervals",
    AttributeSet: "Attribute Set",
    EventSet: "Event Set",
    IntervalSet: "Interval Set",
    TimeIndex: "Time Index",
    TimeSeries: "Time Series",
    TimeSeriesSet: "Time Series Set"
}