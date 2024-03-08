from numba import njit, jit
import numpy as np

@njit 
def numba_sum(x, t0, t1): return np.nansum(x) if len(x[~np.isnan(x)]) else np.nan
@njit 
def numba_mean(x, t0, t1): return np.nanmean(x) if len(x[~np.isnan(x)]) else np.nan
@njit 
def numba_median(x, t0, t1): return np.nanmedian(x) if len(x[~np.isnan(x)]) else np.nan
@njit 
def numba_min(x, t0, t1): return np.nanmin(x) if len(x[~np.isnan(x)]) else np.nan
@njit 
def numba_max(x, t0, t1): return np.nanmax(x) if len(x[~np.isnan(x)]) else np.nan
@jit(nopython=False)
def numba_first(x, t0, t1): return x[~np.isnan(x)][0] if (~np.isnan(x)).sum() else np.nan
@jit(nopython=False)
def numba_last(x, t0, t1): return x[~np.isnan(x)][-1] if (~np.isnan(x)).sum() else np.nan
@jit(nopython=False)
def numba_exists(x, t0, t1): return 1.0 if len(x) else 0.0
@jit(nopython=False)
def numba_exists_nonnull(x, t0, t1): return 1.0 if len(x[~np.isnan(x)]) else 0.0
@jit(nopython=False)
def numba_count(x, t0, t1): return len(x)
@jit(nopython=False)
def numba_count_distinct(x, t0, t1): return len(set(x))
@jit(nopython=False)
def numba_count_nonnull(x, t0, t1): return len(x[~np.isnan(x)])
@jit(nopython=False)
def numba_count_distinct_nonnull(x, t0, t1): return len(set(x[~np.isnan(x)]))
@njit
def numba_integral(x, t0, t1): return np.nansum(x) * (t1 - t0) if len(x) else np.nan
    
AGG_FUNCTIONS = {
    "sum": numba_sum,
    "mean": numba_mean,
    "median": numba_median,
    "min": numba_min,
    "max": numba_max,
    "first": numba_first,
    "last": numba_last,
    "exists": numba_exists,
    "exists nonnull": numba_exists_nonnull,
    "count": numba_count,
    "count distinct": numba_count_distinct,
    "count distinct nonnull": numba_count_distinct_nonnull,
    "count nonnull": numba_count_nonnull,
    "integral": numba_integral
}

CATEGORICAL_SUPPORT_AGG_FUNCTIONS = {"first", "last", "exists", "exists nonnull", "count", "count distinct", "count distinct nonnull", "count nonnull"}
TYPE_PRESERVING_AGG_FUNCTIONS = {"sum", "mean", "median", "min", "max", "first", "last", "integral"}

def convert_numba_result_dtype(x, agg_func):
    """Converts the output of a numba join events/intervals call to the right dtype."""
    if agg_func == "exists":
        return (np.array(x) > 0)
    elif agg_func == "count":
        x = np.array(x)
        return np.where(~np.isnan(x), x, 0).astype(np.int64)
    return np.array(x)

@njit
def numba_join_events(ids, starts, ends, event_ids, event_times, event_values, agg_func):
    """
    Assumes both sets of IDs are in sorted order.
    
    Returns a matrix with 2 columns, time_idx and event_idx, where time_idx
    is an index into the starts/ends arrays and event_idx is an index into the
    event_times array.
    """
    last_id = None
    current_id_time_start = 0
    current_id_event_start = 0
    
    grouped_values = [np.float64(x) for x in range(0)]
    
    for i in range(len(ids) + 1):
        if i >= len(ids) or (last_id is not None and ids[i] != last_id):
            
            j = current_id_event_start
            while j < len(event_ids) and event_ids[j] < last_id:
                j += 1
                
            current_id_event_start = j
            
            if i >= len(ids) or event_ids[j] == last_id:
                
                while j < len(event_ids) and event_ids[j] == last_id:
                    j += 1
                current_id_event_end = j
                
                time_idxs = np.arange(current_id_time_start, i)
                event_idxs = np.arange(current_id_event_start, current_id_event_end, dtype=np.int64)
                
                # Match the events and times together
                for t in time_idxs:
                    matched_idxs = event_idxs[np.logical_and(event_times[event_idxs] >= starts[t],
                                                            event_times[event_idxs] < ends[t])]
                    if len(matched_idxs) == 0:
                        matched_values = np.empty((0,), dtype=np.float64)
                    else:
                        matched_values = event_values[matched_idxs]
                    grouped_values.append(agg_func(matched_values, starts[t], ends[t]))
            else:
                time_idxs = list(range(current_id_time_start, i))
                grouped_values += [agg_func(np.empty((0,), dtype=np.float64), 0, 0)] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return grouped_values
    
# Use pyobject mode if needed to work with string arrays
@jit(nopython=False)
def numba_join_events_dynamic(ids, starts, ends, value_fn, event_ids, event_times, event_values, preaggregated_values, agg_func):
    """
    Performs a bin aggregation using a dynamic function to compute the values being
    aggregated at each point in the index.
    
    Assumes both sets of IDs are in sorted order.
    
    Returns a matrix with 2 columns, time_idx and event_idx, where time_idx
    is an index into the starts/ends arrays and event_idx is an index into the
    event_times array.
    """
    last_id = None
    current_id_time_start = 0
    current_id_event_start = 0
    
    grouped_values = [np.float64(x) for x in range(0)]
    
    for i in range(len(ids) + 1):
        if i >= len(ids) or (last_id is not None and ids[i] != last_id):
            
            j = current_id_event_start
            while j < len(event_ids) and event_ids[j] < last_id:
                j += 1
                
            current_id_event_start = j
            
            if i >= len(ids) or event_ids[j] == last_id:
                
                while j < len(event_ids) and event_ids[j] == last_id:
                    j += 1
                current_id_event_end = j
                
                time_idxs = np.arange(current_id_time_start, i)
                event_idxs = np.arange(current_id_event_start, current_id_event_end, dtype=np.int64)
                
                # Match the events and times together
                for t in time_idxs:
                    matched_idxs = event_idxs[np.logical_and(event_times[event_idxs] >= starts[t],
                                                            event_times[event_idxs] < ends[t])]
                    if len(matched_idxs) == 0:
                        grouped_values.append(agg_func(np.empty((0,), dtype=np.float64), starts[t], ends[t]))
                    else:
                        matched_values = event_values[matched_idxs]
                        transformed_values = value_fn(event_times[matched_idxs], matched_values, preaggregated_values[len(grouped_values)])
                        grouped_values.append(agg_func(transformed_values, starts[t], ends[t]))    
                        
            else:
                time_idxs = list(range(current_id_time_start, i))
                grouped_values += [agg_func(np.empty((0,), dtype=np.float64), 0, 0)] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return grouped_values
    
@njit
def numba_join_intervals(ids, starts, ends, interval_ids, interval_starts, interval_ends, interval_values, agg_type, agg_func):
    """
    Assumes both sets of IDs are in sorted order.
    
    Returns a matrix with 2 columns, time_idx and interval_idx, where time_idx
    is an index into the starts/ends arrays and interval_idx is an index into the
    interval_starts/interval_ends array. Intervals are included in each time slot
    if they overlap at all with the slot.
    """
    last_id = None
    current_id_time_start = 0
    current_id_event_start = 0
    
    grouped_values = [np.float64(x) for x in range(0)]
    
    for i in range(len(ids) + 1):
        if i >= len(ids) or (last_id is not None and ids[i] != last_id):
            
            j = current_id_event_start
            while j < len(interval_ids) and interval_ids[j] < last_id:
                j += 1
                
            current_id_event_start = j
            
            if i >= len(ids) or interval_ids[j] == last_id:
                
                while j < len(interval_ids) and interval_ids[j] == last_id:
                    j += 1
                current_id_event_end = j
                
                time_idxs = np.arange(current_id_time_start, i)
                event_idxs = np.arange(current_id_event_start, current_id_event_end)
                
                # Match the events and times together
                for t in time_idxs:
                    matched_idxs = event_idxs[np.logical_and(interval_starts[event_idxs] < ends[t],
                                                            interval_ends[event_idxs] >= starts[t])]
                    if len(matched_idxs) == 0:
                        grouped_values.append(agg_func(np.empty((0,), dtype=np.float64), starts[t], ends[t]))
                    else:
                        matched_intervals = interval_values[matched_idxs]

                        if agg_type == "rate":
                            matched_intervals *= ((np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                   np.maximum(starts[t], interval_starts[matched_idxs])) / 
                                                  (ends[t] - starts[t]))
                        elif agg_type == "amount":
                            interval_durations = interval_ends[matched_idxs] - interval_starts[matched_idxs]
                            matched_intervals *= np.where(interval_durations == 0, 1,
                                                          ((np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                            np.maximum(starts[t], interval_starts[matched_idxs])) / 
                                                           np.maximum(interval_durations, 1)))
                        elif agg_type == "duration":
                            matched_intervals = (np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                   np.maximum(starts[t], interval_starts[matched_idxs]))
                            
                        grouped_values.append(agg_func(matched_intervals, starts[t], ends[t]))
            else:
                time_idxs = list(range(current_id_time_start, i))
                grouped_values += [agg_func(np.empty((0,), dtype=np.float64), 0, 0)] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return grouped_values
    
@jit(nopython=False)
def numba_join_intervals_dynamic(ids, starts, ends, value_fn, interval_ids, interval_starts, interval_ends, interval_values, preaggregated_values, agg_type, agg_func):
    """
    Performs a bin aggregation using a dynamic function to compute the values being
    aggregated at each point in the index.
    
    Assumes both sets of IDs are in sorted order.
    
    Returns a matrix with 2 columns, time_idx and interval_idx, where time_idx
    is an index into the starts/ends arrays and interval_idx is an index into the
    interval_starts/interval_ends array. Intervals are included in each time slot
    if they overlap at all with the slot.
    """
    last_id = None
    current_id_time_start = 0
    current_id_event_start = 0
    
    grouped_values = [np.float64(x) for x in range(0)]
    
    for i in range(len(ids) + 1):
        if i >= len(ids) or (last_id is not None and ids[i] != last_id):
            
            j = current_id_event_start
            while j < len(interval_ids) and interval_ids[j] < last_id:
                j += 1
                
            current_id_event_start = j
            
            if i >= len(ids) or interval_ids[j] == last_id:
                
                while j < len(interval_ids) and interval_ids[j] == last_id:
                    j += 1
                current_id_event_end = j
                
                time_idxs = np.arange(current_id_time_start, i)
                event_idxs = np.arange(current_id_event_start, current_id_event_end)
                
                # Match the events and times together
                for t in time_idxs:
                    matched_idxs = event_idxs[np.logical_and(interval_starts[event_idxs] < ends[t],
                                                            interval_ends[event_idxs] >= starts[t])]
                    if len(matched_idxs) == 0:
                        grouped_values.append(agg_func(np.empty((0,), dtype=np.float64), starts[t], ends[t]))
                    else:
                        matched_intervals = interval_values[matched_idxs]

                        transformed_values = value_fn(np.stack((interval_starts[matched_idxs], interval_ends[matched_idxs]), axis=1), 
                                                      matched_intervals, 
                                                      preaggregated_values[len(grouped_values)])
                        if agg_type == "rate":
                            transformed_values *= ((np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                   np.maximum(starts[t], interval_starts[matched_idxs])) / 
                                                  (ends[t] - starts[t]))
                        elif agg_type == "amount":
                            interval_durations = interval_ends[matched_idxs] - interval_starts[matched_idxs]
                            transformed_values *= np.where(interval_durations == 0, 1,
                                                          ((np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                            np.maximum(starts[t], interval_starts[matched_idxs])) / 
                                                           np.maximum(interval_durations, 1)))
                        elif agg_type == "duration":
                            transformed_values = (np.minimum(ends[t], interval_ends[matched_idxs]) - 
                                                   np.maximum(starts[t], interval_starts[matched_idxs]))
                            
                        grouped_values.append(agg_func(transformed_values, starts[t], ends[t]))
            else:
                time_idxs = list(range(current_id_time_start, i))
                grouped_values += [agg_func(np.empty((0,), dtype=np.float64), 0, 0)] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return grouped_values
    
@njit
def numba_carry_forward(ids, times, values, max_carry_time):
    """
    Returns a new numpy array with the given values, where values within an ID
    are carried forward by the given amount of time. Assumes IDs are in sorted
    order.
    """
    current_id = None
    new_values = values.copy()
    last_value = None
    last_time = None
    for i in range(len(ids)):
        if current_id is None:
            current_id = ids[i]
        elif current_id != ids[i]:
            last_value = None
            last_time = None
            current_id = ids[i]
                
        if np.isnan(values[i]):
            if last_value is not None and times[i] < last_time + max_carry_time:
                new_values[i] = last_value
            else:
                last_value = None
                last_time = None
        else:
            last_value = values[i]
            last_time = times[i]
            
    return new_values