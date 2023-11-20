from numba import njit
import numpy as np

# TODO: make all agg functions numba compiled so that aggregation can happen here
# instead of returning the full merged dataframe

@njit
def numba_join_events(ids, starts, ends, event_ids, event_times):
    """
    Assumes both sets of IDs are in sorted order.
    
    Returns a matrix with 2 columns, time_idx and event_idx, where time_idx
    is an index into the starts/ends arrays and event_idx is an index into the
    event_times array.
    """
    last_id = None
    current_id_time_start = 0
    current_id_event_start = 0
    
    all_time_indexes = [np.float64(x) for x in range(0)]
    all_event_indexes = [np.float64(x) for x in range(0)]
    
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
                event_idxs = np.arange(current_id_event_start, current_id_event_end)
                
                # Match the events and times together
                for t in time_idxs:
                    matched_events = event_idxs[np.logical_and(event_times[event_idxs] >= starts[t],
                                                            event_times[event_idxs] < ends[t])]
                    if len(matched_events) == 0:
                        all_time_indexes.append(t)
                        all_event_indexes.append(-1)
                    else:
                        all_time_indexes += [t] * len(matched_events)
                        all_event_indexes += list(matched_events)
            else:
                time_idxs = list(range(current_id_time_start, i))
                all_time_indexes += time_idxs
                all_event_indexes += [-1] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return all_time_indexes, all_event_indexes
    
@njit
def numba_join_intervals(ids, starts, ends, interval_ids, interval_starts, interval_ends):
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
    
    all_time_indexes = [np.float64(x) for x in range(0)]
    all_event_indexes = [np.float64(x) for x in range(0)]
    
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
                    matched_events = event_idxs[np.logical_and(interval_starts[event_idxs] < ends[t],
                                                            interval_ends[event_idxs] >= starts[t])]
                    if len(matched_events) == 0:
                        all_time_indexes.append(t)
                        all_event_indexes.append(-1)
                    else:
                        all_time_indexes += [t] * len(matched_events)
                        all_event_indexes += list(matched_events)
            else:
                time_idxs = list(range(current_id_time_start, i))
                all_time_indexes += time_idxs
                all_event_indexes += [-1] * len(time_idxs)
                
            current_id_time_start = i
            current_id_event_start = j
            
        if i < len(ids):
            last_id = ids[i]

    return all_time_indexes, all_event_indexes
    
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