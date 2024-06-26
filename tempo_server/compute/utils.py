import pandas as pd
import numpy as np
from tempo_server.query_language.data_types import *

class Commands:
    BUILD_DATASET = "build_dataset"
    TRAIN_MODEL = "train_model"
    FIND_SLICES = "find_slices"
    SUMMARIZE_DATASET = "summarize_dataset"
    GENERATE_QUERY_DOWNLOAD = "generate_query_download"
    
def make_series_summary(values, value_type=None):
    summary = {}
    if pd.isna(values).sum() > 0:
        summary["missingness"] = pd.isna(values).mean()
        values = values[~pd.isna(values)]
    
    if value_type is None:
        num_unique = len(np.unique(values))
        try:
            is_binary = num_unique == 2 and set(np.unique(values).astype(int).tolist()) == set([0, 1])
        except:
            is_binary = False
        try:
            values.astype(float)
            is_quantitative = True
        except:
            is_quantitative = False
        if is_binary: value_type = "binary"
        elif not is_quantitative or ((values.astype(int) == values).all() and num_unique <= 10): value_type = "categorical"
        else: value_type = "continuous"
    
    summary["type"] = value_type
    if value_type == "binary":
        summary["mean"] = values.astype(int).mean().astype(float)
    elif value_type == "categorical":
        uniques, counts = np.unique(values, return_counts=True)
        uniques_to_show = np.argsort(uniques)
        summary["counts"] = {str(uniques[i]): int(counts[i]) for i in uniques_to_show}
    else:
        summary["mean"] = np.mean(values.astype(float))
        summary["std"] = np.std(values.astype(float))
        
        min_val = values.min()
        max_val = values.max()
        data_range = max_val - min_val
        if data_range == 0:
            if min_val == 0: hist_bins = np.arange(0, 5)
            else: hist_bins = np.arange(min_val - 2, max_val + 3)
        else:
            bin_scale = np.floor(np.log10(data_range))
            if data_range / (10 ** bin_scale) < 2.5:
                bin_scale -= 1 # Make sure there aren't only 2-3 bins
            upper_tol = 2 if (np.ceil(max_val / (10 ** bin_scale))) * (10 ** bin_scale) == max_val else 1
            subdivide = 2 if data_range / (10 ** bin_scale) < 5 else 1
            hist_bins = np.arange(np.floor(min_val / (10 ** bin_scale)) * (10 ** bin_scale),
                                    (np.ceil(max_val / (10 ** bin_scale)) + upper_tol) * (10 ** bin_scale),
                                    10 ** bin_scale / subdivide)
        
        summary["hist"] = dict(zip(hist_bins.astype(float).tolist(), np.histogram(values, bins=hist_bins)[0].astype(int).tolist())) 
    return summary

def make_query_result_summary(dataset, query_result):
    """
    Supports describing Attributes, Events, Intervals, TimeSeries, and TimeSeriesSet.
    """
    base = {}
    if hasattr(query_result, "name"): base["name"] = query_result.name
    
    if isinstance(query_result, (Events, Intervals)):
        ids = pd.Series(dataset.get_ids(), name='id')
        sizes = query_result.get_values().groupby(query_result.get_ids()).size().rename("size")
        base["occurrences"] = make_series_summary(pd.merge(ids, sizes, left_on='id', right_index=True, how='left')["size"].fillna(0), value_type="continuous")
    if isinstance(query_result, Intervals):
        base["durations"] = make_series_summary(query_result.get_end_times() - query_result.get_start_times(), value_type="continuous")
    
    if hasattr(query_result, "get_values") and (~pd.isna(query_result.get_values())).sum() > 0:
        base["values"] = make_series_summary(query_result.get_values())
        
    return base

def make_query(variable_definitions, timestep_definition):
    """
    Constructs a query. variable_definitions should be a dictionary mapping
    variable names to dictionaries containing a "query" key. patient_cohort
    and timestep_definition should be strings.
    """
    variable_queries = ',\n\t'.join(f"{name}: {info['query']}" 
                                    for name, info in variable_definitions.items() 
                                    if info.get("enabled", True))
    return f"""
    (
        {variable_queries}
    )
    {timestep_definition}
    """
    
